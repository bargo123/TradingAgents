"""Pair dense and lexical projections into one catalog-authoritative generation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .catalog import KnowledgeCatalog
from .embeddings import EmbeddingSpecMismatch
from .lexical_index import LexicalIndexError, LexicalIndexReader, LexicalIndexWriter
from .models import ChunkRecord, EmbeddingSpec, IndexGeneration
from .vector_index import (
    VectorIndexError,
    VectorIndexReader,
    VectorIndexWriter,
    embedding_spec_row_values,
)


class IncompatibleIndexGeneration(RuntimeError):
    """The vector and lexical halves cannot safely be used as a pair."""


class IndexGenerationManager:
    """Build, validate, and atomically publish a matched local index generation."""

    def __init__(
        self,
        artifact_root: str | Path | None = None,
        *,
        catalog: KnowledgeCatalog | None = None,
        embedding_spec: EmbeddingSpec | None = None,
        vector_writer: VectorIndexWriter | None = None,
        lexical_writer: LexicalIndexWriter | None = None,
        lexical_index_version: str = "fts5-v1",
        lexical_tokenizer_settings: Mapping[str, Any] | None = None,
        index_version: str = "index-v1",
        config: Any | None = None,
    ) -> None:
        if config is not None:
            if artifact_root is not None:
                raise ValueError("pass either config or artifact_root, not both")
            artifact_root = config.artifact_root
            lexical_index_version = config.lexical_index_version
            lexical_tokenizer_settings = config.lexical_tokenizer_settings
            index_version = config.index_version
        if artifact_root is None:
            raise ValueError("artifact_root is required")
        self.artifact_root = Path(artifact_root)
        self.catalog = catalog or KnowledgeCatalog(self.artifact_root / "catalog.sqlite3")
        self.catalog.initialize()
        self.embedding_spec = embedding_spec or _embedding_spec_from_config(config)
        self.vector_writer = vector_writer or VectorIndexWriter()
        self.lexical_writer = lexical_writer or LexicalIndexWriter()
        self.lexical_index_version = str(lexical_index_version)
        self.lexical_tokenizer_settings = dict(lexical_tokenizer_settings or {})
        self.index_version = str(index_version)

    def active_generation(self) -> IndexGeneration | None:
        return self.catalog.active_generation()

    def build_and_activate(
        self, chunks: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]], generation_id: str
    ) -> IndexGeneration:
        generation = self.build_generation(chunks, vectors, generation_id)
        self.activate_generation(generation)
        return self.catalog.active_generation() or generation

    def build_generation(
        self, chunks: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]], generation_id: str
    ) -> IndexGeneration:
        generation_id = _safe_generation_id(generation_id)
        chunks = tuple(chunks)
        vectors = tuple(vectors)
        self._validate_input(chunks, vectors)
        final_vector = self.artifact_root / "vector" / "lancedb" / generation_id
        final_lexical = self.artifact_root / "keyword" / generation_id / "bm25.sqlite3"
        if final_vector.exists() or final_lexical.exists():
            raise IncompatibleIndexGeneration(f"generation already exists: {generation_id}")
        populations = _population_hashes(chunks)
        stage_root = Path(tempfile.mkdtemp(prefix=f"{generation_id}.", dir=self._staging_root()))
        stage_vector = stage_root / "vector" / "lancedb" / generation_id
        stage_lexical = stage_root / "keyword" / generation_id / "bm25.sqlite3"
        try:
            vector_metadata = self.vector_writer.write(
                stage_vector, chunks, vectors, generation_id=generation_id,
                embedding_spec=self.embedding_spec, lexical_index_version=self.lexical_index_version,
                index_version=self.index_version, **populations,
            )
            lexical_metadata = self.lexical_writer.write(
                stage_lexical, chunks, generation_id=generation_id,
                embedding_spec=self.embedding_spec, lexical_index_version=self.lexical_index_version,
                lexical_tokenizer_settings=self.lexical_tokenizer_settings,
                index_version=self.index_version, **populations,
            )
            staged = self._generation(
                generation_id, stage_vector, stage_lexical, chunks, populations,
                vector_metadata=vector_metadata, lexical_metadata=lexical_metadata,
            )
            self._validate_generation(staged, allow_staged_locations=True)
            final_vector.parent.mkdir(parents=True, exist_ok=True)
            final_lexical.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_vector, final_vector)
            os.replace(stage_lexical, final_lexical)
            generation = replace(staged, vector_location=final_vector, lexical_location=final_lexical)
            self.validate_generation(generation)
            return generation
        except (VectorIndexError, LexicalIndexError, IncompatibleIndexGeneration, EmbeddingSpecMismatch):
            raise
        except Exception as exc:
            raise IncompatibleIndexGeneration(f"could not stage generation {generation_id}: {exc}") from exc
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)

    def validate_generation(self, generation: IndexGeneration) -> None:
        self._validate_generation(generation, allow_staged_locations=False)

    def _validate_generation(
        self, generation: IndexGeneration, *, allow_staged_locations: bool
    ) -> None:
        if generation.embedding_spec != self.embedding_spec:
            raise EmbeddingSpecMismatch("generation embedding specification differs from the configured provider")
        if not generation.vector_ready or not generation.lexical_ready:
            raise IncompatibleIndexGeneration("generation must have both projections ready")
        if generation.status != "VALIDATED":
            raise IncompatibleIndexGeneration("generation must be validated before publication")
        if not generation.vector_location.is_dir() or not generation.lexical_location.is_file():
            raise IncompatibleIndexGeneration("generation projection locations are missing")
        self._validate_location_identity(generation, allow_staged_locations=allow_staged_locations)
        vector_reader = VectorIndexReader(
            generation.vector_location, backend=self.vector_writer.backend
        )
        vector = vector_reader.metadata()
        lexical = LexicalIndexReader(generation.lexical_location)
        lexical_metadata = lexical.metadata()
        required = {
            "generation_id": generation.generation_id,
            "population_hash": generation.population_hash,
            "document_count": generation.document_count,
            "chunk_count": generation.chunk_count,
            "embedding_spec": generation.embedding_spec.to_dict(),
            "lexical_index_version": generation.lexical_index_version,
            "index_version": generation.index_version,
        }
        for key, expected in required.items():
            if vector.get(key) != expected or lexical_metadata.get(key) != expected:
                raise IncompatibleIndexGeneration(f"projection metadata mismatch: {key}")
        population_identity = self._population_identity(generation)
        for key, expected in population_identity.items():
            if vector.get(key) != expected or lexical_metadata.get(key) != expected:
                raise IncompatibleIndexGeneration(f"projection population mismatch: {key}")
        if lexical.row_count() != generation.chunk_count:
            raise IncompatibleIndexGeneration("lexical row count does not match generation metadata")
        vector_rows = vector_reader.rows()
        if len(vector_rows) != generation.chunk_count:
            raise IncompatibleIndexGeneration("vector row count does not match generation metadata")
        self._validate_vector_rows(vector_rows, generation)
        self._validate_lexical_rows(lexical.provenance_rows(), vector_rows, generation)
        components = dict(generation.component_versions)
        if components.get("vector_generation_id") != generation.generation_id:
            raise IncompatibleIndexGeneration("vector generation identity differs from registry")
        if components.get("lexical_generation_id") != generation.generation_id:
            raise IncompatibleIndexGeneration("lexical generation identity differs from registry")
        if components.get("vector_population_hash") != generation.population_hash:
            raise IncompatibleIndexGeneration("vector population differs from registry")
        if components.get("lexical_population_hash") != generation.population_hash:
            raise IncompatibleIndexGeneration("lexical population differs from registry")

    def _validate_location_identity(
        self, generation: IndexGeneration, *, allow_staged_locations: bool
    ) -> None:
        vector_parts = generation.vector_location.parts
        lexical_parts = generation.lexical_location.parts
        if vector_parts[-3:] != ("vector", "lancedb", generation.generation_id):
            raise IncompatibleIndexGeneration("vector location does not identify its generation")
        if lexical_parts[-3:] != ("keyword", generation.generation_id, "bm25.sqlite3"):
            raise IncompatibleIndexGeneration("lexical location does not identify its generation")
        if allow_staged_locations:
            return
        expected_vector = self.artifact_root / "vector" / "lancedb" / generation.generation_id
        expected_lexical = self.artifact_root / "keyword" / generation.generation_id / "bm25.sqlite3"
        if (
            generation.vector_location.resolve() != expected_vector.resolve()
            or generation.lexical_location.resolve() != expected_lexical.resolve()
        ):
            raise IncompatibleIndexGeneration("projection locations do not match the active generation identity")

    @staticmethod
    def _population_identity(generation: IndexGeneration) -> dict[str, str]:
        try:
            parsed = json.loads(generation.population_identity or "")
        except json.JSONDecodeError as exc:
            raise IncompatibleIndexGeneration("generation population identity is invalid") from exc
        if not isinstance(parsed, dict):
            raise IncompatibleIndexGeneration("generation population identity is invalid")
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        if generation.population_identity != canonical:
            raise IncompatibleIndexGeneration("generation population identity is not canonical JSON")
        expected = {
            "population_hash": generation.population_hash,
            "document_population_hash": parsed.get("document_population_hash"),
            "chunk_population_hash": parsed.get("chunk_population_hash"),
        }
        if any(not isinstance(value, str) or not value.strip() for value in expected.values()):
            raise IncompatibleIndexGeneration("generation population identity is incomplete")
        return expected

    @staticmethod
    def _validate_vector_rows(
        vector_rows: Sequence[Mapping[str, Any]], generation: IndexGeneration
    ) -> None:
        expected_spec = embedding_spec_row_values(generation.embedding_spec)
        expected_spec_json = generation.embedding_spec.to_json()
        identities: set[tuple[str, str, str]] = set()
        for row in vector_rows:
            identity = _row_identity(row)
            if identity in identities:
                raise IncompatibleIndexGeneration("vector rows contain duplicate chunk provenance")
            identities.add(identity)
            if (
                row.get("projection_generation") != generation.generation_id
                or not row.get("source_filename")
                or not row.get("source_relative_path")
                or row.get("embedding_spec_json") != expected_spec_json
            ):
                raise IncompatibleIndexGeneration("vector provenance does not match generation")
            for field, expected in expected_spec.items():
                if row.get(field) != expected:
                    raise IncompatibleIndexGeneration(f"vector embedding specification mismatch: {field}")
        actual = _population_hashes_from_rows(vector_rows)
        if actual != IndexGenerationManager._population_identity(generation):
            raise IncompatibleIndexGeneration("vector rows do not match generation population identity")

    @staticmethod
    def _validate_lexical_rows(
        lexical_rows: Sequence[Mapping[str, Any]],
        vector_rows: Sequence[Mapping[str, Any]],
        generation: IndexGeneration,
    ) -> None:
        if len(lexical_rows) != generation.chunk_count:
            raise IncompatibleIndexGeneration("lexical provenance row count does not match generation metadata")
        expected_population = IndexGenerationManager._population_identity(generation)
        vector_by_chunk = {str(row.get("chunk_id")): _row_identity(row) for row in vector_rows}
        lexical_by_chunk: dict[str, tuple[str, str, str]] = {}
        for lexical_row in lexical_rows:
            provenance = lexical_row.get("provenance")
            if not isinstance(provenance, Mapping):
                raise IncompatibleIndexGeneration("lexical provenance is invalid")
            identity = _row_identity(lexical_row)
            chunk_id = identity[0]
            if chunk_id in lexical_by_chunk or vector_by_chunk.get(chunk_id) != identity:
                raise IncompatibleIndexGeneration("lexical provenance rows do not match vector rows")
            if (
                lexical_row.get("generation_id") != generation.generation_id
                or lexical_row.get("content_type") != provenance.get("content_type")
                or provenance.get("chunk_id") != chunk_id
                or provenance.get("document_id") != identity[1]
                or provenance.get("source_hash") != identity[2]
                or provenance.get("projection_generation") != generation.generation_id
                or provenance.get("projection_population_hash") != generation.population_hash
                or not provenance.get("source_filename")
                or not provenance.get("source_relative_path")
            ):
                raise IncompatibleIndexGeneration("lexical provenance does not match generation")
            lexical_by_chunk[chunk_id] = identity
        if set(lexical_by_chunk) != set(vector_by_chunk):
            raise IncompatibleIndexGeneration("lexical and vector chunk populations differ")
        actual = _population_hashes_from_rows(lexical_by_chunk.values())
        if actual != expected_population:
            raise IncompatibleIndexGeneration("lexical rows do not match generation population identity")

    def activate_generation(self, generation: IndexGeneration) -> None:
        self.validate_generation(generation)
        # The catalog commit is authority.  The pointer is a replaceable
        # projection, so a crash between them is repaired by resolve().
        self.catalog.set_active_generation(generation)
        self._write_active_pointer(generation)

    def resolve_active_generation(self) -> IndexGeneration:
        generation = self.catalog.active_generation()
        if generation is None:
            raise IncompatibleIndexGeneration("no active knowledge index generation")
        self.validate_generation(generation)
        pointer = self._read_active_pointer()
        if pointer != generation.to_dict():
            self._write_active_pointer(generation)
        return generation

    def _generation(
        self,
        generation_id: str,
        vector_location: Path,
        lexical_location: Path,
        chunks: Sequence[ChunkRecord],
        populations: Mapping[str, str],
        *,
        vector_metadata: Mapping[str, Any],
        lexical_metadata: Mapping[str, Any],
    ) -> IndexGeneration:
        population_hash = populations["population_hash"]
        components = {
            "vector": str(vector_metadata["component_identity"]),
            "lexical": str(lexical_metadata["component_identity"]),
            "vector_generation_id": generation_id,
            "lexical_generation_id": generation_id,
            "vector_population_hash": population_hash,
            "lexical_population_hash": population_hash,
        }
        identity = json.dumps(
            {
                "document_population_hash": populations["document_population_hash"],
                "chunk_population_hash": populations["chunk_population_hash"],
            }, sort_keys=True, separators=(",", ":")
        )
        return IndexGeneration(
            generation_id=generation_id,
            vector_location=vector_location,
            lexical_location=lexical_location,
            embedding_spec=self.embedding_spec,
            lexical_index_version=self.lexical_index_version,
            lexical_tokenizer_settings=self.lexical_tokenizer_settings,
            index_version=self.index_version,
            population_hash=population_hash,
            population_identity=identity,
            document_count=len({chunk.document_id for chunk in chunks}),
            chunk_count=len(chunks),
            vector_ready=True,
            lexical_ready=True,
            status="VALIDATED",
            component_versions=components,
        )

    def _validate_input(self, chunks: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorIndexError("vector count must match chunk count")
        seen: set[str] = set()
        document_sources: dict[str, str] = {}
        for chunk in chunks:
            if not chunk.chunk_id or not chunk.document_id or not chunk.source_hash:
                raise IncompatibleIndexGeneration("every indexed chunk requires ID and source provenance")
            if not chunk.source_filename or not chunk.source_relative_path:
                raise IncompatibleIndexGeneration("every indexed chunk requires source filename/path provenance")
            if chunk.chunk_id in seen:
                raise IncompatibleIndexGeneration("chunk IDs must be unique")
            seen.add(chunk.chunk_id)
            if chunk.embedding_spec is not None and chunk.embedding_spec != self.embedding_spec:
                raise EmbeddingSpecMismatch("chunk embedding specification differs from generation specification")
            known_hash = document_sources.setdefault(chunk.document_id, chunk.source_hash)
            if known_hash != chunk.source_hash:
                raise IncompatibleIndexGeneration("one document ID cannot have multiple source hashes")
        # Validate here too, before any optional backend does filesystem work.
        VectorIndexWriter._rows(
            chunks, vectors, generation_id="validation", embedding_spec=self.embedding_spec,
            lexical_index_version=self.lexical_index_version, index_version=self.index_version,
        )

    def _staging_root(self) -> Path:
        root = self.artifact_root / ".index-staging"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @property
    def _pointer_path(self) -> Path:
        return self.artifact_root / "state" / "active-index.json"

    def _read_active_pointer(self) -> dict[str, Any] | None:
        try:
            return json.loads(self._pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_active_pointer(self, generation: IndexGeneration) -> None:
        path = self._pointer_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(generation.to_json(), encoding="utf-8")
            with temporary.open("r+", encoding="utf-8") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _safe_generation_id(value: str) -> str:
    generation_id = str(value).strip()
    if not generation_id or generation_id in {".", ".."} or "/" in generation_id or "\\" in generation_id:
        raise ValueError("generation_id must be one safe path component")
    return generation_id


def _population_hashes(chunks: Sequence[ChunkRecord]) -> dict[str, str]:
    documents = sorted({(chunk.document_id, chunk.source_hash) for chunk in chunks})
    chunk_rows = sorted((chunk.chunk_id, chunk.document_id, chunk.source_hash) for chunk in chunks)
    document_hash = _hash_rows(documents)
    chunk_hash = _hash_rows(chunk_rows)
    population_hash = _hash_rows((("documents", document_hash), ("chunks", chunk_hash)))
    return {
        "population_hash": population_hash,
        "document_population_hash": document_hash,
        "chunk_population_hash": chunk_hash,
    }


def _row_identity(row: Mapping[str, Any] | Sequence[str]) -> tuple[str, str, str]:
    if isinstance(row, Mapping):
        values = (row.get("chunk_id"), row.get("document_id"), row.get("source_hash"))
    else:
        values = tuple(row)
    if len(values) != 3 or any(not str(value).strip() for value in values):
        raise IncompatibleIndexGeneration("indexed provenance requires chunk, document, and source identities")
    return tuple(str(value) for value in values)  # type: ignore[return-value]


def _population_hashes_from_rows(
    rows: Sequence[Mapping[str, Any]] | Sequence[Sequence[str]],
) -> dict[str, str]:
    chunk_rows = sorted(_row_identity(row) for row in rows)
    documents = sorted({(document_id, source_hash) for _, document_id, source_hash in chunk_rows})
    document_hash = _hash_rows(documents)
    chunk_hash = _hash_rows(chunk_rows)
    return {
        "population_hash": _hash_rows((("documents", document_hash), ("chunks", chunk_hash))),
        "document_population_hash": document_hash,
        "chunk_population_hash": chunk_hash,
    }


def _hash_rows(rows: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update("\x1f".join(str(value) for value in row).encode("utf-8"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _embedding_spec_from_config(config: Any | None) -> EmbeddingSpec:
    if config is None:
        return EmbeddingSpec()
    return EmbeddingSpec(
        model_id=config.embedding_model_id,
        resolved_model_version=config.embedding_model_version,
        runtime=config.embedding_runtime,
        artifact_hash=config.embedding_artifact_hash,
        dimensions=config.embedding_dimensions,
        normalization_policy=config.embedding_normalization,
        tokenizer_fingerprint=config.embedding_tokenizer_fingerprint,
        model_max_input_tokens=config.embedding_max_input_tokens,
        special_token_budget=config.embedding_special_token_budget,
        effective_corpus_content_token_limit=config.embedding_effective_content_token_limit,
        corpus_instruction_policy=config.embedding_corpus_instruction_policy,
        corpus_instruction_version=config.embedding_corpus_instruction_version,
        query_instruction_policy=config.embedding_query_instruction_policy,
        query_instruction_version=config.embedding_query_instruction_version,
        truncation=False,
    )


__all__ = [
    "EmbeddingSpecMismatch",
    "IncompatibleIndexGeneration",
    "IndexGenerationManager",
]
