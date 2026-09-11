"""Optional LanceDB-backed, generation-scoped dense projection.

The module deliberately imports LanceDB only when a real projection is opened.
Unit tests and metadata-only operators can therefore use the package without
the optional dependency or a network connection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .models import ChunkRecord, EmbeddingSpec


_METADATA_FILE = "index-metadata.json"
_SCHEMA_VERSION = "vector-projection-v1"


class VectorIndexError(RuntimeError):
    """A dense projection cannot be built or read safely."""


def embedding_spec_row_values(embedding_spec: EmbeddingSpec) -> dict[str, Any]:
    """Return every embedding identity field in the stable vector-row schema."""

    return {
        "embedding_model_id": embedding_spec.model_id,
        "embedding_model_version": embedding_spec.resolved_model_version,
        "embedding_runtime": embedding_spec.runtime,
        "embedding_artifact_hash": embedding_spec.artifact_hash,
        "embedding_dimensions": embedding_spec.dimensions,
        "embedding_normalization": embedding_spec.normalization_policy,
        "embedding_tokenizer_fingerprint": embedding_spec.tokenizer_fingerprint,
        "embedding_max_input_tokens": embedding_spec.model_max_input_tokens,
        "embedding_special_token_budget": embedding_spec.special_token_budget,
        "embedding_effective_content_token_limit": embedding_spec.effective_corpus_content_token_limit,
        "embedding_corpus_instruction_policy": embedding_spec.corpus_instruction_policy,
        "embedding_corpus_instruction_version": embedding_spec.corpus_instruction_version,
        "embedding_query_instruction_policy": embedding_spec.query_instruction_policy,
        "embedding_query_instruction_version": embedding_spec.query_instruction_version,
        "embedding_truncation": embedding_spec.truncation,
    }


class VectorBackend(Protocol):
    """Small adapter seam so deterministic tests do not require LanceDB."""

    def write(self, location: Path, rows: tuple[dict[str, Any], ...]) -> None: ...

    def read(self, location: Path) -> tuple[dict[str, Any], ...]: ...


class LanceDBBackend:
    """The production adapter; imports optional LanceDB only at use time."""

    identity = "lancedb-v1"

    def write(self, location: Path, rows: tuple[dict[str, Any], ...]) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise VectorIndexError(
                "LanceDB is not installed; install tradingagents[knowledge] to build dense indexes"
            ) from exc
        try:
            location.mkdir(parents=True, exist_ok=True)
            database = lancedb.connect(str(location))
            database.create_table("chunks", data=list(rows), mode="overwrite")
        except Exception as exc:
            raise VectorIndexError(f"could not write LanceDB projection: {exc}") from exc

    def read(self, location: Path) -> tuple[dict[str, Any], ...]:
        try:
            import lancedb
        except ImportError as exc:
            raise VectorIndexError("LanceDB is not installed") from exc
        try:
            table = lancedb.connect(str(location)).open_table("chunks")
            return tuple(dict(row) for row in table.to_arrow().to_pylist())
        except Exception as exc:
            raise VectorIndexError(f"could not read LanceDB projection: {exc}") from exc


class VectorIndexWriter:
    """Write one complete dense generation plus deterministic side metadata."""

    def __init__(self, *, backend: VectorBackend | None = None) -> None:
        self.backend = backend or LanceDBBackend()

    @property
    def component_identity(self) -> str:
        return str(getattr(self.backend, "identity", type(self.backend).__name__))

    def write(
        self,
        location: str | Path,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
        *,
        generation_id: str,
        embedding_spec: EmbeddingSpec,
        lexical_index_version: str,
        index_version: str,
        population_hash: str,
        document_population_hash: str,
        chunk_population_hash: str,
    ) -> dict[str, Any]:
        location = Path(location)
        rows = self._rows(
            chunks,
            vectors,
            generation_id=generation_id,
            embedding_spec=embedding_spec,
            lexical_index_version=lexical_index_version,
            index_version=index_version,
        )
        try:
            self.backend.write(location, rows)
            metadata = {
                "schema_version": _SCHEMA_VERSION,
                "generation_id": generation_id,
                "embedding_spec": embedding_spec.to_dict(),
                "lexical_index_version": lexical_index_version,
                "index_version": index_version,
                "population_hash": population_hash,
                "document_population_hash": document_population_hash,
                "chunk_population_hash": chunk_population_hash,
                "document_count": len({chunk.document_id for chunk in chunks}),
                "chunk_count": len(chunks),
                "component_identity": self.component_identity,
            }
            self._write_metadata(location, metadata)
            return metadata
        except VectorIndexError:
            raise
        except Exception as exc:
            raise VectorIndexError(f"could not write vector generation {generation_id}: {exc}") from exc

    @staticmethod
    def _rows(
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
        *,
        generation_id: str,
        embedding_spec: EmbeddingSpec,
        lexical_index_version: str,
        index_version: str,
    ) -> tuple[dict[str, Any], ...]:
        if len(chunks) != len(vectors):
            raise VectorIndexError("vector count must match chunk count")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk, vector in zip(chunks, vectors, strict=True):
            if not chunk.chunk_id or chunk.chunk_id in seen:
                raise VectorIndexError("chunk IDs must be non-empty and unique")
            seen.add(chunk.chunk_id)
            try:
                normalized = tuple(float(value) for value in vector)
            except (TypeError, ValueError) as exc:
                raise VectorIndexError("vector values must be numeric") from exc
            if len(normalized) != embedding_spec.dimensions:
                raise VectorIndexError("vector dimensions do not match complete embedding specification")
            if not all(math.isfinite(value) for value in normalized):
                raise VectorIndexError("vector values must be finite")
            row = chunk.to_dict()
            # Dynamic provenance maps are retained as canonical JSON.  Keeping
            # the physical LanceDB schema scalar/stable avoids a document's
            # optional table/equation fields changing Arrow's inferred type.
            for key in ("embedding_spec", "table_metadata", "equation_metadata", "extra"):
                row.pop(key, None)
            row.update(
                {
                    "content_type": chunk.content_type.value,
                    "authors_json": json.dumps(chunk.authors, sort_keys=True),
                    "section_path_json": json.dumps(chunk.section_path, sort_keys=True),
                    "epub_location_json": json.dumps(
                        {"spine_item": chunk.epub_spine_item, "anchor": chunk.anchor},
                        sort_keys=True,
                    ),
                    "table_metadata_json": json.dumps(chunk.table_metadata, sort_keys=True),
                    "equation_metadata_json": json.dumps(chunk.equation_metadata, sort_keys=True),
                    "extra_json": json.dumps(chunk.extra, sort_keys=True),
                    "vector": list(normalized),
                    "embedding_spec_json": embedding_spec.to_json(),
                    "lexical_index_version": lexical_index_version,
                    "index_version": index_version,
                    "projection_generation": generation_id,
                    "active": bool(chunk.active),
                }
            )
            row.update(embedding_spec_row_values(embedding_spec))
            rows.append(row)
        return tuple(rows)

    @staticmethod
    def _write_metadata(location: Path, metadata: Mapping[str, Any]) -> None:
        location.mkdir(parents=True, exist_ok=True)
        (location / _METADATA_FILE).write_text(
            json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )


class VectorIndexReader:
    """Read dense metadata before later query code touches optional LanceDB."""

    def __init__(self, location: str | Path, *, backend: VectorBackend | None = None) -> None:
        self.location = Path(location)
        self.backend = backend or LanceDBBackend()

    def metadata(self) -> dict[str, Any]:
        try:
            return json.loads((self.location / _METADATA_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorIndexError("vector generation metadata is unreadable") from exc

    def rows(self) -> tuple[dict[str, Any], ...]:
        try:
            return self.backend.read(self.location)
        except VectorIndexError:
            raise
        except Exception as exc:
            raise VectorIndexError(f"could not read vector rows: {exc}") from exc

    def row_count(self) -> int:
        return len(self.rows())


__all__ = [
    "LanceDBBackend",
    "VectorBackend",
    "VectorIndexError",
    "VectorIndexReader",
    "VectorIndexWriter",
    "embedding_spec_row_values",
]
