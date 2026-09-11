"""Generation-scoped dense and lexical projection tests."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.embeddings import EmbeddingSpecMismatch
from tradingagents.knowledge.index_generation import (
    EmbeddingSpecMismatch as IndexEmbeddingSpecMismatch,
    IncompatibleIndexGeneration,
    IndexGenerationManager,
)
from tradingagents.knowledge.lexical_index import LexicalIndexError, LexicalIndexReader, LexicalIndexWriter
from tradingagents.knowledge.models import ChunkRecord, ContentType, EmbeddingSpec
from tradingagents.knowledge.vector_index import VectorIndexError, VectorIndexReader, VectorIndexWriter


class FakeVectorBackend:
    """Dependency-free LanceDB-shaped adapter used only by projection tests."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def write(self, location: Path, rows: tuple[dict[str, object], ...]) -> None:
        if self.fail:
            raise RuntimeError("fake vector write failure")
        location.mkdir(parents=True, exist_ok=True)
        (location / "rows.json").write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")

    def read(self, location: Path) -> tuple[dict[str, object], ...]:
        return tuple(json.loads((location / "rows.json").read_text(encoding="utf-8")))


class TruncatingVectorBackend(FakeVectorBackend):
    def write(self, location: Path, rows: tuple[dict[str, object], ...]) -> None:
        super().write(location, rows[:-1])


class CorruptingLexicalWriter(LexicalIndexWriter):
    """Real FTS5 writer that injects one invalid persisted provenance value."""

    def __init__(self, corrupt) -> None:
        self._corrupt = corrupt

    def write(self, location, chunks, **kwargs):
        metadata = super().write(location, chunks, **kwargs)
        self._corrupt(Path(location))
        return metadata


def make_embedding_spec(**overrides: object) -> EmbeddingSpec:
    values: dict[str, object] = {
        "model_id": "fixture/bge",
        "resolved_model_version": "fixture-v1",
        "runtime": "onnx-cpu",
        "artifact_hash": "sha256:fixture-model",
        "dimensions": 3,
        "normalization_policy": "l2",
        "tokenizer_fingerprint": "sha256:fixture-tokenizer",
        "model_max_input_tokens": 512,
        "special_token_budget": 2,
        "effective_corpus_content_token_limit": 510,
        "corpus_instruction_policy": "none-v1",
        "corpus_instruction_version": "v1",
        "query_instruction_policy": "bge-search-prefix-v1",
        "query_instruction_version": "v1",
        "truncation": False,
    }
    values.update(overrides)
    return EmbeddingSpec(**values)


def make_chunks(spec: EmbeddingSpec | None = None) -> tuple[ChunkRecord, ...]:
    spec = spec or make_embedding_spec()
    return (
        ChunkRecord(
            chunk_id="chunk-002",
            document_id="doc-beta",
            source_hash="sha256:beta",
            text="DeepLOB and order-flow describe microstructure signals.",
            content_type=ContentType.PROSE,
            source_filename="beta.pdf",
            source_relative_path="beta.pdf",
            title="Beta",
            authors=("B",),
            page_start=4,
            page_end=4,
            chapter="2",
            section_path=("Signals",),
            reading_order=2,
            parser_id="fixture-parser",
            parser_version="1",
            parser_config_hash="fixture-parser-config",
            chunker_version="structure-v2",
            embedding_spec=spec,
            lexical_index_version="fts5-fixture-v1",
            index_version="index-fixture-v1",
        ),
        ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-alpha",
            source_hash="sha256:alpha",
            text="VPIN measures adverse-selection risk in order_flow.",
            content_type=ContentType.EQUATION,
            source_filename="alpha.pdf",
            source_relative_path="alpha.pdf",
            title="Alpha",
            authors=("A",),
            page_start=1,
            page_end=2,
            chapter="1",
            section_path=("Risk", "VPIN"),
            reading_order=1,
            parser_id="fixture-parser",
            parser_version="1",
            parser_config_hash="fixture-parser-config",
            chunker_version="structure-v2",
            embedding_spec=spec,
            lexical_index_version="fts5-fixture-v1",
            index_version="index-fixture-v1",
        ),
    )


def make_vectors() -> tuple[tuple[float, ...], ...]:
    return ((0.0, 0.5, 1.0), (1.0, 0.5, 0.0))


def make_generation_manager(
    root: Path,
    *,
    vector: FakeVectorBackend | None = None,
    embedding_spec: EmbeddingSpec | None = None,
    seed: bool = True,
) -> IndexGenerationManager:
    catalog = KnowledgeCatalog(root / "catalog.sqlite3")
    manager = IndexGenerationManager(
        artifact_root=root,
        catalog=catalog,
        embedding_spec=embedding_spec or make_embedding_spec(),
        vector_writer=VectorIndexWriter(backend=vector or FakeVectorBackend()),
        lexical_writer=LexicalIndexWriter(),
        lexical_index_version="fts5-fixture-v1",
        lexical_tokenizer_settings={"normalization": "NFC", "casefold": True},
        index_version="index-fixture-v1",
    )
    if seed:
        manager.build_and_activate(make_chunks(), make_vectors(), "gen-001")
    return manager


def test_failed_vector_build_does_not_swap_active_generation(tmp_path):
    manager = make_generation_manager(tmp_path)
    old = manager.active_generation()
    assert old is not None
    manager.vector_writer = VectorIndexWriter(backend=FakeVectorBackend(fail=True))

    with pytest.raises(VectorIndexError):
        manager.build_and_activate(make_chunks(), make_vectors(), "gen-002")

    assert manager.active_generation().generation_id == old.generation_id
    assert (tmp_path / "vector" / "lancedb" / old.generation_id).exists()


def test_failed_lexical_build_does_not_swap_active_generation(tmp_path, monkeypatch):
    manager = make_generation_manager(tmp_path)
    old = manager.active_generation()
    assert old is not None

    def fail(*args, **kwargs):
        raise LexicalIndexError("fts5 unavailable")

    monkeypatch.setattr(manager.lexical_writer, "write", fail)
    with pytest.raises(LexicalIndexError):
        manager.build_and_activate(make_chunks(), make_vectors(), "gen-002")

    assert manager.active_generation().generation_id == old.generation_id
    assert old.lexical_location.exists()


def test_successful_rebuild_activates_both_and_keeps_previous(tmp_path):
    manager = make_generation_manager(tmp_path)
    old = manager.active_generation()
    assert old is not None

    new = manager.build_and_activate(make_chunks(), make_vectors(), "gen-002")

    assert manager.active_generation().generation_id == "gen-002"
    assert new.lexical_location.name == "bm25.sqlite3"
    assert old.vector_location.exists()
    assert old.lexical_location.exists()
    assert new.vector_ready is True
    assert new.lexical_ready is True


def test_query_rejects_vector_lexical_generation_mismatch(tmp_path):
    manager = make_generation_manager(tmp_path)
    generation = manager.active_generation()
    assert generation is not None
    writer = LexicalIndexWriter()
    writer.write_metadata(generation.lexical_location, generation_id="other")

    with pytest.raises(IncompatibleIndexGeneration):
        manager.resolve_active_generation()


def test_generation_persists_complete_embedding_spec_not_dimensions_only(tmp_path):
    spec = make_embedding_spec(
        model_id="BAAI/bge-small-en-v1.5",
        resolved_model_version="2026-01",
        artifact_hash="sha256:local-model",
        dimensions=384,
        tokenizer_fingerprint="tok-v1",
        query_instruction_policy="bge-search-prefix-v1",
    )
    chunks = make_chunks(spec)
    vectors = tuple(tuple(float(index) for index in range(spec.dimensions)) for _ in chunks)
    generation = make_generation_manager(tmp_path, embedding_spec=spec, seed=False).build_and_activate(
        chunks, vectors, "gen-001"
    )

    assert generation.embedding_spec == spec
    assert generation.embedding_spec.artifact_hash == "sha256:local-model"
    assert generation.embedding_spec.query_instruction_policy == "bge-search-prefix-v1"


@pytest.mark.parametrize(
    ("column", "expected", "replacement"),
    [
        ("embedding_model_id", "fixture/bge", "other/model"),
        ("embedding_model_version", "fixture-v1", "other-version"),
        ("embedding_runtime", "onnx-cpu", "cuda"),
        ("embedding_artifact_hash", "sha256:fixture-model", "sha256:other"),
        ("embedding_dimensions", 3, 4),
        ("embedding_normalization", "l2", "none"),
        ("embedding_tokenizer_fingerprint", "sha256:fixture-tokenizer", "sha256:other-tokenizer"),
        ("embedding_max_input_tokens", 512, 513),
        ("embedding_special_token_budget", 2, 3),
        ("embedding_effective_content_token_limit", 510, 509),
        ("embedding_corpus_instruction_policy", "none-v1", "other-policy"),
        ("embedding_corpus_instruction_version", "v1", "v2"),
        ("embedding_query_instruction_policy", "bge-search-prefix-v1", "other-query-policy"),
        ("embedding_query_instruction_version", "v1", "v2"),
        ("embedding_truncation", False, True),
    ],
)
def test_vector_rows_persist_and_validate_every_embedding_spec_field(
    tmp_path, column, expected, replacement
):
    manager = make_generation_manager(tmp_path)
    generation = manager.resolve_active_generation()
    rows_path = generation.vector_location / "rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))

    assert rows[0][column] == expected
    assert rows[0]["embedding_spec_json"] == make_embedding_spec().to_json()

    rows[0][column] = replacement
    rows_path.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")

    with pytest.raises(IncompatibleIndexGeneration):
        manager.resolve_active_generation()


def test_build_rejects_invalid_lexical_provenance_row_generation_or_population(tmp_path):
    def corrupt(location: Path) -> None:
        with closing(sqlite3.connect(location)) as connection:
            provenance_json = connection.execute(
                "SELECT provenance_json FROM knowledge_chunk_provenance WHERE chunk_id = ?",
                ("chunk-001",),
            ).fetchone()[0]
            payload = json.loads(provenance_json)
            payload["projection_generation"] = "other-generation"
            payload["projection_population_hash"] = "sha256:other-population"
            connection.execute(
                "UPDATE knowledge_chunk_provenance SET generation_id = ?, provenance_json = ? WHERE chunk_id = ?",
                ("other-generation", json.dumps(payload, sort_keys=True), "chunk-001"),
            )
            connection.commit()

    manager = make_generation_manager(tmp_path, seed=False)
    manager.lexical_writer = CorruptingLexicalWriter(corrupt)

    with pytest.raises(IncompatibleIndexGeneration):
        manager.build_generation(make_chunks(), make_vectors(), "gen-corrupt-lexical")


def test_resolve_rejects_lexical_location_identity_mismatch(tmp_path):
    manager = make_generation_manager(tmp_path)
    generation = manager.active_generation()
    assert generation is not None
    copied_location = tmp_path / "keyword" / "copied" / "bm25.sqlite3"
    copied_location.parent.mkdir(parents=True)
    shutil.copy2(generation.lexical_location, copied_location)
    with closing(sqlite3.connect(tmp_path / "catalog.sqlite3")) as connection:
        connection.execute(
            "UPDATE knowledge_index_generations SET lexical_location = ? WHERE generation_id = ?",
            (str(copied_location), generation.generation_id),
        )
        connection.commit()

    with pytest.raises(IncompatibleIndexGeneration):
        manager.resolve_active_generation()


def test_resolve_rejects_persisted_lexical_provenance_population_mismatch(tmp_path):
    manager = make_generation_manager(tmp_path)
    generation = manager.active_generation()
    assert generation is not None
    with closing(sqlite3.connect(generation.lexical_location)) as connection:
        provenance_json = connection.execute(
            "SELECT provenance_json FROM knowledge_chunk_provenance WHERE chunk_id = ?",
            ("chunk-001",),
        ).fetchone()[0]
        provenance = json.loads(provenance_json)
        provenance["projection_population_hash"] = "sha256:wrong-population"
        connection.execute(
            "UPDATE knowledge_chunk_provenance SET provenance_json = ? WHERE chunk_id = ?",
            (json.dumps(provenance, sort_keys=True), "chunk-001"),
        )
        connection.commit()

    with pytest.raises(IncompatibleIndexGeneration):
        manager.resolve_active_generation()


def test_build_rejects_mixed_chunk_embedding_specs_and_non_finite_or_wrong_vectors(tmp_path):
    manager = make_generation_manager(tmp_path, seed=False)
    incompatible = make_embedding_spec(artifact_hash="sha256:other")
    mixed = list(make_chunks())
    mixed[1] = ChunkRecord.from_dict({**mixed[1].to_dict(), "embedding_spec": incompatible.to_dict()})

    with pytest.raises((EmbeddingSpecMismatch, IndexEmbeddingSpecMismatch)):
        manager.build_generation(tuple(mixed), make_vectors(), "gen-mixed")
    with pytest.raises(VectorIndexError):
        manager.build_generation(make_chunks(), ((1.0, 2.0), (0.0, 1.0, float("nan"))), "gen-invalid")


def test_generation_validation_rejects_a_persisted_vector_row_count_or_provenance_mismatch(tmp_path):
    manager = make_generation_manager(tmp_path, vector=TruncatingVectorBackend(), seed=False)

    with pytest.raises(IncompatibleIndexGeneration):
        manager.build_generation(make_chunks(), make_vectors(), "gen-truncated")

    assert not (tmp_path / "vector" / "lancedb" / "gen-truncated").exists()


def test_lexical_projection_preserves_hft_token_characters_and_normalizes_diacritics(tmp_path):
    manager = make_generation_manager(tmp_path)
    generation = manager.resolve_active_generation()
    reader = LexicalIndexReader(generation.lexical_location)

    assert [row["chunk_id"] for row in reader.search("order_flow")] == ["chunk-001"]
    assert [row["chunk_id"] for row in reader.search('"order-flow"')] == ["chunk-002"]
    assert [row["chunk_id"] for row in reader.search("VPIŃ")] == ["chunk-001"]


def test_resolve_rebuilds_stale_pointer_from_catalog_and_validates_population(tmp_path):
    manager = make_generation_manager(tmp_path)
    active = manager.active_generation()
    assert active is not None
    pointer = tmp_path / "state" / "active-index.json"
    pointer.write_text('{"generation_id":"stale"}', encoding="utf-8")

    resolved = manager.resolve_active_generation()

    assert resolved.generation_id == active.generation_id
    assert json.loads(pointer.read_text(encoding="utf-8"))["generation_id"] == active.generation_id
    metadata = VectorIndexReader(active.vector_location).metadata()
    metadata["chunk_population_hash"] = "mismatch"
    (active.vector_location / "index-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(IncompatibleIndexGeneration):
        manager.resolve_active_generation()
