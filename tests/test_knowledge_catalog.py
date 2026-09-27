"""Transactional catalog tests for the local knowledge pipeline."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.identity import document_id_for
from tradingagents.knowledge.models import (
    AliasRelation,
    ChunkRecord,
    DocumentMetadata,
    EmbeddingSpec,
    IndexGeneration,
    IngestionRunSummary,
    IngestionState,
    ParsedDocument,
)


def make_parsed_document(*, document_id: str, source_hash: str) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        source_hash=source_hash,
        metadata=DocumentMetadata(
            title="Fixture market microstructure paper",
            authors=("Fixture Author",),
            publication_year=2026,
            document_type="paper",
            format="pdf",
            source_filename="fixture.pdf",
            source_relative_path="fixture.pdf",
        ),
        parser_id="fixture-parser",
        parser_version="1.0",
        parser_config_hash="fixture-config",
    )


def make_generation(
    *,
    generation_id: str = "gen-1",
    ready: bool = True,
    status: str = "VALIDATED",
    vector_location: str | None = None,
    lexical_location: str | None = None,
    component_versions: dict[str, str] | None = None,
) -> IndexGeneration:
    population_hash = "population-fixture"
    return IndexGeneration(
        generation_id=generation_id,
        vector_location=vector_location if vector_location is not None else f"vector/lancedb/{generation_id}",
        lexical_location=lexical_location if lexical_location is not None else f"keyword/{generation_id}/bm25.sqlite3",
        embedding_spec=EmbeddingSpec(model_id="fixture", dimensions=3),
        lexical_index_version="fts5-fixture-v1",
        lexical_tokenizer_settings={"unicode": "NFC"},
        index_version=generation_id,
        population_hash=population_hash,
        population_identity=population_hash,
        document_count=1,
        chunk_count=1,
        vector_ready=ready,
        lexical_ready=ready,
        status=status,
        component_versions=component_versions or {
            "vector": "fixture-vector-v1",
            "lexical": "fixture-lexical-v1",
            "vector_generation_id": generation_id,
            "lexical_generation_id": generation_id,
            "vector_population_hash": population_hash,
            "lexical_population_hash": population_hash,
        },
    )


def test_remove_one_duplicate_alias_keeps_document_active(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("11" * 32)
    catalog.register_document(make_parsed_document(document_id=document_id, source_hash="11" * 32))
    catalog.commit_alias("res_a", document_id, "11" * 32, AliasRelation.CURRENT)
    catalog.commit_alias("res_b", document_id, "11" * 32, AliasRelation.CURRENT)

    catalog.mark_removed("res_a")

    assert catalog.current_alias_count(document_id) == 1
    assert catalog.document_is_active(document_id) is True


def test_chunks_for_document_returns_deterministic_chunk_ordinal_order(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("44" * 32)
    catalog.register_document(make_parsed_document(document_id=document_id, source_hash="44" * 32))
    chunks = (
        ChunkRecord(
            chunk_id="chk_z",
            document_id=document_id,
            source_hash="44" * 32,
            text="second",
            chunk_ordinal=1,
        ),
        ChunkRecord(
            chunk_id="chk_a",
            document_id=document_id,
            source_hash="44" * 32,
            text="first",
            chunk_ordinal=0,
        ),
    )
    catalog.store_chunks(document_id, chunks)

    assert [chunk.chunk_id for chunk in catalog.chunks_for_document(document_id)] == [
        "chk_a",
        "chk_z",
    ]


def test_remove_last_alias_deactivates_default_view_but_keeps_rows(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("22" * 32)
    catalog.register_document(make_parsed_document(document_id=document_id, source_hash="22" * 32))
    catalog.commit_alias("res_only", document_id, "22" * 32, AliasRelation.CURRENT)

    catalog.mark_removed("res_only")

    assert catalog.current_alias_count(document_id) == 0
    assert catalog.document_is_active(document_id) is False
    assert catalog.get_document(document_id) is not None
    assert catalog.get_alias("res_only").relation is AliasRelation.REMOVED


def test_commit_alias_rejects_source_hash_mismatch_without_changing_alias(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("aa" * 32)
    catalog.register_document(make_parsed_document(document_id=document_id, source_hash="aa" * 32))
    catalog.commit_alias("res_only", document_id, "aa" * 32, AliasRelation.CURRENT)

    with pytest.raises(ValueError, match="source_hash"):
        catalog.commit_alias("res_only", document_id, "bb" * 32, AliasRelation.CURRENT)

    alias = catalog.get_alias("res_only")
    resource = next(item for item in catalog.list_resources() if item.resource_id == "res_only")
    assert alias is not None and alias.source_hash == "aa" * 32
    assert resource is not None and resource.source_hash == "aa" * 32


def test_changed_duplicate_and_failed_change_preserve_other_alias(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    old_id = document_id_for("33" * 32)
    catalog.register_document(make_parsed_document(document_id=old_id, source_hash="33" * 32))
    catalog.commit_alias("res_a", old_id, "33" * 32, AliasRelation.CURRENT)
    catalog.commit_alias("res_b", old_id, "33" * 32, AliasRelation.CURRENT)

    catalog.retain_previous("res_a", attempted_hash="44" * 32, previous_document_id=old_id)

    assert catalog.current_alias_count(old_id) == 1
    assert catalog.get_alias("res_b").relation is AliasRelation.CURRENT
    assert catalog.get_alias("res_a").relation is AliasRelation.RETAINED_PREVIOUS
    assert catalog.list_resources(state=IngestionState.RETAINED_PREVIOUS)[0].resource_id == "res_a"
    quarantine = catalog.list_quarantine()
    assert quarantine[0].state is IngestionState.RETAINED_PREVIOUS
    assert quarantine[0].resource_id == "res_a"
    assert quarantine[0].attempted_hash == "44" * 32


def test_schema_is_idempotent_and_contains_only_knowledge_catalog_tables(tmp_path):
    path = tmp_path / "catalog.sqlite3"
    catalog = KnowledgeCatalog(path)
    catalog.initialize()
    catalog.initialize()

    with closing(sqlite3.connect(path)) as connection, connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'knowledge_%'"
            )
        }
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(knowledge_documents)")
        }

    assert tables == {
        "knowledge_resources",
        "knowledge_documents",
        "knowledge_aliases",
        "knowledge_chunks",
        "knowledge_ingestion_runs",
        "knowledge_ingestion_events",
        "knowledge_index_generations",
    }
    assert not {"portfolio", "decision", "experience", "mt5"} & {column.casefold() for column in columns}


def test_projection_readiness_requires_active_matched_generation(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("55" * 32)
    catalog.register_document(make_parsed_document(document_id=document_id, source_hash="55" * 32))
    catalog.commit_alias("res_a", document_id, "55" * 32, AliasRelation.CURRENT)
    catalog.record_projection_ready(document_id, "gen-1", vector_ready=True, lexical_ready=True)

    assert catalog.document_is_active(document_id) is True
    assert catalog.document_is_retrieval_ready(document_id) is False

    catalog.set_active_generation(make_generation())

    assert catalog.document_is_retrieval_ready(document_id) is True
    stored = catalog.get_document(document_id)
    assert stored.projection_generation == "gen-1"
    assert stored.projection_population_hash == "population-fixture"
    assert stored.vector_ready is True
    assert stored.lexical_ready is True


def test_replacing_chunks_invalidates_old_projection_readiness(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("66" * 32)
    catalog.register_document(make_parsed_document(document_id=document_id, source_hash="66" * 32))
    catalog.commit_alias("res_a", document_id, "66" * 32, AliasRelation.CURRENT)
    catalog.store_chunks(
        document_id,
        (ChunkRecord(chunk_id="old", document_id=document_id, source_hash="66" * 32, text="old"),),
    )
    catalog.record_projection_ready(document_id, "gen-1", vector_ready=True, lexical_ready=True)
    catalog.set_active_generation(make_generation(generation_id="gen-1"))
    assert catalog.document_is_retrieval_ready(document_id) is True

    catalog.store_chunks(
        document_id,
        (ChunkRecord(chunk_id="new", document_id=document_id, source_hash="66" * 32, text="new"),),
    )

    assert catalog.document_is_active(document_id) is True
    assert catalog.document_is_retrieval_ready(document_id) is False


def test_reregistering_document_invalidates_old_projection_readiness(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("77" * 32)
    original = make_parsed_document(document_id=document_id, source_hash="77" * 32)
    catalog.register_document(original)
    catalog.commit_alias("res_a", document_id, "77" * 32, AliasRelation.CURRENT)
    catalog.record_projection_ready(document_id, "gen-1", vector_ready=True, lexical_ready=True)
    catalog.set_active_generation(make_generation(generation_id="gen-1"))
    assert catalog.document_is_retrieval_ready(document_id) is True

    reparsed = ParsedDocument(
        document_id=document_id,
        source_hash=original.source_hash,
        metadata=original.metadata,
        parser_id=original.parser_id,
        parser_version=original.parser_version,
        parser_config_hash="changed-parser-config",
    )
    catalog.register_document(reparsed)

    assert catalog.document_is_active(document_id) is True
    assert catalog.document_is_retrieval_ready(document_id) is False


def test_store_chunks_rejects_source_identity_mismatch_without_deleting_old_chunks(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("88" * 32)
    catalog.register_document(make_parsed_document(document_id=document_id, source_hash="88" * 32))
    original = ChunkRecord(
        chunk_id="original", document_id=document_id, source_hash="88" * 32, text="original"
    )
    catalog.store_chunks(document_id, (original,))

    mismatched = ChunkRecord(
        chunk_id="mismatched", document_id=document_id, source_hash="99" * 32, text="wrong source"
    )
    with pytest.raises(ValueError, match="source_hash"):
        catalog.store_chunks(document_id, (mismatched,))

    assert catalog.chunks_for_document(document_id) == (original,)


def test_publish_generation_rejects_cross_source_chunks_atomically(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("aa" * 32)
    document = make_parsed_document(document_id=document_id, source_hash="aa" * 32)
    mismatched = ChunkRecord(
        chunk_id="mismatched", document_id=document_id, source_hash="bb" * 32, text="wrong source"
    )
    summary = IngestionRunSummary(
        run_id="run-identity-mismatch",
        state="SUCCEEDED",
        counts={},
        source_count=1,
        document_count=1,
        chunk_count=1,
    )

    with pytest.raises(ValueError, match="source_hash"):
        catalog.publish_generation(
            make_generation(generation_id="gen-identity-mismatch"),
            documents_by_id={document_id: document},
            component_fingerprints={},
            chunks_by_document={document_id: (mismatched,)},
            aliases=(("res_a", document_id, document.source_hash, IngestionState.INDEXED),),
            removed_resource_ids=(),
            ready_document_ids=(document_id,),
            summary=summary,
        )

    assert catalog.active_generation() is None
    assert catalog.get_document(document_id) is None


def test_generation_registry_rejects_invalid_pair_and_preserves_previous_active_generation(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    previous = make_generation(generation_id="gen-previous")
    catalog.set_active_generation(previous)

    invalid_generations = (
        make_generation(generation_id="gen-partial", ready=False),
        make_generation(generation_id="gen-staged", status="STAGED"),
        make_generation(generation_id="gen-blank-location", vector_location=""),
        make_generation(
            generation_id="gen-missing-pair-metadata",
            component_versions={"vector": "fixture-vector-v1", "lexical": "fixture-lexical-v1"},
        ),
        make_generation(
            generation_id="gen-mismatched-component-identity",
            component_versions={
                "vector": "fixture-vector-v1",
                "lexical": "fixture-lexical-v1",
                "vector_generation_id": "gen-mismatched-component-identity",
                "lexical_generation_id": "other-generation",
                "vector_population_hash": "population-fixture",
                "lexical_population_hash": "population-fixture",
            },
        ),
        make_generation(
            generation_id="gen-mismatched-population-metadata",
            component_versions={
                "vector": "fixture-vector-v1",
                "lexical": "fixture-lexical-v1",
                "vector_generation_id": "gen-mismatched-population-metadata",
                "lexical_generation_id": "gen-mismatched-population-metadata",
                "vector_population_hash": "population-fixture",
                "lexical_population_hash": "other-population",
            },
        ),
    )

    for invalid in invalid_generations:
        with pytest.raises(ValueError):
            catalog.set_active_generation(invalid)
        assert catalog.active_generation().generation_id == previous.generation_id

    generation = make_generation(generation_id="gen-persisted")
    catalog.set_active_generation(generation)

    stored = catalog.active_generation()
    assert stored is not None
    assert stored.generation_id == generation.generation_id
    assert stored.vector_location == generation.vector_location
    assert stored.lexical_location == generation.lexical_location
    assert stored.population_hash == generation.population_hash
    assert stored.component_versions == generation.component_versions
    assert stored.activated_at is not None


@pytest.mark.parametrize(
    ("read_operation", "operation_id"),
    (
        (lambda catalog, document_id: catalog.get_document(document_id), "get_document"),
        (lambda catalog, document_id: catalog.get_alias("res_a"), "get_alias"),
        (lambda catalog, document_id: catalog.current_alias_count(document_id), "current_alias_count"),
        (lambda catalog, document_id: catalog.document_is_active(document_id), "document_is_active"),
        (lambda catalog, document_id: catalog.document_is_retrieval_ready(document_id), "document_is_retrieval_ready"),
        (lambda catalog, document_id: catalog.active_generation(), "active_generation"),
        (lambda catalog, document_id: catalog.list_resources(), "list_resources"),
        (lambda catalog, document_id: catalog.list_runs(), "list_runs"),
        (lambda catalog, document_id: catalog.list_quarantine(), "list_quarantine"),
    ),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_catalog_read_releases_database_for_immediate_rename_and_remove(
    tmp_path, read_operation, operation_id
):
    path = tmp_path / f"{operation_id}.sqlite3"
    catalog = KnowledgeCatalog(path)
    catalog.initialize()
    document_id = document_id_for("77" * 32)
    catalog.register_document(make_parsed_document(document_id=document_id, source_hash="77" * 32))
    catalog.commit_alias("res_a", document_id, "77" * 32, AliasRelation.CURRENT)
    catalog.record_projection_ready(document_id, "gen-read", vector_ready=True, lexical_ready=True)
    catalog.set_active_generation(make_generation(generation_id="gen-read"))

    read_operation(catalog, document_id)

    renamed = tmp_path / f"{operation_id}-renamed.sqlite3"
    path.replace(renamed)
    renamed.unlink()


def test_run_and_quarantine_records_preserve_only_bounded_scalar_diagnostics(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    catalog.record_run(
        IngestionRunSummary(
            run_id="run-1",
            state="PARTIAL_FAILURE",
            counts={IngestionState.PARSE_FAILED: 1},
            source_count=1,
            document_count=0,
            chunk_count=0,
        )
    )
    catalog.record_event(
        resource_id="res_failed",
        document_id=None,
        attempted_hash="66" * 32,
        state=IngestionState.PARSE_FAILED,
        stage="PARSE",
        error="x" * 900,
        counters={"pages": 3},
        run_id="run-1",
    )

    run = catalog.list_runs()[0]
    quarantine = catalog.list_quarantine()[0]

    assert run.run_id == "run-1"
    assert run.counts[IngestionState.PARSE_FAILED] == 1
    assert quarantine.resource_id == "res_failed"
    assert quarantine.stage == "PARSE"
    assert quarantine.counters == {"pages": 3}
    assert len(quarantine.error_message or "") == 512
