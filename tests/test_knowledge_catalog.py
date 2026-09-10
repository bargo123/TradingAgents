"""Transactional catalog tests for the local knowledge pipeline."""

from __future__ import annotations

import sqlite3

import pytest

from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.identity import document_id_for
from tradingagents.knowledge.models import (
    AliasRelation,
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


def make_generation(*, generation_id: str = "gen-1", ready: bool = True) -> IndexGeneration:
    return IndexGeneration(
        generation_id=generation_id,
        vector_location=f"vector/lancedb/{generation_id}",
        lexical_location=f"keyword/{generation_id}/bm25.sqlite3",
        embedding_spec=EmbeddingSpec(model_id="fixture", dimensions=3),
        lexical_index_version="fts5-fixture-v1",
        lexical_tokenizer_settings={"unicode": "NFC"},
        index_version=generation_id,
        population_hash="population-fixture",
        document_count=1,
        chunk_count=1,
        vector_ready=ready,
        lexical_ready=ready,
        status="VALIDATED",
        component_versions={"vector": "fixture-v1", "lexical": "fixture-v1"},
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

    with sqlite3.connect(path) as connection:
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


def test_generation_registry_rejects_partial_pair_and_persists_validated_generation(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()

    with pytest.raises(ValueError, match="ready"):
        catalog.set_active_generation(make_generation(ready=False))

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
