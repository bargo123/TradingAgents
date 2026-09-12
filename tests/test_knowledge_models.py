"""Contract tests for the Phase 7 knowledge value objects."""

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from tradingagents.knowledge.models import (
    AliasRelation,
    ChunkRecord,
    ContentType,
    DocumentMetadata,
    EmbeddingSpec,
    IndexGeneration,
    IngestionRunSummary,
    IngestionState,
    KnowledgeHit,
    KnowledgeQuery,
    ParsedBlock,
    ParsedDocument,
)


def test_state_and_content_enums_are_stable_strings():
    assert IngestionState.HASHED.value == "HASHED"
    assert IngestionState.RETAINED_PREVIOUS.value == "RETAINED_PREVIOUS"
    assert AliasRelation.CURRENT.value == "CURRENT"
    assert AliasRelation.REMOVED.value == "REMOVED"
    assert ContentType.EQUATION.value == "EQUATION"
    assert ContentType.TABLE.value == "TABLE"


def test_contracts_are_immutable_and_round_trip_json():
    metadata = DocumentMetadata(
        title="Microstructure Fixture",
        authors=("A. Author",),
        publication_year=2026,
        document_type="paper",
        format="pdf",
    )
    block = ParsedBlock(
        block_id="blk-1",
        content_type=ContentType.PROSE,
        text="Order flow imbalance.",
        reading_order=0,
        page_start=1,
        section_path=("Methods",),
    )
    document = ParsedDocument(
        document_id="doc_" + "ab" * 32,
        source_hash="ab" * 32,
        metadata=metadata,
        blocks=(block,),
        parser_id="fixture",
        parser_version="1",
        parser_config_hash="cfg",
    )

    with pytest.raises(FrozenInstanceError):
        document.source_hash = "cd" * 32

    restored = ParsedDocument.from_dict(document.to_dict())
    assert restored == document
    assert json.loads(document.to_json()) == document.to_dict()
    assert asdict(document)["metadata"]["title"] == "Microstructure Fixture"


def test_embedding_spec_exposes_complete_semantic_identity():
    spec = EmbeddingSpec(
        model_id="BAAI/bge-small-en-v1.5",
        resolved_model_version="1.0",
        runtime="onnx-cpu",
        artifact_hash="artifact-sha",
        dimensions=384,
        normalization_policy="l2",
        tokenizer_fingerprint="tok-v1",
        model_max_input_tokens=512,
        special_token_budget=2,
        effective_corpus_content_token_limit=510,
        corpus_instruction_policy="none-v1",
        corpus_instruction_version="v1",
        query_instruction_policy="bge-search-prefix-v1",
        query_instruction_version="v1",
        truncation=False,
    )

    assert spec.embedding_model_version == "1.0"
    assert spec.embedding_artifact_hash == "artifact-sha"
    assert spec.to_dict()["effective_corpus_content_token_limit"] == 510
    assert spec.to_dict()["truncation"] is False
    assert EmbeddingSpec.from_dict(spec.to_dict()) == spec


def test_query_and_run_summary_normalize_enum_values_for_serialization():
    request = KnowledgeQuery(
        text="queue imbalance",
        top_k=5,
        content_types=(ContentType.EQUATION, "TABLE"),
        document_ids=("doc_a",),
    )
    assert request.content_types == (ContentType.EQUATION, ContentType.TABLE)
    assert KnowledgeQuery.from_dict(request.to_dict()) == request

    summary = IngestionRunSummary(
        run_id="run-1",
        state="SUCCEEDED",
        counts={IngestionState.INDEXED: 2, IngestionState.UNSUPPORTED: 1},
    )
    assert summary.counts[IngestionState.INDEXED] == 2
    assert IngestionRunSummary.from_dict(summary.to_dict()) == summary


def test_chunk_and_hit_contracts_carry_provenance_and_scores():
    chunk = ChunkRecord(
        chunk_id="chk-1",
        document_id="doc-1",
        source_hash="ab" * 32,
        text="VPIN",
        content_type=ContentType.DEFINITION,
        source_filename="paper.pdf",
        source_relative_path="paper.pdf",
        page_start=4,
        section_path=("Definitions",),
        chunk_ordinal=0,
        content_tokens=4,
        embedding_input_tokens=6,
    )
    hit = KnowledgeHit(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        content_type=chunk.content_type,
        text=chunk.text,
        score=0.9,
        fused_score=0.9,
        source_filename=chunk.source_filename,
        source_relative_path=chunk.source_relative_path,
        source_hash=chunk.source_hash,
        page=4,
    )
    assert hit.content_type is ContentType.DEFINITION
    assert hit.page == hit.page_start == 4
    assert KnowledgeHit.from_dict(hit.to_dict()) == hit


def test_index_generation_round_trips_embedding_spec():
    spec = EmbeddingSpec(model_id="fixture", dimensions=3)
    generation = IndexGeneration(
        generation_id="gen-1",
        vector_location="vector/gen-1",
        lexical_location="keyword/gen-1/bm25.sqlite3",
        embedding_spec=spec,
        lexical_index_version="fts5-v1",
        index_version="index-v1",
        population_hash="population-sha",
        document_count=1,
        chunk_count=1,
    )
    assert IndexGeneration.from_dict(generation.to_dict()) == generation
