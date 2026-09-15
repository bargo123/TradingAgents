import pytest

from tradingagents.distillation.phase7 import Phase7KnowledgeSource, _validate_chunk
from tradingagents.knowledge.models import ChunkRecord


def test_chunk_conversion_preserves_provenance():
    c = ChunkRecord(
        chunk_id="c",
        document_id="d",
        source_hash="h",
        text="microprice",
        source_filename="x.pdf",
        page=4,
        section_path=("2", "2.1"),
        reading_order=3,
    )
    b = Phase7KnowledgeSource.block_from_chunk(c, "gen")
    assert (
        b.ref.document_id == "d"
        and b.ref.page == 4
        and b.ref.section == "2 / 2.1"
        and b.ref.generation_id == "gen"
    )


def test_missing_catalog_is_typed(tmp_path):
    with pytest.raises(Exception, match="missing Phase 7 catalog"):
        Phase7KnowledgeSource.open(tmp_path)


def test_phase7_generation_fingerprint_accepts_path_and_datetime():
    from datetime import UTC, datetime
    from pathlib import Path

    from tradingagents.distillation.models import canonical_hash
    from tradingagents.knowledge.models import EmbeddingSpec, IndexGeneration

    generation = IndexGeneration(
        generation_id="g",
        vector_location=Path("v"),
        lexical_location=Path("l"),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        embedding_spec=EmbeddingSpec(),
    )
    assert canonical_hash(generation)


def test_adapter_rejects_incomplete_or_mismatched_chunk_provenance():
    c = ChunkRecord(
        chunk_id="c",
        document_id="d",
        source_hash="wrong",
        text="microprice",
        source_filename="x.pdf",
    )
    with pytest.raises(Exception, match="source hash mismatch"):
        _validate_chunk(c, "expected", "gen")
    c2 = ChunkRecord(chunk_id="c", document_id="d", source_hash="h", text="microprice")
    with pytest.raises(Exception, match="no source filename"):
        _validate_chunk(c2, "h", "gen")
