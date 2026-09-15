from tradingagents.distillation.phase7 import Phase7KnowledgeSource
from tradingagents.distillation.models import SourceBlock, SourceRef
from tradingagents.knowledge.models import ChunkRecord
import pytest

def test_chunk_conversion_preserves_provenance():
    c=ChunkRecord(chunk_id="c",document_id="d",source_hash="h",text="microprice",source_filename="x.pdf",page=4,section_path=("2","2.1"),reading_order=3)
    b=Phase7KnowledgeSource.block_from_chunk(c,"gen")
    assert b.ref.document_id=="d" and b.ref.page==4 and b.ref.section=="2 / 2.1" and b.ref.generation_id=="gen"
def test_missing_catalog_is_typed(tmp_path):
    with pytest.raises(Exception, match="missing Phase 7 catalog"): Phase7KnowledgeSource.open(tmp_path)

