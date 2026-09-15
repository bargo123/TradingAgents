from tradingagents.distillation.models import *
import pytest

def ref(): return SourceRef("d","book.pdf","h","c","g",page=2,content_type="EQUATION")
def test_contracts_are_versioned_and_canonical():
    r=ref(); assert CONTRACT_VERSION.endswith(".v1"); assert canonical_hash(r)==canonical_hash(r)
    with pytest.raises(ValueError): SourceRef("","book.pdf","h","c","g")
    with pytest.raises(ValueError): SourceBlock(r,"x",metadata={"prompt":"bad"})
def test_packet_and_example_contracts_are_json_safe():
    block=SourceBlock(ref(),"OFI measures signed order flow")
    packet=SourcePacket("p",(block,)); assert packet.refs==(ref(),)
    ex=KnowledgeExample("e",LessonType.DEFINITION,"OFI",Difficulty.FOUNDATIONAL,"sys","user","assistant",(ref(),), (GroundingClaim("OFI",(ref(),)),))
    assert ex.to_dict()["lesson_type"]=="DEFINITION"

