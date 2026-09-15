from types import SimpleNamespace
from tradingagents.distillation.grounding import GroundingValidator
from tradingagents.distillation.quality import QualityPolicy

def packet(): return SimpleNamespace(refs=[SimpleNamespace(ref_id="r1")])
def test_grounded_claim_requires_packet_reference():
    good=SimpleNamespace(claims=[SimpleNamespace(refs=["r1"])])
    bad=SimpleNamespace(claims=[SimpleNamespace(refs=["foreign"])])
    assert GroundingValidator().validate(good, packet()).accepted
    assert GroundingValidator().validate(bad, packet()).reason == "GROUNDING_FAILED"

def test_quality_rejects_action_and_hidden_reasoning_fields():
    assert QualityPolicy().validate({"target":"BUY"}, packet()).reason == "UNSAFE_FUTURE_OUTCOME_INFERENCE"
    assert QualityPolicy().validate({"reasoning":"private"}, packet()).reason == "UNSAFE_FUTURE_OUTCOME_INFERENCE"

def test_quality_accepts_bounded_knowledge_lesson():
    assert QualityPolicy().validate({"target":"Explain order flow imbalance"}, packet()).accepted
