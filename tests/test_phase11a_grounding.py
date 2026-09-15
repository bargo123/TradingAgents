from types import SimpleNamespace

from tradingagents.distillation.grounding import GroundingValidator
from tradingagents.distillation.quality import QualityPolicy


def packet():
    return SimpleNamespace(refs=[SimpleNamespace(ref_id="r1")])


def test_grounded_claim_requires_packet_reference():
    good = SimpleNamespace(claims=[SimpleNamespace(refs=["r1"])])
    bad = SimpleNamespace(claims=[SimpleNamespace(refs=["foreign"])])
    assert GroundingValidator().validate(good, packet()).accepted
    assert GroundingValidator().validate(bad, packet()).reason == "GROUNDING_FAILED"


def test_grounding_accepts_chunk_id_and_string_references():
    p = SimpleNamespace(refs=[SimpleNamespace(chunk_id="chunk-1")])
    assert (
        GroundingValidator()
        .validate(SimpleNamespace(claims=[SimpleNamespace(source_refs=["chunk-1"])]), p)
        .valid
    )


def test_grounding_checks_all_candidate_refs_and_full_structured_provenance():
    good = SimpleNamespace(
        source_refs=["r1", "r2"],
        claims=[SimpleNamespace(source_refs=["r1"])],
    )
    bad = SimpleNamespace(
        source_refs=["r1", "foreign"],
        claims=[SimpleNamespace(source_refs=["r1"])],
    )
    p = SimpleNamespace(refs=[SimpleNamespace(ref_id="r1"), SimpleNamespace(ref_id="r2")])
    assert GroundingValidator().validate(good, p).accepted
    assert GroundingValidator().validate(bad, p).reason == "GROUNDING_FAILED"


def test_grounding_rejects_structured_ref_missing_generation_provenance():
    expected = SimpleNamespace(
        chunk_id="c1", document_id="doc", source_hash="hash", generation_id="gen"
    )
    provided = SimpleNamespace(chunk_id="c1", document_id="doc", source_hash="hash")
    candidate = SimpleNamespace(
        source_refs=[provided], claims=[SimpleNamespace(source_refs=[provided])]
    )
    assert GroundingValidator().validate(candidate, SimpleNamespace(refs=[expected])).reason == (
        "SOURCE_PROVENANCE_INCOMPLETE"
    )


def test_dataset_exclusion_can_persist_source_refs():
    from tradingagents.distillation.models import DatasetExclusion, SourceRef

    value = DatasetExclusion("GROUNDING_FAILED", source_refs=(SourceRef("d", "f", "h", "c", "g"),))
    assert value.to_dict()["source_refs"][0]["chunk_id"] == "c"


def test_quality_rejects_action_and_hidden_reasoning_fields():
    assert (
        QualityPolicy().validate({"target": "BUY"}, packet()).reason
        == "UNSAFE_FUTURE_OUTCOME_INFERENCE"
    )
    assert (
        QualityPolicy().validate({"reasoning": "private"}, packet()).reason
        == "UNSAFE_FUTURE_OUTCOME_INFERENCE"
    )


def test_quality_accepts_bounded_knowledge_lesson():
    assert QualityPolicy().validate({"target": "Explain order flow imbalance"}, packet()).accepted


def test_quality_rejects_excessive_verbatim_overlap():
    p = SimpleNamespace(
        blocks=[SimpleNamespace(text="order flow imbalance measures buying and selling pressure")]
    )
    candidate = {"target": "order flow imbalance measures buying and selling pressure"}
    assert (
        QualityPolicy(max_overlap_ratio=0.8).validate(candidate, p).reason
        == "VERBATIM_OVERLAP_EXCESSIVE"
    )
