import json
from collections import deque
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.forex.evidence_context import (
    CanonicalEvidenceItem,
    CanonicalKnowledgeQuery,
    EvidenceAuditStatus,
    EvidenceBundleStatus,
    EvidenceContext,
    EvidenceIntegrationStatus,
    EvidenceQueryPolicy,
    EvidenceReferenceRejection,
    EvidenceReferenceRejectionReason,
    EvidenceSnapshotAdapter,
    EvidenceUseStatus,
)


def test_evidence_context_contains_complete_v1_field_set():
    names = {f.name for f in EvidenceContext.__dataclass_fields__.values()}
    assert names == {
        "version", "integration_status", "bundle_status", "as_of", "knowledge_generation_id",
        "experience_generation_id", "query_normalization_fingerprint", "knowledge_query",
        "knowledge_query_fingerprint", "knowledge_query_policy_version", "knowledge_items",
        "experience_items", "statistics_items", "statistics_status", "diagnostics",
        "source_errors", "rendered_context", "rendered_context_hash", "selected_knowledge_count",
        "selected_experience_count", "selected_statistics_count", "dropped_knowledge_count",
        "dropped_experience_count", "dropped_statistics_count", "rendered_character_count",
        "budget_policy_version",
    }


def test_context_and_policy_are_frozen():
    context = EvidenceContext()
    policy = EvidenceQueryPolicy()
    with pytest.raises(FrozenInstanceError):
        context.version = "x"
    with pytest.raises(FrozenInstanceError):
        policy.knowledge_top_k = 2


def test_context_rejects_naive_or_non_utc_as_of():
    with pytest.raises(ValueError):
        EvidenceContext(as_of=datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        EvidenceContext(as_of=datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=2))))


def test_closed_status_and_rejection_enums():
    assert {x.value for x in EvidenceIntegrationStatus} == {"DISABLED", "INJECTED", "FALLBACK"}
    assert {x.value for x in EvidenceBundleStatus} == {"COMPLETE", "PARTIAL", "EMPTY", "FAILED"}
    assert {x.value for x in EvidenceUseStatus} == {"USED", "NONE_RELEVANT", "UNAVAILABLE", "DISABLED"}
    assert {x.value for x in EvidenceAuditStatus} == {"VALID", "INVALID_REFERENCE", "NOT_RECORDED", "WRITE_FAILED"}
    assert {x.value for x in EvidenceReferenceRejectionReason} == {
        "CONFLICTS_WITH_CURRENT_STATE", "LOW_RELEVANCE", "INSUFFICIENT_SAMPLE", "DIAGNOSTIC_ONLY", "REDUNDANT"
    }
    with pytest.raises(ValueError):
        EvidenceReferenceRejection(ref="x", reason="NOPE")


def test_forex_portfolio_schema_closes_rejection_reason_contract():
    """The generated PM schema must match the runtime's closed reason enum."""
    from tradingagents.agents.schemas import ForexPortfolioDecision

    reason_schema = ForexPortfolioDecision.model_json_schema()["$defs"][
        "EvidenceReferenceRejection"
    ]["properties"]["reason"]

    assert reason_schema["$ref"].endswith("EvidenceReferenceRejectionReason")
    assert "anyOf" not in reason_schema
    with pytest.raises(ValueError):
        ForexPortfolioDecision.model_validate(
            {
                "rating": "Hold",
                "executive_summary": "Remain flat.",
                "investment_thesis": "Evidence is balanced.",
                "evidence_refs_rejected": [
                    {"ref": "K1", "reason": "free-form explanation"}
                ],
            }
        )


def test_forex_portfolio_schema_rejects_used_refs_with_none_relevant_status():
    from tradingagents.agents.schemas import ForexPortfolioDecision

    with pytest.raises(ValueError, match="NONE_RELEVANT"):
        ForexPortfolioDecision.model_validate(
            {
                "rating": "Hold",
                "executive_summary": "Remain flat.",
                "investment_thesis": "Evidence is not relevant.",
                "evidence_use_status": "NONE_RELEVANT",
                "evidence_refs_used": ["K1"],
            }
        )


def test_forex_portfolio_schema_accepts_none_relevant_with_explicit_rejections():
    from tradingagents.agents.schemas import ForexPortfolioDecision

    decision = ForexPortfolioDecision.model_validate(
        {
            "rating": "Hold",
            "executive_summary": "Remain flat.",
            "investment_thesis": "The supplied evidence is not relevant.",
            "evidence_use_status": "NONE_RELEVANT",
            "evidence_refs_used": [],
            "evidence_refs_rejected": [
                {"ref": "K1", "reason": "LOW_RELEVANCE"},
                {"ref": "K2", "reason": "LOW_RELEVANCE"},
            ],
        }
    )

    assert decision.evidence_use_status == "NONE_RELEVANT"
    assert [item.ref for item in decision.evidence_refs_rejected] == ["K1", "K2"]


def test_nested_mappings_are_immutable():
    context = EvidenceContext(diagnostics={"nested": {"x": [1], "queue": deque([2, 3])}})
    with pytest.raises(TypeError):
        context.diagnostics["new"] = 1
    with pytest.raises(TypeError):
        context.diagnostics["nested"]["x"][0] = 2
    with pytest.raises(TypeError):
        context.diagnostics["nested"]["queue"][0] = 4


def test_nested_timestamps_are_rejected_or_normalized_to_utc():
    with pytest.raises(ValueError):
        EvidenceContext(diagnostics={"nested": {"when": datetime(2026, 1, 1)}})
    offset = datetime(2026, 1, 1, 2, tzinfo=timezone(timedelta(hours=2)))
    context = EvidenceContext(diagnostics={"nested": {"when": offset}})
    assert context.diagnostics["nested"]["when"].tzinfo is timezone.utc
    assert context.diagnostics["nested"]["when"].hour == 0


def test_statistics_status_defaults_to_not_requested():
    assert EvidenceContext().statistics_status == "NOT_REQUESTED"


def test_collection_entries_are_deeply_immutable():
    context = EvidenceContext(
        knowledge_items=({"nested": {"items": [1]}},),
        experience_items=([{"x": 1}],),
        statistics_items=({"values": [1, 2]},),
    )
    with pytest.raises(TypeError):
        context.knowledge_items[0]["nested"]["items"][0] = 9
    with pytest.raises(TypeError):
        context.experience_items[0][0]["x"] = 9
    with pytest.raises(TypeError):
        context.statistics_items[0]["values"][0] = 9


def test_contracts_are_json_serializable():
    item = CanonicalEvidenceItem("k1", "KNOWLEDGE", "auth", "text", "rule", 0.5, {"source": "x"})
    context = EvidenceContext(knowledge_query=CanonicalKnowledgeQuery("q", "fp", "v1"), knowledge_items=(item,))
    payload = context.canonical_payload_bytes()
    assert json.loads(payload) == context.to_dict()
    assert isinstance(EvidenceSnapshotAdapter().to_market_state, object)
