import json
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


def test_nested_mappings_are_immutable():
    context = EvidenceContext(diagnostics={"nested": {"x": [1]}})
    with pytest.raises(TypeError):
        context.diagnostics["new"] = 1
    with pytest.raises(TypeError):
        context.diagnostics["nested"]["x"][0] = 2


def test_contracts_are_json_serializable():
    item = CanonicalEvidenceItem("k1", "KNOWLEDGE", "auth", "text", "rule", 0.5, {"source": "x"})
    context = EvidenceContext(knowledge_query=CanonicalKnowledgeQuery("q", "fp", "v1"), knowledge_items=(item,))
    payload = context.canonical_payload_bytes()
    assert json.loads(payload) == context.to_dict()
    assert isinstance(EvidenceSnapshotAdapter().to_market_state, object)
