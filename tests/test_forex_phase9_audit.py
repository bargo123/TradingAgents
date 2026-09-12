import hashlib
import sqlite3
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from tradingagents.agents.schemas import ForexPortfolioDecision, PortfolioDecision
from tradingagents.forex.evidence_audit import (
    AuditWriteError,
    EvidenceAuditStore,
    EvidenceUsageAudit,
)
from tradingagents.forex.evidence_context import (
    CanonicalEvidenceItem,
    EvidenceContext,
    EvidenceIntegrationStatus,
    EvidenceReferenceRejectionReason,
    EvidenceUseStatus,
    validate_evidence_references,
)


def audit(**overrides):
    context = "K1: evidence"
    values = {
        "decision_id": "D1", "source_run_id": "R1", "integration_status": "INJECTED", "bundle_status": "COMPLETE",
        "as_of": datetime(2026, 1, 1, tzinfo=timezone.utc), "rendered_context": context,
        "rendered_context_hash": hashlib.sha256(context.encode()).hexdigest(), "available_knowledge_ids": ("K1",),
        "evidence_refs_used": ("K1",), "diagnostics": {"code": "ok"}, "source_errors": {}, "retrieval_count": 1,
    }
    values.update(overrides)
    return EvidenceUsageAudit(**values)


def test_audit_schema_contains_required_metadata(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    store.append(audit())
    with sqlite3.connect(tmp_path / "audit.sqlite3") as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(evidence_usage_audit)")}
    assert {"decision_id", "source_run_id", "rendered_context", "rendered_context_hash", "as_of",
            "integration_status", "bundle_status", "evidence_refs_used", "diagnostics"} <= columns


def test_audit_retains_exact_context_payload_and_hash(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    store.append(audit())
    row = store.get("D1")
    assert row["rendered_context"] == "K1: evidence"
    assert row["rendered_context_hash"] == hashlib.sha256(b"K1: evidence").hexdigest()


def test_audit_rejects_prompt_completion_reasoning_and_credentials(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    with pytest.raises(ValueError):
        store.append(audit(diagnostics={"nested": {"prompt": "secret"}}))
    with pytest.raises(ValueError):
        store.append(audit(source_errors={"credential": "x"}))


def test_audit_rejects_hyphenated_forbidden_fields_and_sensitive_values(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    with pytest.raises(ValueError):
        store.append(audit(diagnostics={"nested": {"chain-of-thought": "hidden"}}))
    with pytest.raises(ValueError):
        store.append(audit(source_errors={"nested": ["api-key"]}))


def test_audit_is_append_only(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    store.append(audit())
    other = "different"
    store.append(audit(rendered_context=other, rendered_context_hash=hashlib.sha256(other.encode()).hexdigest()))
    assert len(store.list_for_source_run("R1")) == 2


def test_audit_duplicate_append_is_idempotent(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    store.append(audit())
    store.append(audit())
    assert len(store.list_for_source_run("R1")) == 1


def test_audit_write_failure_is_typed(tmp_path):
    path = tmp_path / "audit.sqlite3"
    path.write_text("not sqlite")
    with pytest.raises(AuditWriteError):
        EvidenceAuditStore(path).initialize()


def test_phase9_evidence_fields_are_forex_only():
    assert not {
        "evidence_use_status",
        "evidence_refs_used",
        "evidence_refs_rejected",
    } & set(PortfolioDecision.model_fields)
    assert {
        "evidence_use_status",
        "evidence_refs_used",
        "evidence_refs_rejected",
    } <= set(ForexPortfolioDecision.model_fields)
    decision = ForexPortfolioDecision(rating="Hold", executive_summary="x", investment_thesis="y")
    assert decision.evidence_use_status == "NONE_RELEVANT"
    assert decision.evidence_refs_used == []
    assert decision.evidence_refs_rejected == []


def _context_with_items() -> EvidenceContext:
    return EvidenceContext(
        integration_status=EvidenceIntegrationStatus.INJECTED,
        knowledge_items=(CanonicalEvidenceItem("K1", "KNOWLEDGE", "k-auth", "knowledge", "rule", 0.9, {}),),
        experience_items=(CanonicalEvidenceItem("E2", "EXPERIENCE", "e-auth", "experience", "trade", 0.8, {}),),
        statistics_items=(CanonicalEvidenceItem("S1", "STATISTICS", "s-auth", "statistics", "summary", 0.7, {}),),
    )


def test_injected_valid_refs_are_used():
    result = validate_evidence_references(
        _context_with_items(),
        {
            "rating": "Buy",
            "evidence_use_status": "USED",
            "evidence_refs_used": ["K1", "E2"],
            "evidence_refs_rejected": [{"ref": "S1", "reason": "LOW_RELEVANCE"}],
        },
        runtime_integration_status=EvidenceIntegrationStatus.INJECTED,
    )

    assert result.evidence_use_status is EvidenceUseStatus.USED
    assert result.evidence_refs_used == ("K1", "E2")
    assert result.evidence_refs_rejected[0].ref == "S1"
    assert result.evidence_audit_status == "VALID"


def test_injected_without_relevant_refs_is_none_relevant():
    result = validate_evidence_references(
        _context_with_items(),
        {"rating": "Hold", "evidence_use_status": "NONE_RELEVANT", "evidence_refs_used": []},
        runtime_integration_status=EvidenceIntegrationStatus.INJECTED,
    )

    assert result.evidence_use_status is EvidenceUseStatus.NONE_RELEVANT
    assert result.evidence_refs_used == ()
    assert result.evidence_refs_rejected == ()


def test_disabled_runtime_overrides_model_status():
    result = validate_evidence_references(
        _context_with_items(),
        {"rating": "Buy", "evidence_use_status": "USED", "evidence_refs_used": ["K1"]},
        runtime_integration_status=EvidenceIntegrationStatus.DISABLED,
    )

    assert result.evidence_use_status is EvidenceUseStatus.DISABLED
    assert result.evidence_refs_used == ()


def test_unavailable_runtime_overrides_model_status_without_usable_evidence():
    result = validate_evidence_references(
        EvidenceContext(integration_status=EvidenceIntegrationStatus.FALLBACK),
        {"rating": "Sell", "evidence_use_status": "USED", "evidence_refs_used": ["K1"]},
        runtime_integration_status=EvidenceIntegrationStatus.FALLBACK,
    )

    assert result.evidence_use_status is EvidenceUseStatus.UNAVAILABLE
    assert result.evidence_refs_used == ()


def test_unknown_references_are_rejected_without_action_change():
    raw = {
        "rating": "Buy",
        "action": "BUY",
        "evidence_use_status": "USED",
        "evidence_refs_used": ["K999", "E404", "S88"],
    }
    original = deepcopy(raw)
    result = validate_evidence_references(
        _context_with_items(), raw, runtime_integration_status=EvidenceIntegrationStatus.INJECTED
    )

    assert raw == original
    assert raw["action"] == "BUY"
    assert result.evidence_refs_used == ()
    assert result.evidence_audit_status == "INVALID_REFERENCE"


def test_used_with_all_invalid_refs_is_audit_inconsistent_without_action_change():
    raw = {"rating": "Sell", "action": "SELL", "evidence_use_status": "USED", "evidence_refs_used": ["K999"]}
    result = validate_evidence_references(
        _context_with_items(), raw, runtime_integration_status=EvidenceIntegrationStatus.INJECTED
    )

    assert raw["action"] == "SELL"
    assert result.evidence_use_status is EvidenceUseStatus.USED
    assert result.evidence_refs_used == ()
    assert result.evidence_audit_status == "INVALID_REFERENCE"


def test_rejection_reason_vocabulary_is_closed():
    result = validate_evidence_references(
        _context_with_items(),
        {
            "evidence_use_status": "USED",
            "evidence_refs_used": ["K1"],
            "evidence_refs_rejected": [{"ref": "E2", "reason": "LOW_RELEVANCE"}],
        },
        runtime_integration_status=EvidenceIntegrationStatus.INJECTED,
    )
    assert result.evidence_refs_rejected[0].reason is EvidenceReferenceRejectionReason.LOW_RELEVANCE
    with pytest.raises(ValueError):
        validate_evidence_references(
            _context_with_items(),
            {"evidence_refs_rejected": [{"ref": "E2", "reason": "MADE_UP"}]},
            runtime_integration_status=EvidenceIntegrationStatus.INJECTED,
        )


def test_internal_nodes_do_not_require_references():
    result = validate_evidence_references(
        _context_with_items(),
        {"rating": "Hold", "analysis": {"claim": "internal node"}},
        runtime_integration_status=EvidenceIntegrationStatus.INJECTED,
    )
    assert result.evidence_use_status is EvidenceUseStatus.NONE_RELEVANT
    assert result.evidence_refs_used == ()


def test_runtime_status_override_never_fabricates_refs():
    result = validate_evidence_references(
        EvidenceContext(integration_status=EvidenceIntegrationStatus.FALLBACK),
        {"evidence_use_status": "DISABLED", "evidence_refs_used": ["K999"]},
        runtime_integration_status=EvidenceIntegrationStatus.FALLBACK,
    )
    assert result.evidence_use_status is EvidenceUseStatus.UNAVAILABLE
    assert result.evidence_refs_used == ()


def test_scalar_reference_containers_are_rejected_without_retaining_refs():
    raw = {
        "action": "BUY",
        "evidence_use_status": "USED",
        "evidence_refs_used": "K1",
        "evidence_refs_rejected": {"ref": "E2", "reason": "LOW_RELEVANCE"},
    }
    result = validate_evidence_references(
        _context_with_items(), raw, runtime_integration_status=EvidenceIntegrationStatus.INJECTED
    )

    assert raw["action"] == "BUY"
    assert result.evidence_refs_used == ()
    assert result.evidence_refs_rejected == ()
    assert result.evidence_audit_status == "INVALID_REFERENCE"


def test_injected_none_relevant_with_valid_citations_is_coerced_to_used():
    result = validate_evidence_references(
        _context_with_items(),
        {"evidence_use_status": "NONE_RELEVANT", "evidence_refs_used": ["K1"]},
        runtime_integration_status=EvidenceIntegrationStatus.INJECTED,
    )

    assert result.evidence_use_status is EvidenceUseStatus.USED
    assert result.evidence_refs_used == ("K1",)
    assert result.evidence_audit_status == "VALID"


def test_unknown_rejection_reason_is_closed_before_unknown_ref_filtering():
    with pytest.raises(ValueError, match="reason"):
        validate_evidence_references(
            _context_with_items(),
            {
                "evidence_refs_rejected": [
                    {"ref": "K999", "reason": "MADE_UP"},
                ]
            },
            runtime_integration_status=EvidenceIntegrationStatus.INJECTED,
        )
