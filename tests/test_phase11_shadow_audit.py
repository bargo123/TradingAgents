from __future__ import annotations

from datetime import datetime, timezone

from tradingagents.datasets.models import (
    BuildReport,
    DatasetExclusion,
    DatasetExclusionReason,
    DatasetManifest,
)
from tradingagents.datasets.sources import (
    EvaluationObservation,
    SourceObservation,
    SourceReadResult,
)
from tradingagents.finetuning.shadow_audit import (
    build_exclusion_audit,
    summarize_report,
)

UTC = timezone.utc


def _sources() -> tuple[SourceReadResult, SourceReadResult, SourceReadResult]:
    decision = SourceObservation(
        "d1",
        datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 12, 1, tzinfo=UTC),
        source_run_id="run-1",
        requested_symbol="EURUSD",
        resolved_symbol="EURUSD",
        analysis_profile="INTRADAY",
        analysis_timeframe="M1/M5/M15/H1",
        action=None,
        fields={
            "normalization_status": "FAILED",
            "decision_context_status": "INCOMPLETE",
            "decision_reference_timestamp": "2026-01-01T12:00:30Z",
            "decision_reference_status": "INVALID_TEMPORAL",
        },
    )
    evaluation = EvaluationObservation(
        "d1",
        "ANALYSIS_SNAPSHOT",
        300,
        "INELIGIBLE",
        False,
        {"training_eligibility_reason": "SOURCE_CONTEXT_INELIGIBLE"},
    )
    phase8 = SourceReadResult(
        records=(
            {
                "experience_id": "exp-1",
                "source_decision_id": "d1",
                "trust": "TIER_C_DIAGNOSTIC_ONLY",
            },
        )
    )
    phase9 = SourceReadResult(
        audits=(
            {
                "decision_id": "d1",
                "source_run_id": "run-1",
                "evidence_audit_status": "INVALID_REFERENCE",
                "available_knowledge_ids": ["K1", "K2"],
                "available_experience_ids": [],
                "available_statistics_ids": [],
                "evidence_refs_used": [],
                "evidence_refs_rejected": [],
                "missing_nodes": [],
                "node_context_hashes": None,
            },
        )
    )
    phase56 = SourceReadResult(decisions=(decision,), evaluations=(evaluation,))
    return phase56, phase8, phase9


def _report() -> BuildReport:
    reasons = tuple(DatasetExclusionReason)
    return BuildReport(
        status="EMPTY_ELIGIBLE_SET",
        manifest=DatasetManifest(
            dataset_id="gen-audit",
            examples=0,
            exclusions=1,
            candidate_count=1,
            eligible_count=0,
            excluded_count=1,
            reason_counts={reason.value: 1 for reason in reasons},
        ),
        exclusions=(DatasetExclusion("d1", reasons, {"decision_id": "d1"}),),
    )


def test_exclusion_audit_maps_reasons_to_scalar_upstream_state() -> None:
    phase56, phase8, phase9 = _sources()

    audit = build_exclusion_audit(_report(), phase56, phase8, phase9)

    assert [item["reason"] for item in audit] == [reason.value for reason in DatasetExclusionReason]
    by_reason = {item["reason"]: item["upstream"] for item in audit}
    assert by_reason["UNTRUSTED_TIER"]["phase8_trust"] == "TIER_C_DIAGNOSTIC_ONLY"
    assert by_reason["DECISION_CONTEXT_INCOMPLETE"]["decision_context_status"] == "INCOMPLETE"
    assert by_reason["TEMPORAL_INVALID"]["decision_reference_status"] == "INVALID_TEMPORAL"
    assert by_reason["NORMALIZATION_FAILED"]["normalization_status"] == "FAILED"
    assert by_reason["CITATION_INVALID"]["available_refs"] == ["K1", "K2"]
    assert by_reason["OUTCOME_INELIGIBLE"]["evaluation_status"] == "INELIGIBLE"
    assert all("snapshot_json" not in item["upstream"] for item in audit)


def test_exclusion_audit_and_summary_are_deterministic_and_empty_for_eligible() -> None:
    phase56, phase8, phase9 = _sources()
    report = BuildReport(
        status="PUBLISHED",
        manifest=DatasetManifest(
            dataset_id="gen-ok",
            examples=1,
            exclusions=0,
            candidate_count=1,
            eligible_count=1,
            excluded_count=0,
            reason_counts={},
        ),
        exclusions=(),
    )

    assert build_exclusion_audit(report, phase56, phase8, phase9) == ()
    summary = summarize_report(report, ())
    assert summary["eligible_count"] == 1
    assert summary["exclusion_audit"] == []
