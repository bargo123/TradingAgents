"""Bounded, metadata-only eligibility diagnostics for Phase 11 shadow data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tradingagents.datasets.models import BuildReport, DatasetExclusionReason

_SAFE_DECISION_FIELDS = (
    "decision_context_status",
    "normalization_status",
    "decision_reference_status",
    "decision_reference_timestamp",
    "analysis_snapshot_timestamp",
    "decision_completed_timestamp",
)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _text(value: Any, limit: int = 256) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:limit]


def _safe_value(value: Any) -> Any:
    """Keep audit payloads to bounded scalar/list metadata only."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sorted((_safe_value(item) for item in value), key=lambda item: str(item))[:100]
    return _text(value)


def _decision_map(phase56: Any) -> dict[str, Any]:
    return {str(_get(item, "decision_id")): item for item in (_get(phase56, "decisions", ()) or ())}


def _record_map(phase8: Any) -> dict[str, Any]:
    return {
        str(_get(item, "source_decision_id", _get(item, "decision_id", ""))): item
        for item in (_get(phase8, "records", ()) or ())
    }


def _audit_map(phase9: Any) -> dict[tuple[str, str], Any]:
    return {
        (
            str(_get(item, "decision_id", "")),
            str(_get(item, "source_run_id", "")),
        ): item
        for item in (_get(phase9, "audits", ()) or ())
    }


def _evaluation_for(phase56: Any, decision_id: str) -> Any | None:
    rows = [
        row
        for row in (_get(phase56, "evaluations", ()) or ())
        if str(_get(row, "decision_id", "")) == decision_id
    ]
    return min(
        rows,
        key=lambda row: (
            str(_get(row, "evaluation_basis", "")),
            int(_get(row, "horizon_seconds", 0) or 0),
        ),
        default=None,
    )


def _available_refs(audit: Any) -> list[str]:
    values: set[str] = set()
    for field in (
        "available_knowledge_ids",
        "available_experience_ids",
        "available_statistics_ids",
    ):
        raw = _get(audit, field, ()) or ()
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            values.update(str(item) for item in raw)
    return sorted(values)


def _common(decision: Any, record: Any, audit: Any) -> dict[str, Any]:
    fields = _get(decision, "fields", {}) or {}
    values = {
        name: _safe_value(fields.get(name))
        for name in _SAFE_DECISION_FIELDS
        if name in fields
    }
    values.update(
        {
            "phase8_trust": _safe_value(_get(record, "trust")) if record else None,
            "audit_status": _safe_value(_get(audit, "evidence_audit_status")) if audit else None,
        }
    )
    return values


def _upstream(reason: DatasetExclusionReason, decision: Any, record: Any, audit: Any, evaluation: Any) -> dict[str, Any]:
    values = _common(decision, record, audit)
    if reason is DatasetExclusionReason.UNTRUSTED_TIER:
        values.update({"phase8_trust": _safe_value(_get(record, "trust")) if record else None})
    elif reason is DatasetExclusionReason.DECISION_CONTEXT_INCOMPLETE:
        values.update(
            {
                "missing_nodes": _safe_value(_get(audit, "missing_nodes", ())) if audit else None,
                "node_context_hashes_present": bool(_get(audit, "node_context_hashes")) if audit else False,
            }
        )
    elif reason is DatasetExclusionReason.TEMPORAL_INVALID:
        values.update(
            {
                "analysis_snapshot_timestamp": _safe_value(_get(decision, "analysis_snapshot_timestamp")),
                "decision_completed_timestamp": _safe_value(_get(decision, "decision_completed_timestamp")),
                "decision_reference_timestamp": _safe_value(
                    (_get(decision, "fields", {}) or {}).get("decision_reference_timestamp")
                ),
            }
        )
    elif reason is DatasetExclusionReason.NORMALIZATION_FAILED:
        values.update(
            {
                "action": _safe_value(_get(decision, "action")),
                "normalization_status": _safe_value(
                    (_get(decision, "fields", {}) or {}).get("normalization_status")
                ),
            }
        )
    elif reason is DatasetExclusionReason.CITATION_INVALID:
        values.update(
            {
                "available_refs": _available_refs(audit) if audit else [],
                "used_refs": _safe_value(_get(audit, "evidence_refs_used", ())) if audit else [],
                "rejected_refs": _safe_value(_get(audit, "evidence_refs_rejected", ())) if audit else [],
            }
        )
    elif reason in {
        DatasetExclusionReason.OUTCOME_INELIGIBLE,
        DatasetExclusionReason.OUTCOME_UNAVAILABLE,
    }:
        values.update(
            {
                "evaluation_basis": _safe_value(_get(evaluation, "evaluation_basis")) if evaluation else None,
                "horizon_seconds": _safe_value(_get(evaluation, "horizon_seconds")) if evaluation else None,
                "evaluation_status": _safe_value(_get(evaluation, "evaluation_status")) if evaluation else None,
                "source_context_eligible": _safe_value(_get(evaluation, "source_context_eligible")) if evaluation else None,
                "training_eligibility_reason": _safe_value(
                    (_get(evaluation, "fields", {}) or {}).get("training_eligibility_reason")
                ) if evaluation else None,
            }
        )
    elif reason is DatasetExclusionReason.PROVENANCE_INCOMPLETE:
        values["source_decision_fingerprint_present"] = bool(
            _get(record, "source_decision_fingerprint") if record else False
        )
    elif reason is DatasetExclusionReason.SCHEMA_UNSUPPORTED:
        values["experience_schema_version"] = _safe_value(
            _get(record, "experience_schema_version") if record else None
        )
    elif reason is DatasetExclusionReason.SOURCE_INTEGRITY_FAILED:
        values["source_available"] = bool(_get(audit, "available", True)) if audit else None
    return values


def build_exclusion_audit(
    report: BuildReport,
    phase56: Any,
    phase8: Any,
    phase9: Any,
) -> tuple[dict[str, Any], ...]:
    """Map each Phase 10 exclusion reason to bounded upstream metadata."""
    if not isinstance(report, BuildReport):
        raise TypeError("report must be a BuildReport")
    decisions = _decision_map(phase56)
    records = _record_map(phase8)
    audits = _audit_map(phase9)
    items: list[dict[str, Any]] = []
    for exclusion in sorted(report.exclusions, key=lambda item: (item.decision_id, tuple(item.reasons))):
        decision = decisions.get(str(exclusion.decision_id))
        record = records.get(str(exclusion.decision_id))
        run_id = str(_get(decision, "source_run_id", "")) if decision else ""
        audit = audits.get((str(exclusion.decision_id), run_id))
        evaluation = _evaluation_for(phase56, str(exclusion.decision_id))
        for reason in exclusion.reasons:
            reason_value = DatasetExclusionReason(reason)
            items.append(
                {
                    "decision_id": str(exclusion.decision_id),
                    "reason": reason_value.value,
                    "upstream": _upstream(reason_value, decision, record, audit, evaluation),
                }
            )
    return tuple(items)


def summarize_report(report: BuildReport, audit: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the bounded machine-readable Phase 10 lifecycle summary."""
    manifest = report.manifest
    return {
        "status": report.status,
        "generation_id": None if manifest is None else manifest.dataset_id,
        "candidate_count": report.candidate_count,
        "eligible_count": report.eligible_count,
        "excluded_count": report.excluded_count,
        "reason_counts": dict(sorted(report.reason_counts.items())),
        "exclusion_audit": [dict(item) for item in audit],
    }


__all__ = ["build_exclusion_audit", "summarize_report"]
