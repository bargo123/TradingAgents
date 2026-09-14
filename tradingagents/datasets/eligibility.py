"""Pure, fail-closed joining and eligibility gates for Phase 10."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from tradingagents.experience.features import extract_market_state
from tradingagents.experience.identity import (
    source_decision_fingerprint,
    source_evaluation_fingerprint,
)
from tradingagents.experience.models import TrustTier
from tradingagents.experience.trust import classify_trust

from .models import (
    DatasetConfig,
    DatasetExclusion,
    DatasetExclusionReason,
    EvaluationObservation,
    EvidenceObservation,
    JoinedObservation,
)


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    eligible: bool
    reasons: tuple[DatasetExclusionReason, ...] = ()
    details: Mapping[str, Any] = None

    def __post_init__(self):
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details or {})))

    @property
    def exclusion(self) -> DatasetExclusion | None:
        return None if self.eligible else DatasetExclusion(str(self.details.get("decision_id", "unknown")), self.reasons, self.details)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _dt(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return value if isinstance(value, datetime) and value.tzinfo and value.utcoffset() == timezone.utc.utcoffset(value) else None


def _key(decision_id: Any, run: Any = "", fingerprint: Any = ""):
    return (str(decision_id or ""), str(run or ""), str(fingerprint or ""))


def _decision_row(decision: Any) -> dict[str, Any]:
    """Reconstruct the source row used by the Phase 8 identity helper."""
    row = dict(_get(decision, "fields", {}) or {})
    row.update(
        {
            "decision_id": _get(decision, "decision_id", ""),
            "source_run_id": _get(decision, "source_run_id", ""),
            "requested_symbol": _get(decision, "requested_symbol", ""),
            "resolved_symbol": _get(decision, "resolved_symbol", ""),
            "analysis_profile": _get(decision, "analysis_profile", ""),
            "analysis_timeframe": _get(decision, "analysis_timeframe", ""),
            "action": _get(decision, "action", ""),
            "analysis_snapshot_timestamp": _get(
                decision, "analysis_snapshot_timestamp"
            ),
            "decision_completed_timestamp": _get(
                decision, "decision_completed_timestamp"
            ),
        }
    )
    row.pop("source_decision_fingerprint", None)
    return row


def _decision_fingerprint(decision: Any) -> str:
    value = _get(decision, "fields", {}).get("source_decision_fingerprint")
    return str(value) if value else source_decision_fingerprint(_decision_row(decision))


def _evaluation_row(evaluation: Any) -> dict[str, Any]:
    row = dict(_get(evaluation, "fields", {}) or {})
    row.update(
        {
            "decision_id": _get(evaluation, "decision_id", ""),
            "evaluation_basis": _get(evaluation, "evaluation_basis", ""),
            "horizon_seconds": _get(evaluation, "horizon_seconds", 0),
            "evaluation_status": _get(evaluation, "evaluation_status", ""),
        }
    )
    row.pop("source_evaluation_fingerprint", None)
    row.pop("fingerprint", None)
    return row


def _evaluation_fingerprint(evaluation: Any) -> str:
    fields = _get(evaluation, "fields", {}) or {}
    value = fields.get("source_evaluation_fingerprint") or fields.get("fingerprint")
    return str(value) if value else source_evaluation_fingerprint(_evaluation_row(evaluation))


def _with_evaluation_fingerprint(evaluation: Any):
    """Attach a deterministic source evaluation identity to adapter facts."""
    fields = dict(_get(evaluation, "fields", {}) or {})
    if fields.get("source_evaluation_fingerprint") or fields.get("fingerprint"):
        return evaluation
    fields["source_evaluation_fingerprint"] = _evaluation_fingerprint(evaluation)
    return type(evaluation)(
        _get(evaluation, "decision_id"),
        _get(evaluation, "evaluation_basis"),
        _get(evaluation, "horizon_seconds"),
        _get(evaluation, "evaluation_status"),
        _get(evaluation, "source_context_eligible"),
        fields,
    )


def _evaluation_metadata(evaluation: Any) -> dict[str, Any]:
    """Convert an immutable evaluation contract to JSON-safe join metadata."""
    return {
        "decision_id": _get(evaluation, "decision_id"),
        "evaluation_basis": _get(evaluation, "evaluation_basis"),
        "horizon_seconds": _get(evaluation, "horizon_seconds"),
        "evaluation_status": _get(evaluation, "evaluation_status"),
        "source_context_eligible": _get(evaluation, "source_context_eligible"),
        "fields": dict(_get(evaluation, "fields", {}) or {}),
    }


def _evaluation_contract(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _with_evaluation_fingerprint(
            EvaluationObservation(
                str(value.get("decision_id")),
                str(value.get("evaluation_basis")),
                int(value.get("horizon_seconds")),
                str(value.get("evaluation_status")),
                bool(value.get("source_context_eligible")),
                value.get("fields", {}),
            )
        )
    return value


def _enrich_evaluation(evaluation: Any, record: Any) -> Any:
    """Carry Phase 8 snapshot fingerprint/provenance beside the source row."""
    snapshots = _get(record, "evaluation_snapshots", ()) if record else ()
    if not snapshots:
        return evaluation
    basis = _get(evaluation, "evaluation_basis")
    horizon = int(_get(evaluation, "horizon_seconds", 0) or 0)
    matching = tuple(
        snapshot
        for snapshot in snapshots
        if _get(snapshot, "evaluation_basis") == basis
        and int(_get(snapshot, "horizon_seconds", 0) or 0) == horizon
    )
    if not matching:
        return evaluation
    snapshot = matching[0]
    fields = dict(_get(evaluation, "fields", {}) or {})
    snapshot_fp = _get(snapshot, "fingerprint")
    if snapshot_fp:
        fields["phase8_evaluation_fingerprint"] = str(snapshot_fp)
    provenance = dict(_get(snapshot, "provenance", {}) or {})
    provenance.setdefault("source_decision_id", _get(record, "source_decision_id"))
    if provenance:
        fields.setdefault("provenance", provenance)
    if _get(snapshot, "observed_at") is not None:
        fields.setdefault("observed_at", _get(snapshot, "observed_at"))
    return EvaluationObservation(
        _get(evaluation, "decision_id"),
        _get(evaluation, "evaluation_basis"),
        _get(evaluation, "horizon_seconds"),
        _get(evaluation, "evaluation_status"),
        _get(evaluation, "source_context_eligible"),
        fields,
    )


def join_observations(phase56: Any, experience: Any, audit: Any) -> tuple[JoinedObservation, ...]:
    """Join adapter results without mutation; ambiguous identities remain marked."""
    decisions = tuple(_get(phase56, "decisions", ()) or ())
    evaluations = tuple(_get(phase56, "evaluations", ()) or ())
    records = tuple(_get(experience, "records", ()) or ())
    audits = tuple(_get(audit, "audits", ()) or ())
    unavailable = not bool(_get(audit, "available", True))
    recs = defaultdict(list)
    for row in records:
        recs[_key(_get(row, "source_decision_id", _get(row, "decision_id")), _get(row, "source_run_id"), _get(row, "source_decision_fingerprint"))].append(row)
    evals = defaultdict(list)
    for row in evaluations:
        row = _with_evaluation_fingerprint(row)
        evals[str(_get(row, "decision_id"))].append(row)
    auds = defaultdict(list)
    for row in audits:
        auds[_key(_get(row, "decision_id"), _get(row, "source_run_id"))].append(row)
    decision_counts = Counter(
        _key(_get(d, "decision_id"), _get(d, "source_run_id"), _decision_fingerprint(d))
        for d in decisions
    )
    out = []
    for decision in sorted(decisions, key=lambda x: str(_get(x, "decision_id"))):
        fp = _decision_fingerprint(decision)
        rid = _get(decision, "source_run_id", "")
        did = _get(decision, "decision_id")
        rec = recs.get(_key(did, rid, fp), [])
        record = rec[0] if rec else None
        ev = evals.get(str(did), [])
        ev = [
            x
            for x in ev
            if _get(x, "fields", {}).get("source_run_id", rid) == rid
            and _get(x, "fields", {}).get("source_decision_fingerprint", fp) == fp
        ]
        ev = sorted(
            ev,
            key=lambda x: (
                _get(x, "evaluation_basis", ""),
                int(_get(x, "horizon_seconds", 0) or 0),
                _evaluation_fingerprint(x),
            ),
        )
        ev = [_enrich_evaluation(x, record) for x in ev]
        evaluation_counts = Counter(
            (
                _get(x, "decision_id"),
                _get(x, "evaluation_basis"),
                int(_get(x, "horizon_seconds", 0) or 0),
            )
            for x in ev
        )
        duplicate_evaluation_keys = tuple(
            ":".join((str(key[0]), str(key[1]), str(key[2])))
            for key, count in sorted(evaluation_counts.items())
            if count > 1
        )
        evaluation = ev[0] if ev else None
        ar = auds.get(_key(did, rid), [])
        evidence = None
        if ar:
            row = ar[0]
            rejected = tuple(x.get("ref", "") if isinstance(x, Mapping) else x for x in (_get(row, "evidence_refs_rejected", _get(row, "refs_rejected", ())) or ()))
            evidence = EvidenceObservation(
                context_integrity=str(_get(row, "context_integrity", "INCOMPLETE")),
                evidence_use_status=str(_get(row, "evidence_use_status", "INCOMPLETE")),
                refs_used=tuple(_get(row, "evidence_refs_used", _get(row, "refs_used", ())) or ()),
                refs_rejected=rejected,
                fields=row,
            )
        source_fp = _get(phase56, "fingerprint", None)
        if source_fp is not None and hasattr(source_fp, "to_dict"):
            source_fp = source_fp.to_dict()
        duplicate_non_evaluation = (
            decision_counts[_key(did, rid, fp)] > 1 or len(rec) > 1 or len(ar) > 1
        )
        metadata = {
            "experience": record,
            "audit": (ar[0] if ar else None),
            "source_fingerprint": source_fp,
            # Different basis/horizon rows are distinct outcomes.  Only
            # identical decision/basis/horizon rows are duplicate evidence.
            "duplicate": duplicate_non_evaluation or bool(duplicate_evaluation_keys),
            "duplicate_non_evaluation": duplicate_non_evaluation,
            "duplicate_evaluation_keys": duplicate_evaluation_keys,
            "evaluations": tuple(_evaluation_metadata(x) for x in ev),
            "audit_unavailable": unavailable,
            "source_available": bool(_get(phase56, "available", True)),
        }
        out.append(JoinedObservation(decision, evaluation, evidence, metadata))
    return tuple(out)


def classify_observation(observation: JoinedObservation, config: DatasetConfig) -> EligibilityResult:
    d = observation.decision
    f = dict(d.fields)
    joined_fields = _get(observation, "fields", {})
    record = joined_fields.get("experience")
    audit = joined_fields.get("audit") or {}
    reasons: set[DatasetExclusionReason] = set()
    candidates = tuple(joined_fields.get("evaluations") or ())
    requested = tuple(
        ev
        for ev in candidates
        if _get(ev, "evaluation_basis") == config.evaluation_basis
        and int(_get(ev, "horizon_seconds", 0) or 0) == config.horizon_seconds
    )
    ev = _evaluation_contract(requested[0]) if requested else observation.evaluation
    duplicate_requested = len(requested) > 1
    if joined_fields.get("duplicate_non_evaluation") or duplicate_requested:
        reasons.add(DatasetExclusionReason.DUPLICATE)
    if not joined_fields.get("source_available", True):
        reasons.add(DatasetExclusionReason.SOURCE_INTEGRITY_FAILED)
    trust_ok = False
    if record:
        try:
            trust_row = dict(record)
            trust_row.update({"resolved_symbol": d.resolved_symbol, "analysis_profile": d.analysis_profile, "analysis_timeframe": d.analysis_timeframe, "analysis_snapshot_timestamp": d.analysis_snapshot_timestamp, "action": d.action, "decision_completed_timestamp": d.decision_completed_timestamp, **f})
            if "snapshot_json" not in trust_row and isinstance(trust_row.get("market_state"), Mapping):
                trust_row["snapshot_json"] = trust_row["market_state"]
            trust_ok = classify_trust(trust_row, extract_market_state(trust_row)).tier is TrustTier.TIER_A_HIGH_TRUST
        except Exception:
            trust_ok = False
    if not trust_ok:
        reasons.add(DatasetExclusionReason.UNTRUSTED_TIER)
    if str(f.get("decision_context_status", "")).upper() != "COMPLETE" or not audit or str(_get(observation.evidence, "context_integrity", "")).upper() != "COMPLETE":
        reasons.add(DatasetExclusionReason.DECISION_CONTEXT_INCOMPLETE)
        if audit and str(_get(observation.evidence, "context_integrity", "")).upper() != "COMPLETE":
            reasons.add(DatasetExclusionReason.EVIDENCE_INCOMPLETE)
    analysis = d.analysis_snapshot_timestamp
    completed = d.decision_completed_timestamp
    reference_raw = _get(
        record, "decision_reference_timestamp", f.get("decision_reference_timestamp")
    )
    reference = _dt(reference_raw)
    cutoff = _dt(_get(config, "filters", {}).get("as_of"))
    if completed is None or completed < analysis or (
        reference_raw not in (None, "") and reference is None
    ) or (
        reference is not None
        and (reference < analysis or (completed is not None and reference < completed))
    ) or (
        cutoff
        and (analysis > cutoff or completed > cutoff or (reference and reference > cutoff))
    ):
        reasons.add(DatasetExclusionReason.TEMPORAL_INVALID)
    if str(f.get("normalization_status", "")).upper() != "NORMALIZED" or ("action" in f and str(f["action"]).upper() not in {"BUY", "SELL", "HOLD"}):
        reasons.add(DatasetExclusionReason.NORMALIZATION_FAILED)
    required_audit = ("context_integrity", "evidence_use_status", "evidence_audit_status", "evidence_refs_used", "evidence_refs_rejected")
    if any(name not in audit for name in required_audit) or str(_get(audit, "evidence_audit_status", "")).upper() != "VALID":
        reasons.add(DatasetExclusionReason.CITATION_INVALID)
    if audit and all(name in audit for name in required_audit):
        available = set(_get(audit, "available_knowledge_ids", ()) or ()) | set(_get(audit, "available_experience_ids", ()) or ()) | set(_get(audit, "available_statistics_ids", ()) or ())
        used = set(_get(audit, "evidence_refs_used", ()) or ())
        rejected = set(_get(audit, "evidence_refs_rejected", ()) or ())
        rejected = {str(x.get("ref")) if isinstance(x, Mapping) else str(x) for x in rejected}
        if used & rejected or used | rejected != {str(x) for x in available}:
            reasons.add(DatasetExclusionReason.CITATION_INVALID)
    if not audit or str(_get(observation.evidence, "context_integrity", "")).upper() != "COMPLETE":
        reasons.add(DatasetExclusionReason.EVIDENCE_INCOMPLETE)
    expected_fp = _decision_fingerprint(d)
    source_fp = joined_fields.get("source_fingerprint")
    has_evaluation_provenance = bool(
        _get(record, "source_evaluation_fingerprints")
        or _get(record, "evaluation_snapshots")
    )
    if not record or not expected_fp or _get(record, "source_decision_fingerprint") != expected_fp or not source_fp or not has_evaluation_provenance or not _get(record, "provenance") or not _get(record, "experience_schema_version") or not _get(record, "feature_schema_version") or not _get(record, "feature_extractor_version") or not _get(record, "trust_policy_version"):
        reasons.add(DatasetExclusionReason.PROVENANCE_INCOMPLETE)
    if record and any(str(_get(record, n, "")).startswith("unsupported") for n in ("experience_schema_version", "feature_schema_version")):
        reasons.add(DatasetExclusionReason.SCHEMA_UNSUPPORTED)
    if ev is None:
        reasons.add(DatasetExclusionReason.OUTCOME_UNAVAILABLE)
    elif ev.evaluation_basis != config.evaluation_basis or ev.horizon_seconds != config.horizon_seconds or ev.evaluation_status != "COMPLETE" or not ev.source_context_eligible:
        reasons.add(DatasetExclusionReason.OUTCOME_INELIGIBLE)
    if record and ev is not None:
        provenance = _get(record, "provenance", {}) or {}
        eval_fps = _get(record, "source_evaluation_fingerprints", {}) or {}
        eval_key = f"{ev.evaluation_basis}:{ev.horizon_seconds}"
        actual_eval_fp = _evaluation_fingerprint(ev)
        eval_provenance = _get(ev.fields, "provenance", {}) or {}
        stored_eval_fp = _get(eval_fps, eval_key) or _get(
            ev.fields, "phase8_evaluation_fingerprint"
        )
        phase8_eval_fp = _get(ev.fields, "phase8_evaluation_fingerprint")
        if (_get(provenance, "source_decision_fingerprint") != expected_fp
            or not actual_eval_fp or stored_eval_fp != actual_eval_fp
            or (phase8_eval_fp is not None and phase8_eval_fp != actual_eval_fp)
            or _get(eval_provenance, "evaluation_fingerprint") != actual_eval_fp
                or _get(eval_provenance, "source_decision_id", _get(eval_provenance, "decision_id")) != d.decision_id):
            reasons.add(DatasetExclusionReason.PROVENANCE_INCOMPLETE)
    if ev is not None:
        times = {}
        for name in (
            "known_at",
            "created_at",
            "evaluated_at",
            "recovered_from_unavailable_at",
            "evaluation_timestamp",
            "entry_timestamp",
            "observation_timestamp",
            "target_timestamp",
        ):
            raw_known = _get(ev.fields, name)
            if raw_known is not None and _dt(raw_known) is None:
                reasons.add(DatasetExclusionReason.TEMPORAL_INVALID)
            known = _dt(raw_known)
            times[name] = known
            # Outcome evaluation occurs after the decision by design.  Its
            # lifecycle/observation timestamps are facts about when the result
            # became known, not future inputs to the decision.  An as-of build
            # still cannot use an outcome observed after its cutoff.
            if cutoff and known is not None and known > cutoff:
                reasons.add(DatasetExclusionReason.TEMPORAL_INVALID)
        if times.get("target_timestamp") and times["target_timestamp"] < analysis:
            reasons.add(DatasetExclusionReason.TEMPORAL_INVALID)
        if times.get("target_timestamp") and times.get("entry_timestamp") and times["entry_timestamp"] > times["target_timestamp"]:
            reasons.add(DatasetExclusionReason.TEMPORAL_INVALID)
        if times.get("entry_timestamp") and times.get("observation_timestamp") and times["observation_timestamp"] < times["entry_timestamp"]:
            reasons.add(DatasetExclusionReason.TEMPORAL_INVALID)
    order = tuple(DatasetExclusionReason)
    ordered = tuple(reason for reason in order if reason in reasons)
    return EligibilityResult(not ordered, ordered, {"decision_id": d.decision_id})


__all__ = ["EligibilityResult", "join_observations", "classify_observation"]
