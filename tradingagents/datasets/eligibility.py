"""Pure, fail-closed joining and eligibility gates for Phase 10."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from tradingagents.experience.features import extract_market_state
from tradingagents.experience.models import TrustTier
from tradingagents.experience.trust import classify_trust

from .models import (
    DatasetConfig,
    DatasetExclusion,
    DatasetExclusionReason,
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
        evals[str(_get(row, "decision_id"))].append(row)
    auds = defaultdict(list)
    for row in audits:
        auds[_key(_get(row, "decision_id"), _get(row, "source_run_id"))].append(row)
    decision_counts = Counter(_key(_get(d, "decision_id"), _get(d, "source_run_id"), _get(_get(d, "fields", {}), "source_decision_fingerprint")) for d in decisions)
    out = []
    for decision in sorted(decisions, key=lambda x: str(_get(x, "decision_id"))):
        fields = dict(_get(decision, "fields", {}) or {})
        fp = fields.get("source_decision_fingerprint", "")
        rid = _get(decision, "source_run_id", "")
        did = _get(decision, "decision_id")
        rec = recs.get(_key(did, rid, fp), [])
        record = rec[0] if rec else None
        ev = evals.get(str(did), [])
        ev = [x for x in ev if _get(x, "fields", {}).get("source_run_id", rid) == rid and _get(x, "fields", {}).get("source_decision_fingerprint", fp) == fp]
        evaluation = sorted(ev, key=lambda x: (_get(x, "evaluation_basis", ""), int(_get(x, "horizon_seconds", 0) or 0)))[0] if len(ev) == 1 else (ev[0] if ev else None)
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
        metadata = {"experience": record, "audit": (ar[0] if ar else None), "source_fingerprint": source_fp, "duplicate": decision_counts[_key(did, rid, fp)] > 1 or len(rec) > 1 or len(ev) > 1 or len(ar) > 1, "audit_unavailable": unavailable, "source_available": bool(_get(phase56, "available", True))}
        out.append(JoinedObservation(decision, evaluation, evidence, metadata))
    return tuple(out)


def classify_observation(observation: JoinedObservation, config: DatasetConfig) -> EligibilityResult:
    d = observation.decision
    f = dict(d.fields)
    record = _get(observation, "fields", {}).get("experience")
    audit = _get(observation, "fields", {}).get("audit") or {}
    reasons: set[DatasetExclusionReason] = set()
    if _get(observation, "fields", {}).get("duplicate"):
        reasons.add(DatasetExclusionReason.DUPLICATE)
    if not _get(observation, "fields", {}).get("source_available", True):
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
    reference_raw = _get(record, "decision_reference_timestamp", f.get("decision_reference_timestamp"))
    reference = _dt(reference_raw)
    cutoff = _dt(_get(config, "filters", {}).get("as_of"))
    if completed is None or completed < analysis or (record is not None and "decision_reference_timestamp" in record and reference is None) or (reference is not None and reference < analysis) or (cutoff and (analysis > cutoff or completed > cutoff or (reference and reference > cutoff))):
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
    expected_fp = f.get("source_decision_fingerprint")
    source_fp = _get(observation, "fields", {}).get("source_fingerprint")
    if not record or not expected_fp or _get(record, "source_decision_fingerprint") != expected_fp or not source_fp or not _get(record, "source_evaluation_fingerprints") or not _get(record, "provenance") or not _get(record, "experience_schema_version") or not _get(record, "feature_schema_version") or not _get(record, "feature_extractor_version") or not _get(record, "trust_policy_version"):
        reasons.add(DatasetExclusionReason.PROVENANCE_INCOMPLETE)
    if record and any(str(_get(record, n, "")).startswith("unsupported") for n in ("experience_schema_version", "feature_schema_version")):
        reasons.add(DatasetExclusionReason.SCHEMA_UNSUPPORTED)
    ev = observation.evaluation
    if ev is None:
        reasons.add(DatasetExclusionReason.OUTCOME_UNAVAILABLE)
    elif ev.evaluation_basis != config.evaluation_basis or ev.horizon_seconds != config.horizon_seconds or ev.evaluation_status != "COMPLETE" or not ev.source_context_eligible:
        reasons.add(DatasetExclusionReason.OUTCOME_INELIGIBLE)
    if record and ev is not None:
        provenance = _get(record, "provenance", {}) or {}
        eval_fps = _get(record, "source_evaluation_fingerprints", {}) or {}
        eval_key = f"{ev.evaluation_basis}:{ev.horizon_seconds}"
        actual_eval_fp = _get(ev.fields, "fingerprint")
        eval_provenance = _get(ev.fields, "provenance", {}) or {}
        stored_eval_fp = _get(eval_fps, eval_key)
        if (_get(provenance, "source_decision_fingerprint") != expected_fp
                or not actual_eval_fp or stored_eval_fp != actual_eval_fp
                or _get(eval_provenance, "evaluation_fingerprint") != actual_eval_fp
                or _get(eval_provenance, "source_decision_id", _get(eval_provenance, "decision_id")) != d.decision_id):
            reasons.add(DatasetExclusionReason.PROVENANCE_INCOMPLETE)
    if ev is not None:
        times = {}
        for name in ("known_at", "created_at", "evaluated_at", "recovered_from_unavailable_at", "evaluation_timestamp", "entry_timestamp", "observation_timestamp", "target_timestamp"):
            raw_known = _get(ev.fields, name)
            if raw_known is not None and _dt(raw_known) is None:
                reasons.add(DatasetExclusionReason.TEMPORAL_INVALID)
            known = _dt(raw_known)
            times[name] = known
            if name != "target_timestamp" and known is not None and known > analysis:
                reasons.add(DatasetExclusionReason.TEMPORAL_INVALID)
        if times.get("target_timestamp") and times.get("observation_timestamp") and times["observation_timestamp"] > times["target_timestamp"]:
            reasons.add(DatasetExclusionReason.TEMPORAL_INVALID)
        if times.get("target_timestamp") and times.get("entry_timestamp") and times["entry_timestamp"] > times["target_timestamp"]:
            reasons.add(DatasetExclusionReason.TEMPORAL_INVALID)
    order = tuple(DatasetExclusionReason)
    ordered = tuple(reason for reason in order if reason in reasons)
    return EligibilityResult(not ordered, ordered, {"decision_id": d.decision_id})


__all__ = ["EligibilityResult", "join_observations", "classify_observation"]
