"""Pure canonical projection for eligible Phase 10 observations."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from .models import (
    EXAMPLE_SCHEMA_VERSION,
    CanonicalExampleV1,
    DatasetExclusionReason,
    JoinedObservation,
)

_FORBIDDEN = re.compile(r"prompt|completion|reasoning|chain[ _-]?of[ _-]?thought|\bcot\b|secret|password|credential|api[ _-]?key|token", re.I)
_OUTCOME_FIELDS = (
    "evaluation_basis", "horizon_seconds", "evaluation_status", "source_context_eligible",
    "target_timestamp", "observation_timestamp", "entry_timestamp", "entry_bid", "entry_ask",
    "entry_spread", "entry_spread_points", "future_bid", "future_ask", "future_spread",
    "future_spread_points", "point", "digits", "buy_net_price", "buy_net_points",
    "sell_net_price", "sell_net_points", "selected_action", "selected_action_net_price",
    "selected_action_net_points", "best_counterfactual_action", "best_counterfactual_net_points",
    "hold_opportunity_cost_points", "buy_mfe_price", "buy_mfe_points", "buy_mae_price",
    "buy_mae_points", "sell_mfe_price", "sell_mfe_points", "sell_mae_price", "sell_mae_points",
    "unavailable_reason",
)

def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda i: str(i[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    return value

def _reject(value: Any, key: str = "") -> None:
    if key and _FORBIDDEN.search(key):
        raise ValueError(f"forbidden field: {key}")
    if isinstance(value, Mapping):
        for k, v in value.items():
            _reject(v, str(k))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            _reject(v)
    elif isinstance(value, str) and _FORBIDDEN.search(value):
        raise ValueError("forbidden sensitive value")

def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

def _snapshot(value: Any, symbol: str, timestamp: Any) -> dict[str, Any]:
    """Project only bounded market facts; arbitrary source payloads are excluded."""
    if not isinstance(value, Mapping):
        raise ValueError("snapshot must be a mapping")
    if len(value) > 64:
        raise ValueError("snapshot keys exceed bound")
    out: dict[str, Any] = {"symbol": symbol, "timestamp": timestamp}
    for key in ("point", "digits"):
        if isinstance(value.get(key), (int, float)) and not isinstance(value[key], bool) and math.isfinite(float(value[key])):
            out[key] = value[key]
    quote = value.get("quote")
    if "quote" in value and not isinstance(quote, Mapping):
        raise ValueError("snapshot quote must be a mapping")
    if isinstance(quote, Mapping):
        if len(quote) > 16:
            raise ValueError("snapshot quote keys exceed bound")
        out["quote"] = {k: quote[k] for k in ("bid", "ask", "spread", "spread_points") if isinstance(quote.get(k), (int, float)) and not isinstance(quote[k], bool) and math.isfinite(float(quote[k]))}
    features = value.get("features")
    if "features" in value and not isinstance(features, Mapping):
        raise ValueError("snapshot features must be a mapping")
    if isinstance(features, Mapping):
        if len(features) > 8:
            raise ValueError("snapshot feature sections exceed bound")
        projected = {}
        for tf, section in list(features.items())[:8]:
            if not isinstance(tf, str) or not tf or len(tf) > 32:
                raise ValueError("snapshot timeframe key exceeds bound")
            if not isinstance(section, Mapping):
                raise ValueError("snapshot feature section must be a mapping")
            if len(section) > 32:
                raise ValueError("snapshot feature keys exceed bound")
            if any(k in {"return_over_bars", "range_pct", "close_position", "average_true_range"} and not isinstance(v, (int, float)) for k, v in section.items()):
                raise ValueError("snapshot feature value is malformed")
            vals = {str(k): v for k, v in list(section.items())[:32] if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}
            if vals:
                projected[str(tf)[:32]] = vals
        if projected:
            out["features"] = projected
    return out

def _pair(result: Any, eligibility: Any | None) -> tuple[JoinedObservation, Any]:
    if eligibility is None and isinstance(result, (tuple, list)) and len(result) == 2:
        result, eligibility = result
    if eligibility is None:
        eligibility = _get(result, "eligibility")
        result = _get(result, "observation", result)
    if not isinstance(result, JoinedObservation):
        raise ValueError("canonicalize requires JoinedObservation")
    if eligibility is None or not bool(_get(eligibility, "eligible", False)):
        raise ValueError("observation is not eligible")
    return result, eligibility


_PHASE9_COUNT_KEYS = frozenset({"knowledge", "experience", "statistics"})
_PHASE9_TEXT_FIELDS = frozenset({
    "rendered_context_hash",
    "context_hash",
    "query_normalization_fingerprint",
    "knowledge_query_fingerprint",
    "query_policy_version",
    "knowledge_generation_id",
    "experience_generation_id",
    "phase9_fingerprint",
})
_PHASE9_STATUS_FIELDS = frozenset({
    "integration_status", "integration", "bundle_status",
    "evidence_audit_status", "context_integrity", "evidence_use_status",
})
_PHASE9_STATUS_VALUES = {
    "integration_status": frozenset({"DISABLED", "INJECTED", "FALLBACK"}),
    "integration": frozenset({"DISABLED", "INJECTED", "FALLBACK"}),
    "bundle_status": frozenset({"COMPLETE", "PARTIAL", "EMPTY", "FAILED"}),
    "evidence_audit_status": frozenset({"VALID", "INVALID_REFERENCE", "NOT_RECORDED", "WRITE_FAILED"}),
    "context_integrity": frozenset({"COMPLETE", "INCOMPLETE", "UNAVAILABLE", "INVALID"}),
    "evidence_use_status": frozenset({"USED", "NONE_RELEVANT", "UNAVAILABLE", "DISABLED"}),
}
_PHASE9_ALLOWED_FIELDS = _PHASE9_TEXT_FIELDS | _PHASE9_STATUS_FIELDS | frozenset({
    "selected_counts", "dropped_counts",
})
_PHASE8_PROVENANCE_FIELDS = frozenset({
    "source_decision_id", "decision_id", "source_decision_fingerprint",
    "source_snapshot_fingerprint", "evaluation_fingerprint",
    "source_evaluation_fingerprint", "experience_id", "source_database_id",
    "schema_version", "experience_schema_version", "feature_schema_version",
    "feature_extractor_version", "trust_policy_version",
})
_SOURCE_FINGERPRINT_FIELDS = frozenset({
    "source_id", "canonical_path", "schema_fingerprint", "file_sha256",
    "snapshot_fingerprint", "contract_version",
})
_REJECTION_REASONS = frozenset({
    "CONFLICTS_WITH_CURRENT_STATE", "LOW_RELEVANCE", "INSUFFICIENT_SAMPLE",
    "DIAGNOSTIC_ONLY", "REDUNDANT",
})


def _bounded_phase9_text(name: str, value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"invalid Phase 9 {name}")
    return value


def _phase9_counts(name: str, value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid Phase 9 {name}")
    if set(value) - _PHASE9_COUNT_KEYS:
        raise ValueError(f"invalid Phase 9 {name} keys")
    output: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 10_000:
            raise ValueError(f"invalid Phase 9 {name} value")
        output[key] = count
    return output


def _phase9_metadata(audit: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    """Project only bounded, typed audit metadata into the canonical row."""
    data = {
        key: value for key, value in fallback.items()
        if key in _PHASE9_ALLOWED_FIELDS and value is not None
    }
    data.update({
        key: value for key, value in audit.items()
        if key in _PHASE9_ALLOWED_FIELDS and value is not None
    })
    for name in _PHASE9_TEXT_FIELDS:
        if name in data:
            data[name] = _bounded_phase9_text(name, data[name])
    for name in ("selected_counts", "dropped_counts"):
        if name in data:
            data[name] = _phase9_counts(name, data[name])
    for name in _PHASE9_STATUS_VALUES:
        if name in data and data[name] is not None:
            data[name] = _bounded_phase9_text(name, data[name])
            if data[name] not in _PHASE9_STATUS_VALUES[name]:
                raise ValueError(f"invalid Phase 9 {name}")
    return data


def _source_fingerprint(
    name: str, value: Any, *, require_complete: bool = True
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or (
        require_complete and set(value) != _SOURCE_FINGERPRINT_FIELDS
    ):
        raise ValueError(f"invalid {name} source fingerprint")
    if set(value) - _SOURCE_FINGERPRINT_FIELDS:
        raise ValueError(f"invalid {name} source fingerprint fields")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str) or not item or len(item) > 512:
            raise ValueError(f"invalid {name} source fingerprint value")
        result[key] = item
    return result


def _phase8_provenance(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or len(value) > len(_PHASE8_PROVENANCE_FIELDS):
        raise ValueError("invalid Phase 8 provenance")
    if set(value) - _PHASE8_PROVENANCE_FIELDS:
        raise ValueError("unapproved Phase 8 provenance field")
    output = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str) or not item or len(item) > 512:
            raise ValueError("invalid Phase 8 provenance value")
        output[key] = item
    return output


def _phase8_evaluation_fingerprints(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or len(value) > 64:
        raise ValueError("invalid Phase 8 evaluation fingerprint mapping")
    output = {}
    for key, item in value.items():
        if (
            not isinstance(key, str) or not key or len(key) > 128
            or not isinstance(item, str) or not item or len(item) > 256
        ):
            raise ValueError("invalid Phase 8 evaluation fingerprint")
        output[key] = item
    return output


def _rejection(value: Any) -> tuple[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ValueError("evidence rejection must include a reason")
    ref = value.get("ref")
    reason = value.get("reason")
    if not isinstance(ref, str) or not ref or len(ref) > 256:
        raise ValueError("invalid evidence rejection reference")
    if hasattr(reason, "value"):
        reason = reason.value
    if not isinstance(reason, str) or reason not in _REJECTION_REASONS:
        raise ValueError("invalid evidence rejection reason")
    return ref, {"ref": ref, "reason": reason}


def _available_ids(audit: Mapping[str, Any], evidence: Mapping[str, Any], name: str) -> tuple[Any, ...]:
    """Select available IDs only when authoritative and fallback sources agree."""
    in_audit = name in audit
    in_evidence = name in evidence
    audit_ids = tuple(audit[name]) if in_audit and isinstance(audit[name], (list, tuple)) else audit.get(name)
    evidence_ids = tuple(evidence[name]) if in_evidence and isinstance(evidence[name], (list, tuple)) else evidence.get(name)
    if in_audit and in_evidence:
        if not isinstance(audit_ids, tuple) or not isinstance(evidence_ids, tuple):
            raise ValueError(f"invalid available {name} IDs")
        try:
            same_ids = set(audit_ids) == set(evidence_ids)
        except TypeError as exc:
            raise ValueError(f"invalid available {name} IDs") from exc
        if not same_ids:
            raise ValueError(f"conflicting available {name} IDs")
    selected = audit_ids if in_audit else evidence_ids
    if selected is None:
        return ()
    if not isinstance(selected, tuple):
        raise ValueError(f"invalid available {name} IDs")
    return selected

def canonicalize(result: Any, eligibility: Any | None = None) -> CanonicalExampleV1:
    """Convert one eligible joined observation into a deterministic JSON contract."""
    observation, eligibility = _pair(result, eligibility)
    raw = _plain(observation)
    _reject(raw)
    d, ev, evidence = observation.decision, observation.evaluation, observation.evidence
    jf = dict(observation.fields)
    if jf.get("duplicate") or jf.get("duplicate_non_evaluation") or jf.get("duplicate_evaluation_keys"):
        raise ValueError("duplicate observation")
    record = jf.get("experience") or {}
    audit = jf.get("audit") or {}
    source_fingerprints = jf.get("source_fingerprints") or {}
    dfp = _get(d, "fields", {}).get("source_decision_fingerprint") or record.get("source_decision_fingerprint")
    basis = _get(ev, "evaluation_basis")
    horizon = _get(ev, "horizon_seconds")
    policy = record.get("trust_policy_version", "")
    if not dfp or not basis or not horizon or not policy:
        raise ValueError("canonical identity/provenance incomplete")
    identity = {"decision_id": d.decision_id, "basis": basis, "horizon_seconds": horizon, "source_decision_fingerprint": dfp, "policy_version": policy}
    example_id = "ex_" + _digest(identity)
    snapshot = _snapshot(_get(d, "fields", {}).get("snapshot_json") or record.get("market_state") or {}, d.resolved_symbol, d.analysis_snapshot_timestamp)
    # Keep Phase 9 bounded: IDs/status/hash only; never rendered evidence text.
    used = tuple(_get(evidence, "refs_used", ()) or ())
    rejected_entries = tuple(_get(evidence, "refs_rejected", ()) or ())
    rejected_pairs = tuple(_rejection(entry) for entry in rejected_entries)
    rejected = tuple(ref for ref, _ in rejected_pairs)
    rejected_records = tuple(record for _, record in rejected_pairs)
    ef = dict(_get(evidence, "fields", {}) or {})
    knowledge_ids = _available_ids(audit, ef, "available_knowledge_ids")
    phase8_ids = _available_ids(audit, ef, "available_experience_ids") + _available_ids(audit, ef, "available_statistics_ids")
    context_hash = audit.get("context_hash") or _get(evidence, "fields", {}).get("context_hash")
    all_ids = knowledge_ids + phase8_ids
    for name, ids in (("available", all_ids), ("used", used), ("rejected", rejected)):
        if len(ids) > 64 or len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x or len(x) > 256 for x in ids):
            raise ValueError(f"invalid evidence {name} IDs")
    if set(used) & set(rejected) or set(used) | set(rejected) != set(all_ids):
        raise ValueError("inconsistent evidence partition")
    decision = {"decision_id": d.decision_id, "source_run_id": d.source_run_id, "requested_symbol": d.requested_symbol, "resolved_symbol": d.resolved_symbol, "analysis_profile": d.analysis_profile, "analysis_timeframe": d.analysis_timeframe, "action": d.action, "normalization_status": _get(d, "fields", {}).get("normalization_status"), "decision_context_status": _get(d, "fields", {}).get("decision_context_status"), "analysis_snapshot_timestamp": d.analysis_snapshot_timestamp, "decision_completed_timestamp": d.decision_completed_timestamp, "raw_result_fingerprint": _digest(raw)}
    outcome = {name: _get(ev, name, _get(ev, "fields", {}).get(name)) for name in _OUTCOME_FIELDS}
    outcome.update({name: _get(ev, "fields", {}).get(name) for name in ("training_eligible", "training_eligibility_reason") if _get(ev, "fields", {}).get(name) is not None})
    outcome = {k: v for k, v in outcome.items() if v is not None}
    trust = {"tier": record.get("trust"), "policy_version": policy, "experience_schema_version": record.get("experience_schema_version"), "feature_schema_version": record.get("feature_schema_version"), "feature_extractor_version": record.get("feature_extractor_version")}
    reasons = tuple(_get(eligibility, "reasons", ()))
    if any((x.value if isinstance(x, DatasetExclusionReason) else str(x)) not in {r.value for r in DatasetExclusionReason} for x in reasons):
        raise ValueError("unknown exclusion reason")
    phase8_prov = record.get("provenance") or {}
    phase8_prov = _phase8_provenance(phase8_prov)
    phase8_eval_fps = _phase8_evaluation_fingerprints(record.get("source_evaluation_fingerprints") or {})
    phase9_audit = _phase9_metadata(audit, ef)
    phase56_value = source_fingerprints.get("phase56")
    phase56_fingerprint = _source_fingerprint("phase56", phase56_value)
    if phase56_value is None:
        # The legacy unlabelled fallback predates the phase-separated contract.
        # Preserve it for backwards-compatible rows; any phase-labelled value
        # above must contain the complete SourceFingerprint contract.
        phase56_fingerprint = _source_fingerprint(
            "phase56", jf.get("source_fingerprint"), require_complete=False
        )
    phase8_fingerprint = _source_fingerprint("phase8", source_fingerprints.get("phase8"))
    phase9_fingerprint = _source_fingerprint("phase9", source_fingerprints.get("phase9"))
    provenance = {"phase56": phase56_fingerprint, "source_decision_fingerprint": dfp, "source_evaluation_fingerprint": _get(ev, "fields", {}).get("source_evaluation_fingerprint"), "phase8_record_provenance": phase8_prov, "phase8_source_evaluation_fingerprints": phase8_eval_fps, "closed_rejection_reasons": reasons, "context_hash": context_hash, "knowledge": {"ids": knowledge_ids, "used": tuple(x for x in used if x in knowledge_ids), "generation_id": phase9_audit.get("knowledge_generation_id", ef.get("knowledge_generation_id"))}, "phase8": {"ids": phase8_ids, "used": tuple(x for x in used if x in phase8_ids), "generation_id": phase9_audit.get("experience_generation_id", ef.get("experience_generation_id")), "source_decision_fingerprint": record.get("source_decision_fingerprint"), "source_fingerprint": phase8_fingerprint}, "phase9": {"used": used, "rejected": rejected_records, "audit_status": phase9_audit.get("evidence_audit_status"), "context_integrity": phase9_audit.get("context_integrity"), "available_knowledge_ids": knowledge_ids, "available_phase8_ids": phase8_ids, "source_fingerprint": phase9_fingerprint, "phase9_fingerprint": phase9_audit.get("phase9_fingerprint") or (phase9_fingerprint or {}).get("snapshot_fingerprint"), "rendered_context_hash": phase9_audit.get("rendered_context_hash"), "query_normalization_fingerprint": phase9_audit.get("query_normalization_fingerprint")}, "raw_result_fingerprint": _digest(raw)}
    market = {"snapshot": snapshot, "snapshot_fingerprint": record.get("snapshot_fingerprint") or record.get("market_state_fingerprint") or (phase56_fingerprint or {}).get("snapshot_fingerprint")}
    research = {"context_integrity": _get(evidence, "context_integrity"), "evidence_use_status": _get(evidence, "evidence_use_status"), "refs_used": used, "refs_rejected": rejected, "bundle_status": phase9_audit.get("bundle_status"), "integration_status": phase9_audit.get("integration_status", phase9_audit.get("integration")), "selected_counts": phase9_audit.get("selected_counts", {}), "dropped_counts": phase9_audit.get("dropped_counts", {}), "knowledge_query_fingerprint": phase9_audit.get("knowledge_query_fingerprint"), "query_policy_version": phase9_audit.get("query_policy_version"), "knowledge_generation_id": phase9_audit.get("knowledge_generation_id"), "experience_generation_id": phase9_audit.get("experience_generation_id"), "phase9": provenance["phase9"]}
    return CanonicalExampleV1(example_id, decision, market, research, outcome, trust, provenance, EXAMPLE_SCHEMA_VERSION)

__all__ = ["canonicalize"]
