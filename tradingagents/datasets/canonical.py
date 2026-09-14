"""Pure canonical projection for eligible Phase 10 observations."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from .models import (
    CANONICALIZATION_VERSION,
    EXAMPLE_SCHEMA_VERSION,
    CanonicalExampleV1,
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
    dfp = _get(d, "fields", {}).get("source_decision_fingerprint") or record.get("source_decision_fingerprint")
    basis = _get(ev, "evaluation_basis")
    horizon = _get(ev, "horizon_seconds")
    policy = record.get("trust_policy_version", "")
    if not dfp or not basis or not horizon or not policy:
        raise ValueError("canonical identity/provenance incomplete")
    identity = {"decision_id": d.decision_id, "basis": basis, "horizon_seconds": horizon, "source_decision_fingerprint": dfp, "policy_version": policy, "canonicalization_version": CANONICALIZATION_VERSION}
    example_id = "ex_" + _digest(identity)
    snapshot = _get(d, "fields", {}).get("snapshot_json") or record.get("market_state") or {}
    # Keep Phase 9 bounded: IDs/status/hash only; never rendered evidence text.
    used = tuple(_get(evidence, "refs_used", ()) or ())
    rejected = tuple(_get(evidence, "refs_rejected", ()) or ())
    ef = dict(_get(evidence, "fields", {}) or {})
    knowledge_ids = tuple(audit.get("available_knowledge_ids", ef.get("available_knowledge_ids", ())))
    phase8_ids = tuple(audit.get("available_experience_ids", ef.get("available_experience_ids", ()))) + tuple(audit.get("available_statistics_ids", ef.get("available_statistics_ids", ())))
    context_hash = audit.get("context_hash") or _get(evidence, "fields", {}).get("context_hash")
    decision = {"decision_id": d.decision_id, "source_run_id": d.source_run_id, "requested_symbol": d.requested_symbol, "resolved_symbol": d.resolved_symbol, "analysis_profile": d.analysis_profile, "analysis_timeframe": d.analysis_timeframe, "action": d.action, "analysis_snapshot_timestamp": d.analysis_snapshot_timestamp, "decision_completed_timestamp": d.decision_completed_timestamp, "evaluation_basis": basis, "horizon_seconds": horizon, "source_decision_fingerprint": dfp}
    outcome = {name: _get(ev, name, _get(ev, "fields", {}).get(name)) for name in _OUTCOME_FIELDS}
    outcome = {k: v for k, v in outcome.items() if v is not None}
    trust = {"tier": record.get("trust"), "policy_version": policy, "experience_schema_version": record.get("experience_schema_version"), "feature_schema_version": record.get("feature_schema_version"), "feature_extractor_version": record.get("feature_extractor_version")}
    provenance = {"source_decision_fingerprint": dfp, "source_evaluation_fingerprint": _get(ev, "fields", {}).get("source_evaluation_fingerprint"), "source": jf.get("source_fingerprint"), "closed_rejection_reasons": tuple(_get(eligibility, "reasons", ())), "context_hash": context_hash, "knowledge": {"ids": knowledge_ids, "used": tuple(x for x in used if x in knowledge_ids), "generation_id": audit.get("knowledge_generation_id", ef.get("knowledge_generation_id"))}, "phase8": {"ids": phase8_ids, "used": tuple(x for x in used if x in phase8_ids), "generation_id": audit.get("experience_generation_id", ef.get("experience_generation_id"))}, "phase9": {"used": used, "rejected": rejected, "audit_status": audit.get("evidence_audit_status"), "context_integrity": audit.get("context_integrity"), "available_knowledge_ids": knowledge_ids, "available_phase8_ids": phase8_ids}, "raw_result_fingerprint": _digest(raw)}
    research = {"phase9": provenance["phase9"]}
    return CanonicalExampleV1(example_id, decision, {"snapshot": snapshot}, research, outcome, trust, provenance, EXAMPLE_SCHEMA_VERSION)

__all__ = ["canonicalize"]
