"""Deterministic record-level trust policy."""
from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
from typing import Any
from datetime import datetime, timezone
from .features import MarketStateVector
from .models import TrustTier

POLICY_VERSION = "trust-policy.v1"
MIN_FEATURES = 8

@dataclass(frozen=True, slots=True)
class TrustClassification:
    tier: TrustTier
    reasons: tuple[str, ...]
    policy_version: str = POLICY_VERSION

def classify_trust(record: Mapping[str, Any] | Any, feature_result: MarketStateVector) -> TrustClassification:
    row = record if isinstance(record, Mapping) else ({f.name: getattr(record, f.name) for f in fields(record)} if is_dataclass(record) else vars(record))
    reasons: list[str] = []
    executed = row.get("executed", 0)
    if executed not in (0, False, None): reasons.append("EXECUTED")
    if str(row.get("normalization_status", "")).upper() != "NORMALIZED": reasons.append("NORMALIZATION_FAILED")
    if str(row.get("decision_context_status", "")).upper() != "COMPLETE": reasons.append("CONTEXT_INCOMPLETE")
    if str(row.get("action", "")).upper() not in {"BUY", "SELL", "HOLD"}: reasons.append("ACTION_INVALID")
    completed = row.get("decision_completed_timestamp")
    if completed is None:
        reasons.append("COMPLETION_UNAVAILABLE")
    else:
        try:
            if isinstance(completed, str): completed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
            snapshot = row.get("analysis_snapshot_timestamp")
            if isinstance(snapshot, str): snapshot = datetime.fromisoformat(snapshot.replace("Z", "+00:00"))
            if completed.tzinfo is None or completed.utcoffset() != timezone.utc.utcoffset(completed) or snapshot is None or completed < snapshot:
                reasons.append("TEMPORAL_INVALID")
        except (TypeError, ValueError): reasons.append("TEMPORAL_INVALID")
    if str(row.get("decision_reference_status", "")).upper() == "INVALID_TEMPORAL": reasons.append("TEMPORAL_INVALID")
    if sum(feature_result.mask) < MIN_FEATURES: reasons.append("MARKET_FEATURES_INSUFFICIENT")
    if any(d.code in {"PROVENANCE_INVALID", "SOURCE_CONFLICT"} for d in feature_result.diagnostics): reasons.append("PROVENANCE_INVALID")
    if any(d.code == "QUOTE_INVALID" for d in feature_result.diagnostics): reasons.append("QUOTE_INVALID")
    provenance = row.get("provenance") or {}
    if isinstance(provenance, Mapping):
        expected = row.get("source_decision_fingerprint")
        actual = provenance.get("source_decision_fingerprint")
        if expected and actual and expected != actual: reasons.append("PROVENANCE_INVALID")
        if provenance.get("feature_fingerprint") and provenance["feature_fingerprint"] != feature_result.fingerprint: reasons.append("PROVENANCE_INVALID")
    severe = {"EXECUTED", "NORMALIZATION_FAILED", "CONTEXT_INCOMPLETE", "ACTION_INVALID", "MARKET_FEATURES_INSUFFICIENT", "PROVENANCE_INVALID", "TEMPORAL_INVALID", "QUOTE_INVALID"}
    if any(reason in severe for reason in reasons):
        return TrustClassification(TrustTier.TIER_C_DIAGNOSTIC_ONLY, tuple(dict.fromkeys(reasons)))
    # Partial but comparable state is limited, while a complete valid state is high trust.
    if reasons or not all(feature_result.mask):
        if not all(feature_result.mask): reasons.append("FEATURES_PARTIAL")
        return TrustClassification(TrustTier.TIER_B_LIMITED, tuple(dict.fromkeys(reasons)))
    return TrustClassification(TrustTier.TIER_A_HIGH_TRUST, ())

__all__ = ["POLICY_VERSION", "TrustClassification", "classify_trust"]
