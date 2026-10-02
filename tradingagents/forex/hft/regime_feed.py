"""Fail-closed translation from strategic decisions to regime constraints."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from .regime import DirectionPolicy, Regime, StrategicRegimeState

UTC = timezone.utc


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def build_regime_from_shadow_decision(
    decision: Any,
    *,
    git_commit: str,
) -> StrategicRegimeState | None:
    """Create a filter envelope; BUY/SELL never become per-tick commands."""

    raw_action = getattr(decision, "action", "")
    action = str(getattr(raw_action, "value", raw_action)).strip().upper()
    symbol = str(getattr(decision, "resolved_symbol", "")).strip().upper()
    decision_id = str(getattr(decision, "decision_id", "")).strip()
    source_run_id = str(getattr(decision, "source_run_id", "")).strip()
    if action not in {"BUY", "SELL", "HOLD"} or not symbol or not decision_id or not source_run_id:
        return None
    completed = getattr(decision, "decision_completed_timestamp", None) or getattr(decision, "created_at", None)
    try:
        created = _timestamp(completed, "decision completion")
    except ValueError:
        return None
    expires_value = getattr(decision, "valid_until", None)
    if expires_value is None:
        seconds = getattr(decision, "valid_for_seconds", None)
        try:
            expires = created + timedelta(seconds=float(seconds))
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        try:
            expires = _timestamp(expires_value, "valid_until")
        except ValueError:
            return None
    if expires <= created:
        return None
    # Phase 5 persists confidence as optional metadata.  A missing value is
    # not a malformed decision (the plan feed already treats it as neutral
    # confidence); keep the regime risk multiplier at its bounded floor.
    confidence_value = getattr(decision, "confidence", 0.0)
    if confidence_value is None:
        confidence_value = 0.0
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    if action == "BUY":
        regime = Regime.BULLISH
        policy = DirectionPolicy.LONG_ONLY
        flags = (True, False, True, False)
    elif action == "SELL":
        regime = Regime.BEARISH
        policy = DirectionPolicy.SHORT_ONLY
        flags = (True, False, True, False)
    else:
        regime = Regime.NEUTRAL
        policy = DirectionPolicy.BOTH
        flags = (False, True, False, True)
    return StrategicRegimeState(
        state_id=f"regime-{decision_id}",
        symbol=symbol,
        regime=regime,
        direction_policy=policy,
        risk_multiplier=max(0.25, confidence),
        confidence=confidence,
        created_at=created,
        expires_at=expires,
        momentum_enabled=flags[0],
        range_enabled=flags[1],
        breakout_enabled=flags[2],
        mean_reversion_enabled=flags[3],
        source_run_id=source_run_id,
        source_decision_id=decision_id,
        git_commit=str(git_commit).strip() or "unknown",
    )


__all__ = ["build_regime_from_shadow_decision"]
