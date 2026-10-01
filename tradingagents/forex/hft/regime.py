"""Atomic strategic regime constraints consumed by the deterministic HFT plane."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from numbers import Real

from .models import FastAction, utc

UTC = timezone.utc


class Regime(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"


class DirectionPolicy(str, Enum):
    LONG_ONLY = "LONG_ONLY"
    SHORT_ONLY = "SHORT_ONLY"
    BOTH = "BOTH"
    PAUSE = "PAUSE"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _bounded(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


@dataclass(frozen=True, slots=True)
class StrategicRegimeState:
    """Short-lived strategic envelope; it never contains an entry signal."""

    state_id: str
    symbol: str
    regime: Regime
    direction_policy: DirectionPolicy
    risk_multiplier: float
    confidence: float
    created_at: datetime
    expires_at: datetime
    momentum_enabled: bool = False
    range_enabled: bool = False
    breakout_enabled: bool = False
    mean_reversion_enabled: bool = False
    source_run_id: str | None = None
    source_decision_id: str | None = None
    git_commit: str | None = None
    validity: str = "VALID"

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _text(self.state_id, "state_id"))
        object.__setattr__(self, "symbol", _text(self.symbol, "symbol").upper())
        try:
            object.__setattr__(self, "regime", Regime(self.regime))
            object.__setattr__(self, "direction_policy", DirectionPolicy(self.direction_policy))
        except (TypeError, ValueError) as exc:
            raise ValueError("regime or direction_policy is invalid") from exc
        created = utc(self.created_at, "created_at")
        expires = utc(self.expires_at, "expires_at")
        if expires <= created:
            raise ValueError("expires_at must be after created_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "risk_multiplier", _bounded(self.risk_multiplier, "risk_multiplier"))
        object.__setattr__(self, "confidence", _bounded(self.confidence, "confidence"))
        for name in (
            "momentum_enabled",
            "range_enabled",
            "breakout_enabled",
            "mean_reversion_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        validity = _text(self.validity, "validity").upper()
        if validity not in {"VALID", "EXPIRED", "INVALID"}:
            raise ValueError("validity must be VALID, EXPIRED, or INVALID")
        object.__setattr__(self, "validity", validity)
        for name in ("source_run_id", "source_decision_id", "git_commit"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))

    def is_active(self, at: datetime) -> bool:
        timestamp = utc(at, "at")
        return self.validity == "VALID" and self.created_at <= timestamp < self.expires_at

    def permits_direction(self, action: FastAction) -> bool:
        if action is FastAction.ENTER_LONG:
            return self.direction_policy in {DirectionPolicy.LONG_ONLY, DirectionPolicy.BOTH}
        if action is FastAction.ENTER_SHORT:
            return self.direction_policy in {DirectionPolicy.SHORT_ONLY, DirectionPolicy.BOTH}
        return True

    def permits_strategy(self, strategy_id: str) -> bool:
        return {
            "momentum_continuation": self.momentum_enabled,
            "range_rejection": self.range_enabled,
            "micro_breakout": self.breakout_enabled,
            "mean_reversion": self.mean_reversion_enabled,
        }.get(str(strategy_id), False)

    def to_payload(self) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "symbol": self.symbol,
            "regime": self.regime.value,
            "direction_policy": self.direction_policy.value,
            "risk_multiplier": self.risk_multiplier,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "expires_at": self.expires_at.isoformat().replace("+00:00", "Z"),
            "momentum_enabled": self.momentum_enabled,
            "range_enabled": self.range_enabled,
            "breakout_enabled": self.breakout_enabled,
            "mean_reversion_enabled": self.mean_reversion_enabled,
            "source_run_id": self.source_run_id,
            "source_decision_id": self.source_decision_id,
            "git_commit": self.git_commit,
            "validity": self.validity,
            "executed": False,
        }


__all__ = ["DirectionPolicy", "Regime", "StrategicRegimeState"]
