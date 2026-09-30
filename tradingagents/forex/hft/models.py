"""Typed, bounded contracts shared by Phase 12 replay and shadow runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from numbers import Real
from typing import Any


UTC = timezone.utc


def utc(value: datetime, name: str = "timestamp") -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must be UTC")
    return value.astimezone(UTC)


def _finite(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _text(value: Any, name: str, *, upper: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    result = value.strip()
    return result.upper() if upper else result


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"
    NONE = "NONE"


class RiskPosture(str, Enum):
    CONSERVATIVE = "CONSERVATIVE"
    NORMAL = "NORMAL"
    REDUCED = "REDUCED"
    BLOCKED = "BLOCKED"


class FastAction(str, Enum):
    NO_ACTION = "NO_ACTION"
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT = "EXIT"
    REDUCE = "REDUCE"
    INVALIDATE = "INVALIDATE"


class PositionState(str, Enum):
    FLAT = "FLAT"
    PENDING_ENTRY = "PENDING_ENTRY"
    LONG = "LONG"
    SHORT = "SHORT"
    PENDING_EXIT = "PENDING_EXIT"
    CLOSED = "CLOSED"


@dataclass(frozen=True, slots=True)
class EntryConstraints:
    max_spread_points: float = 20.0
    minimum_momentum: float = 0.0
    minimum_volatility: float = 0.0
    maximum_volatility: float = 1.0
    minimum_confirmation: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_spread_points", _positive(self.max_spread_points, "max_spread_points"))
        object.__setattr__(self, "minimum_momentum", _finite(self.minimum_momentum, "minimum_momentum", minimum=0.0))
        object.__setattr__(self, "minimum_volatility", _finite(self.minimum_volatility, "minimum_volatility", minimum=0.0))
        object.__setattr__(self, "maximum_volatility", _positive(self.maximum_volatility, "maximum_volatility"))
        if self.maximum_volatility < self.minimum_volatility:
            raise ValueError("maximum_volatility must be >= minimum_volatility")
        object.__setattr__(self, "minimum_confirmation", _finite(self.minimum_confirmation, "minimum_confirmation", minimum=0.0, maximum=1.0))


@dataclass(frozen=True, slots=True)
class StopPolicy:
    stop_distance_points: float
    take_profit_distance_points: float
    time_stop_seconds: int
    trailing_distance_points: float | None = None
    breakeven_after_points: float | None = None

    def __post_init__(self) -> None:
        for name in ("stop_distance_points", "take_profit_distance_points"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if isinstance(self.time_stop_seconds, bool) or not isinstance(self.time_stop_seconds, int) or self.time_stop_seconds <= 0:
            raise ValueError("time_stop_seconds must be a positive integer")
        for name in ("trailing_distance_points", "breakeven_after_points"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(value, name))


@dataclass(frozen=True, slots=True)
class StrategicExecutionPlan:
    symbol: str
    created_at: datetime
    expires_at: datetime
    allowed_until: datetime
    timeframe: str
    regime: str
    primary_direction: Direction
    confidence: float
    strategy_family: str
    entry_constraints: EntryConstraints
    risk_posture: RiskPosture
    stop_policy: StopPolicy
    invalidation: tuple[str, ...] = ()
    session_constraints: tuple[str, ...] = ()
    plan_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _text(self.symbol, "symbol", upper=True))
        created = utc(self.created_at, "created_at")
        expires = utc(self.expires_at, "expires_at")
        allowed = utc(self.allowed_until, "allowed_until")
        if expires <= created:
            raise ValueError("expires_at must be after created_at")
        if allowed < created or allowed > expires:
            raise ValueError("allowed_until must be between created_at and expires_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "allowed_until", allowed)
        object.__setattr__(self, "timeframe", _text(self.timeframe, "timeframe", upper=True))
        object.__setattr__(self, "regime", _text(self.regime, "regime", upper=True))
        if not isinstance(self.primary_direction, Direction):
            try:
                object.__setattr__(self, "primary_direction", Direction(self.primary_direction))
            except (TypeError, ValueError) as exc:
                raise ValueError("primary_direction must be LONG, SHORT, BOTH, or NONE") from exc
        object.__setattr__(self, "confidence", _finite(self.confidence, "confidence", minimum=0.0, maximum=1.0))
        object.__setattr__(self, "strategy_family", _text(self.strategy_family, "strategy_family"))
        if not isinstance(self.entry_constraints, EntryConstraints):
            raise ValueError("entry_constraints must be EntryConstraints")
        if not isinstance(self.stop_policy, StopPolicy):
            raise ValueError("stop_policy must be StopPolicy")
        if not isinstance(self.risk_posture, RiskPosture):
            try:
                object.__setattr__(self, "risk_posture", RiskPosture(self.risk_posture))
            except (TypeError, ValueError) as exc:
                raise ValueError("risk_posture is invalid") from exc
        for name in ("invalidation", "session_constraints"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        if self.plan_id is not None:
            object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))

    def is_active(self, at: datetime) -> bool:
        timestamp = utc(at, "at")
        return self.created_at <= timestamp <= self.allowed_until and timestamp < self.expires_at


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    point: float = 0.00001
    sequence: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _text(self.symbol, "tick symbol", upper=True))
        object.__setattr__(self, "timestamp", utc(self.timestamp, "tick timestamp"))
        bid = _positive(self.bid, "bid")
        ask = _positive(self.ask, "ask")
        if ask < bid:
            raise ValueError("ask must be >= bid")
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        object.__setattr__(self, "point", _positive(self.point, "point"))
        if self.sequence is not None and (isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0):
            raise ValueError("sequence must be a non-negative integer")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_points(self) -> float:
        return self.spread / self.point


@dataclass(frozen=True, slots=True)
class FastDecision:
    action: FastAction
    symbol: str
    timestamp: datetime
    reason: str
    score: float = 0.0
    processing_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.action, FastAction):
            object.__setattr__(self, "action", FastAction(self.action))
        object.__setattr__(self, "symbol", _text(self.symbol, "symbol", upper=True))
        object.__setattr__(self, "timestamp", utc(self.timestamp))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "score", _finite(self.score, "score"))
        object.__setattr__(self, "processing_ms", _finite(self.processing_ms, "processing_ms", minimum=0.0))


@dataclass(frozen=True, slots=True)
class RiskDecision:
    accepted: bool
    reason_code: str
    reason: str
    risk_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be bool")
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code", upper=True))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "risk_fraction", _finite(self.risk_fraction, "risk_fraction", minimum=0.0, maximum=1.0))


__all__ = [
    "UTC",
    "Direction",
    "EntryConstraints",
    "FastAction",
    "FastDecision",
    "PositionState",
    "RiskDecision",
    "RiskPosture",
    "StopPolicy",
    "StrategicExecutionPlan",
    "Tick",
    "utc",
]
