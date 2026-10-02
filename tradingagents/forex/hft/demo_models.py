"""Typed contracts for the Phase 12D DEMO execution boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from numbers import Real
from typing import Any

from .models import utc


class ExecutionMode(str, Enum):
    SHADOW = "SHADOW"
    TEST_ONLY = "TEST_ONLY"
    DEMO = "DEMO"


_RESULT_CLASSES = frozenset(
    {
        "FILLED",
        "PARTIAL",
        "REJECTED",
        "REQUOTE",
        "PRICE_CHANGED",
        "MARKET_CLOSED",
        "NO_MONEY",
        "INVALID_STOPS",
        "CONNECTION_ERROR",
        "UNKNOWN",
    }
)


def _text(value: Any, name: str, *, upper: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    result = value.strip()
    return result.upper() if upper else result


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class DemoAccountSnapshot:
    login: int | None
    server: str | None
    company: str | None
    currency: str | None
    trade_mode: int | None
    balance: float | None = None
    equity: float | None = None
    margin: float | None = None
    free_margin: float | None = None
    margin_level: float | None = None
    observed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.login is not None and (
            isinstance(self.login, bool) or not isinstance(self.login, int) or self.login < 0
        ):
            raise ValueError("login must be a non-negative integer")
        for name in ("server", "company", "currency"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        if self.trade_mode is not None and (
            isinstance(self.trade_mode, bool)
            or not isinstance(self.trade_mode, int)
            or self.trade_mode < 0
        ):
            raise ValueError("trade_mode must be a non-negative integer")
        for name in ("balance", "equity", "margin", "free_margin", "margin_level"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name, minimum=0.0))
        if self.observed_at is not None:
            object.__setattr__(self, "observed_at", utc(self.observed_at, "observed_at"))

    def is_demo(self, demo_trade_mode: int) -> bool:
        return self.trade_mode is not None and self.trade_mode == demo_trade_mode


@dataclass(frozen=True, slots=True)
class DemoOrderIntent:
    intent_id: str
    plan_id: str
    source_decision_id: str
    source_run_id: str
    strategy_id: str
    symbol: str
    direction: str
    volume: float
    requested_price: float
    stop_loss: float
    take_profit: float
    deviation_points: int
    created_at: datetime
    git_commit: str
    execution_mode: ExecutionMode = ExecutionMode.DEMO
    account_trade_mode: int | None = None
    real_money: bool = False

    def __post_init__(self) -> None:
        for name in (
            "intent_id",
            "plan_id",
            "source_decision_id",
            "source_run_id",
            "strategy_id",
            "git_commit",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "symbol", _text(self.symbol, "symbol", upper=True))
        direction = _text(self.direction, "direction", upper=True)
        if direction not in {"LONG", "SHORT"}:
            raise ValueError("direction must be LONG or SHORT")
        object.__setattr__(self, "direction", direction)
        for name in ("volume", "requested_price", "stop_loss", "take_profit"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0.0))
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.deviation_points, bool)
            or not isinstance(self.deviation_points, int)
            or self.deviation_points < 0
        ):
            raise ValueError("deviation_points must be a non-negative integer")
        object.__setattr__(self, "created_at", utc(self.created_at, "created_at"))
        object.__setattr__(self, "execution_mode", ExecutionMode(self.execution_mode))
        if self.execution_mode is not ExecutionMode.DEMO:
            raise ValueError("DemoOrderIntent requires execution_mode=DEMO")
        if self.account_trade_mode is not None and (
            isinstance(self.account_trade_mode, bool)
            or not isinstance(self.account_trade_mode, int)
            or self.account_trade_mode < 0
        ):
            raise ValueError("account_trade_mode must be a non-negative integer")
        if self.real_money is not False:
            raise ValueError("DEMO execution must have real_money=False")


@dataclass(frozen=True, slots=True)
class DemoOrderResult:
    intent_id: str
    classification: str
    retcode: int | None
    order_ticket: int | None
    deal_ticket: int | None
    fill_price: float | None
    fill_volume: float | None
    broker_comment: str | None
    broker_timestamp: datetime | None
    request_payload: Mapping[str, Any]
    execution_mode: ExecutionMode = ExecutionMode.DEMO
    broker_order_sent: bool = False
    real_money: bool = False
    position_ticket: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _text(self.intent_id, "intent_id"))
        classification = _text(self.classification, "classification", upper=True)
        if classification not in _RESULT_CLASSES:
            raise ValueError("classification is not supported")
        object.__setattr__(self, "classification", classification)
        if self.retcode is not None and (
            isinstance(self.retcode, bool) or not isinstance(self.retcode, int)
        ):
            raise ValueError("retcode must be an integer when provided")
        for name in ("order_ticket", "deal_ticket"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be positive when provided")
        if self.position_ticket is not None and (
            isinstance(self.position_ticket, bool)
            or not isinstance(self.position_ticket, int)
            or self.position_ticket <= 0
        ):
            raise ValueError("position_ticket must be positive when provided")
        for name in ("fill_price", "fill_volume"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name, minimum=0.0))
                if getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be positive when provided")
        if self.broker_comment is not None:
            object.__setattr__(self, "broker_comment", str(self.broker_comment)[:500])
        if self.broker_timestamp is not None:
            object.__setattr__(self, "broker_timestamp", utc(self.broker_timestamp, "broker_timestamp"))
        if not isinstance(self.request_payload, Mapping):
            raise ValueError("request_payload must be a mapping")
        object.__setattr__(self, "execution_mode", ExecutionMode(self.execution_mode))
        if self.execution_mode is not ExecutionMode.DEMO:
            raise ValueError("DemoOrderResult requires execution_mode=DEMO")
        if not isinstance(self.broker_order_sent, bool):
            raise ValueError("broker_order_sent must be boolean")
        if self.real_money is not False:
            raise ValueError("DEMO execution must have real_money=False")


__all__ = [
    "DemoAccountSnapshot",
    "DemoOrderIntent",
    "DemoOrderResult",
    "ExecutionMode",
]
