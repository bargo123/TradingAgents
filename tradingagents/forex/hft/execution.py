"""Realistic shadow fills and a single-position lifecycle ledger."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from .models import FastAction, PositionState, Tick, utc


@dataclass(frozen=True, slots=True)
class ShadowFill:
    fill_id: str
    symbol: str
    timestamp: datetime
    action: FastAction
    price: float
    size: float
    slippage_points: float
    latency_ms: float
    executed: bool = False

    def __post_init__(self) -> None:
        if self.executed is not False:
            raise ValueError("shadow fills must have executed=False")
        object.__setattr__(self, "timestamp", utc(self.timestamp))
        if self.size <= 0 or self.price <= 0:
            raise ValueError("fill size and price must be positive")


class ShadowFillEngine:
    def __init__(self, *, slippage_points: float = 0.0, latency_ms: float = 0.0) -> None:
        if slippage_points < 0 or latency_ms < 0:
            raise ValueError("slippage and latency must be non-negative")
        self.slippage_points = float(slippage_points)
        self.latency_ms = float(latency_ms)

    def fill(self, action: FastAction, tick: Tick, *, size: float, position_state: PositionState | None = None) -> ShadowFill:
        action = FastAction(action)
        if action is FastAction.ENTER_LONG or (action is FastAction.EXIT and position_state is PositionState.SHORT):
            price = tick.ask + self.slippage_points * tick.point
        elif action is FastAction.ENTER_SHORT or (action is FastAction.EXIT and position_state is PositionState.LONG):
            price = tick.bid - self.slippage_points * tick.point
        elif action is FastAction.EXIT:
            raise ValueError("position_state is required for an EXIT fill")
        else:
            raise ValueError("fill requires an entry or exit action")
        return ShadowFill(str(uuid.uuid4()), tick.symbol, tick.timestamp, action, price, float(size), self.slippage_points, self.latency_ms)


@dataclass(frozen=True, slots=True)
class ShadowPosition:
    position_id: str
    symbol: str
    state: PositionState
    direction: str
    size: float
    entry_price: float
    entry_timestamp: datetime
    strategy_id: str
    stop_price: float | None
    target_price: float | None
    exit_price: float | None = None
    exit_timestamp: datetime | None = None
    exit_reason: str | None = None
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    holding_seconds: float = 0.0
    executed: bool = False


class ShadowPositionLedger:
    def __init__(self) -> None:
        self._position: ShadowPosition | None = None

    @property
    def position(self) -> ShadowPosition | None:
        return self._position

    def restore(self, payload: Mapping[str, object]) -> ShadowPosition:
        """Restore one persisted open shadow position after a restart."""

        if not isinstance(payload, Mapping):
            raise TypeError("position payload must be a mapping")
        if payload.get("executed", False) is not False:
            raise ValueError("restored positions must be shadow-only")
        state = PositionState(payload.get("state"))
        if state not in (PositionState.LONG, PositionState.SHORT):
            raise ValueError("only open LONG/SHORT positions can be restored")

        def timestamp(value: object, name: str) -> datetime:
            if isinstance(value, datetime):
                parsed = value
            elif isinstance(value, str):
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            else:
                raise ValueError(f"{name} must be a timestamp")
            if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
                raise ValueError(f"{name} must be timezone-aware UTC")
            return parsed.astimezone(timezone.utc)

        restored = ShadowPosition(
            position_id=str(payload["position_id"]),
            symbol=str(payload["symbol"]),
            state=state,
            direction=str(payload.get("direction") or state.value),
            size=float(payload["size"]),
            entry_price=float(payload["entry_price"]),
            entry_timestamp=timestamp(payload["entry_timestamp"], "entry_timestamp"),
            strategy_id=str(payload.get("strategy_id") or "restored"),
            stop_price=None if payload.get("stop_price") is None else float(payload["stop_price"]),
            target_price=None if payload.get("target_price") is None else float(payload["target_price"]),
            exit_price=None if payload.get("exit_price") is None else float(payload["exit_price"]),
            exit_timestamp=None if payload.get("exit_timestamp") is None else timestamp(payload["exit_timestamp"], "exit_timestamp"),
            exit_reason=None if payload.get("exit_reason") is None else str(payload["exit_reason"]),
            gross_pnl=float(payload.get("gross_pnl", 0.0)),
            net_pnl=float(payload.get("net_pnl", 0.0)),
            mfe=float(payload.get("mfe", 0.0)),
            mae=float(payload.get("mae", 0.0)),
            holding_seconds=float(payload.get("holding_seconds", 0.0)),
            executed=False,
        )
        if self._position is not None and self._position.state in (PositionState.LONG, PositionState.SHORT):
            raise ValueError("position ledger already has an open position")
        self._position = restored
        return restored

    def open(self, tick: Tick, action: FastAction, *, size: float, stop: float | None, target: float | None, strategy_id: str, entry_price: float | None = None) -> ShadowPosition:
        if self._position is not None and self._position.state is not PositionState.CLOSED:
            raise ValueError("position ledger is not FLAT")
        if action not in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT):
            raise ValueError("open requires an entry action")
        direction = "LONG" if action is FastAction.ENTER_LONG else "SHORT"
        price = (tick.ask if direction == "LONG" else tick.bid) if entry_price is None else float(entry_price)
        if price <= 0:
            raise ValueError("entry_price must be positive")
        self._position = ShadowPosition(
            position_id=str(uuid.uuid4()), symbol=tick.symbol, state=PositionState.LONG if direction == "LONG" else PositionState.SHORT,
            direction=direction, size=float(size), entry_price=price, entry_timestamp=tick.timestamp,
            strategy_id=strategy_id, stop_price=stop, target_price=target,
        )
        return self._position

    def observe(self, tick: Tick) -> ShadowPosition:
        position = self._require_open()
        if position.direction == "LONG":
            favorable = tick.bid - position.entry_price
            adverse = tick.bid - position.entry_price
        else:
            favorable = position.entry_price - tick.ask
            adverse = position.entry_price - tick.ask
        self._position = replace(
            position,
            mfe=max(position.mfe, favorable),
            mae=min(position.mae, adverse),
            holding_seconds=max(0.0, (tick.timestamp - position.entry_timestamp).total_seconds()),
        )
        return self._position

    def close(self, tick: Tick, *, reason: str, exit_price: float | None = None) -> ShadowPosition:
        position = self._require_open()
        price = (tick.bid if position.direction == "LONG" else tick.ask) if exit_price is None else float(exit_price)
        if price <= 0:
            raise ValueError("exit_price must be positive")
        gross = (price - position.entry_price) * position.size if position.direction == "LONG" else (position.entry_price - price) * position.size
        self._position = replace(
            position, state=PositionState.CLOSED, exit_price=price, exit_timestamp=tick.timestamp,
            exit_reason=str(reason), gross_pnl=gross, net_pnl=gross,
            holding_seconds=max(0.0, (tick.timestamp - position.entry_timestamp).total_seconds()),
        )
        return self._position

    def _require_open(self) -> ShadowPosition:
        if self._position is None or self._position.state not in (PositionState.LONG, PositionState.SHORT):
            raise ValueError("position ledger is FLAT")
        return self._position


__all__ = ["ShadowFill", "ShadowFillEngine", "ShadowPosition", "ShadowPositionLedger"]
