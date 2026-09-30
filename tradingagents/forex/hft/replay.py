"""Causal tick replay and chronological walk-forward utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from .account import AccountSimulator, CompoundingMode
from .engines import FastExecutionEngine
from .execution import ShadowFillEngine, ShadowPositionLedger
from .features import TickFeatureEngine
from .models import (
    Direction,
    EntryConstraints,
    FastAction,
    PositionState,
    RiskPosture,
    StopPolicy,
    StrategicExecutionPlan,
    Tick,
)
from .risk import RiskContext, RiskEngine


class ReplayError(ValueError):
    """Raised when replay input is not causally safe."""


class StrategyFamily(str, Enum):
    MOMENTUM_CONTINUATION = "momentum_continuation"
    RANGE_REJECTION = "range_rejection"


@dataclass(frozen=True, slots=True)
class WalkForwardSplits:
    train: tuple[Tick, ...]
    development: tuple[Tick, ...]
    validation: tuple[Tick, ...]
    unseen_test: tuple[Tick, ...]


@dataclass(frozen=True, slots=True)
class ReplayReport:
    run_id: str
    source_fingerprint: str
    ticks_processed: int
    actions: int
    entries: int
    exits: int
    trades: int
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    max_drawdown: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    future_leak_detected: bool
    executed: bool = False
    account_metrics: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "source_fingerprint": self.source_fingerprint,
            "ticks_processed": self.ticks_processed,
            "actions": self.actions,
            "entries": self.entries,
            "exits": self.exits,
            "trades": self.trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "max_drawdown": self.max_drawdown,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "latency_max_ms": self.latency_max_ms,
            "future_leak_detected": self.future_leak_detected,
            "executed": False,
            "account_metrics": self.account_metrics or {},
        }


def _parse_timestamp(value: str) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ReplayError("tick timestamp must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc)


def load_ticks(path: str | Path) -> tuple[Tick, ...]:
    source = Path(path)
    if not source.is_file():
        raise ReplayError(f"tick source does not exist: {source}")
    rows: list[dict[str, object]] = []
    try:
        if source.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif source.suffix.lower() == ".json":
            decoded = json.loads(source.read_text(encoding="utf-8"))
            rows = decoded if isinstance(decoded, list) else decoded["ticks"]
        else:
            with source.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ReplayError("tick source could not be parsed") from exc
    ticks: list[Tick] = []
    for index, row in enumerate(rows):
        try:
            timestamp = _parse_timestamp(str(row["timestamp"]))
            ticks.append(Tick(str(row["symbol"]), timestamp, float(row["bid"]), float(row["ask"]), float(row.get("point", 0.00001)), int(row.get("sequence", index))))
        except (KeyError, TypeError, ValueError, ReplayError) as exc:
            raise ReplayError(f"invalid tick row {index}") from exc
    return tuple(ticks)


def walk_forward_splits(ticks: Iterable[Tick], *, train_fraction: float = 0.5, dev_fraction: float = 0.2, validation_fraction: float = 0.15) -> WalkForwardSplits:
    values = tuple(ticks)
    if len(values) < 8 or any(right.timestamp <= left.timestamp for left, right in zip(values, values[1:], strict=False)):
        raise ReplayError("walk-forward input must contain at least 8 monotonic ticks")
    if not 0 < train_fraction < 1 or not 0 < dev_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + dev_fraction + validation_fraction >= 1:
        raise ValueError("walk-forward fractions must leave an unseen test partition")
    n = len(values)
    train_end = max(1, int(n * train_fraction))
    dev_end = max(train_end + 1, int(n * (train_fraction + dev_fraction)))
    validation_end = max(dev_end + 1, int(n * (train_fraction + dev_fraction + validation_fraction)))
    if validation_end >= n:
        validation_end = n - 1
    return WalkForwardSplits(values[:train_end], values[train_end:dev_end], values[dev_end:validation_end], values[validation_end:])


def build_strategy_plan(tick: Tick, *, strategy: StrategyFamily = StrategyFamily.MOMENTUM_CONTINUATION) -> StrategicExecutionPlan:
    direction = Direction.BOTH if strategy is StrategyFamily.RANGE_REJECTION else Direction.LONG
    return StrategicExecutionPlan(
        symbol=tick.symbol, created_at=tick.timestamp, expires_at=tick.timestamp + timedelta(minutes=15), allowed_until=tick.timestamp + timedelta(minutes=14),
        timeframe="M1", regime="RANGE" if strategy is StrategyFamily.RANGE_REJECTION else "TREND", primary_direction=direction,
        confidence=0.5, strategy_family=strategy.value, entry_constraints=EntryConstraints(max_spread_points=20, minimum_momentum=0.0),
        risk_posture=RiskPosture.NORMAL, stop_policy=StopPolicy(stop_distance_points=20, take_profit_distance_points=30, time_stop_seconds=300), plan_id=str(uuid.uuid4()),
    )


class TickReplay:
    def __init__(self, ticks: Iterable[Tick], *, plan: StrategicExecutionPlan, run_id: str | None = None) -> None:
        self.ticks = tuple(ticks)
        if not self.ticks:
            raise ReplayError("replay requires at least one tick")
        if any(right.timestamp <= left.timestamp for left, right in zip(self.ticks, self.ticks[1:], strict=False)):
            raise ReplayError("replay ticks must be strictly monotonic")
        if self.ticks[-1].timestamp > datetime.now(timezone.utc):
            raise ReplayError("replay cannot consume future ticks")
        if self.ticks[0].symbol != plan.symbol:
            raise ReplayError("replay symbol does not match plan")
        self.plan = plan
        self.run_id = run_id or str(uuid.uuid4())

    def run(self) -> ReplayReport:
        features = TickFeatureEngine()
        fast = FastExecutionEngine()
        risk = RiskEngine()
        fills = ShadowFillEngine()
        positions = ShadowPositionLedger()
        account = AccountSimulator(initial_balance=100.0, risk_fraction=0.005, mode=CompoundingMode.COMPOUNDING_RISK)
        position_state = PositionState.FLAT
        entry_price = None
        entry_at = None
        latencies: list[float] = []
        actions = entries = exits = trades = 0
        pnls: list[float] = []
        for tick in self.ticks:
            started = time.perf_counter()
            snapshot = features.update(tick, plan_created_at=self.plan.created_at)
            if position_state in (PositionState.LONG, PositionState.SHORT):
                positions.observe(tick)
            decision = fast.on_tick(self.plan, snapshot, position_state=position_state, entry_price=entry_price, entry_at=entry_at)
            actions += 1
            if decision.action in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT):
                context = RiskContext(100.0, 0.0, 0.0, 0.0, 0, tick.timestamp, tick.timestamp, snapshot.session)
                gate = risk.evaluate(decision.action, tick, context, allowed_sessions=self.plan.session_constraints)
                if gate.accepted:
                    fill = fills.fill(decision.action, tick, size=1.0)
                    position = positions.open(tick, decision.action, size=1.0, stop=None, target=None, strategy_id=self.plan.strategy_family, entry_price=fill.price)
                    position_state = PositionState.LONG if decision.action is FastAction.ENTER_LONG else PositionState.SHORT
                    entry_price = position.entry_price
                    entry_at = position.entry_timestamp
                    entries += 1
            elif decision.action is FastAction.EXIT and position_state is not PositionState.FLAT:
                fill = fills.fill(decision.action, tick, size=1.0, position_state=position_state)
                closed = positions.close(tick, reason=decision.reason, exit_price=fill.price)
                account.record_trade(tick.timestamp, closed.net_pnl)
                pnls.append(closed.net_pnl)
                position_state = PositionState.FLAT
                entry_price = entry_at = None
                exits += 1
                trades += 1
            latencies.append((time.perf_counter() - started) * 1000.0)
        sorted_latencies = sorted(latencies)
        def percentile(fraction: float) -> float:
            index = min(len(sorted_latencies) - 1, max(0, math.ceil(len(sorted_latencies) * fraction) - 1))
            return sorted_latencies[index]
        wins = [value for value in pnls if value > 0]
        losses = [value for value in pnls if value < 0]
        return ReplayReport(
            run_id=self.run_id, source_fingerprint=hashlib.sha256(repr(self.ticks).encode()).hexdigest(), ticks_processed=len(self.ticks),
            actions=actions, entries=entries, exits=exits, trades=trades, win_rate=len(wins) / trades if trades else None,
            profit_factor=sum(wins) / abs(sum(losses)) if losses else None, expectancy=sum(pnls) / trades if trades else None,
            max_drawdown=float(account.report()["max_drawdown"]), latency_p50_ms=percentile(0.50), latency_p95_ms=percentile(0.95),
            latency_p99_ms=percentile(0.99), latency_max_ms=max(sorted_latencies), future_leak_detected=False, account_metrics=account.report(),
        )


__all__ = ["ReplayError", "ReplayReport", "StrategyFamily", "TickReplay", "WalkForwardSplits", "build_strategy_plan", "load_ticks", "walk_forward_splits"]
