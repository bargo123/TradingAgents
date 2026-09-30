"""Explicit TEST_ONLY lifecycle canary for the Phase 12 shadow engine.

This module supplies only a synthetic strategic direction.  Every tick,
feature, risk decision, fill, and position transition still goes through the
normal deterministic HFT shadow runtime.  It is intentionally separate from
the production watcher database and never exposes a broker mutation API.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradingagents.forex.watcher import SerializedMt5OperationGate

from .models import (
    Direction,
    EntryConstraints,
    RiskPosture,
    StopPolicy,
    StrategicExecutionPlan,
    Tick,
)
from .plan_store import AtomicPlanStore
from .runtime import HftShadowConfig, HftShadowRuntime, ReadOnlyTickSource
from .store import HftShadowStore

UTC = timezone.utc
TEST_ONLY_EXECUTION_MODE = "TEST_ONLY"
TEST_ONLY_SOURCE_FINGERPRINT = "MT5_READ_ONLY_TEST_ONLY"


class CanaryError(RuntimeError):
    """Raised when the isolated canary cannot run safely."""


@dataclass(frozen=True, slots=True)
class CanaryConfig:
    symbol: str = "EURUSD"
    artifact_path: Path = Path("data_cache/phase12c-canary.sqlite3")
    max_ticks: int = 30
    poll_interval_seconds: float = 1.0
    point: float = 0.00001
    time_stop_seconds: int = 5

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol must be non-empty")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "artifact_path", Path(self.artifact_path).expanduser())
        if isinstance(self.max_ticks, bool) or not isinstance(self.max_ticks, int) or self.max_ticks <= 0:
            raise ValueError("max_ticks must be a positive integer")
        if isinstance(self.poll_interval_seconds, bool) or float(self.poll_interval_seconds) < 0:
            raise ValueError("poll_interval_seconds must be non-negative")
        object.__setattr__(self, "poll_interval_seconds", float(self.poll_interval_seconds))
        if isinstance(self.point, bool) or float(self.point) <= 0:
            raise ValueError("point must be positive")
        object.__setattr__(self, "point", float(self.point))
        if isinstance(self.time_stop_seconds, bool) or not isinstance(self.time_stop_seconds, int) or self.time_stop_seconds <= 0 or self.time_stop_seconds > 300:
            raise ValueError("time_stop_seconds must be between 1 and 300")

    def ensure_fresh_artifact(self) -> None:
        """Refuse to reuse any existing canary artifact or production ledger."""

        if self.artifact_path.exists():
            raise CanaryError(
                f"canary artifact must be a new path; refusing non-empty or existing file: {self.artifact_path}"
            )
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class CanarySideReport:
    direction: Direction
    plan_payload: dict[str, object]
    runtime_result: dict[str, object]
    position: dict[str, object] | None
    entry_quote: dict[str, object] | None = None
    exit_quote: dict[str, object] | None = None
    entry_risk: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_mode": TEST_ONLY_EXECUTION_MODE,
            "direction": self.direction.value,
            "plan": dict(self.plan_payload),
            "runtime": dict(self.runtime_result),
            "position": None if self.position is None else dict(self.position),
            "entry_quote": None if self.entry_quote is None else dict(self.entry_quote),
            "exit_quote": None if self.exit_quote is None else dict(self.exit_quote),
            "entry_risk": None if self.entry_risk is None else dict(self.entry_risk),
            "executed": False,
        }


class _PrimedTickSource:
    def __init__(self, first: Tick, source: ReadOnlyTickSource) -> None:
        self._first = first
        self._source = source

    def get_tick(self, symbol: str) -> Tick:
        if self._first is not None:
            first, self._first = self._first, None
            return first
        return self._source.get_tick(symbol)


class _SequencedTickSource:
    """Assign a deterministic per-run sequence when MT5 exposes none."""

    def __init__(self, source: ReadOnlyTickSource) -> None:
        self._source = source
        self._sequence = 0

    def get_tick(self, symbol: str) -> Tick:
        tick = self._source.get_tick(symbol)
        if not isinstance(tick, Tick):
            raise TypeError("tick_source must return Tick")
        if tick.sequence is not None:
            return tick
        self._sequence += 1
        return Tick(
            tick.symbol,
            tick.timestamp,
            tick.bid,
            tick.ask,
            tick.point,
            sequence=self._sequence,
        )


def build_test_only_plan(
    first_tick: Tick,
    direction: Direction,
    *,
    time_stop_seconds: int = 5,
) -> StrategicExecutionPlan:
    """Create the only synthetic input accepted by the canary.

    The permissive entry thresholds are scoped to this explicitly tagged
    integration plan.  The production strategic plan feed is never changed.
    """

    if not isinstance(first_tick, Tick):
        raise TypeError("first_tick must be Tick")
    try:
        direction = Direction(direction)
    except (TypeError, ValueError) as exc:
        raise ValueError("canary direction must be LONG or SHORT") from exc
    if direction not in (Direction.LONG, Direction.SHORT):
        raise ValueError("canary direction must be LONG or SHORT")
    if isinstance(time_stop_seconds, bool) or not isinstance(time_stop_seconds, int) or time_stop_seconds <= 0:
        raise ValueError("time_stop_seconds must be positive")
    expires = first_tick.timestamp + timedelta(seconds=max(30, time_stop_seconds + 15))
    return StrategicExecutionPlan(
        symbol=first_tick.symbol,
        created_at=first_tick.timestamp,
        valid_from=first_tick.timestamp,
        expires_at=expires,
        allowed_until=expires - timedelta(seconds=1),
        timeframe="TICK",
        regime="TEST_ONLY",
        primary_direction=direction,
        confidence=1.0,
        strategy_family=f"TEST_ONLY_CANARY_{direction.value}",
        entry_constraints=EntryConstraints(
            max_spread_points=20.0,
            minimum_momentum=0.0,
            minimum_volatility=0.0,
            maximum_volatility=1.0,
            minimum_confirmation=0.0,
        ),
        risk_posture=RiskPosture.NORMAL,
        stop_policy=StopPolicy(
            stop_distance_points=1000.0,
            take_profit_distance_points=1000.0,
            time_stop_seconds=time_stop_seconds,
        ),
        plan_id=f"test-only-{direction.value.lower()}-{uuid.uuid4()}",
        analysis_profile="INTRADAY",
        test_only=True,
    )


def _latest_position(path: Path, run_id: str) -> dict[str, object] | None:
    if not path.is_file():
        return None
    normalized = str(path.resolve()).replace("\\", "/")
    uri = f"file:{normalized}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=0) as db:
        row = db.execute(
            "SELECT payload_json FROM hft_positions WHERE run_id=? ORDER BY rowid DESC LIMIT 1",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    value = json.loads(row[0])
    if not isinstance(value, dict):
        raise CanaryError("canary position record is not a JSON object")
    return value


def _fill_quotes_and_risk(
    path: Path,
    run_id: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None, dict[str, object] | None]:
    """Read only scalar quote/risk evidence for the canary report."""

    if not path.is_file():
        return None, None, None
    normalized = str(path.resolve()).replace("\\", "/")
    uri = f"file:{normalized}?mode=ro"
    entries: list[dict[str, object]] = []
    exits: list[dict[str, object]] = []
    entry_risk: dict[str, object] | None = None
    with sqlite3.connect(uri, uri=True, timeout=0) as db:
        rows = db.execute(
            """
            SELECT fills.action, fills.timestamp, fills.price, ticks.tick_key,
                   ticks.bid, ticks.ask, ticks.features_json,
                   actions.action_id, risk.accepted, risk.reason_code
              FROM hft_fills AS fills
              LEFT JOIN hft_ticks AS ticks
                ON ticks.run_id=fills.run_id
               AND replace(ticks.timestamp, 'Z', '+00:00')=fills.timestamp
              LEFT JOIN hft_actions AS actions
                ON actions.run_id=fills.run_id
               AND replace(actions.timestamp, 'Z', '+00:00')=fills.timestamp
              LEFT JOIN hft_risk AS risk ON risk.action_id=actions.action_id
             WHERE fills.run_id=?
             ORDER BY fills.timestamp, fills.fill_id
            """,
            (run_id,),
        ).fetchall()
    for row in rows:
        action = str(row[0]).split(".")[-1]
        try:
            features = json.loads(row[6]) if row[6] else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            features = {}
        quote: dict[str, object] = {
            "action": action,
            "timestamp": row[1],
            "price": row[2],
            "bid": row[4],
            "ask": row[5],
            "spread": features.get("spread"),
            "spread_points": features.get("spread_points"),
            "point": features.get("point"),
            "tick_sequence": row[3],
            "executed": False,
        }
        if action.startswith("ENTER_"):
            entries.append(quote)
            if entry_risk is None:
                entry_risk = {
                    "accepted": bool(row[8]) if row[8] is not None else None,
                    "reason_code": row[9],
                    "action_id": row[7],
                }
        elif action in {"EXIT", "REDUCE"}:
            exits.append(quote)
    return (
        entries[0] if entries else None,
        exits[-1] if exits else None,
        entry_risk,
    )


def run_canary_side(
    tick_source: ReadOnlyTickSource,
    *,
    direction: Direction,
    config: CanaryConfig,
    store: HftShadowStore | None = None,
    mt5_gate: SerializedMt5OperationGate | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CanarySideReport:
    """Run one bounded TEST_ONLY side through the production HFT runtime."""

    if not callable(getattr(tick_source, "get_tick", None)):
        raise TypeError("tick_source must expose read-only get_tick")
    sequenced_source = _SequencedTickSource(tick_source)
    first_tick = sequenced_source.get_tick(config.symbol)
    if not isinstance(first_tick, Tick):
        raise TypeError("tick_source must return Tick")
    if first_tick.symbol != config.symbol:
        raise CanaryError("canary tick symbol does not match configured symbol")
    plan = build_test_only_plan(
        first_tick,
        direction,
        time_stop_seconds=config.time_stop_seconds,
    )
    plans = AtomicPlanStore()
    plans.replace(plan, now=first_tick.timestamp)
    resolved_store = store or HftShadowStore(config.artifact_path)
    runtime = HftShadowRuntime(
        _PrimedTickSource(first_tick, sequenced_source),
        plans,
        config=HftShadowConfig(
            symbol=config.symbol,
            artifact_path=config.artifact_path,
            max_ticks=config.max_ticks,
            poll_interval_seconds=config.poll_interval_seconds,
            point=config.point,
            require_plan_provenance=True,
            source_fingerprint=TEST_ONLY_SOURCE_FINGERPRINT,
        ),
        store=resolved_store,
        mt5_gate=mt5_gate,
        clock=clock,
    )
    result = runtime.run(max_ticks=config.max_ticks)
    entry_quote, exit_quote, entry_risk = _fill_quotes_and_risk(
        config.artifact_path,
        str(result["run_id"]),
    )
    return CanarySideReport(
        direction=direction,
        plan_payload=plan.to_payload(),
        runtime_result=result,
        position=_latest_position(config.artifact_path, str(result["run_id"])),
        entry_quote=entry_quote,
        exit_quote=exit_quote,
        entry_risk=entry_risk,
    )


__all__ = [
    "CanaryConfig",
    "CanaryError",
    "CanarySideReport",
    "TEST_ONLY_EXECUTION_MODE",
    "TEST_ONLY_SOURCE_FINGERPRINT",
    "build_test_only_plan",
    "run_canary_side",
]
