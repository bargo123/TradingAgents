"""Read-only MT5 shadow runtime for the deterministic Phase 12 engine."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from tradingagents.forex.watcher import SerializedMt5OperationGate

from .engines import FastExecutionEngine
from .execution import ShadowFillEngine, ShadowPositionLedger
from .features import TickFeatureEngine
from .models import FastAction, PositionState, Tick, utc
from .plan_store import AtomicPlanStore
from .risk import RiskConfig, RiskContext, RiskEngine
from .store import HftShadowStore


class ReadOnlyTickSource(Protocol):
    def get_tick(self, symbol: str) -> Tick:
        ...


class MT5ReadOnlyTickSource:
    """Adapter exposing only the provider's normalized read-only tick call."""

    def __init__(self, provider, *, point: float = 0.00001) -> None:
        self.provider = provider
        self.point = float(point)

    def get_tick(self, symbol: str) -> Tick:
        raw = self.provider.get_tick(symbol)
        return Tick(raw.symbol, raw.timestamp, raw.bid, raw.ask, self.point)


@dataclass(frozen=True, slots=True)
class HftShadowConfig:
    symbol: str = "EURUSD"
    artifact_path: Path = Path("data_cache/hft-shadow.sqlite3")
    max_ticks: int = 0
    poll_interval_seconds: float = 1.0
    point: float = 0.00001

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if isinstance(self.max_ticks, bool) or not isinstance(self.max_ticks, int) or self.max_ticks < 0:
            raise ValueError("max_ticks must be a non-negative integer")
        if self.poll_interval_seconds < 0 or self.point <= 0:
            raise ValueError("poll interval and point must be valid")


class HftShadowRuntime:
    def __init__(
        self,
        tick_source: ReadOnlyTickSource,
        plan_store: AtomicPlanStore,
        *,
        config: HftShadowConfig,
        store: HftShadowStore | None = None,
        mt5_gate: SerializedMt5OperationGate | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(getattr(tick_source, "get_tick", None)):
            raise TypeError("tick_source must expose read-only get_tick")
        self.tick_source = tick_source
        self.plan_store = plan_store
        self.config = config
        self.store = store or HftShadowStore(config.artifact_path)
        self.mt5_gate = mt5_gate or SerializedMt5OperationGate()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.features = TickFeatureEngine()
        self.fast = FastExecutionEngine()
        self.risk = RiskEngine(RiskConfig())
        self.fills = ShadowFillEngine()
        self.positions = ShadowPositionLedger()
        self.run_id = str(uuid.uuid4())
        self._action_count = 0
        self._entry_count = 0
        self._exit_count = 0

    def run_once(self) -> dict[str, object]:
        now = utc(self.clock(), "now")
        with self.mt5_gate.acquire("hft_tick"):
            tick = self.tick_source.get_tick(self.config.symbol)
        if not isinstance(tick, Tick):
            raise TypeError("tick source must return Tick")
        plan = self.plan_store.current(tick.timestamp, self.config.symbol)
        snapshot = self.features.update(tick, plan_created_at=None if plan is None else plan.created_at)
        current = self.positions.position
        state = PositionState.FLAT if current is None else current.state
        decision = self.fast.on_tick(plan, snapshot, position_state=state, entry_price=None if current is None else current.entry_price, entry_at=None if current is None else current.entry_timestamp)
        action_id = str(uuid.uuid4())
        self._action_count += 1
        self.store.record_tick(self.run_id, tick, features=snapshot)
        self.store.record_action(self.run_id, action_id, tick, decision.action, decision.reason, processing_ms=decision.processing_ms)
        context = RiskContext(
            equity=100.0,
            open_exposure=0.0 if state is PositionState.FLAT else 1.0,
            daily_loss=0.0,
            drawdown=0.0,
            consecutive_losses=0,
            tick_timestamp=tick.timestamp,
            observed_at=now,
            session=snapshot.session,
        )
        risk = self.risk.evaluate(decision.action, tick, context, plan_expired=plan is None)
        self.store.record_risk(action_id, asdict(risk))
        if risk.accepted and decision.action in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT):
            fill = self.fills.fill(decision.action, tick, size=1.0)
            self.store.record_fill(self.run_id, {**asdict(fill), "timestamp": fill.timestamp.isoformat()})
            position = self.positions.open(tick, decision.action, size=1.0, stop=None, target=None, strategy_id=plan.strategy_family if plan else "none")
            self.store.record_position(self.run_id, asdict(position))
            self._entry_count += 1
        elif risk.accepted and decision.action is FastAction.EXIT and state in (PositionState.LONG, PositionState.SHORT):
            position = self.positions.close(tick, reason=decision.reason)
            self.store.record_position(self.run_id, asdict(position))
            self._exit_count += 1
        return {
            "action": decision.action.value,
            "risk_accepted": risk.accepted,
            "reason_code": risk.reason_code,
            "processing_ms": decision.processing_ms,
            "executed": False,
        }

    def run(self, *, max_ticks: int | None = None) -> dict[str, object]:
        limit = self.config.max_ticks if max_ticks is None else max_ticks
        if limit < 0:
            raise ValueError("max_ticks must be non-negative")
        self.store.start_run(self.run_id, mode="SHADOW", source_fingerprint="MT5_READ_ONLY")
        started = time.perf_counter()
        status = "STOPPED"
        try:
            count = 0
            while limit == 0 or count < limit:
                self.run_once()
                count += 1
                if limit == 0:
                    time.sleep(self.config.poll_interval_seconds)
            status = "COMPLETED"
        finally:
            self.store.close_run(self.run_id, status=status)
        return {
            "run_id": self.run_id,
            "actions": self._action_count,
            "entries": self._entry_count,
            "exits": self._exit_count,
            "runtime_seconds": max(0.0, time.perf_counter() - started),
            "executed": False,
        }


__all__ = ["HftShadowConfig", "HftShadowRuntime", "MT5ReadOnlyTickSource", "ReadOnlyTickSource"]
