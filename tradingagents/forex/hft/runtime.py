"""Read-only MT5 shadow runtime for the deterministic Phase 12 engine."""

from __future__ import annotations

import math
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from tradingagents.forex.watcher import Mt5OperationBusy, SerializedMt5OperationGate

from .account import AccountSimulator, CompoundingMode
from .engines import FastExecutionEngine
from .execution import ShadowFillEngine, ShadowPositionLedger
from .features import TickFeatureEngine
from .models import FastAction, PositionState, Tick, utc
from .plan_store import AtomicPlanStore
from .risk import RiskConfig, RiskContext, RiskEngine
from .store import HftLeaseOwner, HftLeaseStatus, HftShadowStore


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
    require_plan_provenance: bool = True
    lease_ttl_seconds: int = 30
    max_reconnect_attempts: int = 3
    reconnect_backoff_seconds: float = 5.0
    initial_balance: float = 100.0
    risk_fraction: float = 0.005
    compounding_mode: CompoundingMode = CompoundingMode.COMPOUNDING_RISK
    slippage_points: float = 0.0
    latency_ms: float = 0.0
    source_fingerprint: str = "MT5_READ_ONLY"

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if isinstance(self.max_ticks, bool) or not isinstance(self.max_ticks, int) or self.max_ticks < 0:
            raise ValueError("max_ticks must be a non-negative integer")
        if self.poll_interval_seconds < 0 or self.point <= 0:
            raise ValueError("poll interval and point must be valid")
        if not isinstance(self.require_plan_provenance, bool):
            raise ValueError("require_plan_provenance must be boolean")
        if isinstance(self.lease_ttl_seconds, bool) or not isinstance(self.lease_ttl_seconds, int) or self.lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        if (
            isinstance(self.max_reconnect_attempts, bool)
            or not isinstance(self.max_reconnect_attempts, int)
            or self.max_reconnect_attempts < 0
        ):
            raise ValueError("max reconnect attempts must be a non-negative integer")
        try:
            reconnect_backoff = float(self.reconnect_backoff_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reconnect backoff must be finite and non-negative") from exc
        if not math.isfinite(reconnect_backoff) or reconnect_backoff < 0:
            raise ValueError("reconnect backoff must be finite and non-negative")
        object.__setattr__(self, "reconnect_backoff_seconds", reconnect_backoff)
        if self.initial_balance <= 0 or not 0 < self.risk_fraction <= 0.1:
            raise ValueError("account configuration is invalid")
        object.__setattr__(self, "compounding_mode", CompoundingMode(self.compounding_mode))
        if self.slippage_points < 0 or self.latency_ms < 0:
            raise ValueError("slippage and latency must be non-negative")
        if not isinstance(self.source_fingerprint, str) or not self.source_fingerprint.strip():
            raise ValueError("source_fingerprint must be non-empty")
        object.__setattr__(self, "source_fingerprint", self.source_fingerprint.strip()[:160])


class HftLeaseBusyError(RuntimeError):
    """Raised when another HFT shadow owner holds the HFT lease."""


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
        on_tick: Callable[[], None] | None = None,
    ) -> None:
        if not callable(getattr(tick_source, "get_tick", None)):
            raise TypeError("tick_source must expose read-only get_tick")
        self.tick_source = tick_source
        self.plan_store = plan_store
        self.config = config
        self.store = store or HftShadowStore(config.artifact_path)
        self.mt5_gate = mt5_gate or SerializedMt5OperationGate()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if on_tick is not None and not callable(on_tick):
            raise TypeError("on_tick must be callable")
        self.on_tick = on_tick
        self.features = TickFeatureEngine()
        self.fast = FastExecutionEngine()
        self.risk = RiskEngine(RiskConfig())
        self.fills = ShadowFillEngine(
            slippage_points=config.slippage_points,
            latency_ms=config.latency_ms,
        )
        self.positions = ShadowPositionLedger()
        self.account = AccountSimulator(
            initial_balance=config.initial_balance,
            risk_fraction=config.risk_fraction,
            mode=config.compounding_mode,
        )
        self.store.initialize()
        for payload in self.store.recover_open_positions():
            if str(payload.get("symbol", "")).strip().upper() != self.config.symbol:
                continue
            self.positions.restore(payload)
            break
        self.run_id = str(uuid.uuid4())
        self._action_count = 0
        self._entry_count = 0
        self._exit_count = 0
        self._trade_count = 0
        self._plan_ids_recorded: set[str] = set()
        self._last_tick_timestamp: datetime | None = None
        self._dropped_ticks = 0
        self._stale_ticks = 0
        self._out_of_order_ticks = 0
        self._runtime_error_code: str | None = None

    def run_once(self) -> dict[str, object]:
        now = utc(self.clock(), "now")
        try:
            with self.mt5_gate.acquire("hft_tick"):
                tick = self.tick_source.get_tick(self.config.symbol)
        except Mt5OperationBusy:
            self._dropped_ticks += 1
            return {
                "status": "DROPPED",
                "action": FastAction.NO_ACTION.value,
                "reason_code": "MT5_OPERATION_BUSY",
                "executed": False,
            }
        if not isinstance(tick, Tick):
            raise TypeError("tick source must return Tick")
        if self.on_tick is not None:
            self.on_tick()
        if self._last_tick_timestamp is not None:
            if tick.timestamp == self._last_tick_timestamp:
                self._dropped_ticks += 1
                self._stale_ticks += 1
                return {
                    "status": "DROPPED",
                    "action": FastAction.NO_ACTION.value,
                    "reason_code": "STALE_TICK",
                    "executed": False,
                }
            if tick.timestamp < self._last_tick_timestamp:
                self._dropped_ticks += 1
                self._out_of_order_ticks += 1
                return {
                    "status": "DROPPED",
                    "action": FastAction.NO_ACTION.value,
                    "reason_code": "OUT_OF_ORDER_TICK",
                    "executed": False,
                }
        plan = self.plan_store.current(
            tick.timestamp,
            self.config.symbol,
            require_provenance=self.config.require_plan_provenance,
        )
        if plan is not None and plan.plan_id and plan.plan_id not in self._plan_ids_recorded:
            self.store.record_plan(self.run_id, plan.to_payload())
            self._plan_ids_recorded.add(plan.plan_id)
        snapshot = self.features.update(tick, plan_created_at=None if plan is None else plan.created_at)
        self._last_tick_timestamp = tick.timestamp
        current = self.positions.position
        state = PositionState.FLAT if current is None else current.state
        if current is not None and state in (PositionState.LONG, PositionState.SHORT):
            current = self.positions.observe(tick)
        decision = self.fast.on_tick(plan, snapshot, position_state=state, entry_price=None if current is None else current.entry_price, entry_at=None if current is None else current.entry_timestamp)
        action_id = str(uuid.uuid4())
        self._action_count += 1
        self.store.record_tick(self.run_id, tick, features=snapshot)
        self.store.record_action(self.run_id, action_id, tick, decision.action, decision.reason, processing_ms=decision.processing_ms)
        account_metrics = self.account.report()
        context = RiskContext(
            equity=float(account_metrics["equity"] or self.config.initial_balance),
            open_exposure=0.0 if state is PositionState.FLAT else 1.0,
            daily_loss=0.0,
            drawdown=0.0,
            consecutive_losses=0,
            tick_timestamp=tick.timestamp,
            observed_at=now,
            session=snapshot.session,
        )
        risk = self.risk.evaluate(decision.action, tick, context, plan_expired=plan is None, allowed_sessions=None if plan is None else plan.session_constraints)
        self.store.record_risk(action_id, asdict(risk))
        if risk.accepted and decision.action in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT):
            fill = self.fills.fill(decision.action, tick, size=1.0)
            self.store.record_fill(self.run_id, {**asdict(fill), "timestamp": fill.timestamp.isoformat()})
            position = self.positions.open(tick, decision.action, size=1.0, stop=None, target=None, strategy_id=plan.strategy_family if plan else "none", entry_price=fill.price)
            self.store.record_position(self.run_id, asdict(position))
            self._entry_count += 1
        elif risk.accepted and decision.action is FastAction.EXIT and state in (PositionState.LONG, PositionState.SHORT):
            fill = self.fills.fill(decision.action, tick, size=1.0, position_state=state)
            self.store.record_fill(self.run_id, {**asdict(fill), "timestamp": fill.timestamp.isoformat()})
            position = self.positions.close(tick, reason=decision.reason, exit_price=fill.price)
            self.store.record_position(self.run_id, asdict(position))
            self._exit_count += 1
            self.account.record_trade(tick.timestamp, position.net_pnl)
            self._trade_count += 1
        self.store.record_account(self.run_id, tick.timestamp, self.account.report())
        return {
            "status": "PROCESSED",
            "action": decision.action.value,
            "risk_accepted": risk.accepted,
            "reason_code": risk.reason_code,
            "processing_ms": decision.processing_ms,
            "plan_id": None if plan is None else plan.plan_id,
            "executed": False,
        }

    def run(
        self,
        *,
        max_ticks: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, object]:
        limit = self.config.max_ticks if max_ticks is None else max_ticks
        if limit < 0:
            raise ValueError("max_ticks must be non-negative")
        owner = HftLeaseOwner(
            owner_token=str(uuid.uuid4()),
            pid=os.getpid(),
            host=socket.gethostname(),
            process_started_at=datetime.now(timezone.utc),
        )
        lease = self.store.acquire_lease(owner, datetime.now(timezone.utc))
        if lease.status is not HftLeaseStatus.ACQUIRED:
            raise HftLeaseBusyError("HFT_ALREADY_RUNNING")
        self.store.start_run(self.run_id, mode="SHADOW", source_fingerprint=self.config.source_fingerprint)
        started = time.perf_counter()
        status = "STOPPED"
        count = 0
        observations = 0
        try:
            while (limit == 0 or observations < limit) and not (
                stop_event is not None and stop_event.is_set()
            ):
                result = self.run_once()
                observations += 1
                if result.get("status") != "DROPPED":
                    count += 1
                self.store.heartbeat_lease(owner.owner_token, datetime.now(timezone.utc))
                if limit == 0:
                    time.sleep(self.config.poll_interval_seconds)
            status = "COMPLETED"
        except Exception as exc:
            self._runtime_error_code = type(exc).__name__.upper()
            raise
        finally:
            with suppress(Exception):
                self.store.record_run_health(
                    self.run_id,
                    dropped_ticks=self._dropped_ticks,
                    stale_ticks=self._stale_ticks,
                    out_of_order_ticks=self._out_of_order_ticks,
                    error_code=self._runtime_error_code,
                )
            self.store.close_run(self.run_id, status=status)
            self.store.release_lease(owner.owner_token)
        elapsed = max(0.0, time.perf_counter() - started)
        return {
            "run_id": self.run_id,
            "actions": self._action_count,
            "entries": self._entry_count,
            "exits": self._exit_count,
            "trades": self._trade_count,
            "runtime_seconds": elapsed,
            "ticks_processed": count,
            "observed_ticks": observations,
            "dropped_ticks": self._dropped_ticks,
            "stale_ticks": self._stale_ticks,
            "out_of_order_ticks": self._out_of_order_ticks,
            "error_code": self._runtime_error_code,
            "ticks_per_second": (count / elapsed) if elapsed else None,
            "account_metrics": self.account.report(),
            "executed": False,
        }


__all__ = [
    "HftLeaseBusyError",
    "HftShadowConfig",
    "HftShadowRuntime",
    "MT5ReadOnlyTickSource",
    "ReadOnlyTickSource",
]
