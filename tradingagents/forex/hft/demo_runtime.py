"""Deterministic Phase 12D runtime that delegates broker mutation to the DEMO gateway."""

from __future__ import annotations

import os
import socket
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.forex.watcher import Mt5OperationBusy, SerializedMt5OperationGate

from .demo_gateway import VerifiedDemoExecutionGateway
from .demo_models import DemoAccountSnapshot, DemoOrderIntent
from .demo_risk import DemoCircuitBreaker, normalize_volume, validate_stop_levels
from .demo_store import DemoExecutionStore
from .engines import FastExecutionEngine
from .features import TickFeatureEngine
from .models import FastAction, PositionState, RiskDecision, Tick, utc
from .plan_store import AtomicPlanStore
from .risk import RiskConfig, RiskContext, RiskEngine
from .runtime import HftLeaseBusyError
from .store import HftLeaseOwner, HftLeaseStatus, HftShadowStore


@dataclass(frozen=True, slots=True)
class DemoRuntimeConfig:
    symbol: str = "EURUSD"
    artifact_path: Path = Path("data_cache/live-market-clean-20260923.demo.sqlite3")
    max_ticks: int = 0
    poll_interval_seconds: float = 1.0
    point: float = 0.00001
    require_plan_provenance: bool = True
    risk_fraction: float = 0.0025
    risk_ceiling: float = 0.005
    volume_cap: float = 0.01
    deviation_points: int = 20
    initial_daily_loss_limit: float = 0.02
    max_consecutive_losses: int = 3
    magic: int = 12012012
    comment: str = "TradingAgents-P12D-DEMO"

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if isinstance(self.max_ticks, bool) or not isinstance(self.max_ticks, int) or self.max_ticks < 0:
            raise ValueError("max_ticks must be a non-negative integer")
        if self.poll_interval_seconds < 0 or self.point <= 0:
            raise ValueError("poll interval and point must be valid")
        if not 0 < self.risk_fraction <= self.risk_ceiling <= 0.005:
            raise ValueError("DEMO risk must be bounded at or below 0.5%")
        if self.volume_cap <= 0 or self.deviation_points < 0:
            raise ValueError("DEMO volume/deviation bounds are invalid")
        if self.initial_daily_loss_limit <= 0 or self.max_consecutive_losses <= 0:
            raise ValueError("DEMO circuit bounds are invalid")


class DemoHftRuntime:
    """Run the existing deterministic engines with real DEMO broker fills."""

    def __init__(
        self,
        provider: Any,
        plan_store: AtomicPlanStore,
        *,
        gateway: VerifiedDemoExecutionGateway,
        store: DemoExecutionStore,
        config: DemoRuntimeConfig,
        clock: Callable[[], datetime] | None = None,
        mt5_gate: SerializedMt5OperationGate | None = None,
        lease_store: HftShadowStore | None = None,
        lease_owner: HftLeaseOwner | None = None,
    ) -> None:
        if not isinstance(plan_store, AtomicPlanStore):
            raise TypeError("plan_store must be AtomicPlanStore")
        if not isinstance(gateway, VerifiedDemoExecutionGateway):
            raise TypeError("gateway must be VerifiedDemoExecutionGateway")
        if not isinstance(store, DemoExecutionStore):
            raise TypeError("store must be DemoExecutionStore")
        self.provider = provider
        self.plan_store = plan_store
        self.gateway = gateway
        self.store = store
        self.config = config
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.mt5_gate = mt5_gate or SerializedMt5OperationGate()
        if lease_store is not None and not isinstance(lease_store, HftShadowStore):
            raise TypeError("lease_store must be HftShadowStore")
        if lease_owner is not None and not isinstance(lease_owner, HftLeaseOwner):
            raise TypeError("lease_owner must be HftLeaseOwner")
        self.lease_store = lease_store
        self.lease_owner = lease_owner
        self.features = TickFeatureEngine()
        self.fast = FastExecutionEngine()
        self.risk = RiskEngine(
            RiskConfig(
                max_risk_fraction=config.risk_ceiling,
                max_daily_loss=config.initial_daily_loss_limit,
                max_consecutive_losses=config.max_consecutive_losses,
            )
        )
        self.circuit = DemoCircuitBreaker(
            daily_loss_limit=config.initial_daily_loss_limit,
            max_consecutive_losses=config.max_consecutive_losses,
        )
        self._started = False
        self._last_tick: datetime | None = None
        self._account_equity: float | None = None
        self._ticks = 0
        self._plan_ids_recorded: set[str] = set()

    def _ensure_run(self) -> None:
        if not self._started:
            self.store.initialize()
            self.store.start_run(self.gateway.run_id, git_commit="runtime")
            self._started = True

    def _tick(self) -> Tick:
        with self.mt5_gate.acquire("demo_tick"):
            raw = self.provider.get_tick(self.config.symbol)
        if isinstance(raw, Tick):
            return raw
        return Tick(
            raw.symbol,
            raw.timestamp,
            raw.bid,
            raw.ask,
            self.config.point,
        )

    def _account(self) -> Any:
        with self.mt5_gate.acquire("demo_account"):
            account = self.provider.get_account_info()
            terminal = self.provider.get_terminal_info()
        if account is None or terminal is None:
            raise RuntimeError("DEMO account metadata unavailable")
        snapshot = DemoAccountSnapshot(
            login=getattr(account, "login", None),
            server=getattr(account, "server", None),
            company=getattr(terminal, "company", None),
            currency=getattr(account, "currency", None),
            trade_mode=getattr(account, "trade_mode", None),
            balance=getattr(account, "balance", None),
            equity=getattr(account, "equity", None),
            margin=getattr(account, "margin", None),
            free_margin=getattr(account, "margin_free", getattr(account, "free_margin", None)),
            margin_level=getattr(account, "margin_level", None),
            observed_at=utc(self.clock(), "account_observed_at"),
        )
        self.store.record_account_snapshot(self.gateway.run_id, snapshot)
        return account

    def _symbol_info(self) -> Any:
        getter = getattr(self.provider, "get_symbols", None)
        if not callable(getter):
            return None
        with self.mt5_gate.acquire("demo_symbol_info"):
            records = getter()
        for record in records:
            if str(getattr(record, "name", "")).upper() == self.config.symbol:
                return record
        return None

    def _open_position(self) -> dict[str, Any] | None:
        positions = self.store.read_owned_positions(self.config.symbol)
        open_positions = [item for item in positions if str(item.get("state", "")).upper() == "OPEN"]
        if len(open_positions) > 1:
            raise RuntimeError("DEMO_RECONCILIATION_REQUIRED: multiple owned positions")
        return open_positions[0] if open_positions else None

    def _record_risk(self, signal_id: str, risk: RiskDecision) -> None:
        self.store.record_risk_decision(
            action_id=str(uuid.uuid4()),
            signal_id=signal_id,
            accepted=risk.accepted,
            reason_code=risk.reason_code,
            reason=risk.reason,
            risk_fraction=risk.risk_fraction,
            payload={"execution_mode": "DEMO"},
        )

    def _entry_intent(self, plan: Any, tick: Tick, account: Any, info: Any) -> DemoOrderIntent:
        direction = "LONG" if plan.primary_direction.value == "LONG" else "SHORT"
        entry = tick.ask if direction == "LONG" else tick.bid
        stop_distance = plan.stop_policy.stop_distance_points * tick.point
        target_distance = plan.stop_policy.take_profit_distance_points * tick.point
        stop = entry - stop_distance if direction == "LONG" else entry + stop_distance
        target = entry + target_distance if direction == "LONG" else entry - target_distance
        stops_level = getattr(info, "trade_stops_level", 0) or 0
        validate_stop_levels(
            direction,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            point=tick.point,
            minimum_distance_points=float(stops_level),
        )
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        requested = equity * self.config.risk_fraction / max(stop_distance * 100000.0, 1.0)
        volume = normalize_volume(
            max(requested, float(getattr(info, "volume_min", 0.01) or 0.01)),
            minimum=float(getattr(info, "volume_min", 0.01) or 0.01),
            maximum=float(getattr(info, "volume_max", self.config.volume_cap) or self.config.volume_cap),
            step=float(getattr(info, "volume_step", 0.01) or 0.01),
            cap=self.config.volume_cap,
        )
        return DemoOrderIntent(
            intent_id=str(uuid.uuid4()),
            plan_id=str(plan.plan_id),
            source_decision_id=str(plan.source_decision_id),
            source_run_id=str(plan.source_run_id),
            strategy_id=str(plan.strategy_family),
            symbol=tick.symbol,
            direction=direction,
            volume=volume,
            requested_price=entry,
            stop_loss=stop,
            take_profit=target,
            deviation_points=self.config.deviation_points,
            created_at=tick.timestamp,
            git_commit=str(plan.git_commit),
            account_trade_mode=getattr(account, "trade_mode", None),
        )

    def run_once(self) -> dict[str, Any]:
        self._ensure_run()
        try:
            tick = self._tick()
        except Mt5OperationBusy:
            return {"status": "DROPPED", "action": "NO_ACTION", "execution_mode": "DEMO", "broker_order_sent": False}
        if self._last_tick is not None and tick.timestamp <= self._last_tick:
            return {"status": "DROPPED", "action": "NO_ACTION", "execution_mode": "DEMO", "broker_order_sent": False}
        self._last_tick = tick.timestamp
        self._ticks += 1
        plan = self.plan_store.current(
            tick.timestamp,
            self.config.symbol,
            require_provenance=self.config.require_plan_provenance,
        )
        if plan is not None and plan.plan_id not in self._plan_ids_recorded:
            self.store.record_plan(self.gateway.run_id, plan)
            self._plan_ids_recorded.add(plan.plan_id)
        snapshot = self.features.update(tick, plan_created_at=None if plan is None else plan.created_at)
        current = self._open_position()
        state = PositionState.FLAT if current is None else PositionState.LONG if current["direction"] == "LONG" else PositionState.SHORT
        decision = self.fast.on_tick(
            plan,
            snapshot,
            position_state=state,
            entry_price=None if current is None else float(current["price_open"]),
            entry_at=None,
        )
        signal_id = str(uuid.uuid4())
        self.store.record_signal(
            run_id=self.gateway.run_id,
            signal_id=signal_id,
            tick_timestamp=tick.timestamp,
            symbol=tick.symbol,
            action=decision.action.value,
            reason=decision.reason,
            plan_id=None if plan is None else plan.plan_id,
            payload={"execution_mode": "DEMO", "real_money": False},
        )
        account = self._account()
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        if self.circuit.daily_start_equity is None:
            self.circuit.start_day(equity, tick.timestamp)
        risk_context = RiskContext(
            equity=equity,
            open_exposure=1.0 if current is not None else 0.0,
            daily_loss=self.circuit.daily_loss_fraction,
            drawdown=0.0,
            consecutive_losses=self.circuit.consecutive_losses,
            tick_timestamp=tick.timestamp,
            observed_at=utc(self.clock(), "observed_at"),
            session=snapshot.session,
        )
        risk = self.risk.evaluate(
            decision.action,
            tick,
            risk_context,
            plan_expired=plan is None,
            allowed_sessions=None if plan is None else plan.session_constraints,
        )
        if decision.action in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT) and not self.circuit.entry_allowed(equity):
            risk = RiskDecision(False, self.circuit.status, "DEMO entry circuit is closed", 0.0)
        self._record_risk(signal_id, risk)
        if not risk.accepted or decision.action not in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT):
            return {
                "status": "PROCESSED",
                "action": decision.action.value,
                "risk_accepted": risk.accepted,
                "execution_mode": "DEMO",
                "broker_order_sent": False,
                "classification": None,
            }
        info = self._symbol_info()
        if info is None:
            raise RuntimeError("DEMO symbol metadata unavailable")
        intent = self._entry_intent(plan, tick, account, info)
        result = self.gateway.submit(intent)
        if result.classification in {"FILLED", "PARTIAL"} and result.order_ticket is not None:
            self.store.record_position(
                {
                    "ticket": result.order_ticket,
                    "intent_id": intent.intent_id,
                    "symbol": tick.symbol,
                    "direction": intent.direction,
                    "volume": result.fill_volume or intent.volume,
                    "price_open": result.fill_price or intent.requested_price,
                    "stop_loss": intent.stop_loss,
                    "take_profit": intent.take_profit,
                    "state": "OPEN",
                }
            )
        return {
            "status": "PROCESSED",
            "action": decision.action.value,
            "risk_accepted": risk.accepted,
            "execution_mode": "DEMO",
            "broker_order_sent": result.broker_order_sent,
            "classification": result.classification,
            "order_ticket": result.order_ticket,
            "deal_ticket": result.deal_ticket,
        }

    def run(self, *, max_ticks: int | None = None, stop_event: Any | None = None) -> dict[str, Any]:
        limit = self.config.max_ticks if max_ticks is None else max_ticks
        self._ensure_run()
        started = time.perf_counter()
        status = "STOPPED"
        observations = 0
        lease_token: str | None = None
        if self.lease_store is not None:
            owner = self.lease_owner
            if owner is None:
                owner = HftLeaseOwner(
                    owner_token=str(uuid.uuid4()),
                    pid=os.getpid(),
                    host=socket.gethostname(),
                    process_started_at=datetime.now(timezone.utc),
                )
                self.lease_owner = owner
            lease = self.lease_store.acquire_lease(owner, datetime.now(timezone.utc))
            if lease.status is not HftLeaseStatus.ACQUIRED:
                raise HftLeaseBusyError("HFT_ALREADY_RUNNING")
            lease_token = lease.owner_token
        try:
            while (limit == 0 or observations < limit) and not (stop_event is not None and stop_event.is_set()):
                self.run_once()
                observations += 1
                if lease_token is not None:
                    self.lease_store.heartbeat_lease(lease_token, datetime.now(timezone.utc))
                if limit == 0:
                    time.sleep(self.config.poll_interval_seconds)
            status = "COMPLETED"
        finally:
            self.store.close_run(self.gateway.run_id, status=status)
            if lease_token is not None:
                self.lease_store.release_lease(lease_token)
        return {
            "run_id": self.gateway.run_id,
            "execution_mode": "DEMO",
            "observed_ticks": observations,
            "runtime_seconds": max(0.0, time.perf_counter() - started),
            "broker_order_sent": self.store.snapshot()["orders"],
            "real_money": False,
        }


__all__ = ["DemoHftRuntime", "DemoRuntimeConfig"]
