"""Deterministic Phase 12D runtime that delegates broker mutation to the DEMO gateway."""

from __future__ import annotations

import json
import math
import os
import socket
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tradingagents.dataflows.mt5.errors import (
    Mt5AccountDisconnectedError,
    Mt5BrokerClockError,
    Mt5DataError,
)
from tradingagents.forex.hft.market_calendar import BrokerSessionCalendar, CalendarStatus
from tradingagents.forex.hft.market_lifecycle import (
    MarketLifecycleController,
    MarketObservation,
    MarketReason,
    MarketState,
)
from tradingagents.forex.watcher import Mt5OperationBusy, SerializedMt5OperationGate

from .demo_gateway import VerifiedDemoExecutionGateway, _net_pnl_from_position_deals
from .demo_models import DemoAccountSnapshot, DemoOrderIntent
from .demo_risk import DemoCircuitBreaker, normalize_volume, validate_stop_levels
from .demo_store import DemoExecutionStore
from .engines import (
    MOMENTUM_EXIT_PROFILE,
    RANGE_EXIT_PROFILE,
    DynamicExitProfile,
    FastExecutionEngine,
    HftExecutionEngine,
)
from .features import TickFeatureEngine
from .models import FastAction, PositionState, RiskDecision, Tick, utc
from .persistence import AsyncPersistenceQueue
from .plan_store import AtomicPlanStore
from .regime import StrategicRegimeState, build_hft_bootstrap_neutral_regime
from .regime_store import AtomicRegimeStore
from .risk import RiskConfig, RiskContext, RiskEngine
from .runtime import HftLeaseBusyError
from .store import HftLeaseOwner, HftLeaseStatus, HftShadowStore
from .strategies import SignalArbiter


def _broker_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(len(values) * fraction) - 1))
    return values[index]


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
    automatic_loss_pauses: bool = True
    magic: int = 12012012
    comment: str = "TradingAgents-P12D-DEMO"
    hft_first: bool = False
    no_regime_policy: str = "PAUSE"
    account_refresh_seconds: float = 1.0
    cooldown_seconds: float = 0.5
    max_order_submissions: int = 20
    order_rate_window_seconds: float = 60.0
    cost_safety_margin_points: float = 1.0
    stop_distance_points: float = 20.0
    take_profit_distance_points: float = 30.0
    stop_loss_fraction: float = 1.0
    emergency_stop_distance_points: float = 100.0
    emergency_take_profit_distance_points: float = 500.0
    time_stop_seconds: int = 30
    max_hft_duration_seconds: float = 60.0
    max_exit_spread_points: float = 40.0
    max_exit_spread_expansion_points: float = 10.0
    max_exit_volatility: float = 0.005
    market_calendar_path: Path | None = None
    market_closed_poll_interval_seconds: float = 1.0
    range_exit_profile: DynamicExitProfile = RANGE_EXIT_PROFILE
    momentum_exit_profile: DynamicExitProfile = MOMENTUM_EXIT_PROFILE
    strategy_version: str = "phase12d-hft.v1"
    config_version: str = "demo-runtime.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if self.market_calendar_path is not None:
            object.__setattr__(self, "market_calendar_path", Path(self.market_calendar_path))
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
        if not isinstance(self.automatic_loss_pauses, bool):
            raise ValueError("automatic_loss_pauses must be boolean")
        if not isinstance(self.hft_first, bool):
            raise ValueError("hft_first must be boolean")
        policy = str(self.no_regime_policy).strip().upper()
        if policy not in {"PAUSE", "BOOTSTRAP_NEUTRAL"}:
            raise ValueError("no_regime_policy must be PAUSE or BOOTSTRAP_NEUTRAL")
        object.__setattr__(self, "no_regime_policy", policy)
        for name in ("strategy_version", "config_version"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        for name in ("account_refresh_seconds", "cooldown_seconds", "order_rate_window_seconds", "cost_safety_margin_points"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if isinstance(self.max_order_submissions, bool) or not isinstance(self.max_order_submissions, int) or self.max_order_submissions <= 0:
            raise ValueError("max_order_submissions must be positive")
        for name in (
            "stop_distance_points",
            "take_profit_distance_points",
            "stop_loss_fraction",
            "emergency_stop_distance_points",
            "emergency_take_profit_distance_points",
            "max_exit_spread_points",
            "max_exit_volatility",
        ):
            value = float(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        value = float(self.max_exit_spread_expansion_points)
        if value < 0:
            raise ValueError("max_exit_spread_expansion_points must be non-negative")
        object.__setattr__(self, "max_exit_spread_expansion_points", value)
        try:
            market_poll = float(self.market_closed_poll_interval_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("market_closed_poll_interval_seconds must be finite and between 0 and 60") from exc
        if not math.isfinite(market_poll) or not 0 < market_poll <= 60:
            raise ValueError("market_closed_poll_interval_seconds must be finite and between 0 and 60")
        object.__setattr__(self, "market_closed_poll_interval_seconds", market_poll)
        if isinstance(self.time_stop_seconds, bool) or not isinstance(self.time_stop_seconds, int) or self.time_stop_seconds <= 0:
            raise ValueError("time_stop_seconds must be positive")
        value = float(self.max_hft_duration_seconds)
        if value <= 0:
            raise ValueError("max_hft_duration_seconds must be positive")
        object.__setattr__(self, "max_hft_duration_seconds", value)
        if not isinstance(self.range_exit_profile, DynamicExitProfile) or not isinstance(self.momentum_exit_profile, DynamicExitProfile):
            raise TypeError("range_exit_profile and momentum_exit_profile must be DynamicExitProfile instances")


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
        regime_store: AtomicRegimeStore | None = None,
        shadow_store: HftShadowStore | None = None,
        challenger_observer: Any | None = None,
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
        if regime_store is not None and not isinstance(regime_store, AtomicRegimeStore):
            raise TypeError("regime_store must be AtomicRegimeStore")
        if shadow_store is not None and not isinstance(shadow_store, HftShadowStore):
            raise TypeError("shadow_store must be HftShadowStore")
        if challenger_observer is not None and not callable(getattr(challenger_observer, "on_tick", None)):
            raise TypeError("challenger_observer must provide on_tick(features, regime, risk_context)")
        self.provider = provider
        self.plan_store = plan_store
        self.gateway = gateway
        self.store = store
        self.config = config
        self.regime_store = regime_store
        self.shadow_store = shadow_store
        self.challenger_observer = challenger_observer
        self._challenger_observation_failures = 0
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
        self.hft = HftExecutionEngine(
            arbiter=SignalArbiter(cost_safety_margin_points=config.cost_safety_margin_points),
            stop_distance_points=config.stop_distance_points,
            take_profit_distance_points=config.take_profit_distance_points,
            stop_loss_fraction=config.stop_loss_fraction,
            time_stop_seconds=config.time_stop_seconds,
            emergency_stop_distance_points=config.emergency_stop_distance_points,
            hard_max_duration_seconds=config.max_hft_duration_seconds,
            max_exit_spread_points=config.max_exit_spread_points,
            max_exit_spread_expansion_points=config.max_exit_spread_expansion_points,
            max_exit_volatility=config.max_exit_volatility,
            range_exit_profile=config.range_exit_profile,
            momentum_exit_profile=config.momentum_exit_profile,
        )
        self.risk = RiskEngine(
            RiskConfig(
                max_risk_fraction=config.risk_ceiling,
                max_daily_loss=config.initial_daily_loss_limit,
                max_consecutive_losses=config.max_consecutive_losses,
                enforce_loss_limits=config.automatic_loss_pauses,
            )
        )
        self.circuit = DemoCircuitBreaker(
            daily_loss_limit=config.initial_daily_loss_limit,
            max_consecutive_losses=config.max_consecutive_losses,
            pause_on_loss=config.automatic_loss_pauses,
        )
        self._started = False
        self._last_tick: datetime | None = None
        self._account_equity: float | None = None
        self._ticks = 0
        self._plan_ids_recorded: set[str] = set()
        self._regime_ids_recorded: set[str] = set()
        self._position_loaded = False
        self._owned_position: dict[str, Any] | None = None
        self._account_cache: Any | None = None
        self._account_observed_at: datetime | None = None
        self._submission_times: deque[datetime] = deque()
        self._last_submission_at: datetime | None = None
        self._order_rate_circuit_open = False
        self._persistence: AsyncPersistenceQueue | None = None
        # Continuous runs must not retain one Python object per lifetime tick.
        self._decision_latencies: deque[float] = deque(maxlen=4096)
        self._order_latencies: deque[float] = deque(maxlen=4096)
        self._market_controller = (
            MarketLifecycleController() if config.market_calendar_path is not None else None
        )
        self._market_calendar: BrokerSessionCalendar | None = None
        self._market_calendar_error: str | None = None
        self._market_last_tick_identity: tuple[str, int] | None = None
        self._market_last_quote_change_at: datetime | None = None
        self._market_last_position_check_at: datetime | None = None
        if config.market_calendar_path is not None:
            try:
                self._market_calendar = BrokerSessionCalendar.from_json(config.market_calendar_path)
            except (OSError, TypeError, ValueError) as exc:
                self._market_calendar_error = type(exc).__name__

    @property
    def market_status(self) -> MarketState | None:
        return None if self._market_controller is None else self._market_controller.observation.state

    @property
    def market_reason(self) -> MarketReason | None:
        return None if self._market_controller is None else self._market_controller.observation.reason

    @property
    def challenger_observation_failures(self) -> int:
        return self._challenger_observation_failures

    def _observe_challengers(
        self,
        features: Any,
        regime: StrategicRegimeState | None,
        risk_context: RiskContext,
    ) -> None:
        if self.challenger_observer is None:
            return
        try:
            self.challenger_observer.on_tick(features, regime, risk_context)
        except Exception:
            # Challenger persistence/evaluation is observational only; it must
            # not interrupt the incumbent decision or its verified gateway.
            self._challenger_observation_failures += 1

    def _market_idle_result(self, observation: MarketObservation) -> dict[str, Any]:
        state = observation.state
        return {
            "status": state.value,
            "action": "NO_ACTION",
            "reason_code": None if observation.reason is None else observation.reason.value,
            "execution_mode": "DEMO",
            "broker_order_sent": False,
            "market_status": state.value,
            "market_reason": None if observation.reason is None else observation.reason.value,
        }

    def _market_preflight(self) -> Tick | dict[str, Any] | None:
        """Return a validated resumed tick, or a no-trade lifecycle result."""

        controller = self._market_controller
        if controller is None:
            return None
        if self._market_calendar is None:
            return self._market_idle_result(
                controller.observe(
                    CalendarStatus.UNKNOWN,
                    terminal_healthy=False,
                    account_healthy=False,
                    tick_fresh=False,
                    error_reason=MarketReason.CALENDAR_UNAVAILABLE,
                )
            )
        try:
            with self.mt5_gate.acquire("market_connection_health"):
                connected = self.provider.is_connected()
            if not connected:
                controller.observe(
                    CalendarStatus.UNKNOWN,
                    terminal_healthy=False,
                    account_healthy=False,
                    tick_fresh=False,
                    error_reason=MarketReason.BROKER_DISCONNECTED,
                )
                raise Mt5AccountDisconnectedError("MT5 disconnected during market probe")
            account = self._account()
            account_healthy = getattr(account, "trade_mode", None) == self.gateway.demo_trade_mode
            with self.mt5_gate.acquire("market_tick_probe"):
                probe = self.provider.probe_tick(self.config.symbol)
            server = getattr(account, "server", None)
            calendar_status = self._market_calendar.status_at(
                utc(self.clock(), "market_calendar_now"),
                broker_server=server,
                symbol=probe.symbol,
            )
            identity = getattr(probe, "identity", None)
            first_identity = self._market_last_tick_identity is None
            fresh_identity = bool(
                getattr(probe, "available", False)
                and identity is not None
                and not first_identity
                and identity != self._market_last_tick_identity
            )
            if identity is not None:
                self._market_last_tick_identity = identity
            probe_now = utc(self.clock(), "market_probe_now")
            if fresh_identity:
                self._market_last_quote_change_at = probe_now
            # Duplicate polls are not a feed outage. Retain OPEN only within
            # the provider's existing freshness bound after an observed change.
            max_age = float(getattr(getattr(self.provider, "clock_config", None), "max_tick_age_seconds", 120.0))
            recent_change = (
                controller.observation.state is MarketState.MARKET_OPEN
                and getattr(probe, "available", False)
                and self._market_last_quote_change_at is not None
                and 0 <= (probe_now - self._market_last_quote_change_at).total_seconds() <= max_age
            )
            if first_identity and getattr(probe, "available", False):
                return self._market_idle_result(
                    controller.observe(
                        calendar_status,
                        terminal_healthy=True,
                        account_healthy=account_healthy,
                        tick_fresh=False,
                        error_reason=MarketReason.STALE_DATA,
                    )
                )
            observation = controller.observe(
                calendar_status,
                terminal_healthy=True,
                account_healthy=account_healthy,
                tick_fresh=fresh_identity or recent_change,
                error_reason=None if account_healthy else MarketReason.DATA_FAILURE,
                require_validation=self._order_rate_circuit_open,
            )
            if observation.state is MarketState.MARKET_OPEN:
                # _tick still applies calibrated broker time/freshness checks;
                # the execution gateway still checks DEMO immediately at send.
                if self._open_position() is not None and (
                    self._market_last_position_check_at is None
                    or (probe_now - self._market_last_position_check_at).total_seconds()
                    >= self.config.account_refresh_seconds
                ):
                    self._validate_market_reconciliation()
                    self._market_last_position_check_at = probe_now
                return self._tick()
            if observation.state is not MarketState.MARKET_OPENING_VALIDATION:
                return self._market_idle_result(observation)
            try:
                tick = self._validate_market_reopen()
            except Mt5OperationBusy:
                return {
                    "status": "DROPPED",
                    "action": "NO_ACTION",
                    "reason_code": "MT5_OPERATION_BUSY",
                    "execution_mode": "DEMO",
                    "broker_order_sent": False,
                    "market_status": MarketState.MARKET_OPENING_VALIDATION.value,
                    "market_reason": None if observation.reason is None else observation.reason.value,
                }
            except Mt5BrokerClockError:
                return self._market_idle_result(
                    controller.complete_opening_validation(
                        False, reason=MarketReason.BROKER_CLOCK_ERROR
                    )
                )
            except Mt5AccountDisconnectedError:
                controller.complete_opening_validation(
                    False, reason=MarketReason.BROKER_DISCONNECTED
                )
                raise
            except Exception:
                return self._market_idle_result(
                    controller.complete_opening_validation(False, reason=MarketReason.DATA_FAILURE)
                )
            controller.complete_opening_validation(True, reason=observation.reason)
            self._market_last_position_check_at = probe_now
            return tick
        except Mt5OperationBusy:
            return {
                "status": "DROPPED",
                "action": "NO_ACTION",
                "reason_code": "MT5_OPERATION_BUSY",
                "execution_mode": "DEMO",
                "broker_order_sent": False,
            }
        except Mt5BrokerClockError:
            return self._market_idle_result(
                controller.observe(
                    CalendarStatus.UNKNOWN,
                    terminal_healthy=True,
                    account_healthy=True,
                    tick_fresh=False,
                    error_reason=MarketReason.BROKER_CLOCK_ERROR,
                )
            )
        except Mt5AccountDisconnectedError:
            raise
        except Exception:
            return self._market_idle_result(
                controller.observe(
                    CalendarStatus.UNKNOWN,
                    terminal_healthy=True,
                    account_healthy=False,
                    tick_fresh=False,
                    error_reason=MarketReason.DATA_FAILURE,
                )
            )

    def _validate_market_reopen(self) -> Tick:
        with self.mt5_gate.acquire("market_reopen_connection_health"):
            connected = self.provider.is_connected()
        if not connected:
            raise Mt5AccountDisconnectedError("MT5 disconnected during reopen validation")
        calibrate = getattr(self.provider, "calibrate_broker_clock", None)
        if not callable(calibrate):
            raise Mt5BrokerClockError("broker clock calibration is unavailable")
        with self.mt5_gate.acquire("market_reopen_clock"):
            calibrate(self.config.symbol)
        tick = self._tick()
        now = utc(self.clock(), "market_reopen_now")
        age = (now - tick.timestamp).total_seconds()
        clock_config = getattr(self.provider, "clock_config", None)
        max_age = float(getattr(clock_config, "max_tick_age_seconds", 120.0))
        max_future = float(getattr(clock_config, "max_future_skew_seconds", 2.0))
        if age > max_age or age < -max_future:
            raise Mt5DataError("resumed broker tick is outside freshness limits")
        account_snapshot = self.gateway.verify_demo_account()
        if account_snapshot.server is None or account_snapshot.trade_mode != self.gateway.demo_trade_mode:
            raise Mt5DataError("authoritative DEMO account identity is unavailable")
        self._account(force=True)
        persisted_circuit = self.store.read_circuit_state()
        if persisted_circuit.get("status") == "HFT_ORDER_RATE_CIRCUIT_OPEN":
            if not self._recover_expired_order_rate_circuit(
                now=now,
                broker_tick_timestamp=tick.timestamp,
            ):
                raise Mt5DataError("DEMO order-rate circuit recovery conditions are not met")
        else:
            self._validate_market_reconciliation()
        if self.circuit.status != "READY" or self._order_rate_circuit_open:
            raise Mt5DataError("DEMO risk or order-rate circuit is not ready")
        persisted_circuit = self.store.read_circuit_state()
        if persisted_circuit.get("status") not in (None, "READY"):
            raise Mt5DataError("persisted DEMO risk circuit is not ready")
        return tick

    def _recover_expired_order_rate_circuit(
        self,
        *,
        now: datetime,
        broker_tick_timestamp: datetime,
    ) -> bool:
        """Recover only the expired rate circuit after DEMO/reconciliation gates.

        Called only after the current tick has passed market-open validation.
        It independently rechecks account and broker/local reconciliation
        before the narrowly scoped store transition.
        """

        state = self.store.read_circuit_state()
        status = state.get("status")
        if status in (None, "READY"):
            return True
        if status != "HFT_ORDER_RATE_CIRCUIT_OPEN":
            return False
        if state.get("reason") != "bounded broker submission rate exceeded":
            return False
        try:
            triggered_at = datetime.fromisoformat(
                str(state.get("updated_at", "")).replace("Z", "+00:00")
            )
        except ValueError:
            return False
        if (
            triggered_at.tzinfo is None
            or utc(now, "order_rate_recovery_now")
            < triggered_at.astimezone(timezone.utc)
            + timedelta(seconds=self.config.order_rate_window_seconds)
        ):
            return False
        broker_cutoff = broker_tick_timestamp.timestamp() - self.config.order_rate_window_seconds
        while self._submission_times and self._submission_times[0].timestamp() < broker_cutoff:
            self._submission_times.popleft()
        if len(self._submission_times) >= self.config.max_order_submissions:
            return False
        account = self.gateway.verify_demo_account()
        if account.server is None or account.trade_mode != self.gateway.demo_trade_mode:
            return False
        self._validate_market_reconciliation()
        recovered = self.store.recover_expired_order_rate_circuit(
            now=now,
            cooldown_seconds=self.config.order_rate_window_seconds,
            max_submissions=self.config.max_order_submissions,
        )
        if recovered:
            self._order_rate_circuit_open = False
            self.circuit.status = "READY"
        return recovered

    def _validate_market_reconciliation(self) -> None:
        reconciliation = self.store.read_reconciliation_state()
        if reconciliation.get("status") not in (None, "RECONCILED"):
            raise Mt5DataError("DEMO reconciliation is not clean")
        position_reader = getattr(self.provider, "get_positions", None)
        order_reader = getattr(self.provider, "get_orders", None)
        if not callable(position_reader) or not callable(order_reader):
            raise Mt5DataError("broker position/order reconciliation is unavailable")
        with self.mt5_gate.acquire("market_reopen_reconciliation"):
            broker_positions = tuple(position_reader(self.config.symbol) or ())
            broker_orders = tuple(order_reader(self.config.symbol) or ())
        if broker_orders:
            raise Mt5DataError("broker has pending orders requiring reconciliation")
        owned = tuple(
            row
            for row in self.store.read_owned_positions(self.config.symbol)
            if str(row.get("state", "")).upper() == "OPEN"
        )
        if not broker_positions and not owned:
            return
        if not broker_positions and len(owned) == 1:
            # Only exact broker exit history may clear an owned OPEN record
            # automatically. Missing/foreign/ambiguous history stays blocked.
            result = self.reconcile_position(int(owned[0]["ticket"]), require_closed_history=True)
            if result.get("status") != "RECONCILED" or result.get("terminal_state") != "CLOSED":
                raise Mt5DataError("DEMO reconciliation requires exact closed-position history")
            with self.mt5_gate.acquire("market_reconciliation_confirm_flat"):
                if tuple(position_reader(self.config.symbol) or ()) or tuple(order_reader(self.config.symbol) or ()):
                    raise Mt5DataError("broker state changed during reconciliation")
            return
        if len(broker_positions) != 1 or len(owned) != 1:
            raise Mt5DataError("broker and local open-position state is ambiguous")
        broker, local = broker_positions[0], owned[0]
        def fields(record: Any, name: str, default: Any = None) -> Any:
            return (
                record.get(name, default)
                if isinstance(record, Mapping)
                else getattr(record, name, default)
            )
        if (
            int(fields(broker, "ticket", -1)) != int(local.get("ticket", -2))
            or str(fields(broker, "symbol", "")).upper() != str(local.get("symbol", "")).upper()
            or fields(broker, "magic") != self.gateway.magic
            or str(fields(broker, "comment", "")).strip() != self.gateway.comment
            or not math.isclose(float(fields(broker, "volume", -1)), float(local.get("volume", -2)), rel_tol=0, abs_tol=1e-8)
        ):
            raise Mt5DataError("broker and local open-position identity differs")
        position_type = fields(broker, "type")
        buy_type = getattr(self.gateway.broker_api, "POSITION_TYPE_BUY", 0)
        sell_type = getattr(self.gateway.broker_api, "POSITION_TYPE_SELL", 1)
        expected_type = buy_type if str(local.get("direction", "")).upper() == "LONG" else sell_type
        if position_type != expected_type:
            raise Mt5DataError("broker and local open-position direction differs")

    def _poll_delay_seconds(self, observation: Mapping[str, Any]) -> float:
        status = str(observation.get("status", ""))
        reason = observation.get("market_reason")
        if status == MarketState.MARKET_CLOSED.value:
            return self.config.market_closed_poll_interval_seconds
        if status == MarketState.MARKET_UNKNOWN.value and reason != MarketReason.STALE_DATA.value:
            return self.config.market_closed_poll_interval_seconds
        if status == MarketState.MARKET_DATA_ERROR.value:
            return self.config.market_closed_poll_interval_seconds
        return max(0.01, self.config.poll_interval_seconds)

    def _bootstrap_regime(self, tick: Tick, account: Any | None) -> StrategicRegimeState | None:
        """Return a neutral fallback only after local DEMO safety checks."""

        if not self.config.hft_first or self.config.no_regime_policy != "BOOTSTRAP_NEUTRAL":
            return None
        if account is None or getattr(account, "trade_mode", None) != self.gateway.demo_trade_mode:
            return None
        if self.circuit.status != "READY":
            return None
        bootstrap = build_hft_bootstrap_neutral_regime(
            self.config.symbol,
            tick.timestamp,
            git_commit="runtime",
        )
        if self.regime_store is None:
            return bootstrap
        with suppress(ValueError):
            self.regime_store.replace(bootstrap, now=tick.timestamp)
        return self.regime_store.current(tick.timestamp, self.config.symbol) or bootstrap

    def _ensure_run(self) -> None:
        if not self._started:
            self.store.initialize()
            self.store.start_run(self.gateway.run_id, git_commit="runtime")
            if self.shadow_store is not None:
                self.shadow_store.initialize()
                self.shadow_store.start_run(
                    self.gateway.run_id,
                    mode="SHADOW",
                    source_fingerprint="MT5_DEMO_HFT",
                )
            self._started = True

    def _persist(self, target: Any, method: str, *args: Any, **kwargs: Any) -> bool:
        writer = self._persistence
        if writer is None:
            getattr(target, method)(*args, **kwargs)
            return True
        return writer.submit(target, method, *args, **kwargs)

    def _record_shadow_tick(self, tick: Tick, snapshot: Any, decision: Any, action_id: str, risk: RiskDecision) -> None:
        if self.shadow_store is None:
            return
        self._persist(self.shadow_store, "record_tick", self.gateway.run_id, tick, features=snapshot)
        self._persist(
            self.shadow_store,
            "record_action",
            self.gateway.run_id,
            action_id,
            tick,
            decision.action,
            decision.reason,
            processing_ms=decision.processing_ms,
        )
        self._persist(self.shadow_store, "record_risk", action_id, asdict(risk))

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
            float(getattr(raw, "point", self.config.point) or self.config.point),
        )

    def _account(self, *, force: bool = False) -> Any:
        now = utc(self.clock(), "account_now")
        if (
            not force
            and self._account_cache is not None
            and self._account_observed_at is not None
            and (now - self._account_observed_at).total_seconds() < self.config.account_refresh_seconds
        ):
            return self._account_cache
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
            observed_at=now,
        )
        self._persist(self.store, "record_account_snapshot", self.gateway.run_id, snapshot)
        self._account_cache = account
        self._account_observed_at = now
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
        if not self._position_loaded:
            positions = self.store.read_owned_positions(self.config.symbol)
            open_positions = [item for item in positions if str(item.get("state", "")).upper() == "OPEN"]
            if len(open_positions) > 1:
                raise RuntimeError("DEMO_RECONCILIATION_REQUIRED: multiple owned positions")
            self._owned_position = open_positions[0] if open_positions else None
            self._position_loaded = True
        return self._owned_position

    def _record_risk(self, signal_id: str, risk: RiskDecision) -> None:
        self._persist(self.store, "record_risk_decision",
            action_id=str(uuid.uuid4()),
            signal_id=signal_id,
            accepted=risk.accepted,
            reason_code=risk.reason_code,
            reason=risk.reason,
            risk_fraction=risk.risk_fraction,
            payload={"execution_mode": "DEMO"},
        )

    def _entry_intent(
        self,
        source: Any,
        tick: Tick,
        account: Any,
        info: Any,
        *,
        strategy_id: str | None = None,
        direction: str | None = None,
        expected_move_points: float | None = None,
    ) -> DemoOrderIntent:
        source_direction = getattr(getattr(source, "primary_direction", None), "value", None)
        direction = direction or ("LONG" if source_direction == "LONG" else "SHORT")
        entry = tick.ask if direction == "LONG" else tick.bid
        stop_policy = getattr(source, "stop_policy", None)
        risk_stop_distance_points = float(getattr(stop_policy, "stop_distance_points", self.config.stop_distance_points))
        if self.config.hft_first:
            stop_distance_points = self.config.emergency_stop_distance_points
            target_distance_points = self.config.emergency_take_profit_distance_points
        else:
            stop_distance_points = float(getattr(stop_policy, "stop_distance_points", self.config.stop_distance_points))
            target_distance_points = float(getattr(stop_policy, "take_profit_distance_points", self.config.take_profit_distance_points))
        stop_distance = stop_distance_points * tick.point
        target_distance = target_distance_points * tick.point
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
        risk_multiplier = float(getattr(source, "risk_multiplier", 1.0) or 1.0)
        requested = equity * self.config.risk_fraction * risk_multiplier / max(risk_stop_distance_points * tick.point * 100000.0, 1.0)
        volume = normalize_volume(
            max(requested, float(getattr(info, "volume_min", 0.01) or 0.01)),
            minimum=float(getattr(info, "volume_min", 0.01) or 0.01),
            maximum=float(getattr(info, "volume_max", self.config.volume_cap) or self.config.volume_cap),
            step=float(getattr(info, "volume_step", 0.01) or 0.01),
            cap=self.config.volume_cap,
        )
        plan_id = str(getattr(source, "plan_id", None) or getattr(source, "state_id", "")).strip()
        source_decision_id = str(getattr(source, "source_decision_id", None) or plan_id).strip()
        source_run_id = str(getattr(source, "source_run_id", None) or plan_id).strip()
        git_commit = str(getattr(source, "git_commit", "runtime") or "runtime").strip()
        if not plan_id or not source_decision_id or not source_run_id:
            raise RuntimeError("DEMO strategy provenance unavailable")
        return DemoOrderIntent(
            intent_id=str(uuid.uuid4()),
            plan_id=plan_id,
            source_decision_id=source_decision_id,
            source_run_id=source_run_id,
            strategy_id=str(strategy_id or getattr(source, "strategy_family", "strategic_shadow")),
            symbol=tick.symbol,
            direction=direction,
            volume=volume,
            requested_price=entry,
            stop_loss=stop,
            take_profit=target,
            deviation_points=self.config.deviation_points,
            created_at=tick.timestamp,
            git_commit=git_commit,
            account_trade_mode=getattr(account, "trade_mode", None),
        )

    def _exit_intent(self, source: Any, current: dict[str, Any], tick: Tick, account: Any, *, reason: str) -> DemoOrderIntent:
        original = str(current.get("direction", "")).upper()
        direction = "SHORT" if original == "LONG" else "LONG"
        requested = tick.bid if original == "LONG" else tick.ask
        distance = self.config.emergency_stop_distance_points * tick.point
        target_distance = self.config.emergency_take_profit_distance_points * tick.point
        stop = requested + distance if direction == "SHORT" else requested - distance
        target = requested - target_distance if direction == "SHORT" else requested + target_distance
        return DemoOrderIntent(
            intent_id=str(uuid.uuid4()),
            plan_id=str(getattr(source, "plan_id", None) or getattr(source, "state_id", "exit-state")),
            source_decision_id=str(getattr(source, "source_decision_id", None) or "runtime-exit"),
            source_run_id=str(getattr(source, "source_run_id", None) or "runtime-exit"),
            strategy_id=f"exit:{reason}",
            symbol=tick.symbol,
            direction=direction,
            volume=float(current["volume"]),
            requested_price=requested,
            stop_loss=stop,
            take_profit=target,
            deviation_points=self.config.deviation_points,
            created_at=tick.timestamp,
            git_commit=str(getattr(source, "git_commit", "runtime") or "runtime"),
            account_trade_mode=getattr(account, "trade_mode", None),
        )

    def _order_allowed(self, timestamp: datetime) -> bool:
        if self._order_rate_circuit_open:
            return False
        cutoff = timestamp.timestamp() - self.config.order_rate_window_seconds
        while self._submission_times and self._submission_times[0].timestamp() < cutoff:
            self._submission_times.popleft()
        if len(self._submission_times) >= self.config.max_order_submissions:
            self._order_rate_circuit_open = True
            self.store.set_circuit_state(
                status="HFT_ORDER_RATE_CIRCUIT_OPEN",
                daily_start_equity=self.circuit.daily_start_equity,
                daily_loss_fraction=self.circuit.daily_loss_fraction,
                consecutive_losses=self.circuit.consecutive_losses,
                reason="bounded broker submission rate exceeded",
            )
            return False
        return not (
            self._last_submission_at is not None
            and (timestamp - self._last_submission_at).total_seconds() < self.config.cooldown_seconds
        )

    def _record_submission(self, timestamp: datetime) -> None:
        self._submission_times.append(timestamp)
        self._last_submission_at = timestamp

    def _entry_position_clear(self) -> bool:
        getter = getattr(self.provider, "get_positions", None)
        if not callable(getter):
            self.store.record_reconciliation(
                status="BROKER_STATE_UNKNOWN",
                reason="provider does not expose position reads before entry",
                details={"symbol": self.config.symbol},
            )
            return False
        try:
            with self.mt5_gate.acquire("demo_entry_position_guard"):
                positions = tuple(getter(self.config.symbol) or ())
        except Exception:
            self.store.record_reconciliation(
                status="BROKER_STATE_UNKNOWN",
                reason="position read failed before entry",
                details={"symbol": self.config.symbol},
            )
            return False
        if positions:
            self.store.record_reconciliation(
                status="RECONCILIATION_REQUIRED",
                reason="broker position exists before new entry",
                details={"symbol": self.config.symbol, "count": len(positions)},
            )
            return False
        return True

    def reconcile_position(self, ticket: int, *, require_closed_history: bool = False) -> dict[str, Any]:
        """Reconcile one stale DEMO ledger position using this owner's MT5 session.

        All broker reads are performed under the existing serialized gate.  A
        positive result is append-only in the ledger: the original position
        row remains auditable while its state is updated from authoritative
        broker evidence.  Ambiguous or non-DEMO states remain fail-closed.
        """

        if isinstance(ticket, bool) or not isinstance(ticket, int) or ticket <= 0:
            raise ValueError("ticket must be a positive integer")
        owned = tuple(
            row for row in self.store.read_owned_positions(self.config.symbol)
            if int(row.get("ticket", -1)) == ticket
        )
        if len(owned) != 1:
            self.store.record_reconciliation(
                status="RECONCILIATION_REQUIRED",
                reason="owned ledger ticket is missing or duplicated",
                details={"ticket": ticket, "symbol": self.config.symbol, "count": len(owned)},
            )
            return {"status": "RECONCILIATION_REQUIRED", "reason": "LEDGER_TICKET_AMBIGUOUS"}
        ledger = dict(owned[0])

        def parse_timestamp(value: Any) -> datetime | None:
            if isinstance(value, datetime):
                return value.astimezone(timezone.utc) if value.tzinfo is not None else None
            if isinstance(value, str):
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    return None
                return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
            return None

        def numeric_ids(record: Any) -> set[int]:
            identifiers: set[int] = set()
            for key in ("ticket", "order", "deal", "position_id"):
                value = _broker_field(record, key)
                if isinstance(value, bool):
                    continue
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if number > 0:
                    identifiers.add(number)
            return identifiers

        intent = self.store.read_order_intent(str(ledger.get("intent_id", ""))) or {}
        result_rows = self.store.read_order_results(str(ledger.get("intent_id", "")))
        broker_ids = {ticket}
        for row in result_rows:
            for key in ("order_ticket", "deal_ticket"):
                value = row.get(key)
                if isinstance(value, int) and value > 0:
                    broker_ids.add(value)
        anchor = next(
            (
                candidate
                for candidate in (
                    parse_timestamp(intent.get("created_at")),
                    parse_timestamp(ledger.get("updated_at")),
                    parse_timestamp(ledger.get("opened_at")),
                )
                if candidate is not None
            ),
            None,
        )
        if anchor is None:
            anchor = datetime.now(timezone.utc)
        window = timedelta(seconds=900)
        history_start, history_end = anchor - window, anchor + window

        try:
            with self.mt5_gate.acquire("demo_reconcile"):
                account = VerifiedDemoExecutionGateway.read_account(self.provider)
                VerifiedDemoExecutionGateway.require_demo_account(account, self.gateway.demo_trade_mode)
                get_positions = getattr(self.provider, "get_positions", None)
                get_orders = getattr(self.provider, "get_orders", None)
                get_history_orders_range = getattr(self.provider, "get_history_orders_range", None)
                get_history_deals_range = getattr(self.provider, "get_history_deals_range", None)
                get_history_orders = getattr(self.provider, "get_history_orders", None)
                get_history_deals = getattr(self.provider, "get_history_deals", None)
                if not all(callable(item) for item in (get_positions, get_orders)):
                    raise RuntimeError("authoritative broker reconciliation APIs are unavailable")
                positions = tuple(get_positions(self.config.symbol) or ())
                orders = tuple(get_orders(self.config.symbol) or ())
                if callable(get_history_orders_range) and callable(get_history_deals_range):
                    history_orders = tuple(get_history_orders_range(history_start, history_end) or ())
                    history_deals = tuple(get_history_deals_range(history_start, history_end) or ())
                elif callable(get_history_orders) and callable(get_history_deals):
                    history_orders = tuple(get_history_orders(ticket) or ())
                    history_deals = tuple(get_history_deals(ticket) or ())
                else:
                    raise RuntimeError("authoritative broker history APIs are unavailable")
        except Exception as exc:
            self.store.record_reconciliation(
                status="RECONCILIATION_REQUIRED",
                reason="broker reconciliation read failed",
                details={"ticket": ticket, "symbol": self.config.symbol, "error_type": type(exc).__name__},
            )
            return {"status": "RECONCILIATION_REQUIRED", "reason": "BROKER_READ_FAILED", "error_type": type(exc).__name__}

        symbol = str(ledger.get("symbol", self.config.symbol)).upper()
        direction = str(ledger.get("direction", "")).upper()

        def symbol_matches(record: Any) -> bool:
            return str(_broker_field(record, "symbol", "")).upper() == symbol

        def metadata_matches(record: Any) -> bool:
            return (
                symbol_matches(record)
                and _broker_field(record, "magic") == self.config.magic
                and str(_broker_field(record, "comment", "") or "").strip() == self.config.comment
            )

        def position_direction(record: Any) -> str | None:
            value = _broker_field(record, "type")
            return "LONG" if value == 0 else "SHORT" if value == 1 else None

        def volume_matches(record: Any) -> bool:
            value = _broker_field(record, "volume")
            try:
                return abs(float(value) - float(ledger.get("volume", 0.0))) <= 1e-9
            except (TypeError, ValueError):
                return False

        def adopt_position(broker: Any) -> dict[str, Any]:
            broker_ticket = int(_broker_field(broker, "ticket"))
            updated = {
                **ledger,
                "state": "OPEN",
                "position_ticket": broker_ticket,
                "volume": float(_broker_field(broker, "volume", ledger["volume"])),
                "price_open": float(_broker_field(broker, "price_open", ledger["price_open"])),
                "stop_loss": _broker_field(broker, "sl", ledger.get("stop_loss")),
                "take_profit": _broker_field(broker, "tp", ledger.get("take_profit")),
                "reconciled_at": datetime.now(timezone.utc).isoformat(),
            }
            self.store.record_position(updated)
            self.store.record_reconciliation(
                status="RECONCILED",
                reason="broker position adopted",
                details={
                    "ticket": ticket,
                    "position_ticket": broker_ticket,
                    "symbol": symbol,
                    "terminal_state": "OPEN",
                },
            )
            self._position_loaded = False
            self._owned_position = None
            return {
                "status": "RECONCILED",
                "terminal_state": "OPEN",
                "ticket": ticket,
                "broker_position_ticket": broker_ticket,
            }

        exact_positions = [item for item in positions if numeric_ids(item) & broker_ids]
        owned_positions = [item for item in positions if metadata_matches(item)]
        if exact_positions:
            matches = [item for item in exact_positions if metadata_matches(item)]
            if len(matches) == 1:
                broker = matches[0]
            else:
                self.store.record_reconciliation(
                    status="RECONCILIATION_REQUIRED",
                    reason="broker ticket ownership metadata is ambiguous",
                    details={"ticket": ticket, "symbol": symbol, "count": len(exact_positions)},
                )
                return {"status": "RECONCILIATION_REQUIRED", "reason": "BROKER_OWNERSHIP_AMBIGUOUS"}
            if position_direction(broker) != direction or not volume_matches(broker):
                self.store.record_reconciliation(
                    status="RECONCILIATION_REQUIRED",
                    reason="broker position attributes are inconsistent",
                    details={"ticket": ticket, "symbol": symbol},
                )
                return {"status": "RECONCILIATION_REQUIRED", "reason": "BROKER_POSITION_MISMATCH"}
            return adopt_position(broker)
        if len(owned_positions) == 1:
            broker = owned_positions[0]
            if position_direction(broker) != direction or not volume_matches(broker):
                self.store.record_reconciliation(
                    status="RECONCILIATION_REQUIRED",
                    reason="broker position attributes are inconsistent",
                    details={"ticket": ticket, "symbol": symbol},
                )
                return {"status": "RECONCILIATION_REQUIRED", "reason": "BROKER_POSITION_MISMATCH"}
            return adopt_position(broker)
        if len(owned_positions) > 1:
            self.store.record_reconciliation(
                status="RECONCILIATION_REQUIRED",
                reason="multiple owned broker positions match stale ledger",
                details={"ticket": ticket, "symbol": symbol, "count": len(owned_positions)},
            )
            return {"status": "RECONCILIATION_REQUIRED", "reason": "BROKER_OWNERSHIP_AMBIGUOUS"}

        exact_orders = [item for item in orders if numeric_ids(item) & broker_ids]
        owned_orders = [item for item in orders if metadata_matches(item)]
        if exact_orders or owned_orders:
            self.store.record_reconciliation(
                status="RECONCILIATION_REQUIRED",
                reason="broker order remains without an owned position",
                details={"ticket": ticket, "symbol": symbol},
            )
            return {"status": "RECONCILIATION_REQUIRED", "reason": "PENDING_BROKER_ORDER"}

        def history_matches(record: Any) -> bool:
            if not symbol_matches(record):
                return False
            if require_closed_history:
                # A previous trade's identical magic/comment is not proof for
                # this ticket. Include the mapped position id when available.
                return (
                    _broker_field(record, "magic") == self.config.magic
                    and bool(numeric_ids(record) & (broker_ids | {self._broker_position_ticket(ledger)}))
                )
            return bool(numeric_ids(record) & broker_ids) or metadata_matches(record)

        history_orders = tuple(item for item in history_orders if history_matches(item))
        history_deals = tuple(item for item in history_deals if history_matches(item))

        def text_state(record: Any) -> str:
            return " ".join(
                str(_broker_field(record, key, "") or "")
                for key in ("state", "reason", "retcode")
            ).upper()

        close_deals = [
            item
            for item in history_deals
            if str(_broker_field(item, "entry", "")).upper() in {"OUT", "OUT_BY"}
            or _broker_field(item, "entry") in {1, 2}
        ]
        rejection_records = [
            item
            for item in history_orders
            if any(word in text_state(item) for word in ("REJECT", "CANCEL", "EXPIRE", "INVALID"))
        ]
        filled_records = [
            item
            for item in history_orders
            if "FILL" in text_state(item) or "DONE" in text_state(item)
        ]
        if require_closed_history and not close_deals:
            self.store.record_reconciliation(
                status="RECONCILIATION_REQUIRED",
                reason="exact owned position exit history is unavailable",
                details={"ticket": ticket, "symbol": symbol},
            )
            return {"status": "RECONCILIATION_REQUIRED", "reason": "EXACT_CLOSE_HISTORY_REQUIRED"}
        if close_deals:
            realized, pnl_status, pnl_deal_count, pnl_error_type = (
                _net_pnl_from_position_deals(
                    history_deals,
                    self._broker_position_ticket(ledger),
                )
            )
            exit_price = _broker_field(close_deals[-1], "price")
            self.store.record_exit(
                ticket=ticket,
                reason="BROKER_HISTORY_RECONCILIATION",
                realized_pnl=realized,
                payload={
                    "deal_tickets": [int(_broker_field(item, "ticket", 0) or 0) for item in close_deals],
                    "history_order_count": len(history_orders),
                    "exit_price": exit_price,
                    "realized_pnl_status": pnl_status,
                    "realized_pnl_source": (
                        "MT5_POSITION_DEALS" if realized is not None else None
                    ),
                    "realized_pnl_deal_count": pnl_deal_count,
                    "realized_pnl_error_type": pnl_error_type,
                },
            )
            self.store.record_position(
                {**ledger, "state": "CLOSED", "realized_pnl": realized, "exit_price": exit_price, "exit_reason": "BROKER_HISTORY_RECONCILIATION"}
            )
            terminal_state = "CLOSED"
            result = {
                "status": "RECONCILED",
                "terminal_state": terminal_state,
                "ticket": ticket,
                "realized_pnl": realized,
                "realized_pnl_status": pnl_status,
                "exit_price": exit_price,
            }
        elif rejection_records:
            self.store.record_position({**ledger, "state": "REJECTED", "realized_pnl": None, "exit_price": None, "exit_reason": "BROKER_REJECTED"})
            terminal_state = "REJECTED"
            result = {"status": "RECONCILED", "terminal_state": terminal_state, "ticket": ticket, "realized_pnl": None, "exit_price": None}
        elif filled_records or history_deals:
            self.store.record_reconciliation(
                status="RECONCILIATION_REQUIRED",
                reason="broker history indicates an unmapped fill",
                details={"ticket": ticket, "symbol": symbol},
            )
            return {"status": "RECONCILIATION_REQUIRED", "reason": "HISTORY_MAPPING_AMBIGUOUS"}
        else:
            self.store.record_position(
                {
                    **ledger,
                    "state": "RECONCILED_ABSENT",
                    "current_exposure": "NONE",
                    "performance_inclusion": False,
                    "realized_pnl": None,
                    "exit_price": None,
                    "exit_reason": "RECONCILIATION_BROKER_ABSENT",
                }
            )
            terminal_state = "BROKER_ABSENT_CONFIRMED"
            result = {
                "status": "RECONCILED",
                "terminal_state": terminal_state,
                "ticket": ticket,
                "current_exposure": "NONE",
                "performance_inclusion": False,
                "realized_pnl": None,
                "exit_price": None,
                "exit_reason": "RECONCILIATION_BROKER_ABSENT",
            }

        self.store.record_reconciliation(
            status="RECONCILED",
            reason="broker history reconciled stale ledger position",
            details={
                "ticket": ticket,
                "symbol": symbol,
                "terminal_state": terminal_state,
                "history_window_start": history_start.isoformat(),
                "history_window_end": history_end.isoformat(),
                "history_order_count": len(history_orders),
                "history_deal_count": len(history_deals),
            },
        )
        self._position_loaded = False
        self._owned_position = None
        return result

    def run_once(self) -> dict[str, Any]:
        self._ensure_run()
        market_result = self._market_preflight()
        if isinstance(market_result, dict):
            return market_result
        try:
            tick = market_result if market_result is not None else self._tick()
        except Mt5OperationBusy:
            return {"status": "DROPPED", "action": "NO_ACTION", "execution_mode": "DEMO", "broker_order_sent": False}
        if self._last_tick is not None and tick.timestamp <= self._last_tick:
            return {"status": "DROPPED", "action": "NO_ACTION", "execution_mode": "DEMO", "broker_order_sent": False}
        self._last_tick = tick.timestamp
        self._ticks += 1
        plan = None
        regime: StrategicRegimeState | None = None
        account = None
        if self.config.hft_first and self.regime_store is not None:
            regime = self.regime_store.current(tick.timestamp, self.config.symbol)
            if regime is None and self.config.no_regime_policy == "BOOTSTRAP_NEUTRAL":
                with suppress(Exception):
                    account = self._account()
                regime = self._bootstrap_regime(tick, account)
        else:
            plan = self.plan_store.current(
                tick.timestamp,
                self.config.symbol,
                require_provenance=self.config.require_plan_provenance,
            )
        if plan is not None and plan.plan_id not in self._plan_ids_recorded:
            self._persist(self.store, "record_plan", self.gateway.run_id, plan)
            self._plan_ids_recorded.add(plan.plan_id)
        if regime is not None and regime.state_id not in self._regime_ids_recorded:
            self._persist(self.store, "record_signal",
                run_id=self.gateway.run_id,
                signal_id=str(uuid.uuid4()),
                tick_timestamp=tick.timestamp,
                symbol=tick.symbol,
                action="REGIME_UPDATE",
                reason=regime.regime.value,
                plan_id=regime.state_id,
                payload={"execution_mode": "DEMO", "regime": regime.to_payload()},
            )
            self._regime_ids_recorded.add(regime.state_id)
        snapshot = self.features.update(tick, plan_created_at=None if plan is None else plan.created_at)
        current = self._open_position()
        state = PositionState.FLAT if current is None else PositionState.LONG if current["direction"] == "LONG" else PositionState.SHORT
        if self.config.hft_first and self.regime_store is not None:
            position_payload = self._position_payload(current) if current is not None else {}
            decision = self.hft.on_tick(
                regime,
                snapshot,
                position_state=state,
                entry_price=None if current is None else float(current["price_open"]),
                entry_at=None if current is None else self._position_entry_at(current),
                expected_move_points=None if current is None else position_payload.get("expected_move_points"),
                strategy_id=None if current is None else position_payload.get("strategy_id"),
                exit_state=position_payload if current is not None else None,
            )
        else:
            decision = self.fast.on_tick(
                plan,
                snapshot,
                position_state=state,
                entry_price=None if current is None else float(current["price_open"]),
                entry_at=None,
            )
        self._decision_latencies.append(float(decision.processing_ms))
        signal_id = str(uuid.uuid4())
        self._persist(self.store, "record_signal",
            run_id=self.gateway.run_id,
            signal_id=signal_id,
            tick_timestamp=tick.timestamp,
            symbol=tick.symbol,
            action=decision.action.value,
            reason=decision.reason,
            plan_id=(regime.state_id if regime is not None else None if plan is None else plan.plan_id),
            payload={"execution_mode": "DEMO", "real_money": False},
        )
        try:
            account = account or self._account()
        except Mt5OperationBusy:
            return {
                "status": "DROPPED",
                "action": "NO_ACTION",
                "reason_code": "MT5_OPERATION_BUSY",
                "execution_mode": "DEMO",
                "broker_order_sent": False,
            }
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
            plan_expired=(
                (plan is None and not self.config.hft_first)
                or (self.config.hft_first and regime is None and decision.action in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT))
            ),
            allowed_sessions=None if plan is None else plan.session_constraints,
        )
        if decision.action in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT) and not self.circuit.entry_allowed(equity):
            risk = RiskDecision(False, self.circuit.status, "DEMO entry circuit is closed", 0.0)
        if decision.action in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT) and not self._order_allowed(tick.timestamp):
            risk = RiskDecision(False, "HFT_ORDER_RATE_CIRCUIT_OPEN" if self._order_rate_circuit_open else "HFT_COOLDOWN", "HFT technical order-rate or cooldown gate", 0.0)
        self._record_risk(signal_id, risk)
        self._record_shadow_tick(tick, snapshot, decision, signal_id, risk)
        if current is not None and self.config.hft_first:
            self._persist_exit_telemetry(current, decision, tick)
        if decision.action is FastAction.EXIT and current is not None and risk.accepted:
            source = regime if regime is not None else plan
            order_started = time.perf_counter()
            exit_intent = self._exit_intent(source, current, tick, account, reason=decision.reason)
            exit_intent = replace(
                exit_intent,
                provenance={
                    "strategy_version": self.config.strategy_version,
                    "config_version": self.config.config_version,
                    "signal_id": signal_id,
                    "feature_snapshot": asdict(snapshot),
                    "risk": asdict(risk),
                    "source_position_id": str(current.get("ticket")),
                    "exit_reason": decision.reason,
                },
            )
            result = self.gateway.submit(
                exit_intent,
                position_ticket=self._broker_position_ticket(current),
                exit_reason=decision.reason,
            )
            self._order_latencies.append((time.perf_counter() - order_started) * 1000.0)
            self._record_submission(tick.timestamp)
            if result.classification in {"FILLED", "PARTIAL"}:
                if self.gateway.last_exit_realized_pnl is not None:
                    self.circuit.record_closed_trade(self.gateway.last_exit_realized_pnl)
                self._owned_position = None
                if self.shadow_store is not None:
                    self._persist(
                        self.shadow_store,
                        "record_fill",
                        self.gateway.run_id,
                        {
                            "fill_id": f"demo-shadow-{exit_intent.intent_id}",
                            "action_id": signal_id,
                            "symbol": tick.symbol,
                            "timestamp": tick.timestamp.isoformat(),
                            "action": decision.action.value,
                            "price": exit_intent.requested_price,
                            "size": result.fill_volume or exit_intent.volume,
                            "slippage_points": 0.0,
                            "latency_ms": self._order_latencies[-1],
                            "executed": False,
                        },
                    )
                    closed_position = dict(current)
                    closed_position["position_id"] = str(current["ticket"])
                    closed_position["state"] = "CLOSED"
                    closed_position["executed"] = False
                    self._persist(self.shadow_store, "record_position", self.gateway.run_id, closed_position)
            self._observe_challengers(snapshot, regime, risk_context)
            return {
                "status": "PROCESSED",
                "action": decision.action.value,
                "reason_code": decision.reason,
                "risk_accepted": risk.accepted,
                "execution_mode": "DEMO",
                "broker_order_sent": result.broker_order_sent,
                "classification": result.classification,
                "strategy_id": decision.strategy_id,
            }
        if not risk.accepted or decision.action not in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT):
            self._observe_challengers(snapshot, regime, risk_context)
            return {
                "status": "PROCESSED",
                "action": decision.action.value,
                "reason_code": decision.reason,
                "risk_accepted": risk.accepted,
                "execution_mode": "DEMO",
                "broker_order_sent": False,
                "classification": None,
                "strategy_id": decision.strategy_id,
            }
        if not self._entry_position_clear():
            self._observe_challengers(snapshot, regime, risk_context)
            return {
                "status": "PROCESSED",
                "action": decision.action.value,
                "reason_code": "RECONCILIATION_REQUIRED",
                "risk_accepted": False,
                "execution_mode": "DEMO",
                "broker_order_sent": False,
                "classification": None,
                "strategy_id": decision.strategy_id,
            }
        info = self._symbol_info()
        if info is None:
            self._observe_challengers(snapshot, regime, risk_context)
            raise RuntimeError("DEMO symbol metadata unavailable")
        source = regime if regime is not None else plan
        intent = self._entry_intent(
            source,
            tick,
            account,
            info,
            strategy_id=decision.strategy_id,
            direction="LONG" if decision.action is FastAction.ENTER_LONG else "SHORT",
            expected_move_points=decision.expected_move_points,
        )
        intent = replace(
            intent,
            provenance={
                "strategy_version": self.config.strategy_version,
                "config_version": self.config.config_version,
                "signal_id": signal_id,
                "feature_snapshot": asdict(snapshot),
                "risk": asdict(risk),
                "regime_state_id": None if regime is None else regime.state_id,
            },
        )
        self._record_submission(tick.timestamp)
        order_started = time.perf_counter()
        result = self.gateway.submit(intent)
        self._order_latencies.append((time.perf_counter() - order_started) * 1000.0)
        if result.classification in {"FILLED", "PARTIAL"} and (result.position_ticket or result.order_ticket) is not None:
            position_ticket = result.position_ticket or result.order_ticket
            self.store.record_position(
                {
                    "ticket": position_ticket,
                    "order_ticket": result.order_ticket,
                    "position_ticket": position_ticket,
                    "intent_id": intent.intent_id,
                    "symbol": tick.symbol,
                    "direction": intent.direction,
                    "volume": result.fill_volume or intent.volume,
                    "price_open": result.fill_price or intent.requested_price,
                    "stop_loss": intent.stop_loss,
                    "take_profit": intent.take_profit,
                    "strategy_id": intent.strategy_id,
                    "expected_move_points": decision.expected_move_points,
                    "exit_policy": "DYNAMIC_HFT",
                    "emergency_broker_stop": True,
                    "normal_take_profit": False,
                    "state": "OPEN",
                    "opened_at": tick.timestamp.isoformat(),
                    "provenance": dict(intent.provenance),
                }
            )
            self._owned_position = {
                "ticket": position_ticket,
                "order_ticket": result.order_ticket,
                "position_ticket": position_ticket,
                "intent_id": intent.intent_id,
                "symbol": tick.symbol,
                "direction": intent.direction,
                "volume": result.fill_volume or intent.volume,
                "price_open": result.fill_price or intent.requested_price,
                "stop_loss": intent.stop_loss,
                "take_profit": intent.take_profit,
                "strategy_id": intent.strategy_id,
                "expected_move_points": decision.expected_move_points,
                "exit_policy": "DYNAMIC_HFT",
                "emergency_broker_stop": True,
                "normal_take_profit": False,
                "state": "OPEN",
                "opened_at": tick.timestamp.isoformat(),
                "provenance": dict(intent.provenance),
            }
            if self.shadow_store is not None:
                shadow_position = dict(self._owned_position)
                shadow_position["position_id"] = str(position_ticket)
                shadow_position["state"] = "LONG" if intent.direction == "LONG" else "SHORT"
                shadow_position["executed"] = False
                self._persist(
                    self.shadow_store,
                    "record_fill",
                    self.gateway.run_id,
                    {
                        "fill_id": f"demo-shadow-{intent.intent_id}",
                        "action_id": signal_id,
                        "symbol": tick.symbol,
                        "timestamp": tick.timestamp.isoformat(),
                        "action": decision.action.value,
                        "price": intent.requested_price,
                        "size": result.fill_volume or intent.volume,
                        "slippage_points": 0.0,
                        "latency_ms": self._order_latencies[-1],
                        "executed": False,
                    },
                )
                self._persist(self.shadow_store, "record_position", self.gateway.run_id, shadow_position)
        self._observe_challengers(snapshot, regime, risk_context)
        return {
            "status": "PROCESSED",
            "action": decision.action.value,
            "risk_accepted": risk.accepted,
            "execution_mode": "DEMO",
            "broker_order_sent": result.broker_order_sent,
            "classification": result.classification,
            "order_ticket": result.order_ticket,
            "deal_ticket": result.deal_ticket,
            "strategy_id": decision.strategy_id,
        }

    @staticmethod
    def _position_payload(position: Mapping[str, Any] | None) -> dict[str, Any]:
        if position is None:
            return {}
        payload = position.get("payload_json")
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
            except (TypeError, ValueError):
                decoded = {}
            if isinstance(decoded, Mapping):
                return dict(decoded)
        return {
            key: position[key]
            for key in (
                "strategy_id",
                "expected_move_points",
                "entry_price",
                "entry_at",
                "mfe_points",
                "mae_points",
                "current_pnl_points",
                "last_mfe_at",
                "profit_protection_state",
                "trailing_level",
                "micro_reversal",
            )
            if key in position
        }

    def _persist_exit_telemetry(self, current: Mapping[str, Any], decision: Any, tick: Tick) -> None:
        """Persist bounded causal position metadata without model/private text."""

        updated = dict(current)
        updated.pop("payload_json", None)
        updated.update(self.hft.exit_state_payload())
        updated.update(
            {
                "current_pnl_points": decision.current_pnl_points,
                "mfe_points": decision.mfe_points,
                "mae_points": decision.mae_points,
                "profit_protection_state": decision.profit_protection_state,
                "trailing_level": decision.trailing_level,
                "micro_reversal": decision.micro_reversal,
                "last_telemetry_at": tick.timestamp.isoformat(),
            }
        )
        self._owned_position = updated
        self._persist(self.store, "record_position", updated)

    @staticmethod
    def _position_entry_at(position: Mapping[str, Any]) -> datetime | None:
        payload = DemoHftRuntime._position_payload(position)
        value = position.get("created_at") or position.get("opened_at") or payload.get("entry_at") or payload.get("opened_at")
        if isinstance(value, datetime):
            return utc(value, "position_entry_at")
        if isinstance(value, str):
            try:
                return utc(datetime.fromisoformat(value.replace("Z", "+00:00")), "position_entry_at")
            except ValueError:
                return None
        return None

    @staticmethod
    def _broker_position_ticket(position: Mapping[str, Any]) -> int:
        value = position.get("position_ticket")
        if value is None:
            payload = position.get("payload_json")
            if isinstance(payload, str):
                try:
                    decoded = json.loads(payload)
                except (TypeError, ValueError):
                    decoded = {}
                if isinstance(decoded, Mapping):
                    value = decoded.get("position_ticket")
        return int(position.get("ticket") if value is None else value)

    def run(self, *, max_ticks: int | None = None, stop_event: Any | None = None) -> dict[str, Any]:
        limit = self.config.max_ticks if max_ticks is None else max_ticks
        self._ensure_run()
        started = time.perf_counter()
        status = "STOPPED"
        observations = 0
        mt5_operation_busy_drops = 0
        persistence_dropped = 0
        persistence_error: str | None = None
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
        self._persistence = AsyncPersistenceQueue(capacity=512)
        self._persistence.start()
        try:
            while (limit == 0 or observations < limit) and not (stop_event is not None and stop_event.is_set()):
                try:
                    observation = self.run_once()
                except Mt5OperationBusy:
                    # The shared MT5 gate is intentionally non-blocking. A
                    # concurrent read-only watcher operation can therefore
                    # drop this tick before any MT5 call/order_send occurs.
                    # Keep the worker and lease alive; the next poll gets a
                    # fresh tick and reevaluates the position/strategy.
                    mt5_operation_busy_drops += 1
                    observation = {
                        "status": "DROPPED",
                        "action": "NO_ACTION",
                        "reason_code": "MT5_OPERATION_BUSY",
                        "execution_mode": "DEMO",
                        "broker_order_sent": False,
                    }
                observations += 1
                if lease_token is not None:
                    self.lease_store.heartbeat_lease(lease_token, datetime.now(timezone.utc))
                if limit == 0:
                    time.sleep(self._poll_delay_seconds(observation))
            status = "COMPLETED"
        finally:
            if self._persistence is not None:
                with suppress(Exception):
                    self._persistence.flush()
                persistence_dropped = self._persistence.dropped
                persistence_error = self._persistence.error
                with suppress(Exception):
                    self._persistence.stop()
                self._persistence = None
            self.store.close_run(self.gateway.run_id, status=status)
            if self.shadow_store is not None:
                with suppress(Exception):
                    self.shadow_store.close_run(self.gateway.run_id, status=status)
            if lease_token is not None:
                self.lease_store.release_lease(lease_token)
        ordered = sorted(self._decision_latencies)
        def percentile(fraction: float) -> float | None:
            if not ordered:
                return None
            index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
            return ordered[index]
        order_latencies = sorted(self._order_latencies)
        return {
            "run_id": self.gateway.run_id,
            "execution_mode": "DEMO",
            "market_status": None if self.market_status is None else self.market_status.value,
            "market_reason": None if self.market_reason is None else self.market_reason.value,
            "observed_ticks": observations,
            "mt5_operation_busy_drops": mt5_operation_busy_drops,
            "runtime_seconds": max(0.0, time.perf_counter() - started),
            "broker_order_sent": self.store.snapshot()["orders"],
            "local_decision_p50_ms": percentile(0.50),
            "local_decision_p95_ms": percentile(0.95),
            "local_decision_p99_ms": percentile(0.99),
            "local_decision_max_ms": max(ordered) if ordered else None,
            "order_submission_p50_ms": _percentile(order_latencies, 0.50),
            "order_submission_p95_ms": _percentile(order_latencies, 0.95),
            "order_submission_p99_ms": _percentile(order_latencies, 0.99),
            "order_submission_max_ms": max(order_latencies) if order_latencies else None,
            "latency_sample_capacity": 4096,
            "persistence_dropped": persistence_dropped,
            "persistence_error": persistence_error,
            "challenger_observation_failures": self.challenger_observation_failures,
            "executed": False,
            "real_money": False,
        }


__all__ = ["DemoHftRuntime", "DemoRuntimeConfig"]
