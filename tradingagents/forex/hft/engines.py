"""Deterministic entry and exit decisions; this module never calls an LLM."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from .features import TickFeatures
from .models import FastAction, FastDecision, PositionState, StrategicExecutionPlan, utc
from .regime import Regime, StrategicRegimeState
from .strategies import MomentumContinuationStrategy, RangeRejectionStrategy, SignalArbiter


class FastExecutionEngine:
    def on_tick(
        self,
        plan: StrategicExecutionPlan | None,
        features: TickFeatures,
        *,
        position_state: PositionState = PositionState.FLAT,
        entry_price: float | None = None,
        entry_at: datetime | None = None,
    ) -> FastDecision:
        started = time.perf_counter()
        if plan is None or not plan.is_active(features.timestamp) or plan.symbol != features.symbol:
            return self._decision(FastAction.INVALIDATE, features, "PLAN_EXPIRED_OR_MISMATCH", started)
        if position_state in (PositionState.LONG, PositionState.SHORT):
            exit_reason = self._exit_reason(plan, features, position_state, entry_price, entry_at)
            if exit_reason is not None:
                return self._decision(FastAction.EXIT, features, exit_reason, started)
            return self._decision(FastAction.NO_ACTION, features, "POSITION_HELD", started)
        if position_state is not PositionState.FLAT:
            return self._decision(FastAction.NO_ACTION, features, "POSITION_TRANSITION_PENDING", started)
        if plan.primary_direction is plan.primary_direction.NONE:
            return self._decision(FastAction.NO_ACTION, features, "STRATEGIC_DIRECTION_NONE", started)
        constraints = plan.entry_constraints
        if plan.primary_direction not in (plan.primary_direction.LONG, plan.primary_direction.SHORT, plan.primary_direction.BOTH):
            return self._decision(FastAction.NO_ACTION, features, "DIRECTION_DISABLED", started)
        if features.spread_points > constraints.max_spread_points:
            return self._decision(FastAction.NO_ACTION, features, "SPREAD_TOO_WIDE", started)
        if not constraints.minimum_volatility <= features.volatility <= constraints.maximum_volatility:
            return self._decision(FastAction.NO_ACTION, features, "VOLATILITY_OUT_OF_BOUNDS", started)
        minimum = constraints.minimum_momentum
        long_confirmed = features.momentum >= minimum and features.direction_persistence >= constraints.minimum_confirmation
        short_confirmed = features.momentum <= -minimum and features.direction_persistence >= constraints.minimum_confirmation
        if plan.primary_direction in (plan.primary_direction.LONG, plan.primary_direction.BOTH) and long_confirmed:
            return self._decision(FastAction.ENTER_LONG, features, "LONG_CONFIRMATION", started, features.direction_persistence)
        if plan.primary_direction in (plan.primary_direction.SHORT, plan.primary_direction.BOTH) and short_confirmed:
            return self._decision(FastAction.ENTER_SHORT, features, "SHORT_CONFIRMATION", started, features.direction_persistence)
        return self._decision(FastAction.NO_ACTION, features, "ENTRY_CONFIRMATION_PENDING", started)

    @staticmethod
    def _exit_reason(plan, features, position_state, entry_price, entry_at):
        if entry_price is None or entry_price <= 0:
            return "POSITION_PRICE_UNAVAILABLE"
        point = features.point
        stop = plan.stop_policy.stop_distance_points * point
        target = plan.stop_policy.take_profit_distance_points * point
        if position_state is PositionState.LONG:
            if features.bid <= entry_price - stop:
                return "STOP_LOSS"
            if features.bid >= entry_price + target:
                return "TAKE_PROFIT"
            if features.momentum < 0 and features.direction_persistence >= 0.5:
                return "MOMENTUM_REVERSAL"
        else:
            if features.ask >= entry_price + stop:
                return "STOP_LOSS"
            if features.ask <= entry_price - target:
                return "TAKE_PROFIT"
            if features.momentum > 0 and features.direction_persistence >= 0.5:
                return "MOMENTUM_REVERSAL"
        if entry_at is not None and (features.timestamp - entry_at).total_seconds() >= plan.stop_policy.time_stop_seconds:
            return "TIME_STOP"
        return None

    @staticmethod
    def _decision(action, features, reason, started, score=0.0):
        return FastDecision(
            action=action,
            symbol=features.symbol,
            timestamp=features.timestamp,
            reason=reason,
            score=score,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )


@dataclass(frozen=True, slots=True)
class DynamicExitProfile:
    """Small, strategy-specific causal policy for managing an open position."""

    name: str
    micro_reversal_mfe_fraction: float
    micro_reversal_giveback_fraction: float
    micro_reversal_persistence: float
    protection_arm_fraction: float
    trailing_activation_fraction: float
    giveback_fraction: float
    trailing_distance_fraction: float
    no_progress_floor_seconds: float
    no_progress_expected_seconds_per_point: float
    no_progress_cap_seconds: float
    max_duration_seconds: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("exit profile name must be non-empty")
        fractions = (
            "micro_reversal_mfe_fraction",
            "micro_reversal_giveback_fraction",
            "protection_arm_fraction",
            "trailing_activation_fraction",
            "giveback_fraction",
            "trailing_distance_fraction",
        )
        for field in fractions:
            value = float(getattr(self, field))
            if value <= 0:
                raise ValueError(f"{field} must be positive")
        if not 0 < float(self.micro_reversal_persistence) <= 1:
            raise ValueError("micro_reversal_persistence must be in (0, 1]")
        for field in (
            "no_progress_floor_seconds",
            "no_progress_expected_seconds_per_point",
            "no_progress_cap_seconds",
            "max_duration_seconds",
        ):
            value = float(getattr(self, field))
            if value <= 0:
                raise ValueError(f"{field} must be positive")
        if self.no_progress_cap_seconds < self.no_progress_floor_seconds:
            raise ValueError("no_progress_cap_seconds must be >= no_progress_floor_seconds")


RANGE_EXIT_PROFILE = DynamicExitProfile(
    name="range_rejection",
    micro_reversal_mfe_fraction=0.25,
    micro_reversal_giveback_fraction=0.10,
    micro_reversal_persistence=0.20,
    protection_arm_fraction=0.25,
    trailing_activation_fraction=0.75,
    giveback_fraction=0.35,
    trailing_distance_fraction=0.50,
    no_progress_floor_seconds=6.0,
    no_progress_expected_seconds_per_point=1.0,
    no_progress_cap_seconds=10.0,
    max_duration_seconds=30.0,
)

MOMENTUM_EXIT_PROFILE = DynamicExitProfile(
    name="momentum_continuation",
    micro_reversal_mfe_fraction=0.50,
    micro_reversal_giveback_fraction=0.15,
    micro_reversal_persistence=0.35,
    protection_arm_fraction=0.50,
    trailing_activation_fraction=1.00,
    giveback_fraction=0.50,
    trailing_distance_fraction=0.50,
    no_progress_floor_seconds=12.0,
    no_progress_expected_seconds_per_point=0.50,
    no_progress_cap_seconds=20.0,
    max_duration_seconds=45.0,
)


@dataclass(slots=True)
class _ExitState:
    entry_price: float
    entry_at: datetime
    expected_move_points: float
    strategy_id: str
    current_pnl_points: float = 0.0
    mfe_points: float = 0.0
    mae_points: float = 0.0
    last_mfe_at: datetime | None = None
    profit_protection_state: str = "UNARMED"
    trailing_level: float | None = None
    micro_reversal: bool = False

    def payload(self) -> dict[str, object]:
        return {
            "entry_price": self.entry_price,
            "entry_at": self.entry_at.isoformat().replace("+00:00", "Z"),
            "expected_move_points": self.expected_move_points,
            "strategy_id": self.strategy_id,
            "current_pnl_points": self.current_pnl_points,
            "mfe_points": self.mfe_points,
            "mae_points": self.mae_points,
            "last_mfe_at": None if self.last_mfe_at is None else self.last_mfe_at.isoformat().replace("+00:00", "Z"),
            "profit_protection_state": self.profit_protection_state,
            "trailing_level": self.trailing_level,
            "micro_reversal": self.micro_reversal,
        }


class HftExecutionEngine:
    """Deterministic strategy/arbitration engine used by the DEMO hot path.

    HFT positions are managed from causal tick state.  The legacy stop/target
    constructor arguments remain accepted for compatibility, but they are not
    normal strategy exits; only ``emergency_stop_distance_points`` can produce
    the explicit catastrophic broker-stop reason.
    """

    def __init__(
        self,
        *,
        momentum: MomentumContinuationStrategy | None = None,
        range_rejection: RangeRejectionStrategy | None = None,
        arbiter: SignalArbiter | None = None,
        stop_distance_points: float = 20.0,
        take_profit_distance_points: float = 30.0,
        time_stop_seconds: int = 30,
        emergency_stop_distance_points: float = 100.0,
        hard_max_duration_seconds: float | None = None,
        max_exit_spread_points: float = 40.0,
        max_exit_spread_expansion_points: float = 10.0,
        max_exit_volatility: float = 0.005,
        range_exit_profile: DynamicExitProfile = RANGE_EXIT_PROFILE,
        momentum_exit_profile: DynamicExitProfile = MOMENTUM_EXIT_PROFILE,
    ) -> None:
        if (
            stop_distance_points <= 0
            or take_profit_distance_points <= 0
            or time_stop_seconds <= 0
            or emergency_stop_distance_points <= 0
            or max_exit_spread_points <= 0
            or max_exit_spread_expansion_points < 0
            or max_exit_volatility <= 0
        ):
            raise ValueError("HFT exit bounds must be positive")
        if not isinstance(range_exit_profile, DynamicExitProfile) or not isinstance(momentum_exit_profile, DynamicExitProfile):
            raise TypeError("exit profiles must be DynamicExitProfile instances")
        hard_max = max(60.0, float(time_stop_seconds)) if hard_max_duration_seconds is None else float(hard_max_duration_seconds)
        if hard_max <= 0:
            raise ValueError("hard_max_duration_seconds must be positive")
        self.momentum = momentum or MomentumContinuationStrategy()
        self.range_rejection = range_rejection or RangeRejectionStrategy()
        self.arbiter = arbiter or SignalArbiter()
        self.stop_distance_points = float(stop_distance_points)
        self.take_profit_distance_points = float(take_profit_distance_points)
        self.time_stop_seconds = int(time_stop_seconds)
        self.emergency_stop_distance_points = float(emergency_stop_distance_points)
        self.hard_max_duration_seconds = hard_max
        self.max_exit_spread_points = float(max_exit_spread_points)
        self.max_exit_spread_expansion_points = float(max_exit_spread_expansion_points)
        self.max_exit_volatility = float(max_exit_volatility)
        self.range_exit_profile = range_exit_profile
        self.momentum_exit_profile = momentum_exit_profile
        self._exit_state: _ExitState | None = None

    def on_tick(
        self,
        state: StrategicRegimeState | None,
        features,
        *,
        position_state: PositionState = PositionState.FLAT,
        entry_price: float | None = None,
        entry_at=None,
        expected_move_points: float | None = None,
        strategy_id: str | None = None,
        exit_state: Mapping[str, object] | None = None,
    ) -> FastDecision:
        started = time.perf_counter()
        if position_state in (PositionState.LONG, PositionState.SHORT):
            tracked = self._ensure_exit_state(
                features,
                entry_price,
                entry_at,
                expected_move_points,
                strategy_id,
                exit_state,
            )
            reason = self._exit_reason(state, features, position_state, tracked)
            if reason is not None:
                return self._decision(
                    FastAction.EXIT,
                    features,
                    reason,
                    started,
                    strategy_id=tracked.strategy_id,
                    state=tracked,
                )
            return self._decision(
                FastAction.NO_ACTION,
                features,
                "POSITION_HELD",
                started,
                strategy_id=tracked.strategy_id,
                state=tracked,
            )
        if position_state is not PositionState.FLAT:
            return self._decision(FastAction.NO_ACTION, features, "POSITION_TRANSITION_PENDING", started)
        self._exit_state = None
        if state is None or not state.is_active(features.timestamp):
            return self._decision(FastAction.NO_ACTION, features, "NO_VALID_STRATEGIC_STATE", started)
        if state.regime is Regime.HIGH_UNCERTAINTY:
            return self._decision(FastAction.NO_ACTION, features, "STRATEGIC_HIGH_UNCERTAINTY", started)
        if state.direction_policy.value == "PAUSE":
            return self._decision(FastAction.NO_ACTION, features, "STRATEGIC_PAUSE", started)
        signals = tuple(
            signal
            for signal in (
                self.momentum.evaluate(features),
                self.range_rejection.evaluate(features),
            )
            if signal is not None
        )
        selected, reason = self.arbiter.select(signals, state, spread_points=features.spread_points)
        if selected is None:
            return self._decision(FastAction.NO_ACTION, features, reason, started)
        return self._decision(
            selected.action,
            features,
            selected.reason,
            started,
            selected.strength,
            selected.strategy_id,
            expected_move_points=selected.expected_move_points,
        )

    @staticmethod
    def _profile(strategy_id: str | None, range_profile: DynamicExitProfile, momentum_profile: DynamicExitProfile) -> DynamicExitProfile:
        normalized = str(strategy_id or "").strip().lower()
        return momentum_profile if "momentum" in normalized else range_profile

    def _ensure_exit_state(
        self,
        features,
        entry_price: float | None,
        entry_at,
        expected_move_points: float | None,
        strategy_id: str | None,
        payload: Mapping[str, object] | None,
    ) -> _ExitState:
        if entry_price is None or entry_price <= 0:
            raise ValueError("position entry price must be positive")
        timestamp = utc(entry_at, "entry_at") if isinstance(entry_at, datetime) else features.timestamp
        normalized_strategy = str(strategy_id or (payload or {}).get("strategy_id") or "range_rejection")
        expected = expected_move_points
        if expected is None and payload is not None:
            expected = payload.get("expected_move_points")
        try:
            expected_value = max(float(expected or 0.0), 1.0)
        except (TypeError, ValueError):
            expected_value = 1.0
        current_key = (float(entry_price), timestamp, normalized_strategy)
        existing_key = None if self._exit_state is None else (
            self._exit_state.entry_price,
            self._exit_state.entry_at,
            self._exit_state.strategy_id,
        )
        if existing_key != current_key:
            self._exit_state = _ExitState(
                entry_price=float(entry_price),
                entry_at=timestamp,
                expected_move_points=expected_value,
                strategy_id=normalized_strategy,
            )
            if payload is not None:
                self._restore_exit_state(self._exit_state, payload)
        return self._exit_state

    @staticmethod
    def _restore_exit_state(state: _ExitState, payload: Mapping[str, object]) -> None:
        for field, minimum in (("mfe_points", 0.0), ("mae_points", None), ("current_pnl_points", None)):
            value = payload.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if field == "mfe_points" and numeric < minimum:
                    continue
                if field == "mae_points" and numeric > 0:
                    continue
                setattr(state, field, numeric)
        protection = str(payload.get("profit_protection_state", "UNARMED")).upper()
        if protection in {"UNARMED", "PROFIT_PROTECTION_ARMED", "TRAILING", "EXIT_PENDING"}:
            state.profit_protection_state = protection
        trailing = payload.get("trailing_level")
        if isinstance(trailing, (int, float)) and not isinstance(trailing, bool) and float(trailing) > 0:
            state.trailing_level = float(trailing)
        state.micro_reversal = payload.get("micro_reversal") is True
        last_mfe = payload.get("last_mfe_at")
        if isinstance(last_mfe, str):
            try:
                state.last_mfe_at = utc(datetime.fromisoformat(last_mfe.replace("Z", "+00:00")), "last_mfe_at")
            except ValueError:
                state.last_mfe_at = None

    def exit_state_payload(self) -> dict[str, object]:
        return {} if self._exit_state is None else self._exit_state.payload()

    def _exit_reason(self, state, features, position_state, tracked: _ExitState):
        mark = features.bid if position_state is PositionState.LONG else features.ask
        if mark <= 0:
            return "POSITION_PRICE_UNAVAILABLE"
        tracked.current_pnl_points = (
            (mark - tracked.entry_price) / features.point
            if position_state is PositionState.LONG
            else (tracked.entry_price - mark) / features.point
        )
        previous_mfe = tracked.mfe_points
        if tracked.current_pnl_points > previous_mfe:
            tracked.mfe_points = tracked.current_pnl_points
            tracked.last_mfe_at = features.timestamp
        tracked.mae_points = min(tracked.mae_points, tracked.current_pnl_points)
        if tracked.last_mfe_at is None:
            tracked.last_mfe_at = tracked.entry_at
        if tracked.current_pnl_points <= -self.emergency_stop_distance_points:
            tracked.profit_protection_state = "EXIT_PENDING"
            return "EMERGENCY_BROKER_STOP"
        if state is None or not state.is_active(features.timestamp):
            return "STRATEGIC_REGIME_INVALIDATED"
        if position_state is PositionState.LONG and not state.permits_direction(FastAction.ENTER_LONG):
            return "STRATEGIC_REGIME_INVALIDATED"
        if position_state is PositionState.SHORT and not state.permits_direction(FastAction.ENTER_SHORT):
            return "STRATEGIC_REGIME_INVALIDATED"
        spread_expansion_points = features.spread_expansion / features.point
        if (
            features.spread_points > self.max_exit_spread_points
            or spread_expansion_points > self.max_exit_spread_expansion_points
        ):
            return "SPREAD_BLOWOUT"
        if features.volatility > self.max_exit_volatility:
            return "VOLATILITY_BLOWOUT"
        profile = self._profile(tracked.strategy_id, self.range_exit_profile, self.momentum_exit_profile)
        expected = tracked.expected_move_points
        arm_threshold = max(1.0, expected * profile.protection_arm_fraction)
        micro_threshold = max(1.0, expected * profile.micro_reversal_mfe_fraction)
        reversal = (
            (position_state is PositionState.LONG and features.momentum < 0)
            or (position_state is PositionState.SHORT and features.momentum > 0)
        ) and features.direction_persistence >= profile.micro_reversal_persistence
        if tracked.mfe_points >= arm_threshold and tracked.profit_protection_state == "UNARMED":
            tracked.profit_protection_state = "PROFIT_PROTECTION_ARMED"
        trailing_threshold = max(arm_threshold, expected * profile.trailing_activation_fraction)
        if tracked.mfe_points >= trailing_threshold:
            tracked.profit_protection_state = "TRAILING"
            giveback_points = max(
                0.5,
                expected * profile.giveback_fraction,
                features.spread_points * 0.5,
            )
            distance_points = max(0.5, expected * profile.trailing_distance_fraction)
            tracked.trailing_level = (
                tracked.entry_price + (tracked.mfe_points - distance_points) * features.point
                if position_state is PositionState.LONG
                else tracked.entry_price - (tracked.mfe_points - distance_points) * features.point
            )
            if tracked.current_pnl_points <= tracked.mfe_points - giveback_points:
                tracked.profit_protection_state = "EXIT_PENDING"
                return "MFE_GIVEBACK"
        if reversal and tracked.mfe_points >= micro_threshold:
            micro_giveback = max(0.5, expected * profile.micro_reversal_giveback_fraction, features.spread_points * 0.25)
            if tracked.current_pnl_points <= tracked.mfe_points - micro_giveback:
                tracked.micro_reversal = True
                tracked.profit_protection_state = "EXIT_PENDING"
                return "MICRO_REVERSAL"
        if reversal and tracked.current_pnl_points < -max(1.0, expected * 0.25) and tracked.mfe_points < arm_threshold:
            tracked.micro_reversal = True
            tracked.profit_protection_state = "EXIT_PENDING"
            return "STRATEGY_INVALIDATION"
        if reversal and tracked.mfe_points < micro_threshold:
            return "MOMENTUM_REVERSAL"
        elapsed = max(0.0, (features.timestamp - tracked.entry_at).total_seconds())
        last_mfe_elapsed = max(0.0, (features.timestamp - (tracked.last_mfe_at or tracked.entry_at)).total_seconds())
        no_progress_limit = min(
            profile.no_progress_cap_seconds,
            max(profile.no_progress_floor_seconds, expected * profile.no_progress_expected_seconds_per_point),
        )
        if (
            elapsed >= no_progress_limit
            and last_mfe_elapsed >= no_progress_limit
            and tracked.mfe_points < max(1.0, expected * profile.protection_arm_fraction)
        ):
            return "NO_PROGRESS"
        if elapsed >= min(self.hard_max_duration_seconds, profile.max_duration_seconds):
            return "MAX_DURATION"
        return None

    @staticmethod
    def _decision(action, features, reason, started, score=0.0, strategy_id=None, *, expected_move_points=0.0, state=None):
        return FastDecision(
            action=action,
            symbol=features.symbol,
            timestamp=features.timestamp,
            reason=reason,
            score=score,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            strategy_id=strategy_id,
            expected_move_points=expected_move_points if state is None else state.expected_move_points,
            current_pnl_points=0.0 if state is None else state.current_pnl_points,
            mfe_points=0.0 if state is None else state.mfe_points,
            mae_points=0.0 if state is None else state.mae_points,
            profit_protection_state="UNARMED" if state is None else state.profit_protection_state,
            trailing_level=None if state is None else state.trailing_level,
            micro_reversal=False if state is None else state.micro_reversal,
            time_since_entry_seconds=0.0 if state is None else max(0.0, (features.timestamp - state.entry_at).total_seconds()),
            time_since_last_mfe_seconds=0.0 if state is None else max(0.0, (features.timestamp - (state.last_mfe_at or state.entry_at)).total_seconds()),
        )


__all__ = [
    "DynamicExitProfile",
    "FastExecutionEngine",
    "HftExecutionEngine",
    "MOMENTUM_EXIT_PROFILE",
    "RANGE_EXIT_PROFILE",
]
