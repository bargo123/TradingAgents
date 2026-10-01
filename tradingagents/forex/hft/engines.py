"""Deterministic entry and exit decisions; this module never calls an LLM."""

from __future__ import annotations

import time
from datetime import datetime

from .features import TickFeatures
from .models import FastAction, FastDecision, PositionState, StrategicExecutionPlan
from .regime import StrategicRegimeState
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


class HftExecutionEngine:
    """Deterministic strategy/arbitration engine used by the DEMO hot path."""

    def __init__(
        self,
        *,
        momentum: MomentumContinuationStrategy | None = None,
        range_rejection: RangeRejectionStrategy | None = None,
        arbiter: SignalArbiter | None = None,
        stop_distance_points: float = 20.0,
        take_profit_distance_points: float = 30.0,
        time_stop_seconds: int = 30,
        max_exit_spread_points: float = 40.0,
        max_exit_spread_expansion_points: float = 10.0,
        max_exit_volatility: float = 0.005,
    ) -> None:
        if (
            stop_distance_points <= 0
            or take_profit_distance_points <= 0
            or time_stop_seconds <= 0
            or max_exit_spread_points <= 0
            or max_exit_spread_expansion_points < 0
            or max_exit_volatility <= 0
        ):
            raise ValueError("HFT exit bounds must be positive")
        self.momentum = momentum or MomentumContinuationStrategy()
        self.range_rejection = range_rejection or RangeRejectionStrategy()
        self.arbiter = arbiter or SignalArbiter()
        self.stop_distance_points = float(stop_distance_points)
        self.take_profit_distance_points = float(take_profit_distance_points)
        self.time_stop_seconds = int(time_stop_seconds)
        self.max_exit_spread_points = float(max_exit_spread_points)
        self.max_exit_spread_expansion_points = float(max_exit_spread_expansion_points)
        self.max_exit_volatility = float(max_exit_volatility)

    def on_tick(
        self,
        state: StrategicRegimeState | None,
        features,
        *,
        position_state: PositionState = PositionState.FLAT,
        entry_price: float | None = None,
        entry_at=None,
    ) -> FastDecision:
        started = time.perf_counter()
        if position_state in (PositionState.LONG, PositionState.SHORT):
            reason = self._exit_reason(state, features, position_state, entry_price, entry_at)
            if reason is not None:
                return self._decision(FastAction.EXIT, features, reason, started)
            return self._decision(FastAction.NO_ACTION, features, "POSITION_HELD", started)
        if position_state is not PositionState.FLAT:
            return self._decision(FastAction.NO_ACTION, features, "POSITION_TRANSITION_PENDING", started)
        if state is None or not state.is_active(features.timestamp):
            return self._decision(FastAction.NO_ACTION, features, "NO_VALID_STRATEGIC_STATE", started)
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
        )

    def _exit_reason(self, state, features, position_state, entry_price, entry_at):
        if entry_price is None or entry_price <= 0:
            return "POSITION_PRICE_UNAVAILABLE"
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
        stop = self.stop_distance_points * features.point
        target = self.take_profit_distance_points * features.point
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
        if entry_at is not None and (features.timestamp - entry_at).total_seconds() >= self.time_stop_seconds:
            return "TIME_STOP"
        return None

    @staticmethod
    def _decision(action, features, reason, started, score=0.0, strategy_id=None):
        return FastDecision(
            action=action,
            symbol=features.symbol,
            timestamp=features.timestamp,
            reason=reason,
            score=score,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            strategy_id=strategy_id,
        )


__all__ = ["FastExecutionEngine", "HftExecutionEngine"]
