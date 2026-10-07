"""Causal deterministic HFT strategies and signal arbitration."""

from __future__ import annotations

from dataclasses import dataclass

from .features import TickFeatures
from .models import FastAction
from .regime import StrategicRegimeState


@dataclass(frozen=True, slots=True)
class StrategySignal:
    strategy_id: str
    action: FastAction
    strength: float
    expected_move_points: float
    reason: str

    def __post_init__(self) -> None:
        if self.action not in {FastAction.ENTER_LONG, FastAction.ENTER_SHORT}:
            raise ValueError("strategy signals must be directional entries")
        if not self.strategy_id.strip() or self.strength < 0 or self.expected_move_points < 0:
            raise ValueError("strategy signal values are invalid")


class MomentumContinuationStrategy:
    strategy_id = "momentum_continuation"

    def __init__(self, *, minimum_momentum_points: float = 2.0, minimum_persistence: float = 0.6, minimum_ticks: int = 3) -> None:
        if minimum_momentum_points <= 0 or not 0 <= minimum_persistence <= 1 or minimum_ticks < 2:
            raise ValueError("invalid momentum strategy bounds")
        self.minimum_momentum_points = float(minimum_momentum_points)
        self.minimum_persistence = float(minimum_persistence)
        self.minimum_ticks = int(minimum_ticks)

    def evaluate(self, features: TickFeatures) -> StrategySignal | None:
        if features.tick_count < self.minimum_ticks:
            return None
        move = features.momentum / features.point
        magnitude = abs(move)
        if magnitude < self.minimum_momentum_points or features.direction_persistence < self.minimum_persistence:
            return None
        action = FastAction.ENTER_LONG if move > 0 else FastAction.ENTER_SHORT
        strength = min(1.0, features.direction_persistence * magnitude / self.minimum_momentum_points)
        return StrategySignal(self.strategy_id, action, strength, magnitude, "MOMENTUM_CONTINUATION")


class RangeRejectionStrategy:
    strategy_id = "range_rejection"

    def __init__(self, *, edge_fraction: float = 0.2, minimum_range_points: float = 2.0, minimum_ticks: int = 4) -> None:
        if not 0 < edge_fraction < 0.5 or minimum_range_points <= 0 or minimum_ticks < 3:
            raise ValueError("invalid range strategy bounds")
        self.edge_fraction = float(edge_fraction)
        self.minimum_range_points = float(minimum_range_points)
        self.minimum_ticks = int(minimum_ticks)

    def evaluate(self, features: TickFeatures) -> StrategySignal | None:
        range_points = features.rolling_range / features.point
        if features.tick_count < self.minimum_ticks or range_points < self.minimum_range_points:
            return None
        position = features.range_position
        # The target for a range rejection is the range midpoint, not the
        # entire observed high-to-low width.  This value also drives the
        # transaction-cost gate and dynamic exit sizing, so using full width
        # would overstate the move available from the current edge.
        expected_move_points = abs(0.5 - position) * range_points
        if position <= self.edge_fraction and features.return_1 > 0:
            return StrategySignal(self.strategy_id, FastAction.ENTER_LONG, min(1.0, 1.0 - position), expected_move_points, "RANGE_REJECTION_LOW")
        if position >= 1.0 - self.edge_fraction and features.return_1 < 0:
            return StrategySignal(self.strategy_id, FastAction.ENTER_SHORT, min(1.0, position), expected_move_points, "RANGE_REJECTION_HIGH")
        return None


class SignalArbiter:
    """Choose one non-competing candidate without model/network calls."""

    _priority = {"momentum_continuation": 20, "range_rejection": 10, "micro_breakout": 5, "mean_reversion": 5}

    def __init__(self, *, cost_safety_margin_points: float = 1.0) -> None:
        if cost_safety_margin_points < 0:
            raise ValueError("cost_safety_margin_points must be non-negative")
        self.cost_safety_margin_points = float(cost_safety_margin_points)

    def select(
        self,
        signals: tuple[StrategySignal, ...],
        state: StrategicRegimeState,
        *,
        spread_points: float,
    ) -> tuple[StrategySignal | None, str]:
        allowed: list[StrategySignal] = []
        saw_direction_rejection = False
        saw_cost_rejection = False
        for signal in signals:
            if not state.permits_strategy(signal.strategy_id):
                continue
            if not state.permits_direction(signal.action):
                saw_direction_rejection = True
                continue
            if signal.expected_move_points <= float(spread_points) + self.cost_safety_margin_points:
                saw_cost_rejection = True
                continue
            allowed.append(signal)
        if not allowed:
            if saw_direction_rejection:
                return None, "DIRECTION_POLICY"
            if saw_cost_rejection:
                return None, "TRANSACTION_COST"
            return None, "NO_VALID_SIGNAL"
        allowed.sort(key=lambda item: (-self._priority.get(item.strategy_id, 0), -item.strength, item.strategy_id))
        return allowed[0], "SELECTED"


__all__ = [
    "MomentumContinuationStrategy",
    "RangeRejectionStrategy",
    "SignalArbiter",
    "StrategySignal",
]
