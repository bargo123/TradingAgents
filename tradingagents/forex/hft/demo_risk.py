"""Bounded DEMO sizing, stop validation, and entry circuit breakers."""

from __future__ import annotations

import math
from datetime import datetime
from decimal import ROUND_DOWN, Decimal


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def normalize_volume(
    requested: float,
    *,
    minimum: float,
    maximum: float,
    step: float,
    cap: float,
) -> float:
    requested = _positive(requested, "requested volume")
    minimum = _positive(minimum, "volume minimum")
    maximum = _positive(maximum, "volume maximum")
    step = _positive(step, "volume step")
    cap = _positive(cap, "volume cap")
    upper = min(maximum, cap)
    if upper < minimum:
        raise ValueError("volume cap is below broker minimum")
    bounded = min(requested, upper)
    units = (Decimal(str(bounded)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
    normalized = float(units * Decimal(str(step)))
    if normalized < minimum:
        minimum_units = (Decimal(str(minimum)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
        normalized = float(minimum_units * Decimal(str(step)))
        if normalized < minimum:
            normalized += step
    if normalized > upper + 1e-12:
        raise ValueError("requested volume cannot satisfy broker step and cap")
    return normalized


def validate_stop_levels(
    direction: str,
    *,
    entry: float,
    stop_loss: float,
    take_profit: float,
    point: float,
    minimum_distance_points: float,
) -> None:
    direction = str(direction).upper()
    entry = _positive(entry, "entry")
    stop_loss = _positive(stop_loss, "stop_loss")
    take_profit = _positive(take_profit, "take_profit")
    point = _positive(point, "point")
    minimum_distance_points = float(minimum_distance_points)
    if not math.isfinite(minimum_distance_points) or minimum_distance_points < 0:
        raise ValueError("minimum_distance_points must be non-negative")
    distance = minimum_distance_points * point
    if direction == "LONG":
        if stop_loss >= entry or entry - stop_loss < distance:
            raise ValueError("stop_loss is invalid for LONG")
        if take_profit <= entry or take_profit - entry < distance:
            raise ValueError("take_profit is invalid for LONG")
    elif direction == "SHORT":
        if stop_loss <= entry or stop_loss - entry < distance:
            raise ValueError("stop_loss is invalid for SHORT")
        if take_profit >= entry or entry - take_profit < distance:
            raise ValueError("take_profit is invalid for SHORT")
    else:
        raise ValueError("direction must be LONG or SHORT")


class DemoCircuitBreaker:
    """In-memory bounded entry circuits; persistence is owned by DemoExecutionStore."""

    def __init__(self, *, daily_loss_limit: float = 0.02, max_consecutive_losses: int = 3) -> None:
        self.daily_loss_limit = _positive(daily_loss_limit, "daily_loss_limit")
        if isinstance(max_consecutive_losses, bool) or not isinstance(max_consecutive_losses, int) or max_consecutive_losses <= 0:
            raise ValueError("max_consecutive_losses must be positive")
        self.max_consecutive_losses = max_consecutive_losses
        self.daily_start_equity: float | None = None
        self.daily_loss_fraction = 0.0
        self.consecutive_losses = 0
        self.status = "READY"
        self.day: datetime | None = None

    def start_day(self, equity: float, at: datetime) -> None:
        self.daily_start_equity = _positive(equity, "daily start equity")
        self.day = at
        self.daily_loss_fraction = 0.0
        self.consecutive_losses = 0
        self.status = "READY"

    def reset_day(self, equity: float, at: datetime) -> None:
        self.start_day(equity, at)

    def entry_allowed(self, equity: float) -> bool:
        if self.daily_start_equity is None:
            self.status = "OPERATOR_REVIEW_REQUIRED"
            return False
        current = _positive(equity, "equity")
        self.daily_loss_fraction = max(0.0, (self.daily_start_equity - current) / self.daily_start_equity)
        if self.daily_loss_fraction >= self.daily_loss_limit:
            self.status = "DAILY_LOSS_LIMIT"
            return False
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.status = "DEMO_RISK_COOLDOWN"
            return False
        self.status = "READY"
        return True

    def record_closed_trade(self, net_pnl: float) -> None:
        value = float(net_pnl)
        if not math.isfinite(value):
            raise ValueError("net_pnl must be finite")
        if value < 0:
            self.consecutive_losses += 1
        elif value > 0:
            self.consecutive_losses = 0
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.status = "DEMO_RISK_COOLDOWN"


__all__ = ["DemoCircuitBreaker", "normalize_volume", "validate_stop_levels"]
