"""Cheap causal tick features for the Phase 12 fast path."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev

from .models import Tick, utc


def _session(timestamp: datetime) -> str:
    hour = timestamp.hour
    if 7 <= hour < 12:
        return "LONDON"
    if 12 <= hour < 16:
        return "LONDON_NEW_YORK_OVERLAP"
    if 16 <= hour < 21:
        return "NEW_YORK"
    return "ASIA"


@dataclass(frozen=True, slots=True)
class TickFeatures:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    mid: float
    spread: float
    spread_points: float
    return_1: float
    momentum: float
    velocity: float
    acceleration: float
    volatility: float
    rolling_range: float
    spread_expansion: float
    tick_frequency: float
    burst: float
    direction_persistence: float
    tick_count: int
    session: str
    plan_age_seconds: float = 0.0
    m1_return: float | None = None
    m5_return: float | None = None


class TickFeatureEngine:
    def __init__(self, *, max_history: int = 120, window: int = 20) -> None:
        if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 2:
            raise ValueError("max_history must be an integer >= 2")
        if isinstance(window, bool) or not isinstance(window, int) or window < 2 or window > max_history:
            raise ValueError("window must be between 2 and max_history")
        self.max_history = max_history
        self.window = window
        self._ticks: deque[Tick] = deque(maxlen=max_history)
        self._velocities: deque[float] = deque(maxlen=max_history)

    @property
    def history(self) -> tuple[Tick, ...]:
        return tuple(self._ticks)

    def update(self, tick: Tick, *, plan_created_at: datetime | None = None) -> TickFeatures:
        if not isinstance(tick, Tick):
            raise TypeError("tick must be Tick")
        if self._ticks and tick.symbol != self._ticks[-1].symbol:
            raise ValueError("feature symbol cannot change without resetting the engine")
        if self._ticks and tick.timestamp <= self._ticks[-1].timestamp:
            raise ValueError("tick timestamps must be strictly monotonic")
        previous = self._ticks[-1] if self._ticks else None
        velocity = 0.0
        return_1 = 0.0
        if previous is not None:
            elapsed = (tick.timestamp - previous.timestamp).total_seconds()
            if elapsed <= 0:
                raise ValueError("tick timestamps must be strictly monotonic")
            return_1 = (tick.mid - previous.mid) / previous.mid
            velocity = (tick.mid - previous.mid) / elapsed
        acceleration = 0.0
        if self._velocities and previous is not None:
            elapsed = (tick.timestamp - previous.timestamp).total_seconds()
            acceleration = (velocity - self._velocities[-1]) / elapsed
        self._ticks.append(tick)
        self._velocities.append(velocity)
        window_ticks = tuple(self._ticks)[-self.window :]
        mids = [item.mid for item in window_ticks]
        returns = [
            (right.mid - left.mid) / left.mid
            for left, right in zip(window_ticks, window_ticks[1:])
            if left.mid > 0
        ]
        momentum = mids[-1] - mids[0] if len(mids) > 1 else 0.0
        volatility = pstdev(returns) if len(returns) > 1 else (abs(returns[0]) if returns else 0.0)
        directions = [1 if value > 0 else -1 if value < 0 else 0 for value in returns]
        nonzero = [value for value in directions if value]
        persistence = abs(sum(nonzero)) / len(nonzero) if nonzero else 0.0
        elapsed_window = (window_ticks[-1].timestamp - window_ticks[0].timestamp).total_seconds()
        frequency = (len(window_ticks) - 1) / elapsed_window if elapsed_window > 0 else 0.0
        spreads = [item.spread for item in window_ticks]
        average_spread = mean(spreads) if spreads else tick.spread
        median_interval = mean(
            (right.timestamp - left.timestamp).total_seconds()
            for left, right in zip(window_ticks, window_ticks[1:])
        ) if len(window_ticks) > 1 else 0.0
        interval = (tick.timestamp - previous.timestamp).total_seconds() if previous else 0.0
        burst = (median_interval / interval) if interval > 0 and median_interval > 0 else 0.0
        plan_age = 0.0
        if plan_created_at is not None:
            created = utc(plan_created_at, "plan_created_at")
            plan_age = max(0.0, (tick.timestamp - created).total_seconds())
        return TickFeatures(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            bid=tick.bid,
            ask=tick.ask,
            mid=tick.mid,
            spread=tick.spread,
            spread_points=tick.spread_points,
            return_1=return_1,
            momentum=momentum,
            velocity=velocity,
            acceleration=acceleration,
            volatility=volatility,
            rolling_range=max(mids) - min(mids) if mids else 0.0,
            spread_expansion=tick.spread - average_spread,
            tick_frequency=frequency,
            burst=burst,
            direction_persistence=persistence,
            tick_count=len(self._ticks),
            session=_session(tick.timestamp),
            plan_age_seconds=plan_age,
        )


__all__ = ["TickFeatureEngine", "TickFeatures"]
