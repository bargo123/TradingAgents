"""Fail-closed market lifecycle states for a single supervised HFT owner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .market_calendar import CalendarStatus


class MarketState(str, Enum):
    MARKET_CLOSED = "MARKET_CLOSED"
    MARKET_OPENING_VALIDATION = "MARKET_OPENING_VALIDATION"
    MARKET_OPEN = "MARKET_OPEN"
    MARKET_UNKNOWN = "MARKET_UNKNOWN"
    MARKET_DATA_ERROR = "MARKET_DATA_ERROR"


class MarketReason(str, Enum):
    STALE_DATA = "STALE_DATA"
    BROKER_DISCONNECTED = "BROKER_DISCONNECTED"
    BROKER_CLOCK_ERROR = "BROKER_CLOCK_ERROR"
    CALENDAR_BROKER_MISMATCH = "CALENDAR_BROKER_MISMATCH"
    CALENDAR_UNAVAILABLE = "CALENDAR_UNAVAILABLE"
    DATA_FAILURE = "DATA_FAILURE"


@dataclass(frozen=True, slots=True)
class MarketObservation:
    state: MarketState
    reason: MarketReason | None = None

    @property
    def permits_strategy_evaluation(self) -> bool:
        return self.state is MarketState.MARKET_OPEN


class MarketLifecycleController:
    """Track closure and require a successful validation before market-open."""

    def __init__(self) -> None:
        self._observation = MarketObservation(MarketState.MARKET_UNKNOWN, MarketReason.CALENDAR_UNAVAILABLE)

    @property
    def observation(self) -> MarketObservation:
        return self._observation

    def observe(
        self,
        calendar_status: CalendarStatus,
        *,
        terminal_healthy: bool,
        account_healthy: bool,
        tick_fresh: bool,
        error_reason: MarketReason | str | None = None,
    ) -> MarketObservation:
        if error_reason is not None:
            reason = _reason(error_reason)
            state = MarketState.MARKET_DATA_ERROR
        elif not terminal_healthy:
            state, reason = MarketState.MARKET_DATA_ERROR, MarketReason.BROKER_DISCONNECTED
        elif not account_healthy:
            state, reason = MarketState.MARKET_DATA_ERROR, MarketReason.DATA_FAILURE
        elif calendar_status is CalendarStatus.UNKNOWN:
            state, reason = MarketState.MARKET_UNKNOWN, MarketReason.CALENDAR_UNAVAILABLE
        elif calendar_status is CalendarStatus.CLOSED:
            if tick_fresh:
                state, reason = MarketState.MARKET_OPENING_VALIDATION, MarketReason.CALENDAR_BROKER_MISMATCH
            else:
                state, reason = MarketState.MARKET_CLOSED, None
        elif calendar_status is CalendarStatus.OPEN:
            if tick_fresh:
                state, reason = MarketState.MARKET_OPENING_VALIDATION, None
            else:
                state, reason = MarketState.MARKET_UNKNOWN, MarketReason.STALE_DATA
        else:
            state, reason = MarketState.MARKET_UNKNOWN, MarketReason.CALENDAR_UNAVAILABLE
        self._observation = MarketObservation(state, reason)
        return self._observation

    def complete_opening_validation(
        self, success: bool, *, reason: MarketReason | str | None = None
    ) -> MarketObservation:
        if self._observation.state is not MarketState.MARKET_OPENING_VALIDATION:
            raise RuntimeError("opening validation is not pending")
        if success:
            self._observation = MarketObservation(MarketState.MARKET_OPEN)
        else:
            self._observation = MarketObservation(
                MarketState.MARKET_DATA_ERROR,
                _reason(reason or MarketReason.DATA_FAILURE),
            )
        return self._observation


def _reason(value: MarketReason | str) -> MarketReason:
    if isinstance(value, MarketReason):
        return value
    try:
        return MarketReason(value)
    except ValueError:
        return MarketReason.DATA_FAILURE


__all__ = ["MarketLifecycleController", "MarketObservation", "MarketReason", "MarketState"]
