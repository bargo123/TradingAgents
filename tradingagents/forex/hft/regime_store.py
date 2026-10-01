"""Lock-free-on-tick atomic publication of the latest strategic regime."""

from __future__ import annotations

from datetime import datetime
from threading import RLock

from .models import utc
from .regime import StrategicRegimeState


class RegimeRejectedError(ValueError):
    """Raised when a regime cannot become the active state."""


class AtomicRegimeStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._state: StrategicRegimeState | None = None

    def replace(self, state: StrategicRegimeState, *, now: datetime) -> None:
        if not isinstance(state, StrategicRegimeState):
            raise RegimeRejectedError("state must be StrategicRegimeState")
        timestamp = utc(now, "now")
        if not state.is_active(timestamp):
            raise RegimeRejectedError("state is not active")
        with self._lock:
            current = self._state
            if current is not None and state.created_at < current.created_at:
                raise RegimeRejectedError("replacement state predates current state")
            self._state = state

    def current(self, now: datetime, symbol: str) -> StrategicRegimeState | None:
        timestamp = utc(now, "now")
        requested = str(symbol).strip().upper()
        with self._lock:
            state = self._state
            if state is None or state.symbol != requested or not state.is_active(timestamp):
                return None
            return state

    def clear(self) -> None:
        with self._lock:
            self._state = None


__all__ = ["AtomicRegimeStore", "RegimeRejectedError"]
