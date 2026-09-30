"""Atomic validated strategic-plan replacement for the fast path."""

from __future__ import annotations

from threading import RLock
from datetime import datetime

from .models import StrategicExecutionPlan, utc


class PlanRejectedError(ValueError):
    """Raised when a plan cannot become the active plan."""


class AtomicPlanStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._plan: StrategicExecutionPlan | None = None

    def replace(self, plan: StrategicExecutionPlan, *, now: datetime) -> None:
        if not isinstance(plan, StrategicExecutionPlan):
            raise PlanRejectedError("plan must be StrategicExecutionPlan")
        timestamp = utc(now, "now")
        if not plan.is_active(timestamp):
            raise PlanRejectedError("plan is not active")
        with self._lock:
            current = self._plan
            if current is not None and plan.created_at < current.created_at:
                raise PlanRejectedError("replacement plan created_at must not predate current plan")
            self._plan = plan

    def current(self, now: datetime, symbol: str) -> StrategicExecutionPlan | None:
        timestamp = utc(now, "now")
        requested = str(symbol).strip().upper()
        with self._lock:
            plan = self._plan
            if plan is None or plan.symbol != requested or not plan.is_active(timestamp):
                return None
            return plan

    def clear(self) -> None:
        with self._lock:
            self._plan = None


__all__ = ["AtomicPlanStore", "PlanRejectedError"]
