from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.forex.hft.models import (
    Direction,
    EntryConstraints,
    RiskPosture,
    StopPolicy,
    StrategicExecutionPlan,
)
from tradingagents.forex.hft.plan_store import AtomicPlanStore, PlanRejectedError

UTC = timezone.utc


def _plan(symbol="EURUSD", *, offset=0):
    created = datetime(2026, 1, 1, 12, 0, tzinfo=UTC) + timedelta(minutes=offset)
    return StrategicExecutionPlan(
        symbol=symbol,
        created_at=created,
        expires_at=created + timedelta(minutes=15),
        allowed_until=created + timedelta(minutes=10),
        timeframe="M15",
        regime="RANGE",
        primary_direction=Direction.BOTH,
        confidence=0.6,
        strategy_family="range_rejection",
        entry_constraints=EntryConstraints(max_spread_points=20),
        risk_posture=RiskPosture.CONSERVATIVE,
        stop_policy=StopPolicy(stop_distance_points=20, take_profit_distance_points=20, time_stop_seconds=300),
    )


def test_atomic_plan_store_replaces_only_valid_current_plan():
    store = AtomicPlanStore()
    first = _plan()
    store.replace(first, now=first.created_at)
    assert store.current(first.created_at, "EURUSD") == first
    second = _plan(offset=1)
    store.replace(second, now=second.created_at)
    assert store.current(second.created_at, "EURUSD") == second
    assert store.current(second.created_at, "USDJPY") is None


def test_atomic_plan_store_rejects_expired_or_mismatched_replacement():
    store = AtomicPlanStore()
    plan = _plan()
    with pytest.raises(PlanRejectedError, match="active"):
        store.replace(plan, now=plan.expires_at)
    store.replace(plan, now=plan.created_at)
    with pytest.raises(PlanRejectedError, match="created"):
        store.replace(_plan(offset=-1), now=plan.created_at)

