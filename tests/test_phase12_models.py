from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.forex.hft.models import (
    Direction,
    EntryConstraints,
    RiskPosture,
    StopPolicy,
    StrategicExecutionPlan,
    Tick,
)


UTC = timezone.utc


def _plan(**overrides):
    created = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    values = {
        "symbol": "EURUSD",
        "created_at": created,
        "expires_at": created + timedelta(minutes=15),
        "allowed_until": created + timedelta(minutes=10),
        "timeframe": "M15",
        "regime": "TREND",
        "primary_direction": Direction.LONG,
        "confidence": 0.8,
        "strategy_family": "momentum_continuation",
        "entry_constraints": EntryConstraints(
            max_spread_points=20,
            minimum_momentum=0.00005,
            minimum_volatility=0.00001,
            maximum_volatility=0.01,
        ),
        "risk_posture": RiskPosture.NORMAL,
        "stop_policy": StopPolicy(
            stop_distance_points=20,
            take_profit_distance_points=40,
            time_stop_seconds=600,
        ),
    }
    values.update(overrides)
    return StrategicExecutionPlan(**values)


def test_plan_requires_aware_utc_and_valid_bounds():
    plan = _plan()
    assert plan.symbol == "EURUSD"
    assert plan.is_active(plan.created_at)
    assert not plan.is_active(plan.expires_at)

    with pytest.raises(ValueError, match="UTC"):
        _plan(created_at=datetime(2026, 1, 1, 12, 0))

    with pytest.raises(ValueError, match="expires_at"):
        _plan(expires_at=plan.created_at)


def test_plan_rejects_unbounded_confidence_and_unknown_direction():
    with pytest.raises(ValueError, match="confidence"):
        _plan(confidence=1.1)
    with pytest.raises(ValueError, match="primary_direction"):
        _plan(primary_direction="BUY")


def test_tick_validates_quote_and_utc():
    tick = Tick(
        symbol="EURUSD",
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        bid=1.10000,
        ask=1.10010,
        sequence=1,
    )
    assert tick.mid == pytest.approx(1.10005)
    assert tick.spread == pytest.approx(0.00010)
    with pytest.raises(ValueError, match="ask"):
        Tick("EURUSD", tick.timestamp, 1.1, 1.0)

