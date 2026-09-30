from datetime import datetime, timedelta, timezone

from tradingagents.forex.hft.engines import FastExecutionEngine
from tradingagents.forex.hft.features import TickFeatureEngine
from tradingagents.forex.hft.models import (
    Direction,
    EntryConstraints,
    FastAction,
    PositionState,
    RiskPosture,
    StopPolicy,
    StrategicExecutionPlan,
    Tick,
)

UTC = timezone.utc


def _plan(direction=Direction.LONG):
    created = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    return StrategicExecutionPlan(
        symbol="EURUSD",
        created_at=created,
        expires_at=created + timedelta(minutes=5),
        allowed_until=created + timedelta(minutes=4),
        timeframe="M1",
        regime="TREND",
        primary_direction=direction,
        confidence=0.9,
        strategy_family="momentum_continuation",
        entry_constraints=EntryConstraints(max_spread_points=20, minimum_momentum=0.00005, minimum_volatility=0.0, maximum_volatility=1.0, minimum_confirmation=0.5),
        risk_posture=RiskPosture.NORMAL,
        stop_policy=StopPolicy(stop_distance_points=10, take_profit_distance_points=20, time_stop_seconds=30),
    )


def test_fast_engine_requires_confirmation_before_entry():
    features = TickFeatureEngine(window=3)
    engine = FastExecutionEngine()
    plan = _plan()
    for second, bid in enumerate((1.1, 1.10002, 1.10004)):
        decision = engine.on_tick(plan, features.update(Tick("EURUSD", datetime(2026, 1, 1, 12, 0, second, tzinfo=UTC), bid, bid + 0.0001)), position_state=PositionState.FLAT)
    assert decision.action is FastAction.NO_ACTION
    for second, bid in enumerate((1.1002, 1.1004, 1.1006), start=3):
        decision = engine.on_tick(plan, features.update(Tick("EURUSD", datetime(2026, 1, 1, 12, 0, second, tzinfo=UTC), bid, bid + 0.0001)), position_state=PositionState.FLAT)
    assert decision.action is FastAction.ENTER_LONG


def test_fast_engine_invalidates_expired_plan_and_exits_stop():
    features = TickFeatureEngine(window=2)
    engine = FastExecutionEngine()
    plan = _plan()
    tick = Tick("EURUSD", plan.created_at, 1.1, 1.1001)
    assert engine.on_tick(plan, features.update(tick), position_state=PositionState.FLAT).action is FastAction.NO_ACTION
    stop_tick = Tick("EURUSD", plan.created_at + timedelta(seconds=2), 1.0998, 1.0999)
    assert engine.on_tick(plan, features.update(stop_tick), position_state=PositionState.LONG, entry_price=1.1, entry_at=plan.created_at).action is FastAction.EXIT


def test_fast_engine_uses_tick_point_for_stop_distance():
    features = TickFeatureEngine(window=2)
    engine = FastExecutionEngine()
    plan = _plan()
    start = plan.created_at
    engine.on_tick(plan, features.update(Tick("EURUSD", start, 1.1, 1.1001, point=0.001)), position_state=PositionState.FLAT)
    tick = Tick("EURUSD", start + timedelta(seconds=1), 1.0985, 1.0995, point=0.001)
    assert engine.on_tick(plan, features.update(tick), position_state=PositionState.LONG, entry_price=1.1, entry_at=start).action is FastAction.EXIT
