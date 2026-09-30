from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.forex.hft.models import (
    Direction,
    EntryConstraints,
    RiskPosture,
    StopPolicy,
    StrategicExecutionPlan,
    Tick,
)
from tradingagents.forex.hft.replay import ReplayError, TickReplay, load_ticks

UTC = timezone.utc


def _plan(start):
    return StrategicExecutionPlan(
        symbol="EURUSD", created_at=start, expires_at=start + timedelta(minutes=10), allowed_until=start + timedelta(minutes=9),
        timeframe="M1", regime="TREND", primary_direction=Direction.LONG, confidence=0.8, strategy_family="momentum_continuation",
        entry_constraints=EntryConstraints(max_spread_points=20, minimum_momentum=0.00001), risk_posture=RiskPosture.NORMAL,
        stop_policy=StopPolicy(stop_distance_points=10, take_profit_distance_points=20, time_stop_seconds=30),
    )


def _ticks(count=8):
    start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    return tuple(Tick("EURUSD", start + timedelta(seconds=i), 1.1 + i * 0.0001, 1.1001 + i * 0.0001, sequence=i) for i in range(count))


def test_replay_uses_same_causal_engine_and_reports_latency():
    ticks = _ticks()
    report = TickReplay(ticks, plan=_plan(ticks[0].timestamp)).run()
    assert report.ticks_processed == len(ticks)
    assert report.future_leak_detected is False
    assert report.latency_p50_ms >= 0
    assert report.to_dict()["executed"] is False


def test_replay_rejects_non_monotonic_and_future_ticks():
    ticks = list(_ticks())
    ticks[3] = ticks[1]
    with pytest.raises(ReplayError, match="monotonic"):
        TickReplay(ticks, plan=_plan(ticks[0].timestamp))
    future = datetime.now(UTC) + timedelta(minutes=1)
    with pytest.raises(ReplayError, match="future"):
        TickReplay([Tick("EURUSD", future, 1.1, 1.1001)], plan=_plan(future))


def test_load_ticks_reads_csv_without_future_leakage(tmp_path: Path):
    path = tmp_path / "ticks.csv"
    path.write_text("timestamp,symbol,bid,ask,point,sequence\n2026-01-01T12:00:00Z,EURUSD,1.1,1.1001,0.00001,0\n", encoding="utf-8")
    loaded = load_ticks(path)
    assert loaded[0].symbol == "EURUSD"

