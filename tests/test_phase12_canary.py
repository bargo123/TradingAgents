from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.forex.hft.canary import (
    CanaryConfig,
    CanaryError,
    build_test_only_plan,
    run_canary_side,
)
from tradingagents.forex.hft.models import Direction, Tick
from tradingagents.forex.hft.store import HftShadowStore

UTC = timezone.utc


class _TickSource:
    def __init__(self, ticks):
        self._ticks = iter(ticks)

    def get_tick(self, symbol):
        return next(self._ticks)


def _unsequenced_ticks(start: datetime):
    return [
        Tick("EURUSD", start + timedelta(seconds=offset), 1.1000, 1.1001)
        for offset in range(4)
    ]


def _ticks(start: datetime):
    return [
        Tick("EURUSD", start + timedelta(seconds=offset), 1.1000, 1.1001, sequence=offset + 1)
        for offset in range(4)
    ]


def test_test_only_plan_is_explicitly_tagged_and_shadow_only():
    first = _ticks(datetime.now(UTC))[0]

    plan = build_test_only_plan(first, Direction.LONG, time_stop_seconds=2)
    payload = plan.to_payload()

    assert plan.test_only is True
    assert payload["execution_mode"] == "TEST_ONLY"
    assert payload["synthetic_plan"] is True
    assert payload["excluded_from_performance"] is True
    assert payload["executed"] is False
    assert payload["primary_direction"] == "LONG"


def test_production_plan_payload_has_no_test_only_execution_mode():
    first = _ticks(datetime.now(UTC))[0]
    from tradingagents.forex.hft.models import (
        EntryConstraints,
        RiskPosture,
        StopPolicy,
        StrategicExecutionPlan,
    )

    plan = StrategicExecutionPlan(
        symbol=first.symbol,
        created_at=first.timestamp,
        expires_at=first.timestamp + timedelta(minutes=1),
        allowed_until=first.timestamp + timedelta(seconds=30),
        timeframe="M1",
        regime="TREND",
        primary_direction=Direction.LONG,
        confidence=0.5,
        strategy_family="strategic_shadow",
        entry_constraints=EntryConstraints(),
        risk_posture=RiskPosture.NORMAL,
        stop_policy=StopPolicy(
            stop_distance_points=20,
            take_profit_distance_points=30,
            time_stop_seconds=30,
        ),
    )

    assert "execution_mode" not in plan.to_payload()


def test_canary_long_uses_real_engine_risk_fill_and_exit_path(tmp_path):
    start = datetime.now(UTC)
    path = tmp_path / "canary-long.sqlite3"
    source = _TickSource(_ticks(start))
    config = CanaryConfig(
        symbol="EURUSD",
        artifact_path=path,
        max_ticks=3,
        poll_interval_seconds=0.0,
        time_stop_seconds=2,
    )

    report = run_canary_side(
        source,
        direction=Direction.LONG,
        config=config,
        clock=lambda: start + timedelta(seconds=3),
    )

    assert report.runtime_result["executed"] is False
    assert report.runtime_result["entries"] == 1
    assert report.runtime_result["exits"] == 1
    assert report.position is not None
    assert report.position["state"] == "CLOSED"
    assert report.position["executed"] is False
    assert report.plan_payload["execution_mode"] == "TEST_ONLY"
    assert report.plan_payload["excluded_from_performance"] is True
    assert report.entry_quote is not None
    assert report.entry_quote["tick_sequence"] == "1"
    assert report.entry_quote["bid"] == pytest.approx(1.1)
    assert report.entry_risk is not None
    assert report.entry_risk["accepted"] is True
    assert report.exit_quote is not None


def test_canary_short_uses_real_engine_risk_fill_and_exit_path(tmp_path):
    start = datetime.now(UTC)
    path = tmp_path / "canary-short.sqlite3"
    source = _TickSource(_ticks(start))
    config = CanaryConfig(
        symbol="EURUSD",
        artifact_path=path,
        max_ticks=3,
        poll_interval_seconds=0.0,
        time_stop_seconds=2,
    )

    report = run_canary_side(
        source,
        direction=Direction.SHORT,
        config=config,
        clock=lambda: start + timedelta(seconds=3),
    )

    assert report.runtime_result["executed"] is False
    assert report.runtime_result["entries"] == 1
    assert report.runtime_result["exits"] == 1
    assert report.position is not None
    assert report.position["state"] == "CLOSED"


def test_canary_assigns_local_sequence_when_mt5_tick_has_none(tmp_path):
    start = datetime.now(UTC)
    path = tmp_path / "canary-sequence.sqlite3"
    report = run_canary_side(
        _TickSource(_unsequenced_ticks(start)),
        direction=Direction.LONG,
        config=CanaryConfig(
            artifact_path=path,
            max_ticks=3,
            poll_interval_seconds=0.0,
            time_stop_seconds=2,
        ),
        clock=lambda: start + timedelta(seconds=3),
    )

    assert report.entry_quote is not None
    assert report.entry_quote["tick_sequence"] == "1"


def test_canary_refuses_existing_artifact_data(tmp_path):
    path = tmp_path / "existing.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("existing", mode="SHADOW", source_fingerprint="TEST_ONLY")

    with pytest.raises(CanaryError, match="non-empty"):
        CanaryConfig(artifact_path=path).ensure_fresh_artifact()


def test_store_rejects_unlabeled_test_only_plan(tmp_path):
    path = tmp_path / "metadata.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run", mode="SHADOW", source_fingerprint="MT5_READ_ONLY_TEST_ONLY")
    payload = {
        "plan_id": "plan",
        "symbol": "EURUSD",
        "created_at": "2026-01-01T00:00:00Z",
        "expires_at": "2026-01-01T00:01:00Z",
        "test_only": True,
        "executed": False,
    }

    with pytest.raises(ValueError, match="isolation metadata"):
        store.record_plan("run", payload)
