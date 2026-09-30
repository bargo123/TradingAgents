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
from tradingagents.forex.hft.plan_store import AtomicPlanStore
from tradingagents.forex.hft.runtime import HftShadowConfig, HftShadowRuntime, MT5ReadOnlyTickSource
from tradingagents.forex.hft.store import HftLeaseOwner, HftLeaseStatus, HftShadowStore
from tradingagents.forex.watcher import SerializedMt5OperationGate

UTC = timezone.utc


def _plan(start):
    return StrategicExecutionPlan(
        symbol="EURUSD", created_at=start, expires_at=start + timedelta(minutes=5), allowed_until=start + timedelta(minutes=4), timeframe="M1", regime="TREND", primary_direction=Direction.BOTH, confidence=0.5, strategy_family="range_rejection", entry_constraints=EntryConstraints(max_spread_points=20), risk_posture=RiskPosture.NORMAL, stop_policy=StopPolicy(stop_distance_points=20, take_profit_distance_points=30, time_stop_seconds=60), plan_id="plan-1",
    )


class _Source:
    def __init__(self, ticks):
        self.ticks = iter(ticks)
        self.calls = 0

    def get_tick(self, symbol):
        self.calls += 1
        return next(self.ticks)


def test_runtime_uses_one_read_only_tick_operation_and_persists_shadow_state(tmp_path):
    start = datetime.now(UTC) - timedelta(minutes=1)
    source = _Source([Tick("EURUSD", start, 1.1, 1.1001, sequence=1)])
    store = HftShadowStore(tmp_path / "hft.sqlite3")
    plans = AtomicPlanStore()
    plans.replace(_plan(start), now=start)
    runtime = HftShadowRuntime(source, plans, config=HftShadowConfig(symbol="EURUSD", artifact_path=tmp_path / "hft.sqlite3", max_ticks=1), store=store, mt5_gate=SerializedMt5OperationGate())
    result = runtime.run(max_ticks=1)
    assert source.calls == 1
    assert result["executed"] is False
    assert store.snapshot()["ticks"] == 1


def test_runtime_fails_closed_when_plan_is_expired(tmp_path):
    now = datetime.now(UTC)
    source = _Source([Tick("EURUSD", now, 1.1, 1.1001, sequence=1)])
    store = HftShadowStore(tmp_path / "hft.sqlite3")
    plans = AtomicPlanStore()
    runtime = HftShadowRuntime(source, plans, config=HftShadowConfig(symbol="EURUSD", artifact_path=tmp_path / "hft.sqlite3", max_ticks=1), store=store)
    result = runtime.run(max_ticks=1)
    assert result["executed"] is False
    assert result["actions"] == 1


def test_runtime_requires_provenance_for_live_plan(tmp_path):
    start = datetime.now(UTC) - timedelta(minutes=1)
    source = _Source([Tick("EURUSD", start, 1.1, 1.1001, sequence=1)])
    store = HftShadowStore(tmp_path / "hft.sqlite3")
    plans = AtomicPlanStore()
    plans.replace(_plan(start), now=start)
    runtime = HftShadowRuntime(
        source,
        plans,
        config=HftShadowConfig(
            symbol="EURUSD",
            artifact_path=tmp_path / "hft.sqlite3",
            max_ticks=1,
            require_plan_provenance=True,
        ),
        store=store,
    )
    result = runtime.run(max_ticks=1)
    assert result["executed"] is False
    assert result["actions"] == 1


def test_runtime_drops_non_monotonic_ticks_without_stopping(tmp_path):
    start = datetime.now(UTC)
    source = _Source(
        [
            Tick("EURUSD", start, 1.1, 1.1001, sequence=1),
            Tick("EURUSD", start, 1.1, 1.1001, sequence=2),
            Tick("EURUSD", start - timedelta(seconds=1), 1.1, 1.1001, sequence=3),
            Tick("EURUSD", start + timedelta(seconds=1), 1.1001, 1.1002, sequence=4),
        ]
    )
    path = tmp_path / "hft.sqlite3"
    runtime = HftShadowRuntime(
        source,
        AtomicPlanStore(),
        config=HftShadowConfig(symbol="EURUSD", artifact_path=path, max_ticks=4),
        store=HftShadowStore(path),
    )

    result = runtime.run(max_ticks=4)

    assert result["executed"] is False
    assert result["ticks_processed"] == 2
    assert result["stale_ticks"] == 1
    assert result["out_of_order_ticks"] == 1
    assert result["dropped_ticks"] == 2


def test_runtime_skips_tick_when_strategic_mt5_operation_is_busy(tmp_path):
    now = datetime.now(UTC)
    source = _Source([Tick("EURUSD", now, 1.1, 1.1001, sequence=1)])
    gate = SerializedMt5OperationGate()
    hold = gate.acquire("strategic_analysis")
    hold.__enter__()
    try:
        path = tmp_path / "hft.sqlite3"
        runtime = HftShadowRuntime(
            source,
            AtomicPlanStore(),
            config=HftShadowConfig(symbol="EURUSD", artifact_path=path, max_ticks=1),
            store=HftShadowStore(path),
            mt5_gate=gate,
        )

        result = runtime.run(max_ticks=1)
    finally:
        hold.__exit__(None, None, None)

    assert result["executed"] is False
    assert result["ticks_processed"] == 0
    assert result["dropped_ticks"] == 1
    assert result["error_code"] is None


def test_runtime_refuses_an_active_hft_owner_before_reading_ticks(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    now = datetime.now(UTC)
    assert store.acquire_lease(
        HftLeaseOwner("existing", 999, "host", now), now
    ).status is HftLeaseStatus.ACQUIRED
    source = _Source([Tick("EURUSD", now, 1.1, 1.1001, sequence=1)])
    plans = AtomicPlanStore()
    runtime = HftShadowRuntime(
        source,
        plans,
        config=HftShadowConfig(symbol="EURUSD", artifact_path=path, max_ticks=1),
        store=store,
    )
    with pytest.raises(RuntimeError, match="HFT_ALREADY_RUNNING"):
        runtime.run(max_ticks=1)
    assert source.calls == 0


def test_runtime_restores_open_shadow_position_after_restart(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("prior", mode="SHADOW", source_fingerprint="abc")
    entry = datetime.now(UTC) - timedelta(minutes=1)
    store.record_position(
        "prior",
        {
            "position_id": "persisted",
            "symbol": "EURUSD",
            "state": "LONG",
            "direction": "LONG",
            "size": 1.0,
            "entry_price": 1.1001,
            "entry_timestamp": entry.isoformat(),
            "strategy_id": "strategic_shadow",
            "stop_price": None,
            "target_price": None,
            "executed": False,
        },
    )
    runtime = HftShadowRuntime(
        _Source([Tick("EURUSD", entry + timedelta(seconds=1), 1.1, 1.1001, sequence=1)]),
        AtomicPlanStore(),
        config=HftShadowConfig(symbol="EURUSD", artifact_path=path, max_ticks=1),
        store=store,
    )
    assert runtime.positions.position is not None
    assert runtime.positions.position.position_id == "persisted"


def test_mt5_tick_adapter_exposes_no_mutation_api():
    names = {name.casefold() for name in dir(MT5ReadOnlyTickSource)}
    assert not any(token in name for name in names for token in ("order_send", "buy", "sell", "close_position", "modify_position"))
