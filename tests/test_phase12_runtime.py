from datetime import datetime, timedelta, timezone

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
from tradingagents.forex.hft.store import HftShadowStore
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


def test_mt5_tick_adapter_exposes_no_mutation_api():
    names = {name.casefold() for name in dir(MT5ReadOnlyTickSource)}
    assert not any(token in name for name in names for token in ("order_send", "buy", "sell", "close_position", "modify_position"))
