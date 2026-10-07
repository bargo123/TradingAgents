import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from tradingagents.forex.hft.demo_gateway import VerifiedDemoExecutionGateway
from tradingagents.forex.hft.demo_runtime import DemoHftRuntime, DemoRuntimeConfig
from tradingagents.forex.hft.demo_store import DemoExecutionStore
from tradingagents.forex.hft.models import (
    Direction,
    EntryConstraints,
    RiskPosture,
    StopPolicy,
    StrategicExecutionPlan,
)
from tradingagents.forex.hft.plan_store import AtomicPlanStore
from tradingagents.forex.hft.store import HftShadowStore

UTC = timezone.utc


def _plan(now: datetime, direction=Direction.LONG):
    return StrategicExecutionPlan(
        symbol="EURUSD",
        created_at=now - timedelta(seconds=1),
        expires_at=now + timedelta(minutes=5),
        allowed_until=now + timedelta(minutes=5),
        timeframe="M1",
        regime="DEMO",
        primary_direction=direction,
        confidence=0.8,
        strategy_family="strategic_shadow",
        entry_constraints=EntryConstraints(minimum_momentum=0.0, minimum_confirmation=0.0),
        risk_posture=RiskPosture.NORMAL,
        stop_policy=StopPolicy(stop_distance_points=10, take_profit_distance_points=20, time_stop_seconds=300),
        plan_id="plan-1",
        source_decision_id="decision-1",
        source_run_id="source-run-1",
        git_commit="abc123",
    )


class _Provider:
    def __init__(self):
        self.order_api = None

    def get_tick(self, symbol):
        return SimpleNamespace(symbol=symbol, timestamp=datetime.now(UTC), bid=1.1000, ask=1.1001)

    def get_account_info(self):
        return SimpleNamespace(login=7, server="Broker-Demo", currency="USD", trade_mode=0, balance=1000.0, equity=1000.0, margin=0.0, margin_free=1000.0)

    def get_terminal_info(self):
        return SimpleNamespace(company="Broker", connected=True)

    def get_symbols(self):
        return (SimpleNamespace(name="EURUSD", point=0.00001, volume_min=0.01, volume_max=10.0, volume_step=0.01, trade_stops_level=0),)

    def get_positions(self, symbol=None):
        return ()


class _Api:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_FILLING_IOC = 1
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE = 10009

    def __init__(self):
        self.calls = []

    def order_send(self, request):
        self.calls.append(request)
        return SimpleNamespace(retcode=10009, order=11, deal=12, price=request["price"], volume=request["volume"], comment="filled")


def _runtime(tmp_path: Path, plan=None):
    now = datetime.now(UTC)
    provider = _Provider()
    api = _Api()
    demo_store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    gateway = VerifiedDemoExecutionGateway(provider, api, demo_store, run_id="demo-run", demo_trade_mode=0, magic=12012012)
    plans = AtomicPlanStore()
    if plan is not None:
        plans.replace(plan, now=now)
    runtime = DemoHftRuntime(
        provider,
        plans,
        gateway=gateway,
        store=demo_store,
        config=DemoRuntimeConfig(artifact_path=tmp_path / "demo.sqlite3", symbol="EURUSD"),
        clock=lambda: now,
    )
    return runtime, api, demo_store


def test_demo_runtime_none_plan_never_sends_order(tmp_path: Path):
    runtime, api, _ = _runtime(tmp_path)

    result = runtime.run_once()

    assert result["execution_mode"] == "DEMO"
    assert result["broker_order_sent"] is False
    assert api.calls == []


def test_continuous_demo_runtime_bounds_latency_history(tmp_path):
    runtime, _, _ = _runtime(tmp_path)
    for sample in range(10000):
        runtime._decision_latencies.append(float(sample))
        runtime._order_latencies.append(float(sample))
    assert len(runtime._decision_latencies) == 4096
    assert len(runtime._order_latencies) == 4096
    assert list(runtime._decision_latencies)[0] == 5904.0
    report = runtime.run(max_ticks=1)
    assert report["order_submission_p99_ms"] == 9958.0
    assert report["latency_sample_capacity"] == 4096


def test_demo_runtime_natural_long_uses_demo_gateway(tmp_path: Path):
    now = datetime.now(UTC)
    runtime, api, store = _runtime(tmp_path, _plan(now))

    result = runtime.run_once()

    assert result["action"] == "ENTER_LONG"
    assert result["broker_order_sent"] is True
    assert result["classification"] == "FILLED"
    assert len(api.calls) == 1
    assert store.snapshot()["orders"] == 1


def test_demo_runtime_links_signal_features_risk_and_strategy_version_to_intent(tmp_path: Path):
    now = datetime.now(UTC)
    runtime, _, store = _runtime(tmp_path, _plan(now))

    result = runtime.run_once()

    assert result["classification"] == "FILLED"
    with sqlite3.connect(tmp_path / "demo.sqlite3") as db:
        intent_id = db.execute("SELECT intent_id FROM demo_order_intents LIMIT 1").fetchone()[0]
    stored = store.read_order_intent(intent_id)
    assert stored is not None
    payload = __import__("json").loads(stored["request_payload"])
    assert payload["provenance"]["strategy_version"] == "phase12d-hft.v1"
    assert payload["provenance"]["config_version"] == "demo-runtime.v1"
    assert "feature_snapshot" in payload["provenance"]
    assert payload["provenance"]["risk"]["accepted"] is True


def test_demo_runtime_uses_hft_lease_and_releases_it(tmp_path: Path):
    now = datetime.now(UTC)
    hft_path = tmp_path / "hft.sqlite3"
    runtime, api, _ = _runtime(tmp_path, _plan(now))
    runtime.lease_store = HftShadowStore(hft_path)
    runtime.config = DemoRuntimeConfig(
        artifact_path=tmp_path / "demo.sqlite3", symbol="EURUSD", max_ticks=1
    )

    result = runtime.run(max_ticks=1)

    assert result["execution_mode"] == "DEMO"
    assert HftShadowStore(hft_path).read_only_active_lease() is None
