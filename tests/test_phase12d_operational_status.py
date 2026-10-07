import sqlite3
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from tradingagents.forex.hft.dashboard import read_hft_dashboard
from tradingagents.forex.hft.demo_store import DemoExecutionStore
from tradingagents.forex.hft.models import Tick
from tradingagents.forex.hft.store import HftLeaseOwner, HftShadowStore
from tradingagents.forex.supervisor import ForexSupervisor

UTC = timezone.utc


def _insert_demo_position(path, *, ticket, state):
    with sqlite3.connect(path) as db:
        db.execute(
            """
            INSERT INTO demo_positions(
                ticket,intent_id,symbol,direction,volume,price_open,state,owned,
                payload_json,updated_at,execution_mode,real_money
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ticket,
                f"intent-{ticket}",
                "EURUSD",
                "LONG",
                0.01,
                1.1,
                state,
                1,
                "{}",
                "2026-10-05T16:00:00Z",
                "DEMO",
                0,
            ),
        )


def test_demo_status_does_not_treat_closed_history_as_an_open_position(tmp_path):
    path = tmp_path / "demo.sqlite3"
    store = DemoExecutionStore(path)
    store.initialize()
    _insert_demo_position(path, ticket=101, state="CLOSED")

    status = ForexSupervisor().demo_status(path)

    assert status["positions"] == 1
    assert status["open_positions"] == 0
    assert status["status"] == "READY"


def test_demo_status_reports_a_current_open_position(tmp_path):
    path = tmp_path / "demo.sqlite3"
    store = DemoExecutionStore(path)
    store.initialize()
    _insert_demo_position(path, ticket=102, state="OPEN")

    status = ForexSupervisor().demo_status(path)

    assert status["positions"] == 1
    assert status["open_positions"] == 1
    assert status["status"] == "POSITION_OPEN"


def test_hft_dashboard_exposes_age_of_latest_persisted_tick(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run-1", mode="SHADOW", source_fingerprint="fixture")
    tick_time = datetime(2026, 10, 5, 16, 0, tzinfo=UTC)
    store.record_tick(
        "run-1",
        Tick("EURUSD", tick_time, 1.1, 1.1001),
        features={},
    )

    snapshot = read_hft_dashboard(
        path,
        observed_at=tick_time + timedelta(seconds=3000),
    )

    assert snapshot.dataset_last_timestamp == "2026-10-05T16:00:00Z"
    assert snapshot.last_tick_age_seconds == 3000.0


def test_hft_status_degrades_when_lease_is_active_but_no_run_is_active(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    now = datetime.now(UTC)
    store.set_runtime_state("RUNNING")
    lease = store.acquire_lease(
        HftLeaseOwner("owner-token", 123, "test-host", now), now
    )
    assert lease.status.value == "ACQUIRED"
    store.start_run("previous-run", mode="SHADOW", source_fingerprint="fixture")
    store.close_run("previous-run", status="STOPPED")

    report = ForexSupervisor().hft_status(path)

    assert report["active_runs"] == 0
    assert report["status"] == "DEGRADED"


def test_supervisor_status_exposes_latest_tick_age(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run-1", mode="SHADOW", source_fingerprint="fixture")
    store.record_tick(
        "run-1",
        Tick("EURUSD", datetime.now(UTC) - timedelta(minutes=5), 1.1, 1.1001),
        features={},
    )

    report = ForexSupervisor().hft_status(path)

    assert report["last_tick_age_seconds"] is not None
    assert report["last_tick_age_seconds"] >= 299.0


def test_active_lease_cannot_mask_a_stalled_tick_feed(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    now = datetime.now(UTC)
    store.set_runtime_state("RUNNING")
    store.acquire_lease(HftLeaseOwner("owner", 123, "host", now), now)
    store.start_run("run", mode="SHADOW", source_fingerprint="fixture")
    store.record_tick("run", Tick("EURUSD", now - timedelta(seconds=300), 1.1, 1.1001), features={})
    snapshot = read_hft_dashboard(path, observed_at=now)
    assert snapshot.status == "DEGRADED"
    assert snapshot.last_error_code == "TICK_FEED_STALE"
    assert snapshot.runtime_status == "RUNNING"


def test_recreated_demo_runtime_gets_a_new_ledger_run_id(tmp_path, monkeypatch):
    import tradingagents.dataflows.mt5.provider as provider_module
    import tradingagents.forex.hft.demo_gateway as gateway_module
    import tradingagents.forex.hft.demo_runtime as demo_runtime_module
    import tradingagents.forex.hft.demo_store as demo_store_module
    import tradingagents.forex.hft.plan_store as plan_store_module
    import tradingagents.forex.hft.regime_store as regime_store_module
    import tradingagents.forex.hft.runtime as hft_runtime_module
    import tradingagents.forex.hft.store as hft_store_module
    import tradingagents.forex.hft.supervisor as hft_supervisor_module
    import tradingagents.forex.runtime_config as runtime_config_module
    import tradingagents.forex.shadow as shadow_module
    import tradingagents.forex.supervisor as supervisor_module

    class Provider:
        _api = SimpleNamespace(ACCOUNT_TRADE_MODE_DEMO=0)

        def __init__(self, terminal_path=None):
            self.terminal_path = terminal_path

    class Gate:
        def acquire(self, _operation):
            return nullcontext()

    class SharedProvider:
        def __init__(self, _provider):
            pass

        def demo_api(self):
            return Provider._api

    class Gateway:
        def __init__(self, *_args, **kwargs):
            self.run_id = kwargs["run_id"]

        @staticmethod
        def read_account(_provider):
            return SimpleNamespace(server="MetaQuotes-Demo", trade_mode=0)

        @staticmethod
        def require_demo_account(account, demo_trade_mode):
            assert account.trade_mode == demo_trade_mode == 0

    class Runtime:
        def __init__(self, *_args, gateway, **_kwargs):
            self.gateway = gateway

    class Worker:
        def __init__(self, *_args, runtime_factory, **_kwargs):
            self.runtime_factory = runtime_factory

        def handle_decision(self, _decision):
            return False

    class ShadowStore:
        def __init__(self, *_args):
            pass

        def latest_eligible(self, _symbol):
            return None

        def is_execution_eligible(self, _decision):
            return False

    class Ledger:
        def __init__(self, *_args):
            pass

    monkeypatch.setattr(provider_module, "MT5Provider", Provider)
    monkeypatch.setattr(supervisor_module, "_SharedReadOnlyMt5Provider", SharedProvider)
    monkeypatch.setattr(gateway_module, "VerifiedDemoExecutionGateway", Gateway)
    monkeypatch.setattr(demo_runtime_module, "DemoHftRuntime", Runtime)
    monkeypatch.setattr(demo_runtime_module, "DemoRuntimeConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(demo_store_module, "DemoExecutionStore", Ledger)
    monkeypatch.setattr(plan_store_module, "AtomicPlanStore", lambda: object())
    monkeypatch.setattr(regime_store_module, "AtomicRegimeStore", lambda: object())
    monkeypatch.setattr(hft_runtime_module, "HftShadowConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(hft_store_module, "HftShadowStore", lambda *_args: object())
    monkeypatch.setattr(hft_supervisor_module, "HftShadowWorker", Worker)
    monkeypatch.setattr(runtime_config_module, "collect_runtime_provenance", lambda _config: {"git_commit": "test"})
    monkeypatch.setattr(shadow_module, "ShadowDecisionStore", ShadowStore)

    context = ForexSupervisor()._hft_context(
        terminal_path=None,
        source_db_path=tmp_path / "source.sqlite3",
        symbol="EURUSD",
        hft_db_path=tmp_path / "hft.sqlite3",
        max_ticks=0,
        poll_interval_seconds=0.05,
        gate=Gate(),
        demo_execute=True,
        demo_db_path=tmp_path / "demo.sqlite3",
        market_session_calendar=tmp_path / "calendar.json",
    )

    first = context.worker.runtime_factory(context.provider_factory())
    second = context.worker.runtime_factory(context.provider_factory())

    assert first.gateway.run_id != second.gateway.run_id
