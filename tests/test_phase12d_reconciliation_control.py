from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli.forex_supervisor import main as supervisor_main
from tradingagents.forex.hft.demo_gateway import VerifiedDemoExecutionGateway
from tradingagents.forex.hft.demo_models import DemoOrderIntent
from tradingagents.forex.hft.demo_runtime import DemoHftRuntime, DemoRuntimeConfig
from tradingagents.forex.hft.demo_store import DemoExecutionStore
from tradingagents.forex.hft.plan_store import AtomicPlanStore
from tradingagents.forex.hft.runtime import HftShadowConfig
from tradingagents.forex.hft.store import HftShadowStore
from tradingagents.forex.hft.supervisor import HftShadowWorker
from tradingagents.forex.supervisor import HftShadowSupervisorContext
from tradingagents.forex.watcher import SerializedMt5OperationGate

UTC = timezone.utc


class _Api:
    def order_send(self, _request):  # pragma: no cover - reconciliation must not send
        raise AssertionError("reconciliation must not call order_send")


class _Provider:
    def __init__(
        self,
        *,
        positions=(),
        orders=(),
        history_orders=(),
        history_deals=(),
        history_orders_range=None,
        history_deals_range=None,
        trade_mode=0,
    ):
        self.positions = tuple(positions)
        self.orders = tuple(orders)
        self.history_orders = tuple(history_orders)
        self.history_deals = tuple(history_deals)
        self.history_orders_range = (
            tuple(history_orders)
            if history_orders_range is None
            else tuple(history_orders_range)
        )
        self.history_deals_range = (
            tuple(history_deals)
            if history_deals_range is None
            else tuple(history_deals_range)
        )
        self.trade_mode = trade_mode
        self.calls: list[str] = []

    def get_account_info(self):
        self.calls.append("account")
        return SimpleNamespace(login=1, server="Demo", currency="USD", trade_mode=self.trade_mode)

    def get_terminal_info(self):
        self.calls.append("terminal")
        return SimpleNamespace(company="Broker", connected=True)

    def get_positions(self, symbol=None):
        self.calls.append("positions")
        return tuple(item for item in self.positions if symbol is None or item.symbol == symbol)

    def get_orders(self, symbol=None):
        self.calls.append("orders")
        return tuple(item for item in self.orders if symbol is None or item.symbol == symbol)

    def get_history_orders(self, ticket):
        self.calls.append(f"history_orders:{ticket}")
        return self.history_orders

    def get_history_deals(self, ticket):
        self.calls.append(f"history_deals:{ticket}")
        return self.history_deals

    def get_history_orders_range(self, start, end):
        self.calls.append(f"history_orders_range:{start.isoformat()}:{end.isoformat()}")
        return self.history_orders_range

    def get_history_deals_range(self, start, end):
        self.calls.append(f"history_deals_range:{start.isoformat()}:{end.isoformat()}")
        return self.history_deals_range


def _seed_open_position(store: DemoExecutionStore, ticket: int = 152717255467) -> dict:
    store.initialize()
    store.start_run("demo-run", git_commit="test")
    now = datetime(2026, 10, 2, 12, 40, tzinfo=UTC)
    intent = DemoOrderIntent(
        intent_id="intent-1",
        plan_id="plan-1",
        source_decision_id="decision-1",
        source_run_id="run-1",
        strategy_id="hft",
        symbol="EURUSD",
        direction="LONG",
        volume=0.01,
        requested_price=1.12493,
        stop_loss=1.12474,
        take_profit=1.12524,
        deviation_points=20,
        created_at=now,
        git_commit="test",
        account_trade_mode=0,
    )
    store.record_order_intent("demo-run", intent)
    position = {
        "ticket": ticket,
        "intent_id": intent.intent_id,
        "symbol": "EURUSD",
        "direction": "LONG",
        "volume": 0.01,
        "price_open": 1.12493,
        "stop_loss": 1.12474,
        "take_profit": 1.12524,
        "state": "OPEN",
    }
    store.record_position(position)
    return position


def _runtime(tmp_path: Path, provider: _Provider, store: DemoExecutionStore, gate=None):
    return DemoHftRuntime(
        provider,
        AtomicPlanStore(),
        gateway=VerifiedDemoExecutionGateway(
            provider,
            _Api(),
            store,
            run_id="demo-run",
            demo_trade_mode=0,
            magic=12012012,
            comment="TradingAgents-P12D-DEMO",
            gate=gate,
        ),
        store=store,
        config=DemoRuntimeConfig(
            artifact_path=tmp_path / "demo.sqlite3",
            symbol="EURUSD",
            magic=12012012,
            comment="TradingAgents-P12D-DEMO",
        ),
        mt5_gate=gate,
    )


def test_external_reconciliation_request_only_enqueues_without_constructing_supervisor(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("external request must not construct a supervisor or MT5 provider")

    monkeypatch.setattr("cli.forex_supervisor.ForexSupervisor", forbidden)
    monkeypatch.setattr("tradingagents.dataflows.mt5.provider.MT5Provider", forbidden)
    db = tmp_path / "demo.sqlite3"
    assert supervisor_main(
        [
            "control",
            "reconcile-demo-position",
            "--demo-db-path",
            str(db),
            "--ticket",
            "152717255467",
            "--symbol",
            "EURUSD",
        ]
    ) == 0
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT command,ticket,status FROM demo_control_requests"
        ).fetchone()
    assert row == ("RECONCILE_DEMO_POSITION", 152717255467, "PENDING")


def test_active_supervisor_consumes_control_request_without_duplicate_worker(tmp_path):
    class Worker:
        running = True

        def reconciliation_ready(self):
            return False

        def __init__(self):
            self.started = 0
            self.stopped = 0
            self.tickets = []

        def start(self):
            self.started += 1

        def stop(self):
            self.stopped += 1

        def reconcile_demo_position(self, ticket):
            self.tickets.append(ticket)
            return {"status": "RECONCILED", "resumed": True}

    db = tmp_path / "demo.sqlite3"
    store = DemoExecutionStore(db)
    store.request_reconciliation(ticket=152717255467, symbol="EURUSD")
    worker = Worker()
    context = HftShadowSupervisorContext(
        worker=worker,
        gate=SerializedMt5OperationGate(),
        provider_factory=lambda: None,
        plan_store=AtomicPlanStore(),
        hft_db_path=tmp_path / "hft.sqlite3",
        control_store=store,
        control_poll_interval_seconds=0.01,
    )
    context.start()
    deadline = time.monotonic() + 2
    while not worker.tickets and time.monotonic() < deadline:
        time.sleep(0.01)
    context.stop()
    assert worker.started == 1
    assert worker.tickets == [152717255467]
    assert worker.stopped == 1
    request = store.read_control_requests()[0]
    assert request["status"] == "COMPLETED"


def test_reconcile_existing_position_uses_serialized_gate_and_adopts_it(tmp_path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    ledger = _seed_open_position(store)
    broker_position = SimpleNamespace(
        ticket=ledger["ticket"],
        symbol="EURUSD",
        type=0,
        volume=0.01,
        price_open=1.12493,
        price_current=1.12450,
        profit=-0.43,
        magic=12012012,
        comment="TradingAgents-P12D-DEMO",
        sl=1.12474,
        tp=1.12524,
    )
    provider = _Provider(positions=(broker_position,))
    gate = SerializedMt5OperationGate()
    runtime = _runtime(tmp_path, provider, store, gate)
    result = runtime.reconcile_position(ledger["ticket"])
    assert result["status"] == "RECONCILED"
    assert "positions" in provider.calls
    assert store.read_owned_positions("EURUSD")[0]["state"] == "OPEN"
    assert store.read_reconciliation_state()["status"] == "RECONCILED"


def test_reconcile_adopts_owned_position_when_position_ticket_differs_from_order_ticket(tmp_path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    ledger = _seed_open_position(store)
    provider = _Provider(
        positions=(
            SimpleNamespace(
                ticket=9001,
                symbol="EURUSD",
                type=0,
                volume=0.01,
                price_open=1.12493,
                magic=12012012,
                comment="TradingAgents-P12D-DEMO",
            ),
        )
    )

    result = _runtime(tmp_path, provider, store).reconcile_position(ledger["ticket"])

    assert result == {
        "status": "RECONCILED",
        "terminal_state": "OPEN",
        "ticket": ledger["ticket"],
        "broker_position_ticket": 9001,
    }
    assert store.read_owned_positions("EURUSD")[0]["state"] == "OPEN"
    assert provider.calls.count("positions") == 1


def test_reconcile_broker_flat_without_history_records_explicit_absence(tmp_path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    ledger = _seed_open_position(store)
    provider = _Provider()

    result = _runtime(tmp_path, provider, store).reconcile_position(ledger["ticket"])

    assert result["status"] == "RECONCILED"
    assert result["terminal_state"] == "BROKER_ABSENT_CONFIRMED"
    assert result["current_exposure"] == "NONE"
    assert result["performance_inclusion"] is False
    assert result["realized_pnl"] is None
    assert result["exit_price"] is None
    assert result["exit_reason"] == "RECONCILIATION_BROKER_ABSENT"
    assert store.read_owned_positions("EURUSD")[0]["state"] == "RECONCILED_ABSENT"


def test_reconcile_matches_close_history_by_position_and_order_correlations(tmp_path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    ledger = _seed_open_position(store)
    provider = _Provider(
        history_orders_range=(
            {
                "ticket": 7001,
                "order": 7001,
                "position_id": 9001,
                "symbol": "EURUSD",
                "state": "FILLED",
                "magic": 12012012,
                "comment": "TradingAgents-P12D-DEMO",
            },
        ),
        history_deals_range=(
            {
                "ticket": 8001,
                "order": 7001,
                "position_id": 9001,
                "symbol": "EURUSD",
                "entry": "OUT",
                "volume": 0.01,
                "price": 1.12450,
                "profit": -4.3,
                "magic": 12012012,
                "comment": "TradingAgents-P12D-DEMO",
            },
        ),
    )

    result = _runtime(tmp_path, provider, store).reconcile_position(ledger["ticket"])

    assert result["status"] == "RECONCILED"
    assert result["terminal_state"] == "CLOSED"
    assert store.read_owned_positions("EURUSD")[0]["state"] == "CLOSED"
    assert any(call.startswith("history_orders_range:") for call in provider.calls)
    assert any(call.startswith("history_deals_range:") for call in provider.calls)


def test_reconcile_closed_history_persists_terminal_state(tmp_path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    ledger = _seed_open_position(store)
    provider = _Provider(
        history_orders=(SimpleNamespace(ticket=ledger["ticket"], symbol="EURUSD", state="FILLED", magic=12012012, comment="TradingAgents-P12D-DEMO"),),
        history_deals=(SimpleNamespace(ticket=77, order=ledger["ticket"], position_id=ledger["ticket"], symbol="EURUSD", entry="OUT", volume=0.01, price=1.12450, profit=-4.3),),
    )
    result = _runtime(tmp_path, provider, store).reconcile_position(ledger["ticket"])
    assert result["status"] == "RECONCILED"
    assert result["terminal_state"] == "CLOSED"
    assert store.read_owned_positions("EURUSD")[0]["state"] == "CLOSED"


def test_reopen_recovers_exact_broker_closed_ledger_position(tmp_path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    ledger = _seed_open_position(store)
    provider = _Provider(history_deals=(SimpleNamespace(
        ticket=77, position_id=ledger["ticket"], symbol="EURUSD", entry="OUT",
        volume=0.01, price=1.12450, profit=-4.3, magic=12012012,
    ),))
    runtime = _runtime(tmp_path, provider, store)
    runtime._validate_market_reconciliation()
    assert store.read_owned_positions("EURUSD")[0]["state"] == "CLOSED"
    assert runtime._open_position() is None
    assert store.read_reconciliation_state()["status"] == "RECONCILED"


@pytest.mark.parametrize("position_id,magic", [(999, 12012012), (152717255467, 999), (152717255467, None)])
def test_reopen_cannot_use_another_owned_trades_exit_as_close_proof(tmp_path, position_id, magic):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    _seed_open_position(store)
    provider = _Provider(history_deals=(SimpleNamespace(
        ticket=77, position_id=position_id, symbol="EURUSD", entry="OUT",
        volume=0.01, price=1.12450, profit=-4.3,
        magic=magic, comment="TradingAgents-P12D-DEMO",
    ),))
    runtime = _runtime(tmp_path, provider, store)
    with pytest.raises(Exception):
        runtime._validate_market_reconciliation()
    assert store.read_owned_positions("EURUSD")[0]["state"] == "OPEN"
    assert store.read_reconciliation_state()["status"] == "RECONCILIATION_REQUIRED"


def test_reconcile_closed_history_persists_net_pnl_from_entry_and_exit_deals(tmp_path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    ledger = _seed_open_position(store)
    history_deals = (
        {
            "ticket": 8000,
            "order": 7000,
            "position_id": ledger["ticket"],
            "symbol": "EURUSD",
            "entry": "IN",
            "profit": 0.0,
            "commission": -0.15,
            "swap": 0.0,
            "fee": 0.01,
            "magic": 12012012,
            "comment": "TradingAgents-P12D-DEMO",
        },
        {
            "ticket": 8001,
            "order": 7001,
            "position_id": ledger["ticket"],
            "symbol": "EURUSD",
            "entry": "OUT",
            "profit": 0.40,
            "commission": -0.15,
            "swap": -0.01,
            "fee": 0.01,
            "magic": 12012012,
            "comment": "TradingAgents-P12D-DEMO",
        },
    )
    provider = _Provider(
        history_orders_range=(
            {
                "ticket": 7001,
                "order": 7001,
                "position_id": ledger["ticket"],
                "symbol": "EURUSD",
                "state": "FILLED",
                "magic": 12012012,
                "comment": "TradingAgents-P12D-DEMO",
            },
        ),
        history_deals_range=history_deals,
    )

    result = _runtime(tmp_path, provider, store).reconcile_position(ledger["ticket"])

    assert result["status"] == "RECONCILED"
    assert result["realized_pnl"] == pytest.approx(0.11)
    with sqlite3.connect(tmp_path / "demo.sqlite3") as db:
        realized_pnl = db.execute(
            "SELECT realized_pnl FROM demo_exits ORDER BY observed_at DESC LIMIT 1"
        ).fetchone()[0]
    assert realized_pnl == pytest.approx(0.11)


def test_reconcile_ambiguous_state_remains_blocked(tmp_path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    ledger = _seed_open_position(store)
    provider = _Provider(
        positions=(SimpleNamespace(ticket=ledger["ticket"], symbol="EURUSD", type=0, volume=0.01, price_open=1.12493, magic=999, comment="other"),)
    )
    result = _runtime(tmp_path, provider, store).reconcile_position(ledger["ticket"])
    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert store.read_owned_positions("EURUSD")[0]["state"] == "OPEN"


def test_reconcile_non_demo_account_fails_closed(tmp_path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    ledger = _seed_open_position(store)
    provider = _Provider(trade_mode=1)
    result = _runtime(tmp_path, provider, store).reconcile_position(ledger["ticket"])
    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert store.read_owned_positions("EURUSD")[0]["state"] == "OPEN"


def test_worker_cannot_resume_before_successful_reconciliation(tmp_path):
    artifact = tmp_path / "hft.sqlite3"
    worker = HftShadowWorker(
        lambda: SimpleNamespace(initialize=lambda: False),
        AtomicPlanStore(),
        config=HftShadowConfig(artifact_path=artifact, max_ticks=1),
        store=HftShadowStore(artifact),
    )
    with pytest.raises(RuntimeError):
        worker.reconcile_demo_position(152717255467)


def test_failed_reconciliation_does_not_resume_worker(tmp_path):
    artifact = tmp_path / "hft.sqlite3"

    class Runtime:
        def reconcile_position(self, _ticket):
            return {"status": "RECONCILIATION_REQUIRED"}

        def run(self, *, stop_event):  # pragma: no cover - must not be reached
            raise AssertionError("worker must not resume after ambiguous reconciliation")

    worker = HftShadowWorker(
        lambda: SimpleNamespace(initialize=lambda: True, shutdown=lambda: None),
        AtomicPlanStore(),
        config=HftShadowConfig(artifact_path=artifact, max_ticks=1),
        store=HftShadowStore(artifact),
        runtime_factory=lambda _provider: Runtime(),
    )
    worker._runtime = Runtime()
    result = worker.reconcile_demo_position(152717255467)
    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert worker.running is False


def _clock_recovery_worker(
    tmp_path,
    *,
    server="MetaQuotes-Demo",
    account_trade_mode=0,
    tick_timestamp=None,
    second_result=None,
):
    artifact = tmp_path / "clock-recovery.hft.sqlite3"
    now = datetime.now(UTC)
    tick_timestamp = now if tick_timestamp is None else tick_timestamp
    events = []
    started = threading.Event()

    class Provider:
        clock_config = SimpleNamespace(max_tick_age_seconds=120.0, max_future_skew_seconds=2.0)
        broker_clock = SimpleNamespace(status="CALIBRATED")

        def initialize(self):
            events.append("initialize")
            return True

        def shutdown(self):
            events.append("shutdown")

        def reinitialize(self):
            events.append("reinitialize")
            return True

        def is_connected(self):
            events.append("connected")
            return True

        def calibrate_broker_clock(self, symbol):
            events.append(f"calibrate:{symbol}")
            return SimpleNamespace(status="CALIBRATED")

        def get_tick(self, symbol):
            events.append(f"tick:{symbol}")
            return SimpleNamespace(
                symbol=symbol, timestamp=tick_timestamp, bid=1.1, ask=1.1001, point=0.0001
            )

    provider = Provider()

    class Runtime:
        config = SimpleNamespace(symbol="EURUSD")
        clock = staticmethod(lambda: now)
        gateway = SimpleNamespace(
            demo_trade_mode=0,
            verify_demo_account=lambda: (
                events.append("account")
                or SimpleNamespace(server=server, trade_mode=account_trade_mode)
            ),
        )

        def __init__(self):
            self.reconcile_calls = 0

        def _tick(self):
            events.append("tick")
            return SimpleNamespace(
                symbol="EURUSD", timestamp=now, bid=1.1, ask=1.1001, point=0.0001
            )

        def _validate_market_reopen(self):
            events.append("full_safety_validation")
            return self._tick()

        def reconcile_position(self, ticket):
            assert ticket == 152720525521
            self.reconcile_calls += 1
            events.append(f"reconcile:{self.reconcile_calls}")
            if self.reconcile_calls == 1:
                return {
                    "status": "RECONCILIATION_REQUIRED",
                    "reason": "BROKER_READ_FAILED",
                    "error_type": "Mt5BrokerClockError",
                }
            return second_result or {"status": "RECONCILED", "terminal_state": "CLOSED"}

    runtime = Runtime()

    class RunningRuntime:
        def run(self, *, stop_event):
            started.set()
            stop_event.wait(5)
            return {"executed": False}

    worker = HftShadowWorker(
        lambda: provider,
        AtomicPlanStore(),
        config=HftShadowConfig(
            execution_mode="DEMO",
            artifact_path=artifact,
            max_ticks=1,
            max_reconnect_attempts=1,
            reconnect_backoff_seconds=0,
        ),
        store=HftShadowStore(artifact),
        runtime_factory=lambda _provider: RunningRuntime(),
    )
    worker._active_provider = provider
    worker._runtime = runtime
    return worker, runtime, provider, events, started


def test_stale_broker_clock_reinitializes_verifies_reconciles_once_and_resumes(tmp_path):
    worker, runtime, provider, events, started = _clock_recovery_worker(tmp_path)

    result = worker.reconcile_demo_position(152720525521)

    assert result["status"] == "RECONCILED"
    assert result["recovery"] == "BROKER_CLOCK_REINITIALIZED"
    assert result["resumed"] is True
    assert runtime.reconcile_calls == 2
    assert started.wait(2)
    assert events.index("reinitialize") < events.index("calibrate:EURUSD")
    assert events.index("account") < events.index("calibrate:EURUSD")
    assert events.index("tick:EURUSD") < events.index("reconcile:2")
    assert events.index("full_safety_validation") < events.index("initialize")
    worker.stop()
    assert provider.broker_clock.status == "CALIBRATED"


def test_clock_recovery_refuses_wrong_demo_server_without_retry_or_resume(tmp_path):
    worker, runtime, _provider, _events, _started = _clock_recovery_worker(
        tmp_path, server="Other-Demo"
    )

    result = worker.reconcile_demo_position(152720525521)

    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert result["reason"] == "BROKER_RECOVERY_FAILED"
    assert runtime.reconcile_calls == 1
    assert worker.running is False


def test_clock_recovery_refuses_non_demo_trade_mode(tmp_path):
    worker, runtime, _provider, _events, _started = _clock_recovery_worker(
        tmp_path, account_trade_mode=1
    )

    result = worker.reconcile_demo_position(152720525521)

    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert result["reason"] == "BROKER_RECOVERY_FAILED"
    assert runtime.reconcile_calls == 1
    assert worker.running is False


def test_clock_recovery_refuses_stale_tick_without_reconciliation_retry(tmp_path):
    worker, runtime, _provider, _events, _started = _clock_recovery_worker(
        tmp_path, tick_timestamp=datetime.now(UTC) - timedelta(minutes=5)
    )

    result = worker.reconcile_demo_position(152720525521)

    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert result["reason"] == "BROKER_RECOVERY_FAILED"
    assert runtime.reconcile_calls == 1
    assert worker.running is False


def test_clock_recovery_retries_reconciliation_at_most_once(tmp_path):
    second_failure = {
        "status": "RECONCILIATION_REQUIRED",
        "reason": "BROKER_READ_FAILED",
        "error_type": "Mt5BrokerClockError",
    }
    worker, runtime, _provider, events, _started = _clock_recovery_worker(
        tmp_path, second_result=second_failure
    )

    result = worker.reconcile_demo_position(152720525521)

    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert result["reason"] == "BROKER_READ_FAILED"
    assert runtime.reconcile_calls == 2
    assert events.count("reinitialize") == 1
    assert worker.running is False


def test_broker_clock_error_uses_bounded_automatic_recovery_path(tmp_path):
    worker, runtime, _provider, events, _started = _clock_recovery_worker(tmp_path)

    assert worker._is_recoverable("MT5BROKERCLOCKERROR") is True
    assert worker._recover_provider(
        worker._active_provider, "MT5BROKERCLOCKERROR", runtime
    ) is True
    assert events.count("reinitialize") == 1
    assert "calibrate:EURUSD" in events
    assert "tick:EURUSD" in events


def test_control_request_completion_is_owner_fenced(tmp_path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    request_id = store.request_reconciliation(ticket=152717255467, symbol="EURUSD")
    claimed = store.claim_control_request(claimed_by="owner-a")
    assert claimed is not None and claimed["request_id"] == request_id
    store.complete_control_request(
        request_id,
        claimed_by="owner-b",
        status="COMPLETED",
        result={"status": "RECONCILED"},
    )
    assert store.read_control_requests()[0]["status"] == "CLAIMED"
    store.complete_control_request(
        request_id,
        claimed_by="owner-a",
        status="COMPLETED",
        result={"status": "RECONCILED"},
    )
    assert store.read_control_requests()[0]["status"] == "COMPLETED"


def test_reconciled_ticket_does_not_resume_when_full_safety_validation_fails(tmp_path):
    artifact = tmp_path / "hft.sqlite3"

    class Runtime:
        def reconcile_position(self, _ticket):
            return {"status": "RECONCILED", "terminal_state": "CLOSED"}

        def _validate_market_reopen(self):
            raise RuntimeError("owned exposure mismatch")

        def run(self, *, stop_event):  # pragma: no cover - must not be reached
            raise AssertionError("worker must not resume before full safety validation")

    provider = SimpleNamespace(initialize=lambda: True, shutdown=lambda: None)
    worker = HftShadowWorker(
        lambda: provider,
        AtomicPlanStore(),
        config=HftShadowConfig(artifact_path=artifact, max_ticks=1),
        store=HftShadowStore(artifact),
        runtime_factory=lambda _provider: Runtime(),
    )
    worker._runtime = Runtime()

    result = worker.reconcile_demo_position(152717255467)

    assert result == {
        "status": "RECONCILIATION_REQUIRED",
        "reason": "POST_RECONCILIATION_VALIDATION_FAILED",
        "error_type": "RuntimeError",
    }
    assert worker.running is False


def test_successful_reconciliation_resumes_worker(tmp_path):
    artifact = tmp_path / "hft.sqlite3"
    started = []

    class Runtime:
        def reconcile_position(self, ticket):
            assert ticket == 152717255467
            return {"status": "RECONCILED", "terminal_state": "CLOSED"}

        def _validate_market_reopen(self):
            return SimpleNamespace(timestamp=datetime.now(UTC))

        def run(self, *, stop_event):
            started.append(True)
            return {"executed": False}

    provider = SimpleNamespace(initialize=lambda: True, shutdown=lambda: None)
    worker = HftShadowWorker(
        lambda: provider,
        AtomicPlanStore(),
        config=HftShadowConfig(artifact_path=artifact, max_ticks=1),
        store=HftShadowStore(artifact),
        runtime_factory=lambda _provider: Runtime(),
    )
    worker._runtime = Runtime()
    result = worker.reconcile_demo_position(152717255467)
    assert result["resumed"] is True
    assert worker.join(timeout=5) is True
    assert started == [True]


def test_running_worker_pauses_before_reconciliation_and_resumes_serially(tmp_path):
    artifact = tmp_path / "hft.sqlite3"
    active = 0
    max_active = 0
    reconciled_while_active = []
    entered = threading.Event()

    class Runtime:
        def reconcile_position(self, ticket):
            reconciled_while_active.append((ticket, active))
            return {"status": "RECONCILED", "terminal_state": "BROKER_ABSENT_CONFIRMED"}

        def _validate_market_reopen(self):
            return SimpleNamespace(timestamp=datetime.now(UTC))

        def run(self, *, stop_event):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            entered.set()
            stop_event.wait(5)
            active -= 1
            return {"executed": False}

    provider = SimpleNamespace(initialize=lambda: True, shutdown=lambda: None)
    worker = HftShadowWorker(
        lambda: provider,
        AtomicPlanStore(),
        config=HftShadowConfig(artifact_path=artifact, max_ticks=1),
        store=HftShadowStore(artifact),
        runtime_factory=lambda _provider: Runtime(),
    )
    worker.start()
    assert entered.wait(2)

    result = worker.reconcile_demo_position(152717255467)

    assert result["resumed"] is True
    assert reconciled_while_active == [(152717255467, 0)]
    assert max_active == 1
    assert worker.running is True
    worker.stop()
    assert worker.join(timeout=2) is True
