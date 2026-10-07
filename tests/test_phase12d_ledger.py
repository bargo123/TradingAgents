from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.forex.hft.demo_models import DemoAccountSnapshot, DemoOrderIntent, ExecutionMode
from tradingagents.forex.hft.demo_store import DemoExecutionStore

UTC = timezone.utc


def _intent():
    return DemoOrderIntent(
        intent_id="intent-1",
        plan_id="plan-1",
        source_decision_id="decision-1",
        source_run_id="run-1",
        strategy_id="strategic_shadow",
        symbol="EURUSD",
        direction="LONG",
        volume=0.01,
        requested_price=1.1,
        stop_loss=1.099,
        take_profit=1.102,
        deviation_points=20,
        created_at=datetime(2026, 9, 30, tzinfo=UTC),
        git_commit="abc123",
        account_trade_mode=0,
    )


def test_demo_store_initializes_separate_schema_and_is_idempotent(tmp_path: Path):
    path = tmp_path / "demo.sqlite3"
    store = DemoExecutionStore(path)

    store.initialize()
    store.initialize()

    assert store.integrity_check() == "ok"
    assert store.snapshot()["orders"] == 0


def test_demo_store_persists_account_intent_result_and_owned_position(tmp_path: Path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    store.initialize()
    store.start_run("run-1", git_commit="abc123")
    store.record_account_snapshot(
        "run-1",
        DemoAccountSnapshot(
            login=7,
            server="Broker-Demo",
            company="Broker",
            currency="USD",
            trade_mode=0,
            balance=1000.0,
            equity=1000.0,
        ),
    )
    intent = _intent()
    store.record_order_intent("run-1", intent)
    store.record_order_intent("run-1", intent)
    store.record_order_result(
        intent_id=intent.intent_id,
        classification="FILLED",
        retcode=10009,
        order_ticket=11,
        deal_ticket=12,
        fill_price=1.1001,
        fill_volume=0.01,
        broker_comment="done",
        broker_timestamp=datetime(2026, 9, 30, tzinfo=UTC),
        request_payload={"symbol": "EURUSD"},
    )
    store.record_position(
        {
            "ticket": 11,
            "intent_id": intent.intent_id,
            "symbol": "EURUSD",
            "direction": "LONG",
            "volume": 0.01,
            "price_open": 1.1001,
            "stop_loss": 1.099,
            "take_profit": 1.102,
            "state": "OPEN",
        }
    )

    assert store.snapshot()["orders"] == 1
    assert store.snapshot()["results"] == 1
    assert store.read_owned_positions("EURUSD")[0]["ticket"] == 11


def test_demo_store_rejects_non_demo_execution_mode(tmp_path: Path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    store.initialize()
    intent = _intent()
    object.__setattr__(intent, "execution_mode", ExecutionMode.SHADOW)

    with pytest.raises(ValueError, match="DEMO"):
        store.record_order_intent("run-1", intent)


def test_demo_store_does_not_reopen_closed_position_from_delayed_telemetry(tmp_path: Path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    store.initialize()
    store.start_run("run-1", git_commit="abc123")
    intent = _intent()
    store.record_order_intent("run-1", intent)
    position = {
        "ticket": 11,
        "intent_id": intent.intent_id,
        "symbol": "EURUSD",
        "direction": "LONG",
        "volume": 0.01,
        "price_open": 1.1001,
        "stop_loss": 1.099,
        "take_profit": 1.102,
        "state": "OPEN",
    }

    store.record_position(position)
    store.record_position({**position, "state": "CLOSED"})
    store.record_position({**position, "state": "OPEN", "current_pnl_points": 3.0})

    assert store.read_owned_positions("EURUSD")[0]["state"] == "CLOSED"


def _record_broker_submission(store: DemoExecutionStore, *, sent: bool = True) -> None:
    store.initialize()
    store.start_run("run-1", git_commit="abc123")
    intent = _intent()
    store.record_order_intent("run-1", intent)
    store.record_order_result(
        intent_id=intent.intent_id,
        classification="REJECTED",
        retcode=10030,
        order_ticket=None,
        deal_ticket=None,
        fill_price=None,
        fill_volume=None,
        broker_comment="unsupported filling mode",
        broker_timestamp=None,
        request_payload={"symbol": "EURUSD"},
        broker_order_sent=sent,
    )


def test_order_rate_recovery_requires_exact_reason_elapsed_cooldown_and_preserves_risk(tmp_path: Path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    store.initialize()
    now = datetime.now(UTC)
    store.set_circuit_state(
        status="HFT_ORDER_RATE_CIRCUIT_OPEN",
        daily_start_equity=1234.5,
        daily_loss_fraction=0.012,
        consecutive_losses=2,
        reason="bounded broker submission rate exceeded",
    )

    assert store.recover_expired_order_rate_circuit(
        now=now + timedelta(seconds=59), cooldown_seconds=60, max_submissions=20
    ) is False
    assert store.recover_expired_order_rate_circuit(
        now=now + timedelta(seconds=61), cooldown_seconds=60, max_submissions=20
    ) is True

    state = store.read_circuit_state()
    assert state["status"] == "READY"
    assert state["reason"] is None
    assert state["daily_start_equity"] == 1234.5
    assert state["daily_loss_fraction"] == pytest.approx(0.012)
    assert state["consecutive_losses"] == 2


def test_order_rate_recovery_fails_closed_for_recent_sends_or_other_circuits(tmp_path: Path):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    store.initialize()
    now = datetime.now(UTC)
    store.set_circuit_state(
        status="HFT_ORDER_RATE_CIRCUIT_OPEN",
        reason="bounded broker submission rate exceeded",
    )
    _record_broker_submission(store, sent=True)

    assert store.count_recent_broker_submissions(now - timedelta(seconds=60)) == 1
    assert store.recover_expired_order_rate_circuit(
        now=now + timedelta(seconds=60), cooldown_seconds=60, max_submissions=1
    ) is False

    store.set_circuit_state(status="DAILY_LOSS_LIMIT", reason="daily loss limit")
    assert store.recover_expired_order_rate_circuit(
        now=now + timedelta(hours=1), cooldown_seconds=60, max_submissions=20
    ) is False
    assert store.read_circuit_state()["status"] == "DAILY_LOSS_LIMIT"
