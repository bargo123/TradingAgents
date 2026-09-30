from datetime import datetime, timezone
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
