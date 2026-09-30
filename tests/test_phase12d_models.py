from datetime import datetime, timezone

import pytest

from tradingagents.forex.hft.demo_models import (
    DemoAccountSnapshot,
    DemoOrderIntent,
    DemoOrderResult,
    ExecutionMode,
)

UTC = timezone.utc


def _intent(**overrides):
    values = {
        "intent_id": "intent-1",
        "plan_id": "plan-1",
        "source_decision_id": "decision-1",
        "source_run_id": "run-1",
        "strategy_id": "strategic_shadow",
        "symbol": "EURUSD",
        "direction": "LONG",
        "volume": 0.01,
        "requested_price": 1.1001,
        "stop_loss": 1.0991,
        "take_profit": 1.1021,
        "deviation_points": 20,
        "created_at": datetime(2026, 9, 30, tzinfo=UTC),
        "git_commit": "abc123",
        "execution_mode": ExecutionMode.DEMO,
        "account_trade_mode": 0,
    }
    values.update(overrides)
    return DemoOrderIntent(**values)


def test_demo_order_intent_is_bounded_and_real_money_false():
    intent = _intent()

    assert intent.execution_mode is ExecutionMode.DEMO
    assert intent.real_money is False
    assert intent.direction == "LONG"


@pytest.mark.parametrize("direction", ["BUY", "SELL", "NONE", "BOTH"])
def test_demo_order_intent_rejects_non_directional_plan_values(direction):
    with pytest.raises(ValueError, match="direction"):
        _intent(direction=direction)


def test_demo_order_result_requires_closed_execution_classification():
    result = DemoOrderResult(
        intent_id="intent-1",
        classification="FILLED",
        retcode=10009,
        order_ticket=11,
        deal_ticket=12,
        fill_price=1.1002,
        fill_volume=0.01,
        broker_comment="done",
        broker_timestamp=datetime(2026, 9, 30, tzinfo=UTC),
        request_payload={"symbol": "EURUSD"},
        execution_mode=ExecutionMode.DEMO,
        broker_order_sent=True,
    )

    assert result.real_money is False
    assert result.broker_order_sent is True


def test_demo_account_snapshot_requires_authoritative_trade_mode():
    account = DemoAccountSnapshot(
        login=123,
        server="Broker-Demo",
        company="Broker",
        currency="USD",
        trade_mode=0,
    )

    assert account.trade_mode == 0
    assert account.is_demo(0) is True
    assert account.is_demo(1) is False
