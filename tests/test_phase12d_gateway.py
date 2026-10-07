import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradingagents.forex.hft.demo_gateway import (
    DemoExecutionForbidden,
    VerifiedDemoExecutionGateway,
)
from tradingagents.forex.hft.demo_models import DemoOrderIntent
from tradingagents.forex.hft.demo_store import DemoExecutionStore

UTC = timezone.utc


def _intent(**overrides):
    values = {
        "intent_id": "intent-1",
        "plan_id": "plan-1",
        "source_decision_id": "decision-1",
        "source_run_id": "source-run-1",
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
        "account_trade_mode": 0,
    }
    values.update(overrides)
    return DemoOrderIntent(**values)


class _Provider:
    def __init__(self, modes):
        self.modes = list(modes)
        self.calls = 0

    def get_account_info(self):
        index = min(self.calls, len(self.modes) - 1)
        self.calls += 1
        return SimpleNamespace(
            login=7,
            server="Broker-Demo",
            currency="USD",
            trade_mode=self.modes[index],
            balance=1000.0,
            equity=1000.0,
            margin=0.0,
            free_margin=1000.0,
        )

    def get_terminal_info(self):
        return SimpleNamespace(company="Broker", connected=True)

    def get_positions(self, symbol=None):
        return (SimpleNamespace(ticket=11, symbol=symbol or "EURUSD", magic=12012012, comment="TradingAgents-P12D-DEMO"),)


class _Api:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_RETURN = 2
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE = 10009

    def __init__(self):
        self.order_calls = []

    def order_send(self, request):
        self.order_calls.append(request)
        return SimpleNamespace(
            retcode=10009,
            order=11,
            deal=12,
            price=1.1002,
            volume=0.01,
            comment="done",
            time_msc=1_790_000_000_000,
        )


def _gateway(tmp_path: Path, *, modes=(0, 0), api=None):
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    store.initialize()
    store.start_run("demo-run", git_commit="abc123")
    api = api or _Api()
    gateway = VerifiedDemoExecutionGateway(
        _Provider(modes),
        api,
        store,
        run_id="demo-run",
        demo_trade_mode=0,
        magic=12012012,
        comment="TradingAgents-P12D-DEMO",
    )
    return gateway, api


def test_demo_account_can_submit_and_persists_result(tmp_path: Path):
    gateway, api = _gateway(tmp_path)

    result = gateway.submit(_intent())

    assert result.classification == "FILLED"
    assert result.broker_order_sent is True
    assert result.real_money is False
    assert len(api.order_calls) == 1
    assert api.order_calls[0]["comment"] == "TradingAgents-P12D-DEMO"


def test_order_intent_persists_additive_experience_provenance(tmp_path: Path):
    gateway, _ = _gateway(tmp_path)
    intent = _intent(provenance={"strategy_version": "hft-v1", "feature_tick_key": "tick-1"})

    gateway.submit(intent)

    stored = gateway.store.read_order_intent(intent.intent_id)
    assert stored is not None
    payload = __import__("json").loads(stored["request_payload"])
    assert payload["provenance"]["strategy_version"] == "hft-v1"


def test_gateway_preserves_broker_position_identity_when_order_differs(tmp_path: Path):
    class PositionApi(_Api):
        def order_send(self, request):
            self.order_calls.append(request)
            return SimpleNamespace(
                retcode=10009,
                order=11,
                deal=12,
                position=99,
                price=1.1002,
                volume=0.01,
                comment="done",
            )

    gateway, _ = _gateway(tmp_path, api=PositionApi())
    result = gateway.submit(_intent())

    assert result.order_ticket == 11
    assert result.position_ticket == 99
    assert result.request_payload["broker_position_ticket"] == 99


def test_gateway_uses_broker_advertised_filling_mode(tmp_path: Path):
    gateway, api = _gateway(tmp_path)
    gateway.provider.get_symbol_filling_mode = lambda symbol: 1

    result = gateway.submit(_intent())

    assert result.classification == "FILLED"
    assert api.order_calls[0]["type_filling"] == api.ORDER_FILLING_FOK


def test_gateway_prefers_ioc_when_broker_advertises_it(tmp_path: Path):
    gateway, api = _gateway(tmp_path)
    gateway.provider.get_symbol_filling_mode = lambda symbol: 2

    result = gateway.submit(_intent())

    assert result.classification == "FILLED"
    assert api.order_calls[0]["type_filling"] == api.ORDER_FILLING_IOC


@pytest.mark.parametrize("trade_mode", [1, 2, None])
def test_non_demo_account_fails_before_order_send(tmp_path: Path, trade_mode):
    gateway, api = _gateway(tmp_path, modes=(trade_mode,))

    with pytest.raises(DemoExecutionForbidden, match="DEMO_EXECUTION_FORBIDDEN"):
        gateway.submit(_intent())

    assert api.order_calls == []


def test_account_preflight_can_reject_before_gateway_construction(tmp_path: Path):
    provider = _Provider((2,))
    account = VerifiedDemoExecutionGateway.read_account(provider)

    with pytest.raises(DemoExecutionForbidden, match="authoritative trade_mode"):
        VerifiedDemoExecutionGateway.require_demo_account(account, 0)

    assert not (tmp_path / "demo.sqlite3").exists()


def test_demo_server_name_does_not_override_real_trade_mode(tmp_path: Path):
    gateway, api = _gateway(tmp_path, modes=(1,))

    with pytest.raises(DemoExecutionForbidden):
        gateway.submit(_intent())

    assert api.order_calls == []


def test_account_mode_change_before_send_blocks_second_validation(tmp_path: Path):
    gateway, api = _gateway(tmp_path, modes=(0, 1))

    with pytest.raises(DemoExecutionForbidden, match="DEMO_EXECUTION_FORBIDDEN"):
        gateway.submit(_intent())

    assert api.order_calls == []


def test_close_requires_owned_broker_position_and_uses_opposite_order(tmp_path: Path):
    gateway, api = _gateway(tmp_path)
    entry = gateway.submit(_intent())
    gateway.store.record_position(
        {
            "ticket": entry.order_ticket,
            "intent_id": "intent-1",
            "symbol": "EURUSD",
            "direction": "LONG",
            "volume": 0.01,
            "price_open": 1.1002,
            "stop_loss": 1.0991,
            "take_profit": 1.1021,
            "state": "OPEN",
        }
    )
    exit_intent = _intent(intent_id="exit-1", direction="SHORT", requested_price=1.1, stop_loss=1.099, take_profit=1.102)

    result = gateway.submit(exit_intent, position_ticket=entry.order_ticket, exit_reason="TAKE_PROFIT")

    assert result.classification == "FILLED"
    assert api.order_calls[-1]["position"] == entry.order_ticket
    assert api.order_calls[-1]["type"] == api.ORDER_TYPE_SELL
    with sqlite3.connect(tmp_path / "demo.sqlite3") as db:
        payload = json.loads(db.execute("SELECT payload_json FROM demo_exits ORDER BY observed_at DESC LIMIT 1").fetchone()[0])
    assert payload["position_ticket"] == entry.order_ticket
    assert payload["fill_price"] == result.fill_price
    assert "provenance" in payload


def test_successful_close_persists_net_pnl_from_all_position_deals(tmp_path: Path):
    gateway, _ = _gateway(tmp_path)
    entry = gateway.submit(_intent())
    gateway.store.record_position(
        {
            "ticket": entry.order_ticket,
            "position_ticket": entry.order_ticket,
            "intent_id": "intent-1",
            "symbol": "EURUSD",
            "direction": "LONG",
            "volume": 0.01,
            "price_open": 1.1002,
            "stop_loss": 1.0991,
            "take_profit": 1.1021,
            "state": "OPEN",
        }
    )
    history_calls = []
    gateway.provider.get_history_deals = lambda ticket: (
        history_calls.append(ticket)
        or (
            {
                "ticket": 21,
                "position_id": entry.order_ticket,
                "entry": 0,
                "profit": 0.0,
                "commission": -0.15,
                "swap": 0.0,
                "fee": 0.01,
            },
            {
                "ticket": 22,
                "position_id": entry.order_ticket,
                "entry": 1,
                "profit": 0.40,
                "commission": -0.15,
                "swap": -0.01,
                "fee": 0.01,
            },
        )
    )

    result = gateway.submit(
        _intent(intent_id="exit-1", direction="SHORT", requested_price=1.1),
        position_ticket=entry.order_ticket,
        exit_reason="TAKE_PROFIT",
    )

    assert result.classification == "FILLED"
    assert history_calls == [entry.order_ticket]
    with sqlite3.connect(tmp_path / "demo.sqlite3") as db:
        row = db.execute(
            "SELECT realized_pnl, payload_json FROM demo_exits ORDER BY observed_at DESC LIMIT 1"
        ).fetchone()
    assert row[0] == pytest.approx(0.11)
    payload = json.loads(row[1])
    assert payload["realized_pnl"] == pytest.approx(0.11)
    assert payload["realized_pnl_status"] == "AVAILABLE"
    assert payload["realized_pnl_source"] == "MT5_POSITION_DEALS"


def test_submit_exposes_last_exit_realized_pnl_for_circuit_breaker_wiring(tmp_path: Path):
    """The runtime's loss-streak circuit breaker only has a chance to work if a
    closed trade's realized P&L is actually observable right after submit();
    this was previously computed and persisted to the store but never surfaced
    anywhere the runtime could feed it back into DemoCircuitBreaker."""
    gateway, _ = _gateway(tmp_path)
    assert gateway.last_exit_realized_pnl is None
    entry = gateway.submit(_intent())
    assert gateway.last_exit_realized_pnl is None
    gateway.store.record_position(
        {
            "ticket": entry.order_ticket,
            "position_ticket": entry.order_ticket,
            "intent_id": "intent-1",
            "symbol": "EURUSD",
            "direction": "LONG",
            "volume": 0.01,
            "price_open": 1.1002,
            "stop_loss": 1.0991,
            "take_profit": 1.1021,
            "state": "OPEN",
        }
    )
    gateway.provider.get_history_deals = lambda ticket: (
        {
            "ticket": 21,
            "position_id": entry.order_ticket,
            "entry": 0,
            "profit": 0.0,
            "commission": -0.15,
            "swap": 0.0,
            "fee": 0.0,
        },
        {
            "ticket": 22,
            "position_id": entry.order_ticket,
            "entry": 1,
            "profit": -0.80,
            "commission": -0.15,
            "swap": -0.01,
            "fee": 0.0,
        },
    )

    result = gateway.submit(
        _intent(intent_id="exit-1", direction="SHORT", requested_price=1.1),
        position_ticket=entry.order_ticket,
        exit_reason="EMERGENCY_BROKER_STOP",
    )

    assert result.classification == "FILLED"
    assert gateway.last_exit_realized_pnl == pytest.approx(-1.11)


def test_close_keeps_realized_pnl_unknown_when_broker_history_is_unavailable(tmp_path: Path):
    gateway, _ = _gateway(tmp_path)
    entry = gateway.submit(_intent())
    gateway.store.record_position(
        {
            "ticket": entry.order_ticket,
            "position_ticket": entry.order_ticket,
            "intent_id": "intent-1",
            "symbol": "EURUSD",
            "direction": "LONG",
            "volume": 0.01,
            "price_open": 1.1002,
            "stop_loss": 1.0991,
            "take_profit": 1.1021,
            "state": "OPEN",
        }
    )

    def history_failure(_ticket):
        raise RuntimeError("private provider details must not be persisted")

    gateway.provider.get_history_deals = history_failure

    result = gateway.submit(
        _intent(intent_id="exit-1", direction="SHORT", requested_price=1.1),
        position_ticket=entry.order_ticket,
        exit_reason="TAKE_PROFIT",
    )

    assert result.classification == "FILLED"
    with sqlite3.connect(tmp_path / "demo.sqlite3") as db:
        row = db.execute(
            "SELECT realized_pnl, payload_json FROM demo_exits ORDER BY observed_at DESC LIMIT 1"
        ).fetchone()
    assert row[0] is None
    payload = json.loads(row[1])
    assert payload["realized_pnl"] is None
    assert payload["realized_pnl_status"] == "HISTORY_READ_FAILED"
    assert payload["realized_pnl_error_type"] == "RuntimeError"
    assert "private provider details" not in row[1]


def test_close_of_unowned_position_is_rejected(tmp_path: Path):
    gateway, api = _gateway(tmp_path)

    with pytest.raises(DemoExecutionForbidden, match="owned"):
        gateway.submit(_intent(intent_id="exit-1", direction="SHORT"), position_ticket=999, exit_reason="STOP_LOSS")

    assert api.order_calls == []


def test_close_uses_reconciled_position_ticket_when_order_ticket_differs(tmp_path: Path):
    gateway, api = _gateway(tmp_path)
    gateway.provider.get_positions = lambda symbol=None: (
        SimpleNamespace(
            ticket=99,
            symbol=symbol or "EURUSD",
            magic=12012012,
            comment="TradingAgents-P12D-DEMO",
        ),
    )
    gateway.store.record_order_intent("demo-run", _intent())
    gateway.store.record_position(
        {
            "ticket": 11,
            "position_ticket": 99,
            "intent_id": "intent-1",
            "symbol": "EURUSD",
            "direction": "LONG",
            "volume": 0.01,
            "price_open": 1.1002,
            "stop_loss": 1.0991,
            "take_profit": 1.1021,
            "state": "OPEN",
        }
    )

    result = gateway.submit(
        _intent(intent_id="exit-1", direction="SHORT"),
        position_ticket=99,
        exit_reason="TAKE_PROFIT",
    )

    assert result.classification == "FILLED"
    assert api.order_calls[-1]["position"] == 99
    assert gateway.store.read_owned_positions("EURUSD")[0]["state"] == "CLOSED"
