from datetime import datetime, timezone

import pytest

from tradingagents.forex.hft.execution import ShadowFillEngine, ShadowPositionLedger
from tradingagents.forex.hft.models import FastAction, PositionState, Tick

UTC = timezone.utc


def _tick(bid=1.1, ask=1.1001, second=0):
    return Tick("EURUSD", datetime(2026, 1, 1, 12, 0, second, tzinfo=UTC), bid, ask)


def test_fill_semantics_use_ask_for_buy_and_bid_for_sell_with_slippage():
    engine = ShadowFillEngine(slippage_points=2, latency_ms=5)
    buy = engine.fill(FastAction.ENTER_LONG, _tick(), size=1000)
    sell = engine.fill(FastAction.ENTER_SHORT, _tick(), size=1000)
    assert buy.price == pytest.approx(1.10012)
    assert sell.price == pytest.approx(1.09998)
    assert buy.executed is False
    assert sell.executed is False


def test_exit_fill_requires_position_side_instead_of_midpoint_fallback():
    engine = ShadowFillEngine()
    with pytest.raises(ValueError, match="position_state"):
        engine.fill(FastAction.EXIT, _tick(), size=1000)


def test_position_ledger_tracks_open_close_and_excursions():
    ledger = ShadowPositionLedger()
    entry = ledger.open(_tick(), FastAction.ENTER_LONG, size=1000, stop=1.099, target=1.102, strategy_id="test", entry_price=1.10012)
    assert entry.state is PositionState.LONG
    assert entry.entry_price == pytest.approx(1.10012)
    ledger.observe(_tick(bid=1.101, ask=1.1011, second=1))
    closed = ledger.close(_tick(bid=1.1008, ask=1.1009, second=2), reason="TAKE_PROFIT", exit_price=1.10075)
    assert closed.state is PositionState.CLOSED
    assert closed.exit_price == pytest.approx(1.10075)
    assert closed.mfe > 0
    assert closed.holding_seconds == pytest.approx(2.0)
    with pytest.raises(ValueError, match="FLAT"):
        ledger.close(_tick(second=3), reason="duplicate")


def test_position_ledger_restores_persisted_open_position_without_execution():
    ledger = ShadowPositionLedger()
    restored = ledger.restore(
        {
            "position_id": "p1",
            "symbol": "EURUSD",
            "state": "LONG",
            "direction": "LONG",
            "size": 1.0,
            "entry_price": 1.1001,
            "entry_timestamp": "2026-09-30T09:00:00+00:00",
            "strategy_id": "strategic_shadow",
            "stop_price": None,
            "target_price": None,
            "executed": False,
        }
    )
    assert restored.position_id == "p1"
    assert restored.state is PositionState.LONG
    assert restored.executed is False
