"""RED tests defining the read-only MT5 smoke-report contract."""

from datetime import datetime, timezone

import pytest

from scripts.test_mt5_connection import render_report
from tradingagents.dataflows.mt5.models import (
    ForexMarketSnapshot,
    Mt5AccountInfo,
    Mt5Bar,
    Mt5Position,
)


@pytest.fixture
def snapshot_fixture():
    timestamp = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    def bar(close):
        return Mt5Bar(
            timestamp=timestamp,
            open=close - 0.00020,
            high=close + 0.00010,
            low=close - 0.00030,
            close=close,
            tick_volume=120,
            spread=2,
            real_volume=120,
        )

    m1 = (bar(1.10020), bar(1.10020))
    m5 = (bar(1.10040), bar(1.10040))
    m15 = (bar(1.10060), bar(1.10060))
    h1 = (bar(1.10080), bar(1.10080))
    return ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSD.a",
        bid=1.10000,
        ask=1.10020,
        spread=0.00020,
        spread_points=20.0,
        m1_candles=m1,
        m5_candles=m5,
        m15_candles=m15,
        h1_candles=h1,
        account=Mt5AccountInfo(
            login=123456,
            server="Fake-Demo",
            currency="USD",
            balance=10_000.0,
            equity=9_900.0,
            profit=-100.0,
            margin=400.0,
            free_margin=9_500.0,
            leverage=100,
        ),
        positions=(
            Mt5Position(
                ticket=101,
                symbol="EURUSD.a",
                type=0,
                volume=0.10,
                price_open=1.09980,
                price_current=1.10000,
                profit=2.0,
                time=timestamp,
            ),
        ),
    )


@pytest.mark.unit
def test_render_report_is_secret_free(snapshot_fixture):
    output = render_report(
        snapshot_fixture, terminal_name="MetaTrader 5", requested_symbol="EURUSD"
    )
    assert "MT5 CONNECTED" in output
    assert "Requested symbol: EURUSD" in output
    assert "Resolved symbol: EURUSD.a" in output
    assert "Bid: 1.1" in output and "Ask: 1.1002" in output
    assert "Last candle:" in output
    assert "C=1.10040" in output
    assert "Balance:" in output and "Free margin:" in output
    assert "M1:" not in output and "M15:" not in output and "H1:" not in output
    output_lower = output.lower()
    for credential in ("password", "token", "secret", "api key"):
        assert credential not in output_lower
    for mutation in (
        "order_send",
        "open_position",
        "close_position",
        "modify_position",
        "modify_order",
        "place_order",
        "cancel_order",
    ):
        assert mutation not in output_lower
    for raw_repr in ("ForexMarketSnapshot(", "Mt5AccountInfo(", "Mt5Bar(", "Mt5Position("):
        assert raw_repr not in output
