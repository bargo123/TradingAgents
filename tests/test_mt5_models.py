from datetime import datetime, timezone
from math import nan

import pytest

from tradingagents.dataflows.mt5.clock import Mt5BrokerClock
from tradingagents.dataflows.mt5.models import (
    ForexMarketSnapshot,
    Mt5AccountInfo,
    Mt5Bar,
    Mt5Order,
    Mt5Position,
    Mt5Spread,
    Mt5SymbolInfo,
    Mt5Tick,
)
from tradingagents.dataflows.mt5.timeframes import SUPPORTED_TIMEFRAMES, TIMEFRAME_ATTRIBUTES


@pytest.mark.unit
def test_supported_timeframes_have_mt5_constant_names():
    assert SUPPORTED_TIMEFRAMES == ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
    assert TIMEFRAME_ATTRIBUTES["M5"] == "TIMEFRAME_M5"
    assert TIMEFRAME_ATTRIBUTES["D1"] == "TIMEFRAME_D1"


@pytest.mark.unit
def test_symbol_info_carries_digits_and_point():
    info = Mt5SymbolInfo(name="EURUSD.a", digits=5, point=0.00001)
    assert info.name == "EURUSD.a"
    assert info.digits == 5
    assert info.point == 0.00001


@pytest.mark.unit
def test_snapshot_uses_immutable_candle_tuples_and_utc_timestamp():
    timestamp = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    bar = Mt5Bar(timestamp, 1.0, 1.1, 0.9, 1.05, 10, 2, 10)
    tick = Mt5Tick("EURUSD.a", timestamp, 1.04999, 1.05001)
    snapshot = ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSD.a",
        bid=tick.bid,
        ask=tick.ask,
        spread=0.00002,
        spread_points=2.0,
        m1_candles=(bar,),
        m5_candles=(bar,),
        m15_candles=(bar,),
        h1_candles=(bar,),
        account=None,
        positions=(),
        broker_clock=Mt5BrokerClock(
            offset_seconds=0,
            status="CALIBRATED",
            calibrated_at_utc=timestamp,
            server="Test",
            symbol="EURUSD.a",
            sample_count=1,
            max_residual_seconds=0.0,
            source="TEST",
        ),
    )
    assert snapshot.timestamp.tzinfo == timezone.utc
    assert isinstance(snapshot.m5_candles, tuple)
    assert snapshot.m5_candles[0].close == 1.05
    assert snapshot.broker_clock is not None


@pytest.mark.unit
def test_market_models_reject_non_finite_or_inconsistent_quotes():
    timestamp = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="finite"):
        Mt5Tick("EURUSD.a", timestamp, nan, 1.05001)
    with pytest.raises(ValueError, match="ask"):
        Mt5Tick("EURUSD.a", timestamp, 1.05002, 1.05001)
    with pytest.raises(ValueError, match="OHLC"):
        Mt5Bar(timestamp, 1.0, 0.9, 0.8, 1.05, 10)
    with pytest.raises(ValueError, match="spread"):
        Mt5Spread("EURUSD.a", 1.05, 1.05001, 0.0, 1.0, timestamp)
    with pytest.raises(ValueError, match="finite"):
        Mt5AccountInfo(balance=nan)
    with pytest.raises(ValueError, match="finite"):
        Mt5Position(ticket=1, symbol="EURUSD.a", volume=nan)
    with pytest.raises(ValueError, match="finite"):
        Mt5Order(ticket=1, symbol="EURUSD.a", price_open=nan)


@pytest.mark.unit
def test_snapshot_normalizes_sequences_and_rejects_quote_mismatch():
    timestamp = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    bar = Mt5Bar(timestamp, 1.0, 1.1, 0.9, 1.05, 10, 2, 10)
    with pytest.raises(ValueError, match="spread"):
        ForexMarketSnapshot(
            timestamp=timestamp,
            symbol="EURUSD.a",
            bid=1.04999,
            ask=1.05001,
            spread=0.00001,
            spread_points=1.0,
            m1_candles=[bar],
            m5_candles=[bar],
            m15_candles=[bar],
            h1_candles=[bar],
            account=None,
            positions=[],
        )

    snapshot = ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSD.a",
        bid=1.04999,
        ask=1.05001,
        spread=0.00002,
        spread_points=2.0,
        m1_candles=[bar],
        m5_candles=[bar],
        m15_candles=[bar],
        h1_candles=[bar],
        account=None,
        positions=[],
    )
    assert isinstance(snapshot.m1_candles, tuple)
    assert isinstance(snapshot.positions, tuple)
