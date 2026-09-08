from datetime import datetime, timezone

import pytest

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5Bar, Mt5SymbolInfo, Mt5Tick
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
    )
    assert snapshot.timestamp.tzinfo == timezone.utc
    assert isinstance(snapshot.m5_candles, tuple)
    assert snapshot.m5_candles[0].close == 1.05
