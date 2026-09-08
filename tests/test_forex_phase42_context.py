from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tradingagents.dataflows.mt5.models import (
    ForexMarketSnapshot,
    Mt5Bar,
    Mt5SymbolInfo,
)
from tradingagents.forex.context import build_forex_market_context, snapshot_to_dict
from tradingagents.forex.profile import (
    INTRADAY_PROFILE,
    calculate_timeframe_features,
    resolve_forex_profile,
)
from tradingagents.graph.propagation import Propagator


def _bars() -> tuple[Mt5Bar, ...]:
    return tuple(
        Mt5Bar(
            timestamp=datetime(2026, 9, 8, index, tzinfo=timezone.utc),
            open=open_price,
            high=high,
            low=low,
            close=close,
            tick_volume=100 + index,
        )
        for index, (open_price, high, low, close) in enumerate(
            (
                (1.1000, 1.1020, 1.0990, 1.1010),
                (1.1010, 1.1040, 1.1000, 1.1030),
                (1.1030, 1.1050, 1.1020, 1.1040),
            )
        )
    )


def _snapshot(bars: tuple[Mt5Bar, ...]) -> ForexMarketSnapshot:
    timestamp = bars[-1].timestamp
    return ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSD",
        bid=1.1039,
        ask=1.1040,
        spread=0.0001,
        spread_points=1.0,
        m1_candles=bars,
        m5_candles=bars,
        m15_candles=bars,
        h1_candles=bars,
        account=None,
        positions=(),
        symbol_info=Mt5SymbolInfo(name="EURUSD", digits=5, point=0.00001),
    )


def test_intraday_profile_is_explicit_and_fail_closed() -> None:
    assert resolve_forex_profile("INTRADAY") == INTRADAY_PROFILE
    assert INTRADAY_PROFILE.horizon_label == "minutes to hours"
    assert INTRADAY_PROFILE.valid_for_seconds == 3600
    with pytest.raises(ValueError, match="unknown forex analysis profile"):
        resolve_forex_profile("LONG_TERM")


def test_multi_bar_features_are_deterministic() -> None:
    features = calculate_timeframe_features(_bars())

    assert features["candle_count"] == 3
    assert features["latest"]["open"] == 1.103
    assert features["latest"]["high"] == 1.105
    assert features["latest"]["low"] == 1.102
    assert features["latest"]["close"] == 1.104
    assert features["return_over_bars"] == pytest.approx((1.104 - 1.1) / 1.1)
    assert features["recent_high"] == 1.105
    assert features["recent_low"] == 1.099
    assert features["range"] == pytest.approx(0.006)
    assert features["direction"] == "UP"
    assert features["average_true_range"] == pytest.approx((0.003 + 0.004 + 0.003) / 3)
    assert features["close_position"] == pytest.approx((1.104 - 1.099) / 0.006)


def test_one_candle_is_not_described_as_a_trend() -> None:
    features = calculate_timeframe_features((_bars()[0],))

    assert features["candle_count"] == 1
    assert features["return_over_bars"] is None
    assert features["direction"] == "INSUFFICIENT_DATA"


def test_forex_context_contains_profile_features_and_macro_uncertainty() -> None:
    snapshot = _snapshot(_bars())
    context = build_forex_market_context(snapshot)

    assert "FOREX ANALYSIS PROFILE: INTRADAY" in context
    assert "DECISION HORIZON: minutes to hours" in context
    assert "VALID FOR SECONDS: 3600" in context
    assert "MACRO/EVENT DATA UNAVAILABLE" in context
    assert "M1: count=3" in context
    assert "return_over_bars=" in context
    assert "direction=UP" in context
    assert "average_true_range=" in context
    assert "close_position=" in context
    assert "1.1000" not in context  # raw bars are summarized, not dumped
    assert len(context) < 12000


def test_snapshot_serialization_exposes_features_without_changing_bar_bound() -> None:
    payload = snapshot_to_dict(_snapshot(_bars()))

    assert payload["features"]["M15"]["candle_count"] == 3
    assert payload["features"]["H1"]["direction"] == "UP"
    assert len(payload["candles"]["M1"]) == 3


def test_forex_profile_is_carried_in_initial_state_without_stock_mode_change() -> None:
    state = Propagator().create_initial_state(
        "EURUSD",
        "2026-09-08",
        asset_type="forex",
        market_data_mode="forex_mt5",
        forex_analysis_profile="INTRADAY",
    )

    assert state["forex_analysis_profile"] == "INTRADAY"
    stock_state = Propagator().create_initial_state("AAPL", "2026-09-08")
    assert stock_state["market_data_mode"] == "stock"
    assert "forex_analysis_profile" not in stock_state
