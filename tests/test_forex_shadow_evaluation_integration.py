"""Opt-in real-terminal proof for read-only historical shadow evaluation."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.dataflows.mt5.errors import (
    Mt5AccountDisconnectedError,
    Mt5DependencyError,
    Mt5InitializationError,
    Mt5ProviderError,
)
from tradingagents.dataflows.mt5.provider import MT5Provider


@pytest.mark.integration
def test_real_mt5_historical_ticks_are_read_only() -> None:
    if os.getenv("RUN_MT5_INTEGRATION") != "1":
        pytest.skip("set RUN_MT5_INTEGRATION=1 to use the local MT5 terminal")
    pytest.importorskip("MetaTrader5")

    provider = MT5Provider(terminal_path=os.getenv("MT5_TERMINAL_PATH"))
    symbol = os.getenv("MT5_TEST_SYMBOL", "EURUSD")
    try:
        try:
            provider.initialize()
        except (Mt5DependencyError, Mt5InitializationError, Mt5AccountDisconnectedError) as exc:
            pytest.skip(f"MT5 unavailable: {exc}")
        resolved = provider.ensure_symbol(symbol)
        before_positions = provider.get_positions(resolved)
        before_orders = provider.get_orders(resolved)
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=5)
        try:
            ticks = provider.get_ticks_range(resolved, start, end)
        except Mt5ProviderError as exc:
            pytest.fail(f"historical MT5 read failed: {exc}")
        after_positions = provider.get_positions(resolved)
        after_orders = provider.get_orders(resolved)

        assert all(tick.timestamp.tzinfo == timezone.utc for tick in ticks)
        assert all(tick.ask >= tick.bid for tick in ticks)
        assert before_positions == after_positions
        assert before_orders == after_orders
        for forbidden in (
            "order_send",
            "buy",
            "sell",
            "open_position",
            "close_position",
            "modify_position",
            "modify_order",
            "place_order",
            "cancel_order",
        ):
            assert not hasattr(provider, forbidden)
    finally:
        provider.shutdown()
