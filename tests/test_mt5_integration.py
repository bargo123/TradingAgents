"""Opt-in live smoke coverage for the read-only MT5 provider."""

from __future__ import annotations

import os

import pytest

from tradingagents.dataflows.mt5.errors import (
    Mt5AccountDisconnectedError,
    Mt5DependencyError,
    Mt5InitializationError,
)
from tradingagents.dataflows.mt5.provider import MT5Provider


@pytest.mark.integration
def test_live_mt5_read_only_snapshot() -> None:
    """Read one live normalized market snapshot from the local terminal."""
    if os.getenv("RUN_MT5_INTEGRATION") != "1":
        pytest.skip("set RUN_MT5_INTEGRATION=1 to use the local terminal")

    provider = MT5Provider()
    try:
        try:
            provider.initialize()
            snapshot = provider.get_market_snapshot(
                os.getenv("MT5_TEST_SYMBOL", "EURUSD"), count=2
            )
        except (Mt5DependencyError, Mt5InitializationError, Mt5AccountDisconnectedError) as exc:
            pytest.skip(f"MT5 unavailable: {exc}")
        assert snapshot.bid > 0
        assert snapshot.ask >= snapshot.bid
        assert snapshot.m5_candles
        assert snapshot.account.balance >= 0
    finally:
        provider.shutdown()
