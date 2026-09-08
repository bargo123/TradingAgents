from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool

from tradingagents.dataflows.mt5.models import (
    ForexMarketSnapshot,
    Mt5AccountInfo,
    Mt5Bar,
    Mt5Position,
    Mt5Spread,
    Mt5SymbolInfo,
)
from tradingagents.forex.context import build_forex_market_context, snapshot_to_dict
from tradingagents.forex.tools import MT5ToolAdapter


@pytest.fixture
def fake_snapshot() -> ForexMarketSnapshot:
    timestamp = datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc)
    bars = (
        Mt5Bar(
            timestamp=timestamp,
            open=1.10000,
            high=1.10040,
            low=1.09980,
            close=1.10020,
            tick_volume=120,
            spread=2,
            real_volume=120,
        ),
    )
    return ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSD",
        bid=1.10000,
        ask=1.10020,
        spread=0.00020,
        spread_points=20.0,
        symbol_info=Mt5SymbolInfo(
            name="EURUSD",
            description="Euro / US Dollar",
            digits=5,
            point=0.00001,
            visible=True,
            trade_mode=0,
            currency_base="EUR",
            currency_profit="USD",
        ),
        m1_candles=bars,
        m5_candles=bars,
        m15_candles=bars,
        h1_candles=bars,
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
                symbol="EURUSD",
                type=0,
                volume=0.10,
                price_open=1.10000,
                price_current=1.10020,
                profit=2.0,
                time=timestamp,
            ),
        ),
    )


class FakeProvider:
    def __init__(self, snapshot: ForexMarketSnapshot) -> None:
        self.snapshot = snapshot
        self.market_snapshot_calls = 0
        self.tick_calls = 0
        self.bars_calls = 0
        self.account_calls = 0
        self.positions_calls = 0
        self.spread_calls = 0

    def get_market_snapshot(self, symbol: str, count: int = 100) -> ForexMarketSnapshot:
        self.market_snapshot_calls += 1
        return self.snapshot

    def get_tick(self, symbol: str):
        self.tick_calls += 1
        return SimpleNamespace(
            symbol=symbol,
            timestamp=self.snapshot.timestamp,
            bid=self.snapshot.bid,
            ask=self.snapshot.ask,
            last=self.snapshot.ask,
            volume=12,
            volume_real=12.0,
        )

    def get_bars(self, symbol: str, timeframe: str, count: int):
        self.bars_calls += 1
        return getattr(self.snapshot, f"{timeframe.lower()}_candles")

    def get_account_info(self):
        self.account_calls += 1
        return self.snapshot.account

    def get_positions(self, symbol: str | None = None):
        self.positions_calls += 1
        return self.snapshot.positions

    def get_spread(self, symbol: str):
        self.spread_calls += 1
        return Mt5Spread(
            symbol=symbol,
            bid=self.snapshot.bid,
            ask=self.snapshot.ask,
            price=self.snapshot.spread,
            points=self.snapshot.spread_points,
            timestamp=self.snapshot.timestamp,
        )


def test_snapshot_to_dict_is_json_safe_and_utc_serialized(fake_snapshot: ForexMarketSnapshot):
    payload = snapshot_to_dict(fake_snapshot)

    json.dumps(payload)
    assert payload["timestamp"] == "2026-09-08T00:00:00Z"
    assert payload["symbol"] == "EURUSD"
    assert payload["quote"]["bid"] == 1.1
    assert payload["quote"]["ask"] == 1.1002
    assert payload["symbol_metadata"]["digits"] == 5
    assert payload["symbol_metadata"]["point"] == 0.00001
    assert payload["positions"][0]["ticket"] == 101


def test_build_forex_market_context_is_bounded_and_explicit(
    fake_snapshot: ForexMarketSnapshot,
):
    context = build_forex_market_context(fake_snapshot)

    assert context.startswith("SOURCE: LIVE MT5 BROKER DATA (read-only; not Yahoo Finance)")
    assert "EURUSD" in context
    assert "bid=1.1" in context
    assert "ask=1.1002" in context
    assert "spread_points=20.0" in context
    assert "digits=5" in context
    assert "point=1e-05" in context or "point=0.00001" in context
    assert "object at 0x" not in context
    assert len(context) < 12000


def test_mt5_tool_adapter_uses_cached_snapshot_without_second_provider_snapshot(
    fake_snapshot: ForexMarketSnapshot,
):
    provider = FakeProvider(fake_snapshot)
    adapter = MT5ToolAdapter(provider, fake_snapshot)

    first = adapter.get_mt5_market_snapshot("EURUSD")
    second = adapter.get_mt5_market_snapshot("EURUSD")

    assert first == second
    assert provider.market_snapshot_calls == 0


def test_mt5_tool_adapter_returns_json_safe_primitives_and_tools(
    fake_snapshot: ForexMarketSnapshot,
):
    provider = FakeProvider(fake_snapshot)
    adapter = MT5ToolAdapter(provider, fake_snapshot)

    json.dumps(adapter.get_mt5_tick("EURUSD"))
    json.dumps(adapter.get_mt5_bars("EURUSD", "M5", 1))
    json.dumps(adapter.get_mt5_account_context())
    json.dumps(adapter.get_mt5_positions("EURUSD"))
    json.dumps(adapter.get_mt5_spread("EURUSD"))

    tool_names = [tool.name for tool in adapter.as_tools()]
    assert tool_names == [
        "get_mt5_market_snapshot",
        "get_mt5_tick",
        "get_mt5_bars",
        "get_mt5_account_context",
        "get_mt5_positions",
        "get_mt5_spread",
    ]
    assert not any(name.startswith("order") for name in dir(adapter))


def test_mt5_tool_adapter_as_tools_returns_langchain_structured_tools(
    fake_snapshot: ForexMarketSnapshot,
):
    provider = FakeProvider(fake_snapshot)
    adapter = MT5ToolAdapter(provider, fake_snapshot)

    tools = adapter.as_tools()

    assert all(isinstance(tool, StructuredTool) for tool in tools)
    assert [tool.name for tool in tools] == [
        "get_mt5_market_snapshot",
        "get_mt5_tick",
        "get_mt5_bars",
        "get_mt5_account_context",
        "get_mt5_positions",
        "get_mt5_spread",
    ]
    assert tools[0].invoke({"symbol": "EURUSD"})["symbol"] == "EURUSD"


def test_mt5_tool_adapter_rejects_cached_snapshot_symbol_mismatch(
    fake_snapshot: ForexMarketSnapshot,
):
    provider = FakeProvider(fake_snapshot)
    adapter = MT5ToolAdapter(provider, fake_snapshot)

    with pytest.raises(RuntimeError, match="cached snapshot symbol"):
        adapter.get_mt5_market_snapshot("USDJPY")


def test_mt5_tool_adapter_rejects_market_snapshot_without_cache(
    fake_snapshot: ForexMarketSnapshot,
):
    provider = FakeProvider(fake_snapshot)
    adapter = MT5ToolAdapter(provider)

    with pytest.raises(RuntimeError, match="cached snapshot"):
        adapter.get_mt5_market_snapshot("EURUSD")
