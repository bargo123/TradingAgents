from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph


class _FakeLLM:
    def with_structured_output(self, schema):
        return None

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        return SimpleNamespace(content="", tool_calls=[])


class _FakeMt5Tools:
    def as_tools(self):
        def get_mt5_market_snapshot(symbol):
            """Return a market snapshot."""
            return {"symbol": symbol}

        def get_mt5_tick(symbol):
            """Return a tick."""
            return {"symbol": symbol}

        def get_mt5_bars(symbol, timeframe, count):
            """Return bars."""
            return {"symbol": symbol, "timeframe": timeframe, "count": count}

        def get_mt5_account_context():
            """Return account context."""
            return {"balance": 10_000}

        def get_mt5_positions(symbol=None):
            """Return positions."""
            return []

        def get_mt5_spread(symbol):
            """Return spread."""
            return {"symbol": symbol, "spread": 0.0002}

        return [
            get_mt5_market_snapshot,
            get_mt5_tick,
            get_mt5_bars,
            get_mt5_account_context,
            get_mt5_positions,
            get_mt5_spread,
        ]


def _config(tmp_path):
    return {
        "llm_provider": "local",
        "quick_think_llm": "qwen",
        "deep_think_llm": "qwen",
        "backend_url": None,
        "data_cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
        "max_recur_limit": 16,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    }


def _patch_llm(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.create_llm_client",
        lambda **kwargs: SimpleNamespace(get_llm=lambda: _FakeLLM()),
    )


def test_forex_initial_state_carries_mode_and_context():
    state = Propagator().create_initial_state(
        "EURUSD",
        "2026-09-08",
        asset_type="forex",
        market_data_mode="forex_mt5",
        market_context="SOURCE: LIVE MT5 BROKER DATA",
    )

    assert state["market_data_mode"] == "forex_mt5"
    assert state["market_context"].startswith("SOURCE: LIVE MT5 BROKER DATA")


def test_stock_initial_state_keeps_mode_and_shadow_fields_defaults():
    state = Propagator().create_initial_state("AAPL", "2026-09-08")

    assert state["asset_type"] == "stock"
    assert state["market_data_mode"] == "stock"
    assert state["market_context"] == ""
    assert state["portfolio_manager_raw_result"] == ""
    assert state["normalization_status"] == ""
    assert state["normalization_error"] == ""


def test_instrument_context_appends_market_context_once():
    state = {
        "company_of_interest": "EURUSD",
        "asset_type": "forex",
        "instrument_context": "Instrument context for EURUSD",
        "market_context": "SOURCE: LIVE MT5 BROKER DATA",
    }

    context = get_instrument_context_from_state(state)

    assert context.count("SOURCE: LIVE MT5 BROKER DATA") == 1
    assert context.startswith("Instrument context for EURUSD")


def test_instrument_context_dedupes_existing_source_and_bounds_context():
    source = "SOURCE: LIVE MT5 BROKER DATA"
    state = {
        "company_of_interest": "EURUSD",
        "asset_type": "forex",
        "instrument_context": f"Instrument context for EURUSD\n{source}",
        "market_context": f"{source}\n" + ("x" * 10_000),
    }

    context = get_instrument_context_from_state(state)

    assert context.count(source) == 1
    assert len(context) <= len(state["instrument_context"]) + 1 + 4096


def test_graph_rejects_unknown_market_data_mode_before_building(tmp_path, monkeypatch):
    _patch_llm(monkeypatch)

    with pytest.raises(ValueError, match="market_data_mode"):
        TradingAgentsGraph(
            selected_analysts=("market",),
            market_data_mode="crypto",
            config=_config(tmp_path),
            callbacks=[],
        )


def test_forex_graph_requires_mt5_adapter_before_building(tmp_path, monkeypatch):
    _patch_llm(monkeypatch)

    with pytest.raises(ValueError, match="mt5"):
        TradingAgentsGraph(
            selected_analysts=("market", "news"),
            market_data_mode="forex_mt5",
            mt5_tools=None,
            config=_config(tmp_path),
            callbacks=[],
        )


def test_graph_supports_forex_mode_and_rejects_stock_specific_analysts(
    monkeypatch, tmp_path
):
    config = _config(tmp_path)
    _patch_llm(monkeypatch)

    graph = TradingAgentsGraph(
        selected_analysts=("market", "news"),
        market_data_mode="forex_mt5",
        mt5_tools=_FakeMt5Tools(),
        config=config,
        callbacks=[],
    )

    assert graph.market_data_mode == "forex_mt5"
    assert "get_mt5_market_snapshot" in graph.tool_nodes["market"].tools_by_name
    assert set(graph.tool_nodes["news"].tools_by_name) == {"get_global_news"}

    with pytest.raises(ValueError, match="social|fundamentals"):
        TradingAgentsGraph(
            selected_analysts=("market", "fundamentals"),
            market_data_mode="forex_mt5",
            mt5_tools=_FakeMt5Tools(),
            config=config,
            callbacks=[],
        )
