from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state
from tradingagents.dataflows.mt5.models import (
    ForexMarketSnapshot,
    Mt5AccountInfo,
    Mt5Bar,
    Mt5Position,
    Mt5SymbolInfo,
)
from tradingagents.forex.context import build_forex_market_context
from tradingagents.forex.evidence_context import EvidenceContext
from tradingagents.forex.tools import MT5ToolAdapter
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.fixture
def forex_snapshot() -> ForexMarketSnapshot:
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
    )


def _bare_graph() -> TradingAgentsGraph:
    graph = object.__new__(TradingAgentsGraph)
    graph.selected_analysts = ("market", "news")
    graph.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
    graph.market_data_mode = "stock"
    return graph


@pytest.mark.unit
def test_forex_initial_state_carries_mode_and_market_context() -> None:
    state = Propagator().create_initial_state(
        "EURUSD",
        "2026-09-08",
        asset_type="forex",
        market_data_mode="forex_mt5",
        market_context="SOURCE: LIVE MT5 BROKER DATA",
    )

    assert state["asset_type"] == "forex"
    assert state["market_data_mode"] == "forex_mt5"
    assert state["market_context"] == "SOURCE: LIVE MT5 BROKER DATA"
    assert state["portfolio_manager_raw_result"] == ""
    assert state["normalization_status"] == ""
    assert state["normalization_error"] == ""
    assert AgentState.__annotations__["market_data_mode"]


@pytest.mark.unit
def test_forex_initial_state_accepts_evidence_context() -> None:
    context = EvidenceContext()
    state = Propagator().create_initial_state(
        "EURUSD", "2026-09-08", asset_type="forex", evidence_context=context
    )
    assert state["evidence_context"] is context


@pytest.mark.unit
def test_shared_context_appends_market_context_without_yahoo_lookup(
    forex_snapshot: ForexMarketSnapshot,
) -> None:
    base_context = build_forex_market_context(forex_snapshot)
    state = {
        "company_of_interest": "EURUSD",
        "asset_type": "forex",
        "instrument_context": "The pair is EURUSD.",
        "market_context": base_context,
    }

    context = get_instrument_context_from_state(state)

    assert context.startswith("The pair is EURUSD.")
    assert context.count("SOURCE: LIVE MT5 BROKER DATA") == 1


@pytest.mark.unit
def test_forex_run_signature_includes_market_data_mode() -> None:
    graph = _bare_graph()
    graph.market_data_mode = "forex_mt5"

    signature = graph._run_signature("forex")

    assert "market_data_mode=forex_mt5" in signature
    assert "asset=forex" in signature


@pytest.mark.unit
def test_graph_config_accepts_evidence_identity() -> None:
    graph = _bare_graph()
    graph.market_data_mode = "forex_mt5"
    graph.config.update(
        forex_evidence_enabled=True,
        forex_evidence_context_hash="ctx-123",
    )
    signature = graph._run_signature("forex")
    assert "forex_evidence=enabled/ctx-123" in signature


@pytest.mark.unit
def test_forex_graph_setup_rejects_stock_specific_analysts(
    forex_snapshot: ForexMarketSnapshot,
) -> None:
    adapter = MT5ToolAdapter(SimpleNamespace(), forex_snapshot)
    setup = GraphSetup(
        SimpleNamespace(),
        SimpleNamespace(),
        {"market": SimpleNamespace()},
        ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1),
        market_data_mode="forex_mt5",
        mt5_tools=adapter,
    )

    with pytest.raises(ValueError, match="social|fundamentals"):
        setup.setup_graph(("market", "fundamentals"))


@pytest.mark.unit
def test_trading_agents_graph_accepts_forex_mode_arguments(
    monkeypatch: pytest.MonkeyPatch,
    forex_snapshot: ForexMarketSnapshot,
) -> None:
    class DummyLLMClient:
        def get_llm(self):
            return SimpleNamespace(bind_tools=lambda tools: self, invoke=lambda prompt: None)

    class DummyWorkflow:
        def compile(self, checkpointer=None):
            return SimpleNamespace()

    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.create_llm_client",
        lambda **kwargs: DummyLLMClient(),
    )
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.GraphSetup.setup_graph",
        lambda self, selected_analysts=("market", "social", "news", "fundamentals"): DummyWorkflow(),
    )

    adapter = MT5ToolAdapter(SimpleNamespace(), forex_snapshot)
    graph = TradingAgentsGraph(
        selected_analysts=("market", "news"),
        config={
            "data_cache_dir": "data",
            "results_dir": "results",
            "llm_provider": "openai",
            "deep_think_llm": "deep",
            "quick_think_llm": "quick",
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
        },
        market_data_mode="forex_mt5",
        mt5_tools=adapter,
    )

    assert graph.market_data_mode == "forex_mt5"
    assert graph.mt5_tools is adapter
