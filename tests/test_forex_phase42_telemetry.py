from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from cli.stats_handler import StatsCallbackHandler
from tradingagents.forex.telemetry import (
    agent_context,
    current_agent_context,
    instrument_agent_node,
)
from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _response() -> LLMResult:
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="PRIVATE REASONING MUST NOT BE STORED",
                        usage_metadata={
                            "input_tokens": 11,
                            "output_tokens": 7,
                            "total_tokens": 18,
                            "output_token_details": {"reasoning": 3},
                        },
                    )
                )
            ]
        ]
    )


def test_agent_context_is_scoped_and_restored():
    assert current_agent_context() is None
    with agent_context("Trader", SimpleNamespace(model_name="quick-model")) as context:
        assert context.name == "Trader"
        assert context.model == "quick-model"
        assert current_agent_context() == context
    assert current_agent_context() is None


def test_instrumented_node_sets_agent_context():
    seen = []

    def node(state):
        seen.append(current_agent_context())
        return state

    wrapped = instrument_agent_node(node, "Portfolio Manager", SimpleNamespace(model="deep-model"))
    assert wrapped({"ok": True}) == {"ok": True}
    assert seen[0].name == "Portfolio Manager"
    assert seen[0].model == "deep-model"
    assert current_agent_context() is None


def test_stats_collects_usage_per_agent_without_private_content():
    handler = StatsCallbackHandler()
    with agent_context("Trader", SimpleNamespace(model_name="quick-model")):
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[]],
            run_id="run-1",
            invocation_params={"model": "quick-model"},
        )
        # Some callback managers expose both start hooks for one run; the
        # stable run id must prevent double-counting.
        handler.on_llm_start(
            {"name": "ChatOpenAI"},
            ["PRIVATE PROMPT"],
            run_id="run-1",
            invocation_params={"model": "quick-model"},
        )
        handler.on_llm_end(_response(), run_id="run-1")
        handler.on_llm_end(_response(), run_id="run-1")

    stats = handler.get_stats()
    assert stats["llm_calls"] == 1
    assert stats["tokens_in"] == 11
    assert stats["tokens_out"] == 7
    assert stats["reasoning_tokens"] == 3
    trader = stats["agents"]["Trader"]
    assert trader["model"] == "quick-model"
    assert trader["calls"] == 1
    assert trader["tokens_in"] == 11
    assert trader["tokens_out"] == 7
    assert trader["reasoning_tokens"] == 3
    assert trader["elapsed_seconds"] >= 0
    assert "PRIVATE" not in repr(stats)
    assert not hasattr(handler, "prompts")


def _bare_graph(config, mode="forex_mt5"):
    graph = object.__new__(TradingAgentsGraph)
    graph.config = config
    graph.market_data_mode = mode
    return graph


def test_forex_provider_role_controls_are_separate_from_stock_defaults():
    config = {
        "llm_provider": "openai",
        "openai_reasoning_effort": "medium",
        "forex_quick_reasoning_effort": "low",
        "forex_deep_reasoning_effort": "high",
    }
    graph = _bare_graph(config)
    assert graph._get_provider_kwargs(role="quick")["reasoning_effort"] == "low"
    assert graph._get_provider_kwargs(role="deep")["reasoning_effort"] == "high"
    graph.market_data_mode = "stock"
    assert graph._get_provider_kwargs(role="quick")["reasoning_effort"] == "medium"


def test_ollama_forex_thinking_controls_use_reasoning_effort_for_quick_and_extra_body_for_deep():
    config = {
        "llm_provider": "ollama",
        "forex_quick_thinking": False,
        "forex_deep_thinking": True,
    }
    graph = _bare_graph(config)
    quick_kwargs = graph._get_provider_kwargs(role="quick")
    assert quick_kwargs["reasoning_effort"] == "none"
    assert "extra_body" not in quick_kwargs
    assert graph._get_provider_kwargs(role="deep")["extra_body"] == {"think": True}
    graph.market_data_mode = "stock"
    assert "extra_body" not in graph._get_provider_kwargs(role="quick")


def test_forex_trader_keeps_deep_model_available_without_changing_stock():
    setup = object.__new__(GraphSetup)
    setup.market_data_mode = "forex_mt5"
    deep = object()
    quick = object()
    setup.deep_thinking_llm = deep
    setup.quick_thinking_llm = quick
    forex_trader_llm = (
        setup.deep_thinking_llm
        if setup.market_data_mode == "forex_mt5"
        else setup.quick_thinking_llm
    )
    assert forex_trader_llm is deep
    setup.market_data_mode = "stock"
    stock_trader_llm = (
        setup.deep_thinking_llm
        if setup.market_data_mode == "forex_mt5"
        else setup.quick_thinking_llm
    )
    assert stock_trader_llm is quick
