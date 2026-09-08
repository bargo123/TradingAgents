from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from tests.test_forex_shadow_runner import _snapshot
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.schemas import (
    ForexPortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.trader.trader import create_trader
from tradingagents.forex.context_integrity import evaluate_context_integrity
from tradingagents.forex.telemetry import capture_state_trace, instrument_agent_node
from tradingagents.forex.tools import MT5ToolAdapter
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.setup import GraphSetup


def _prompt_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(
            item.get("content", str(item)) if isinstance(item, dict) else str(item)
            for item in prompt
        )
    if hasattr(prompt, "to_messages"):
        return "\n".join(str(message.content) for message in prompt.to_messages())
    return str(prompt)


class _Structured:
    def __init__(self, owner: _TraceLLM, schema):
        self.owner = owner
        self.schema = schema

    def invoke(self, prompt):
        self.owner.prompts.append(prompt)
        if self.schema is ResearchPlan:
            return ResearchPlan(
                recommendation=PortfolioRating.HOLD,
                rationale="RM_RESULT",
                strategic_actions="RM_ACTION",
            )
        if self.schema is TraderProposal:
            return TraderProposal(
                action=TraderAction.HOLD,
                reasoning="TRADER_RESULT",
            )
        if self.schema is ForexPortfolioDecision:
            return ForexPortfolioDecision(
                rating=PortfolioRating.HOLD,
                executive_summary="PM_RESULT",
                investment_thesis="PM_THESIS",
            )
        raise AssertionError(f"unexpected structured schema: {self.schema!r}")


class _BoundTools:
    def __init__(self, owner: _TraceLLM):
        self.owner = owner

    def invoke(self, prompt):
        return self.owner.invoke(prompt)

    __call__ = invoke


class _TraceLLM:
    """Deterministic LLM stub that exposes normal reports, never private CoT."""

    model_name = "phase43-test-model"

    def __init__(self):
        self.prompts = []

    def with_structured_output(self, schema):
        return _Structured(self, schema)

    def bind_tools(self, tools):
        return _BoundTools(self)

    def invoke(self, prompt):
        self.prompts.append(prompt)
        text = _prompt_text(prompt)
        if "Bull Analyst for" in text:
            content = "BULL_REPORT"
        elif "Bear Analyst for" in text:
            content = "BEAR_REPORT"
        elif "Aggressive Risk Analyst for" in text:
            content = "AGGRESSIVE_REPORT"
        elif "Conservative Risk Analyst for" in text:
            content = "CONSERVATIVE_REPORT"
        elif "Neutral Risk Analyst for" in text:
            content = "NEUTRAL_REPORT"
        elif "forex market analyst" in text:
            content = "MARKET_REPORT"
        elif "forex news researcher" in text:
            content = "NEWS_REPORT"
        else:
            content = "UNCLASSIFIED_REPORT"
        return AIMessage(content=content)


def _merge(state: dict, update: dict) -> dict:
    state.update(update)
    return state


def test_forex_required_artifacts_reach_each_downstream_boundary():
    """The complete deterministic forex hand-off must retain every report.

    This is deliberately a node-level integration trace: it uses the normal
    agent factories and state updates, while the LLM returns only deterministic
    visible reports/structured values.  The Portfolio Manager must receive the
    original bull/bear debate as well as the downstream plan and risk debate;
    the current implementation omits that debate from its prompt.
    """

    llm = _TraceLLM()
    state = Propagator().create_initial_state(
        "EURUSD",
        "2026-09-08",
        asset_type="forex",
        market_data_mode="forex_mt5",
        instrument_context="EURUSD currency pair",
        market_context="SOURCE: LIVE MT5 BROKER DATA",
        forex_analysis_profile="INTRADAY",
    )
    # These fields model values owned by a different node.  Nested LangGraph
    # channels use last-write replacement, so a speaker must not erase them.
    state["investment_debate_state"]["judge_decision"] = "PRIOR_RESEARCH_JUDGE"
    state["risk_debate_state"]["judge_decision"] = "PRIOR_RISK_JUDGE"
    mt5_tool = SimpleNamespace(name="get_mt5_market_snapshot")
    mt5_adapter = SimpleNamespace(as_tools=lambda: [mt5_tool])

    _merge(state, create_market_analyst(llm, "forex_mt5", mt5_adapter)(state))
    _merge(state, create_news_analyst(llm, "forex_mt5")(state))
    assert state["market_report"] == "MARKET_REPORT"
    assert state["news_report"] == "NEWS_REPORT"

    _merge(state, create_bull_researcher(llm)(state))
    assert "Bull Analyst: BULL_REPORT" in state["investment_debate_state"]["bull_history"]
    assert state["investment_debate_state"]["judge_decision"] == "PRIOR_RESEARCH_JUDGE"
    _merge(state, create_bear_researcher(llm)(state))
    investment_debate = state["investment_debate_state"]
    assert "Bull Analyst: BULL_REPORT" in investment_debate["history"]
    assert "Bear Analyst: BEAR_REPORT" in investment_debate["history"]
    assert "Bear Analyst: BEAR_REPORT" in investment_debate["bear_history"]

    _merge(state, create_research_manager(llm)(state))
    rm_prompt = _prompt_text(llm.prompts[-1])
    assert "Bull Analyst: BULL_REPORT" in rm_prompt
    assert "Bear Analyst: BEAR_REPORT" in rm_prompt
    assert "RM_RESULT" in state["investment_plan"]

    _merge(state, create_trader(llm)(state))
    trader_prompt = _prompt_text(llm.prompts[-1])
    assert "RM_RESULT" in trader_prompt
    assert "TRADER_RESULT" in state["trader_investment_plan"]

    for node_factory in (
        create_aggressive_debator,
        create_conservative_debator,
        create_neutral_debator,
    ):
        _merge(state, node_factory(llm)(state))
        risk = state["risk_debate_state"]
        assert risk["history"]
        assert "TRADER_RESULT" in _prompt_text(llm.prompts[-1])

    risk = state["risk_debate_state"]
    assert all(risk[key] for key in ("aggressive_history", "conservative_history", "neutral_history"))
    assert risk["judge_decision"] == "PRIOR_RISK_JUDGE"

    _merge(state, create_portfolio_manager(llm)(state))
    pm_prompt = _prompt_text(llm.prompts[-1])
    assert "Bull Analyst: BULL_REPORT" in pm_prompt
    assert "Bear Analyst: BEAR_REPORT" in pm_prompt
    assert "RM_RESULT" in pm_prompt
    assert "TRADER_RESULT" in pm_prompt
    assert "AGGRESSIVE_REPORT" in pm_prompt
    assert "CONSERVATIVE_REPORT" in pm_prompt
    assert "NEUTRAL_REPORT" in pm_prompt
    assert state["portfolio_manager_raw_result"]["rating"] == "Hold"


def test_context_integrity_rejects_label_only_debate_artifacts():
    state = {
        "market_report": "MARKET_REPORT",
        "news_report": "NEWS_REPORT",
        "investment_debate_state": {
            "history": "\nBull Analyst: \nBear Analyst: ",
            "bull_history": "\nBull Analyst: ",
            "bear_history": "\nBear Analyst: ",
        },
        "investment_plan": "RM_PLAN",
        "trader_investment_plan": "TRADER_PLAN",
        "risk_debate_state": {
            "history": "\nAggressive Analyst: ",
            "aggressive_history": "\nAggressive Analyst: ",
            "conservative_history": "\nConservative Analyst: ",
            "neutral_history": "\nNeutral Analyst: ",
        },
        "portfolio_manager_raw_result": {"rating": "Hold"},
        "final_trade_decision": "PM_RESULT",
    }

    result = evaluate_context_integrity(state)

    assert result["status"] == "INCOMPLETE"
    assert {"bull", "bear", "risk_debate"}.issubset(result["missing"])
    assert result["artifacts"]["bull"]["present"] is False
    assert result["artifacts"]["bull"]["chars"] > 0


def test_instrumented_state_trace_contains_metadata_only():
    state = {"market_report": "PRIVATE REPORT CONTENT"}

    def node(current_state):
        return {"news_report": "NEWS"}

    with capture_state_trace() as trace:
        instrument_agent_node(node, "News Analyst", SimpleNamespace(model="test"))(state)

    assert [entry["phase"] for entry in trace] == ["before", "after"]
    assert trace[1]["artifacts"]["market"]["present"] is True
    assert trace[1]["artifacts"]["news"]["present"] is True
    assert "PRIVATE REPORT CONTENT" not in repr(trace)


def test_compiled_forex_graph_trace_is_complete_without_report_retention():
    llm = _TraceLLM()
    adapter = MT5ToolAdapter(SimpleNamespace(), _snapshot())
    tools = adapter.as_tools()
    setup = GraphSetup(
        llm,
        llm,
        {"market": ToolNode(tools), "news": ToolNode([tools[0]])},
        ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1),
        market_data_mode="forex_mt5",
        mt5_tools=adapter,
    )
    graph = setup.setup_graph(("market", "news")).compile()
    initial_state = Propagator().create_initial_state(
        "EURUSD",
        "2026-09-08",
        asset_type="forex",
        market_data_mode="forex_mt5",
        instrument_context="EURUSD currency pair",
        market_context="SOURCE: LIVE MT5 BROKER DATA",
        forex_analysis_profile="INTRADAY",
    )

    with capture_state_trace() as trace:
        final_state = graph.invoke(initial_state, config={"recursion_limit": 50})

    integrity = evaluate_context_integrity(final_state, trace=trace)

    assert integrity["status"] == "COMPLETE"
    assert integrity["missing"] == []
    assert set(integrity["nodes_seen"]) == {
        "Market Analyst",
        "News Analyst",
        "Bull Researcher",
        "Bear Researcher",
        "Research Manager",
        "Trader",
        "Aggressive Analyst",
        "Conservative Analyst",
        "Neutral Analyst",
        "Portfolio Manager",
    }
    assert all(entry["artifacts"]["portfolio_manager"]["present"] is False for entry in trace if entry["phase"] == "before")
    assert "BULL_REPORT" not in repr(trace)
    assert "BEAR_REPORT" not in repr(trace)
