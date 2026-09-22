# TradingAgents/graph/setup.py

from typing import Any

try:  # pragma: no cover - fallback for minimal test environments
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode
except ModuleNotFoundError:  # pragma: no cover
    END = "END"
    START = "START"

    class ToolNode:
        def __init__(self, tools):
            self.tools = list(tools)
            self.tools_by_name = {getattr(tool, "name", getattr(tool, "__name__", str(i))): tool for i, tool in enumerate(self.tools)}

    class StateGraph:
        def __init__(self, state_type):
            self.state_type = state_type

        def add_node(self, *args, **kwargs):
            return None

        def add_edge(self, *args, **kwargs):
            return None

        def add_conditional_edges(self, *args, **kwargs):
            return None

        def compile(self):
            return self

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.forex.profile import resolve_forex_profile
from tradingagents.forex.telemetry import instrument_agent_node

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic
from .parallel_analysts import ParallelAnalystBranch, run_parallel_analysts

_FORBIDDEN_MT5_TOOL_TOKENS = (
    "order_send",
    "send_order",
    "place_order",
    "cancel_order",
    "pending_order",
    "buy",
    "sell",
    "open_position",
    "close_position",
    "modify_position",
    "modify_order",
)


def validate_read_only_mt5_tools(mt5_tools: Any) -> list[Any]:
    """Return injected MT5 tools only when their names are read-only."""
    tools = list(mt5_tools.as_tools())
    for tool in tools:
        name = str(getattr(tool, "name", getattr(tool, "__name__", ""))).casefold()
        if any(token in name for token in _FORBIDDEN_MT5_TOOL_TOKENS):
            raise ValueError(
                f"forex_mt5 injected tool {name!r} is not allowed in read-only execution"
            )
    return tools

# Every target a shared conditional router can return. Each edge driven by the
# router maps all of them, so a fall-through return (e.g. under prompt/i18n/
# refactor drift in the speaker labels) can never hit a missing path_map entry
# and crash LangGraph mid-run (#1088).
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}
RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}

_FOREX_PARALLEL_ANALYST_NODE = "Forex Market and News Analysts"


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        market_data_mode: str = "stock",
        mt5_tools: Any | None = None,
        forex_profile: str = "INTRADAY",
        forex_pm_max_tokens: int | None = None,
    ):
        """Initialize with required components."""
        if market_data_mode not in {"stock", "forex_mt5"}:
            raise ValueError("market_data_mode must be one of: stock, forex_mt5")
        if market_data_mode == "forex_mt5" and mt5_tools is None:
            raise ValueError("forex_mt5 market_data_mode requires an MT5 adapter")
        if market_data_mode == "forex_mt5" and not callable(
            getattr(mt5_tools, "as_tools", None)
        ):
            raise ValueError(
                "forex_mt5 market_data_mode requires an MT5 adapter with as_tools()"
            )
        if market_data_mode == "forex_mt5":
            validate_read_only_mt5_tools(mt5_tools)
            forex_profile = resolve_forex_profile(forex_profile).name
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.market_data_mode = market_data_mode
        self.mt5_tools = mt5_tools
        self.forex_profile = forex_profile
        self.forex_pm_max_tokens = forex_pm_max_tokens

    def _instrument(self, node, agent_name: str, llm: Any):
        """Add forex-only callback attribution around an LLM-bearing node."""
        if self.market_data_mode != "forex_mt5":
            return node
        return instrument_agent_node(node, agent_name, llm)

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if self.market_data_mode == "forex_mt5":
            validate_read_only_mt5_tools(self.mt5_tools)
            forbidden = [name for name in selected_analysts if name in {"social", "fundamentals"}]
            if forbidden:
                raise ValueError(
                    f"Forex mode does not allow these analysts: {', '.join(forbidden)}"
                )
        plan = build_analyst_execution_plan(selected_analysts)

        if self.market_data_mode == "forex_mt5":
            analyst_factories = {
                "market": lambda: create_market_analyst(
                    self.quick_thinking_llm,
                    market_data_mode="forex_mt5",
                    mt5_tools=self.mt5_tools,
                ),
                "news": lambda: create_news_analyst(
                    self.quick_thinking_llm,
                    market_data_mode="forex_mt5",
                ),
            }
        else:
            analyst_factories = {
                "market": lambda: create_market_analyst(self.quick_thinking_llm),
                "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
                "news": lambda: create_news_analyst(self.quick_thinking_llm),
                "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
            }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_llm = (
            self.deep_thinking_llm
            if self.market_data_mode == "forex_mt5"
            else self.quick_thinking_llm
        )
        trader_node = create_trader(
            trader_llm,
            forex_mode=self.market_data_mode == "forex_mt5",
        )

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(
            self.deep_thinking_llm,
            forex_profile=self.forex_profile,
            forex_pm_max_tokens=self.forex_pm_max_tokens,
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        parallel_forex_analysts = (
            self.market_data_mode == "forex_mt5"
            and len(plan.specs) == 2
            and {spec.key for spec in plan.specs} == {"market", "news"}
        )
        parallel_branches: list[ParallelAnalystBranch] = []

        # Add analyst nodes to the graph
        for spec in plan.specs:
            analyst_node = self._instrument(
                analyst_factories[spec.key](), spec.agent_node, self.quick_thinking_llm
            )
            clear_node = create_msg_delete()
            if parallel_forex_analysts:
                parallel_branches.append(
                    ParallelAnalystBranch(
                        key=spec.key,
                        agent_node=spec.agent_node,
                        clear_node=spec.clear_node,
                        tool_node=spec.tool_node,
                        report_key=spec.report_key,
                        node=analyst_node,
                        clear=clear_node,
                        tool=self.tool_nodes[spec.key],
                        route=getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                    )
                )
            else:
                workflow.add_node(spec.agent_node, analyst_node)
                workflow.add_node(spec.clear_node, clear_node)
                workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])

        if parallel_forex_analysts:
            workflow.add_node(
                _FOREX_PARALLEL_ANALYST_NODE,
                lambda state, config=None: run_parallel_analysts(
                    tuple(parallel_branches), state, config=config
                ),
            )

        # Add other nodes
        workflow.add_node(
            "Bull Researcher",
            self._instrument(bull_researcher_node, "Bull Researcher", self.quick_thinking_llm),
        )
        workflow.add_node(
            "Bear Researcher",
            self._instrument(bear_researcher_node, "Bear Researcher", self.quick_thinking_llm),
        )
        workflow.add_node(
            "Research Manager",
            self._instrument(research_manager_node, "Research Manager", self.deep_thinking_llm),
        )
        workflow.add_node(
            "Trader",
            self._instrument(trader_node, "Trader", trader_llm),
        )
        workflow.add_node(
            "Aggressive Analyst",
            self._instrument(aggressive_analyst, "Aggressive Analyst", self.quick_thinking_llm),
        )
        workflow.add_node(
            "Neutral Analyst",
            self._instrument(neutral_analyst, "Neutral Analyst", self.quick_thinking_llm),
        )
        workflow.add_node(
            "Conservative Analyst",
            self._instrument(
                conservative_analyst, "Conservative Analyst", self.quick_thinking_llm
            ),
        )
        workflow.add_node(
            "Portfolio Manager",
            self._instrument(portfolio_manager_node, "Portfolio Manager", self.deep_thinking_llm),
        )

        # Define edges. Market and News begin from the same immutable snapshot;
        # their tool loops run in isolated branch state and merge by report key.
        if parallel_forex_analysts:
            workflow.add_edge(START, _FOREX_PARALLEL_ANALYST_NODE)
            workflow.add_edge(_FOREX_PARALLEL_ANALYST_NODE, "Bull Researcher")
        else:
            workflow.add_edge(START, plan.specs[0].agent_node)

            # Connect analysts in sequence
            for i, spec in enumerate(plan.specs):
                current_analyst = spec.agent_node
                current_tools = spec.tool_node
                current_clear = spec.clear_node

                workflow.add_conditional_edges(
                    current_analyst,
                    getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                    [current_tools, current_clear],
                )
                workflow.add_edge(current_tools, current_analyst)

                if i < len(plan.specs) - 1:
                    workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
                else:
                    workflow.add_edge(current_clear, "Bull Researcher")

        # Both research-debate edges share the complete DEBATE_PATH_MAP (#1088).
        for debate_node in ("Bull Researcher", "Bear Researcher"):
            workflow.add_conditional_edges(
                debate_node,
                self.conditional_logic.should_continue_debate,
                DEBATE_PATH_MAP,
            )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        # All three risk edges share the complete RISK_ANALYSIS_PATH_MAP (#1088).
        for risk_node in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
            workflow.add_conditional_edges(
                risk_node,
                self.conditional_logic.should_continue_risk_analysis,
                RISK_ANALYSIS_PATH_MAP,
            )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
