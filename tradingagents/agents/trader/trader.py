"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    render_supporting_evidence,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_forex_ollama_structured,
    bind_structured,
    invoke_structured_only,
    invoke_structured_or_freetext,
    is_ollama_chat_model,
)
from tradingagents.forex.profile import build_forex_profile_context


def create_trader(llm, *, forex_mode: bool = False):
    # Ollama's OpenAI-compatible endpoint does not support tool_choice and
    # Qwen can answer with prose instead of voluntarily selecting the schema
    # tool on production-sized Trader prompts.  Its response_format JSON-schema
    # path is reliable, so use that path only for the forex Ollama Trader.
    ollama_forex = forex_mode and is_ollama_chat_model(llm)
    structured_llm = (
        bind_forex_ollama_structured(llm, TraderProposal, "Trader")
        if ollama_forex
        else bind_structured(llm, TraderProposal, "Trader")
    )

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        if state.get("asset_type") == "forex":
            evidence_block = render_supporting_evidence(state)
            forex_suffix = NO_EXTERNAL_TOOLS + get_language_instruction()
            messages = [
                {
                    "role": "system",
                    "content": (
                    "You are a forex shadow-mode trader analyzing one currency pair. "
                        + build_forex_profile_context(state.get("forex_analysis_profile"))
                        + "\n"
                        "This is an intraday decision: use minutes-to-hours horizons only. "
                        "Return a structured hypothetical Buy, Sell, or Hold proposal using "
                        "only the supplied price action, spread, volatility, and macro context. "
                        "Any entry or stop levels are observational only; no order is sent. "
                        "Issuer-level fundamentals are unavailable for forex; do not infer them. "
                        + grounding
                        + ("" if evidence_block else forex_suffix)
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Here is the research team's plan for currency pair {company_name}. "
                        f"{instrument_context}\n\n"
                        f"{report_section}"
                        f"Proposed Plan:\n{investment_plan}\n\n"
                        f"Make an evidence-based hypothetical trading proposal."
                    ),
                },
            ]
            if evidence_block:
                messages.append(
                    {
                        "role": "system",
                        "content": f"{evidence_block}\n\n{forex_suffix}",
                    }
                )
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a trading agent analyzing market data to make investment decisions. "
                        "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                        + grounding
                        + NO_EXTERNAL_TOOLS
                        + get_language_instruction()
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Here is the research team's investment plan for {company_name}. "
                        f"{instrument_context}\n\n"
                        f"{report_section}"
                        f"Proposed Investment Plan:\n{investment_plan}\n\n"
                        f"Make an informed, strategic trading decision."
                    ),
                },
            ]

        if ollama_forex:
            trader_plan = render_trader_proposal(
                invoke_structured_only(structured_llm, messages, "Trader")
            )
        else:
            trader_plan = invoke_structured_or_freetext(
                structured_llm,
                llm,
                messages,
                render_trader_proposal,
                "Trader",
            )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
