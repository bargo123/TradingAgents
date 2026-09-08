"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.forex.profile import build_forex_profile_context


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        is_forex = state.get("asset_type") == "forex"
        if is_forex:
            prompt = f"""As the Research Manager for an intraday forex shadow analysis, critically evaluate the bull/bear debate and deliver a clear, actionable plan for the trader. This is a minutes-to-hours currency-pair decision, not a long-term equity investment. Shadow analysis only; no order is sent.

{build_forex_profile_context(state.get("forex_analysis_profile"))}

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong evidence to enter or add to the currency-pair position
- **Overweight**: Favorable intraday bias, increase exposure only hypothetically
- **Hold**: Wait or maintain the current position while evidence is mixed
- **Underweight**: Reduce hypothetical exposure because risk dominates
- **Sell**: Strong evidence to exit or avoid the currency-pair position

Commit to a directional stance only when observed price action, spread, volatility, and broad macro evidence clearly warrant one. Choose Hold when evidence is balanced, materially conflicting, unavailable, or ambiguous. Do not use months, years, company valuation, issuer fundamentals, dividends, earnings, or equity portfolio-allocation language.

---

**Debate History:**
{history}

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()
        else:
            prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a directional stance only when the debate's strongest arguments clearly warrant one. Choose Hold when the evidence is balanced, materially conflicting, ambiguous, or insufficient to justify changing exposure; do not manufacture a direction merely to appear decisive. Weigh the bull and bear cases on their merits, independent of which side spoke first or last.

---

**Debate History:**
{history}

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
