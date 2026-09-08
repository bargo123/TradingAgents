"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import json

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_only,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        is_forex = state.get("asset_type") == "forex"

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        if is_forex:
            prompt = f"""As the Portfolio Manager for a currency pair, synthesize the risk analysts' debate and return one structured portfolio rating.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Ground the rating in observed currency-pair price action, spread, volatility, and broad macro context from the analysts. Commit to a directional call only when evidence clearly supports one; choose Hold when the case is balanced, materially conflicting, ambiguous, or insufficient. Entry and risk levels are hypothetical observations only; no order is sent. Do not infer issuer-level business data.

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""
        else:
            prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Ground every conclusion in specific evidence from the analysts. Commit to a directional call only when the evidence clearly supports one; choose Hold when the case is balanced, materially conflicting, ambiguous, or insufficient to justify changing exposure, rather than forcing a direction to appear decisive. Weigh the analysts on their merits, independent of speaking order.

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        if is_forex:
            try:
                structured_result = invoke_structured_only(
                    structured_llm,
                    prompt,
                    "Portfolio Manager",
                )
                raw_result = structured_result.model_dump(mode="json")
                json.dumps(raw_result, allow_nan=False)
                final_trade_decision = render_pm_decision(structured_result)
                normalization_status = "NORMALIZED"
                normalization_error = None
            except Exception as exc:  # noqa: BLE001 — shadow mode must fail closed
                raw_result = {
                    "status": "FAILED",
                    "error": "STRUCTURED_OUTPUT_REQUIRED",
                }
                final_trade_decision = "FOREX_PORTFOLIO_MANAGER_FAILED"
                normalization_status = "FAILED"
                normalization_error = str(exc)
        else:
            final_trade_decision = invoke_structured_or_freetext(
                structured_llm,
                llm,
                prompt,
                render_pm_decision,
                "Portfolio Manager",
            )
            raw_result = None
            normalization_status = ""
            normalization_error = ""

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            **(
                {
                    "portfolio_manager_raw_result": raw_result,
                    "normalization_status": normalization_status,
                    "normalization_error": normalization_error,
                }
                if is_forex
                else {}
            ),
        }

    return portfolio_manager_node
