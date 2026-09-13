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
from collections.abc import Mapping

from tradingagents.agents.schemas import (
    ForexPortfolioDecision,
    PortfolioDecision,
    render_forex_pm_decision,
    render_pm_decision,
)
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    render_final_pm_evidence_instruction,
    render_supporting_evidence,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_forex_ollama_structured,
    bind_structured,
    invoke_structured_only,
    invoke_structured_or_freetext,
)
from tradingagents.forex.profile import build_forex_profile_context


def create_portfolio_manager(llm, forex_profile: str = "INTRADAY"):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")
    forex_structured_llm = None

    def portfolio_manager_node(state) -> dict:
        nonlocal forex_structured_llm
        instrument_context = get_instrument_context_from_state(state)
        is_forex = state.get("asset_type") == "forex"

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        investment_debate_history = ""
        bull_history = ""
        bear_history = ""
        if is_forex:
            investment_debate_state = state.get("investment_debate_state", {})
            if not isinstance(investment_debate_state, Mapping):
                investment_debate_state = {}
            investment_debate_history = investment_debate_state.get("history", "")
            bull_history = investment_debate_state.get("bull_history", "")
            bear_history = investment_debate_state.get("bear_history", "")
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        if is_forex:
            profile_context = build_forex_profile_context(state.get("forex_analysis_profile", forex_profile))
            evidence_block = render_supporting_evidence(state)
            evidence_prompt_section = (
                f"{evidence_block}\n\n{render_final_pm_evidence_instruction()}"
                if evidence_block
                else ""
            )
            prompt = f"""As the Portfolio Manager for a currency pair, synthesize the risk analysts' debate and return one structured portfolio rating.

{instrument_context}

{profile_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Bull/Bear Research Debate History:
{investment_debate_history}
- Bull Research History:
{bull_history}
- Bear Research History:
{bear_history}
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Ground the rating in observed currency-pair price action, spread, volatility, and broad macro context from the analysts. Commit to a directional call only when evidence clearly supports one; choose Hold when the case is balanced, materially conflicting, ambiguous, or insufficient. Entry and risk levels are hypothetical observations only; no order is sent. Return the exact analysis profile and a bounded valid_for_seconds value. Use minutes-to-hours horizons only; never use months, years, long-term equity language, issuer valuation, dividends, or company fundamentals. Do not infer issuer-level business data.

{evidence_prompt_section}
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
            if forex_structured_llm is None:
                forex_structured_llm = bind_forex_ollama_structured(
                    llm,
                    ForexPortfolioDecision,
                    "Forex Portfolio Manager",
                )
            try:
                structured_result = invoke_structured_only(
                    forex_structured_llm,
                    prompt,
                    "Portfolio Manager",
                )
                if not isinstance(structured_result, ForexPortfolioDecision):
                    structured_result = ForexPortfolioDecision.model_validate(
                        structured_result.model_dump(mode="python")
                    )
                raw_result = structured_result.model_dump(mode="json")
                json.dumps(raw_result, allow_nan=False)
                final_trade_decision = render_forex_pm_decision(structured_result)
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
        if is_forex:
            new_risk_debate_state = {**risk_debate_state, **new_risk_debate_state}

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
