from __future__ import annotations

from types import SimpleNamespace

import pytest

import tradingagents.forex.news as forex_news
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.schemas import ForexPortfolioDecision, PortfolioRating
from tradingagents.agents.trader.trader import create_trader


class _PromptCaptureLLM:
    def __init__(self, structured_result=None):
        self.prompts = []
        self.structured_result = structured_result

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content="captured", tool_calls=[])

    def bind_tools(self, tools):
        outer = self

        class _Bound:
            def invoke(self, prompt):
                outer.prompts.append(prompt)
                return SimpleNamespace(content="captured", tool_calls=[])

            __call__ = invoke

        return _Bound()

    def with_structured_output(self, schema):
        outer = self

        class _Structured:
            def invoke(self, prompt):
                outer.prompts.append(prompt)
                if isinstance(outer.structured_result, Exception):
                    raise outer.structured_result
                return outer.structured_result

        return _Structured()


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


def _forex_state():
    return {
        "trade_date": "2026-09-08",
        "asset_type": "forex",
        "company_of_interest": "EURUSD",
        "forex_analysis_profile": "INTRADAY",
        "instrument_context": "EURUSD currency pair",
        "market_context": "SOURCE: LIVE MT5 BROKER DATA\nFOREX ANALYSIS PROFILE: INTRADAY",
        "market_report": "M1/M5/M15/H1 multi-bar price-action context.",
        "sentiment_report": "",
        "news_report": "MACRO/EVENT DATA UNAVAILABLE",
        "fundamentals_report": "",
        "investment_plan": "Hold while evidence is mixed.",
        "trader_investment_plan": "Hold; price levels are observational only.",
        "past_context": "",
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "count": 0,
        },
        "risk_debate_state": {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "latest_speaker": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 0,
        },
    }


def test_forex_news_wrapper_uses_exact_unavailable_semantics(monkeypatch):
    monkeypatch.setattr(forex_news, "route_to_vendor", lambda *args, **kwargs: "No global news found for 2026-09-08")
    assert forex_news.get_forex_global_news.invoke({"curr_date": "2026-09-08"}) == (
        "MACRO/EVENT DATA UNAVAILABLE"
    )

    monkeypatch.setattr(forex_news, "route_to_vendor", lambda *args, **kwargs: "Error fetching global news: timeout")
    assert forex_news.get_forex_global_news.invoke({"curr_date": "2026-09-08"}) == (
        "MACRO/EVENT DATA UNAVAILABLE"
    )


def test_forex_news_wrapper_preserves_broad_report_and_tool_name(monkeypatch):
    report = "## Global Market News\n\n### ECB policy outlook"
    monkeypatch.setattr(forex_news, "route_to_vendor", lambda *args, **kwargs: report)

    assert forex_news.get_forex_global_news.name == "get_global_news"
    assert forex_news.get_forex_global_news.invoke({"curr_date": "2026-09-08"}) == report


def test_forex_news_prompt_mentions_profile_and_macro_uncertainty():
    llm = _PromptCaptureLLM()
    node = create_news_analyst(llm, market_data_mode="forex_mt5")
    state = _forex_state()
    state["messages"] = [("human", "analyze EURUSD")]
    node(state)

    prompt = _prompt_text(llm.prompts[0])
    assert "MACRO/EVENT DATA UNAVAILABLE" in prompt
    assert "minutes to hours" in prompt
    assert "months" in prompt.lower()  # appears only in the prohibition
    assert "company earnings" not in prompt.lower()


@pytest.mark.parametrize(
    "factory",
    [
        create_bull_researcher,
        create_bear_researcher,
        create_aggressive_debator,
        create_conservative_debator,
        create_neutral_debator,
        create_trader,
        create_research_manager,
    ],
)
def test_forex_downstream_agents_use_intraday_language(factory):
    llm = _PromptCaptureLLM()
    factory(llm)(_forex_state())
    text = "\n".join(_prompt_text(prompt) for prompt in llm.prompts).lower()

    assert "intraday" in text
    assert "minutes to hours" in text
    assert "no order is sent" in text or "no order" in text
    assert "p/e" not in text
    assert "stocktwits" not in text


def test_forex_portfolio_manager_uses_forex_schema_and_validity_language():
    llm = _PromptCaptureLLM(
        ForexPortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Wait for confirmation.",
            investment_thesis="Evidence is mixed.",
            time_horizon="minutes to hours",
        )
    )
    result = create_portfolio_manager(llm)(_forex_state())
    text = _prompt_text(llm.prompts[0]).lower()

    assert result["portfolio_manager_raw_result"]["analysis_profile"] == "INTRADAY"
    assert result["portfolio_manager_raw_result"]["valid_for_seconds"] == 3600
    assert "minutes to hours" in text
    assert "validity" in text or "valid_for_seconds" in text
    assert "3-6 months" not in text
    assert "issuer-level" in text
    assert "3-6 months" not in ForexPortfolioDecision.model_json_schema()["properties"]["time_horizon"]["description"]


def test_stock_research_manager_prompt_does_not_receive_forex_profile():
    llm = _PromptCaptureLLM()
    create_research_manager(llm)(
        {
            "company_of_interest": "AAPL",
            "investment_debate_state": {"history": "Bull and bear.", "count": 0},
        }
    )
    text = _prompt_text(llm.prompts[0])
    assert "FOREX ANALYSIS PROFILE" not in text
    assert "investment plan" in text.lower()
