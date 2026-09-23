from __future__ import annotations

from types import SimpleNamespace

from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.agents.trader.trader import create_trader


class _CapturedLLM:
    def __init__(self):
        self.bound_tool_names = []
        self.last_prompt = None

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]

        class _Bound:
            def __init__(self, outer):
                self.outer = outer

            def invoke(self, prompt):
                self.outer.last_prompt = prompt
                return SimpleNamespace(content="", tool_calls=[])

            __call__ = invoke

        return _Bound(self)


class _FakeMt5Tools:
    def as_tools(self):
        def get_mt5_market_snapshot(symbol):
            return {"symbol": symbol}

        def get_mt5_tick(symbol):
            return {"symbol": symbol}

        def get_mt5_bars(symbol, timeframe, count):
            return {"symbol": symbol, "timeframe": timeframe, "count": count}

        def get_mt5_account_context():
            return {"balance": 10_000}

        def get_mt5_positions(symbol=None):
            return []

        def get_mt5_spread(symbol):
            return {"symbol": symbol, "spread": 0.0002}

        for function in (
            get_mt5_market_snapshot,
            get_mt5_tick,
            get_mt5_bars,
            get_mt5_account_context,
            get_mt5_positions,
            get_mt5_spread,
        ):
            function.name = function.__name__

        return [
            get_mt5_market_snapshot,
            get_mt5_tick,
            get_mt5_bars,
            get_mt5_account_context,
            get_mt5_positions,
            get_mt5_spread,
        ]


def _flatten_prompt(prompt) -> str:
    if hasattr(prompt, "to_messages"):
        return "\n".join(getattr(message, "content", "") or "" for message in prompt.to_messages())
    if isinstance(prompt, list):
        return "\n".join(getattr(item, "content", "") or str(item) for item in prompt)
    return str(prompt)


def test_forex_market_prompt_binds_mt5_tools_only():
    llm = _CapturedLLM()
    node = create_market_analyst(
        llm,
        market_data_mode="forex_mt5",
        mt5_tools=_FakeMt5Tools(),
    )

    node(
        {
            "trade_date": "2026-09-08",
            "asset_type": "forex",
            "company_of_interest": "EURUSD",
            "instrument_context": "EURUSD pair",
            "market_context": "SOURCE: LIVE MT5 BROKER DATA",
            "messages": [("human", "analyze EURUSD")],
        }
    )

    assert llm.bound_tool_names == ["get_mt5_market_snapshot"]
    assert "get_stock_data" not in _flatten_prompt(llm.last_prompt)


def test_forex_news_prompt_binds_global_news_only():
    llm = _CapturedLLM()
    node = create_news_analyst(llm, market_data_mode="forex_mt5")

    node(
        {
            "trade_date": "2026-09-08",
            "asset_type": "forex",
            "company_of_interest": "EURUSD",
            "instrument_context": "EURUSD pair",
            "market_context": "SOURCE: LIVE MT5 BROKER DATA",
            "messages": [("human", "analyze EURUSD")],
        }
    )

    assert llm.bound_tool_names == ["get_global_news"]
    prompt_text = _flatten_prompt(llm.last_prompt)
    assert "P/E" not in prompt_text
    assert "EPS" not in prompt_text
    assert "StockTwits" not in prompt_text


def _forex_state():
    return {
        "trade_date": "2026-09-08",
        "asset_type": "forex",
        "company_of_interest": "EURUSD",
        "instrument_context": "EURUSD currency pair; bid 1.1000 ask 1.1002",
        "market_context": "SOURCE: LIVE MT5 BROKER DATA",
        "market_report": "EURUSD price action is range-bound near 1.1001.",
        "sentiment_report": "Unavailable for forex; do not infer social sentiment.",
        "news_report": "Global macro news is mixed.",
        "fundamentals_report": "Unavailable for forex; do not infer company fundamentals.",
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


class _PromptCaptureLLM:
    def __init__(self, structured_result=None):
        self.prompts = []
        self.structured_result = structured_result

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content="captured", tool_calls=[])

    def with_structured_output(self, schema):
        result = self.structured_result
        outer = self

        class _Structured:
            def invoke(self, prompt):
                outer.prompts.append(prompt)
                if isinstance(result, Exception):
                    raise result
                return result

        return _Structured()


def _prompt_text(prompt):
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in prompt)
    if hasattr(prompt, "to_messages"):
        return "\n".join(str(message.content) for message in prompt.to_messages())
    return str(prompt)


def test_forex_portfolio_manager_keeps_structured_raw_result():
    raw = PortfolioDecision(
        rating=PortfolioRating.SELL,
        executive_summary="x",
        investment_thesis="y",
    )
    llm = _PromptCaptureLLM(raw)
    result = create_portfolio_manager(llm)(_forex_state())

    assert result["portfolio_manager_raw_result"]["rating"] == "Sell"
    assert result["final_trade_decision"].startswith("**Rating**: Sell")
    assert result["normalization_status"] == "NORMALIZED"
    assert result["normalization_error"] is None


def test_forex_portfolio_manager_requests_concise_json_only_output():
    llm = _PromptCaptureLLM(
        PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Remain flat.",
            investment_thesis="Evidence is mixed.",
        )
    )

    create_portfolio_manager(llm)(_forex_state())

    prompt = _prompt_text(llm.prompts[0]).lower()
    assert "return only the required json object" in prompt
    assert "no markdown or prose outside json" in prompt
    assert "keep each narrative field concise" in prompt
    assert "avoid repeating the same rationale" in prompt


def test_forex_portfolio_manager_fails_closed_without_structured_output():
    llm = _PromptCaptureLLM(RuntimeError("provider rejected schema"))
    result = create_portfolio_manager(llm)(_forex_state())

    assert result["portfolio_manager_raw_result"]["error"] == "STRUCTURED_OUTPUT_REQUIRED"
    assert result["final_trade_decision"] == "FOREX_PORTFOLIO_MANAGER_FAILED"
    assert result["normalization_status"] == "FAILED"
    assert result["normalization_error"]
    # Forex structured output permits one bounded retry, never a prose
    # fallback or an unbounded loop.
    assert len(llm.prompts) == 2


def test_forex_portfolio_manager_retains_safe_structured_failure_category():
    llm = _PromptCaptureLLM(RuntimeError("provider rejected schema"))
    result = create_portfolio_manager(llm)(_forex_state())

    assert "cause_type=RuntimeError" in result["normalization_error"]
    assert "provider rejected schema" not in result["normalization_error"]


def test_forex_downstream_prompts_remove_stock_specific_sections():
    factories = [
        create_bull_researcher,
        create_bear_researcher,
        create_aggressive_debator,
        create_conservative_debator,
        create_neutral_debator,
        create_trader,
    ]
    for factory in factories:
        llm = _PromptCaptureLLM()
        factory(llm)(_forex_state())
        text = "\n".join(_prompt_text(prompt) for prompt in llm.prompts)
        assert "Company Fundamentals Report" not in text
        assert "Social Media Sentiment Report" not in text
        assert "StockTwits" not in text
        assert "fundamentals are unavailable" in text.lower()


def test_forex_trader_treats_price_levels_as_observational_only():
    llm = _PromptCaptureLLM()
    create_trader(llm)(_forex_state())
    text = _prompt_text(llm.prompts[0]).lower()
    assert "observational only" in text
    assert "no order is sent" in text


def test_stock_portfolio_manager_keeps_general_structured_path():
    llm = _PromptCaptureLLM(
        PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Remain flat.",
            investment_thesis="Evidence is mixed.",
        )
    )
    state = _forex_state()
    state["asset_type"] = "stock"
    state["company_of_interest"] = "AAPL"

    result = create_portfolio_manager(llm)(state)

    assert "normalization_status" not in result
    assert result["final_trade_decision"].startswith("**Rating**: Hold")
    assert len(llm.prompts) == 1
    assert "return only the required json object" not in _prompt_text(llm.prompts[0]).lower()
