from __future__ import annotations

import json

import httpx
import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import ForexPortfolioDecision, ResearchPlan, TraderProposal
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.structured import StructuredOutputRequiredError
from tradingagents.llm_clients.openai_client import OllamaChatOpenAI


def _forex_state() -> dict:
    return {
        "trade_date": "2026-09-13",
        "asset_type": "forex",
        "company_of_interest": "EURUSD",
        "forex_analysis_profile": "INTRADAY",
        "instrument_context": "EURUSD pair; bid=1.1000 ask=1.1002 spread=2 points",
        "market_context": "SOURCE: LIVE MT5 BROKER DATA",
        "market_report": "Range-bound price action; ATR remains bounded.",
        "news_report": "Global macro news is mixed.",
        "investment_plan": "**Recommendation**: Hold\n**Rationale**: balanced\n**Strategic Actions**: wait",
        "trader_investment_plan": "**Action**: Hold\n**Reasoning**: no edge",
        "past_context": "",
        "investment_debate_state": {
            "history": "Bull: momentum is constructive.\nBear: spread-adjusted edge is weak.",
            "bull_history": "Bull: momentum is constructive.",
            "bear_history": "Bear: spread-adjusted edge is weak.",
            "current_response": "",
            "judge_decision": "",
            "count": 1,
        },
        "risk_debate_state": {
            "history": "Aggressive: small size.\nConservative: wait for confirmation.",
            "aggressive_history": "Aggressive: small size.",
            "conservative_history": "Conservative: wait for confirmation.",
            "neutral_history": "Neutral: balanced.",
            "latest_speaker": "Neutral Analyst",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
    }


_VALID_RESPONSES = {
    "ResearchPlan": {
        "recommendation": "Hold",
        "rationale": "The bull and bear cases are balanced after spread costs.",
        "strategic_actions": "Wait for a clearer intraday edge.",
    },
    "TraderProposal": {
        "action": "Hold",
        "reasoning": "No spread-adjusted edge is present.",
    },
    "ForexPortfolioDecision": {
        "rating": "Hold",
        "executive_summary": "Remain flat while evidence is balanced.",
        "investment_thesis": "The risk debate does not support a directional entry.",
        "analysis_profile": "INTRADAY",
        "valid_for_seconds": 3600,
        "evidence_use_status": "DISABLED",
        "evidence_refs_used": [],
        "evidence_refs_rejected": [],
    },
}


def _ollama(response_by_schema: dict[str, str | dict], calls: list[dict]) -> OllamaChatOpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        schema_name = body["response_format"]["json_schema"]["name"]
        response = response_by_schema[schema_name]
        content = response if isinstance(response, str) else json.dumps(response)
        return httpx.Response(
            200,
            json={
                "id": "structured-probe",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.5:2b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
            request=request,
        )

    return OllamaChatOpenAI(
        model="qwen3.5:2b",
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        temperature=0,
        max_tokens=1024,
        # Simulate the graph's deep default. The forex structured binding must
        # override this with the documented reasoning_effort=none request.
        extra_body={"think": True},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _assert_json_schema_wire(call: dict, schema_name: str) -> None:
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["name"] == schema_name
    assert call["reasoning_effort"] == "none"
    assert call["max_tokens"] == 1024
    assert "tools" not in call
    assert "tool_choice" not in call
    assert (call.get("extra_body") or {}).get("think") is not True


@pytest.mark.unit
def test_ollama_forex_research_manager_uses_json_schema_and_parses() -> None:
    calls: list[dict] = []
    llm = _ollama(_VALID_RESPONSES, calls)

    result = create_research_manager(llm)(_forex_state())

    assert "**Recommendation**: Hold" in result["investment_plan"]
    assert len(calls) == 1
    _assert_json_schema_wire(calls[0], ResearchPlan.__name__)


@pytest.mark.unit
def test_ollama_forex_trader_uses_json_schema_and_parses() -> None:
    calls: list[dict] = []
    llm = _ollama(_VALID_RESPONSES, calls)

    result = create_trader(llm, forex_mode=True)(_forex_state())

    assert "**Action**: Hold" in result["trader_investment_plan"]
    assert len(calls) == 1
    _assert_json_schema_wire(calls[0], TraderProposal.__name__)


@pytest.mark.unit
def test_ollama_forex_portfolio_manager_uses_json_schema_and_parses() -> None:
    calls: list[dict] = []
    llm = _ollama(_VALID_RESPONSES, calls)

    result = create_portfolio_manager(llm)(_forex_state())

    assert result["normalization_status"] == "NORMALIZED"
    assert "**Rating**: Hold" in result["final_trade_decision"]
    assert len(calls) == 1
    _assert_json_schema_wire(calls[0], ForexPortfolioDecision.__name__)


@pytest.mark.unit
@pytest.mark.parametrize("node", ("research", "trader", "portfolio"))
def test_ollama_forex_structured_nodes_fail_closed_on_malformed_output(node: str) -> None:
    calls: list[dict] = []
    malformed = dict.fromkeys(_VALID_RESPONSES, '{"unexpected": "value"}')
    llm = _ollama(malformed, calls)
    state = _forex_state()

    if node == "research":
        with pytest.raises(StructuredOutputRequiredError):
            create_research_manager(llm)(state)
        schema_name = ResearchPlan.__name__
    elif node == "trader":
        with pytest.raises(StructuredOutputRequiredError):
            create_trader(llm, forex_mode=True)(state)
        schema_name = TraderProposal.__name__
    else:
        result = create_portfolio_manager(llm)(state)
        assert result["normalization_status"] == "FAILED"
        assert result["final_trade_decision"] == "FOREX_PORTFOLIO_MANAGER_FAILED"
        schema_name = ForexPortfolioDecision.__name__

    assert len(calls) == 1
    _assert_json_schema_wire(calls[0], schema_name)
