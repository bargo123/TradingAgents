from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

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


def _ollama_sequence(
    responses: list[str | dict], calls: list[dict]
) -> OllamaChatOpenAI:
    """Return an Ollama mock that consumes one PM response per invocation."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        response = responses.pop(0)
        content = response if isinstance(response, str) else json.dumps(response)
        return httpx.Response(
            200,
            json={
                "id": "structured-sequence-probe",
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
    # Graph-created deep clients carry their normal thinking default.  A
    # schema-constrained forex call must override that at the request boundary
    # so Ollama receives an explicit non-thinking request.
    assert call.get("think") is False


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
def test_graph_created_deep_ollama_portfolio_manager_overrides_thinking() -> None:
    """A graph deep client carries think=true until the PM binding overrides it."""

    calls: list[dict] = []
    llm = _ollama(_VALID_RESPONSES, calls)
    result = create_portfolio_manager(llm)(_forex_state())

    assert result["normalization_status"] == "NORMALIZED"
    assert len(calls) == 1
    _assert_json_schema_wire(calls[0], ForexPortfolioDecision.__name__)


@pytest.mark.unit
@pytest.mark.parametrize("model_profile", ("intraday", "SWING"))
def test_ollama_forex_portfolio_rebinds_runtime_analysis_profile(model_profile: str) -> None:
    calls: list[dict] = []
    payload = dict(_VALID_RESPONSES["ForexPortfolioDecision"])
    payload["analysis_profile"] = model_profile
    llm = _ollama({"ForexPortfolioDecision": payload}, calls)

    result = create_portfolio_manager(llm, forex_profile="INTRADAY")(_forex_state())

    assert result["normalization_status"] == "NORMALIZED"
    assert result["portfolio_manager_raw_result"]["analysis_profile"] == "INTRADAY"
    assert len(calls) == 1


@pytest.mark.unit
def test_ollama_forex_portfolio_supplies_missing_runtime_analysis_profile() -> None:
    calls: list[dict] = []
    payload = dict(_VALID_RESPONSES["ForexPortfolioDecision"])
    payload.pop("analysis_profile")
    llm = _ollama({"ForexPortfolioDecision": payload}, calls)

    result = create_portfolio_manager(llm, forex_profile="INTRADAY")(_forex_state())

    assert result["normalization_status"] == "NORMALIZED"
    assert result["portfolio_manager_raw_result"]["analysis_profile"] == "INTRADAY"
    assert len(calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation,expected_path",
    (
        (lambda payload: payload.pop("executive_summary"), "executive_summary"),
        (lambda payload: payload.__setitem__("rating", "NotARating"), "rating"),
    ),
)
def test_ollama_forex_portfolio_still_fails_closed_for_non_runtime_invalid_fields(
    mutation, expected_path: str
) -> None:
    calls: list[dict] = []
    payload = dict(_VALID_RESPONSES["ForexPortfolioDecision"])
    mutation(payload)
    llm = _ollama({"ForexPortfolioDecision": payload}, calls)

    result = create_portfolio_manager(llm, forex_profile="INTRADAY")(_forex_state())

    assert result["normalization_status"] == "FAILED"
    assert result["final_trade_decision"] == "FOREX_PORTFOLIO_MANAGER_FAILED"
    assert f'"path":"{expected_path}"' in result["normalization_error"]
    assert len(calls) == 2


@pytest.mark.unit
def test_forex_portfolio_schema_validator_still_rejects_model_profile_without_rebinding() -> None:
    payload = dict(_VALID_RESPONSES["ForexPortfolioDecision"])
    payload["analysis_profile"] = "SWING"

    with pytest.raises(ValidationError):
        ForexPortfolioDecision.model_validate(payload)


@pytest.mark.unit
def test_ollama_forex_portfolio_manager_retries_only_the_failed_structured_call() -> None:
    calls: list[dict] = []
    llm = _ollama_sequence(
        ['{"unexpected": "value"}', _VALID_RESPONSES["ForexPortfolioDecision"]],
        calls,
    )

    result = create_portfolio_manager(llm)(_forex_state())

    assert result["normalization_status"] == "NORMALIZED"
    assert "**Rating**: Hold" in result["final_trade_decision"]
    assert len(calls) == 2
    for call in calls:
        _assert_json_schema_wire(call, ForexPortfolioDecision.__name__)
        assert call["response_format"]["json_schema"]["schema"] == calls[0]["response_format"]["json_schema"]["schema"]
    wire_schema = calls[0]["response_format"]["json_schema"]["schema"]
    model_schema = ForexPortfolioDecision.model_json_schema()
    assert set(wire_schema["properties"]) == set(model_schema["properties"])
    assert wire_schema["properties"]["rating"]["enum"] == model_schema["$defs"]["PortfolioRating"]["enum"]
    assert wire_schema["additionalProperties"] is False


@pytest.mark.unit
def test_ollama_forex_portfolio_manager_retries_once_then_fails_closed() -> None:
    calls: list[dict] = []
    llm = _ollama_sequence(['{"unexpected": "value"}', '{"unexpected": "value"}'], calls)

    result = create_portfolio_manager(llm)(_forex_state())

    assert result["normalization_status"] == "FAILED"
    assert result["final_trade_decision"] == "FOREX_PORTFOLIO_MANAGER_FAILED"
    assert "cause_type=ValidationError" in result["normalization_error"]
    assert '"attempt":1' in result["normalization_error"]
    assert '"attempt":2' in result["normalization_error"]
    assert '"exception_type":"ValidationError"' in result["normalization_error"]
    assert '"path":"rating"' in result["normalization_error"]
    assert '"path":"executive_summary"' in result["normalization_error"]
    assert '"path":"investment_thesis"' in result["normalization_error"]
    assert '"type":"missing"' in result["normalization_error"]
    assert '"input_type":"dict"' in result["normalization_error"]
    # Diagnostics must not retain the malformed payload itself.
    assert '"unexpected"' not in result["normalization_error"]
    assert len(calls) == 2
    for call in calls:
        _assert_json_schema_wire(call, ForexPortfolioDecision.__name__)


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

    expected_calls = 2 if node == "portfolio" else 1
    assert len(calls) == expected_calls
    for call in calls:
        _assert_json_schema_wire(call, schema_name)
