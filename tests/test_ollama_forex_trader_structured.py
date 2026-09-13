"""Focused Ollama forex Trader structured-output coverage."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from tradingagents.agents.schemas import TraderAction, TraderProposal
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.structured import StructuredOutputRequiredError
from tradingagents.llm_clients.openai_client import OllamaChatOpenAI


def _state() -> dict:
    return {
        "company_of_interest": "EURUSD",
        "asset_type": "forex",
        "market_data_mode": "forex_mt5",
        "forex_analysis_profile": "INTRADAY",
        "instrument_context": "The instrument to analyze is `EURUSD`.",
        "market_context": "",
        "market_report": "Bid 1.10000; ask 1.10010; spread 1 point; ATR is bounded.",
        "investment_plan": "**Recommendation**: Hold\n**Rationale**: balanced\n**Strategic Actions**: wait",
    }


def _ollama(response_content: str, calls: list[dict]) -> OllamaChatOpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "probe",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.5:2b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
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
        reasoning_effort="none",
        max_tokens=1024,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.unit
def test_ollama_forex_trader_uses_response_format_and_parses_trader_proposal() -> None:
    calls: list[dict] = []
    llm = _ollama(json.dumps({"action": "Hold", "reasoning": "No edge after spread."}), calls)

    result = create_trader(llm, forex_mode=True)(_state())

    assert "**Action**: Hold" in result["trader_investment_plan"]
    assert calls and calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "TraderProposal"
    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]
    assert calls[0]["reasoning_effort"] == "none"
    assert calls[0]["max_tokens"] == 1024


@pytest.mark.unit
def test_ollama_forex_trader_malformed_json_fails_closed_without_prose_fallback() -> None:
    calls: list[dict] = []
    llm = _ollama('{"action": "Hold",', calls)

    with pytest.raises(StructuredOutputRequiredError):
        create_trader(llm, forex_mode=True)(_state())

    assert len(calls) == 1
    assert calls[0]["response_format"]["type"] == "json_schema"


@pytest.mark.unit
def test_non_ollama_forex_trader_path_keeps_existing_binding() -> None:
    structured = MagicMock()
    structured.invoke.return_value = TraderProposal(action=TraderAction.HOLD, reasoning="No edge.")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    result = create_trader(llm, forex_mode=True)(_state())

    assert result["trader_investment_plan"]
    llm.with_structured_output.assert_called_once_with(TraderProposal)
