"""Deterministic Ollama OpenAI-compatible wire-control contracts."""

from __future__ import annotations

import json

import httpx

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.openai_client import OpenAIClient, _supports_reasoning_effort


def _bare_graph(config, mode="forex_mt5"):
    graph = object.__new__(TradingAgentsGraph)
    graph.config = config
    graph.market_data_mode = mode
    return graph


def test_ollama_forex_quick_kwargs_disable_thinking_with_supported_field():
    graph = _bare_graph(
        {
            "llm_provider": "ollama",
            "forex_quick_thinking": False,
            "forex_deep_thinking": True,
        }
    )

    quick_kwargs = graph._get_provider_kwargs(role="quick")

    assert quick_kwargs["reasoning_effort"] == "none"
    assert "extra_body" not in quick_kwargs


def test_ollama_wire_preserves_documented_max_tokens_and_reasoning_effort():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.5:2b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        llm = OpenAIClient(
            model="qwen3.5:2b",
            provider="ollama",
            base_url="http://localhost:11434/v1",
            reasoning_effort="none",
            max_tokens=1024,
            temperature=0.1,
            http_client=http_client,
        ).get_llm()
        llm.invoke("synthetic prompt omitted")
    finally:
        http_client.close()

    body = captured["body"]
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert body["model"] == "qwen3.5:2b"
    assert body["reasoning_effort"] == "none"
    assert body["max_tokens"] == 1024
    assert "max_completion_tokens" not in body
    assert "think" not in body
    assert body["temperature"] == 0.1


def test_ollama_deep_thinking_remains_separate_from_quick_control():
    graph = _bare_graph(
        {
            "llm_provider": "ollama",
            "forex_quick_thinking": False,
            "forex_deep_thinking": True,
        }
    )

    deep_kwargs = graph._get_provider_kwargs(role="deep")

    assert deep_kwargs["extra_body"] == {"think": True}
    assert "reasoning_effort" not in deep_kwargs


def test_reasoning_effort_support_is_provider_scoped():
    assert _supports_reasoning_effort("qwen3.5:2b", provider="ollama") is True
    assert _supports_reasoning_effort("qwen3.5:2b", provider="openai") is False
