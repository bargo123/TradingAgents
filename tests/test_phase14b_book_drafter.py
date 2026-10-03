from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tradingagents.knowledge.models import KnowledgeHit
from tradingagents.self_enhancement.book_drafter import (
    OllamaStrategyDrafter,
    StrategyDraft,
    StrategyDraftBatch,
    StrategyDraftClaim,
)

SOURCE_TEXT = "LONG when momentum > 1.2 points after confirmation"
SECRET_PROMPT_MARKER = "private-prompt-marker-should-not-be-returned"
PRIVATE_REASONING_MARKER = "private-reasoning-marker-should-not-be-returned"


def _hit() -> KnowledgeHit:
    return KnowledgeHit(
        chunk_id="chunk-14",
        document_id="doc-book-1",
        source_hash="a" * 64,
        text=SOURCE_TEXT,
        source_filename="book.pdf",
        title="Microstructure",
        page=14,
        section="Entry rules",
        score=0.92,
    )


def _draft_payload() -> dict:
    quote = SOURCE_TEXT
    offset = 0
    return {
        "specs": [
            {
                "name": "Short-horizon continuation",
                "family": "MOMENTUM_CONTINUATION",
                "required_data": ["momentum"],
                "implementation_confidence": 0.65,
                "rule_claims": [
                    {
                        "stage": "ENTRY",
                        "operator": "GREATER_THAN",
                        "direction": "LONG",
                        "value": 1.2,
                        "unit": "points",
                        "condition": "after confirmation",
                        "horizon_seconds": None,
                        "origin": "SOURCE_SUPPORTED_CONCEPT",
                        "evidence_ref": "E1",
                        "start_offset": offset,
                        "end_offset": offset + len(quote),
                        "quote": quote,
                    }
                ],
            }
        ]
    }


class _Response:
    def __init__(self, body: object):
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.body


class _FakeTransport:
    def __init__(self, body: object | None = None, error: Exception | None = None):
        self.body = body
        self.error = error
        self.calls: list[dict] = []

    def post(self, url: str, *, json: dict, timeout: float) -> _Response:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.error is not None:
            raise self.error
        if self.body is not None:
            return _Response(self.body)
        return _Response(
            {
                "message": {"content": json_draft_content},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 137,
                "eval_count": 81,
            }
        )


json_draft_content = json.dumps(_draft_payload(), separators=(",", ":"))


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    ],
)
def test_drafter_accepts_loopback_ollama_endpoint(endpoint: str) -> None:
    drafter = OllamaStrategyDrafter(endpoint, "qwen3.5:2b", transport=_FakeTransport())

    assert drafter.endpoint == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://localhost:11434",
        "http://localhost.evil.example:11434",
        "http://192.168.1.5:11434",
        "http://user@127.0.0.1:11434",
        "http://127.0.0.1:11434/v1",
    ],
)
def test_drafter_rejects_non_native_or_non_loopback_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError, match="loopback Ollama"):
        OllamaStrategyDrafter(endpoint, "qwen3.5:2b", transport=_FakeTransport())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": 601},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": True},
        {"max_output_tokens": 0},
        {"max_output_tokens": 8193},
        {"max_output_tokens": True},
    ],
)
def test_drafter_rejects_unbounded_timeout_and_output(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        OllamaStrategyDrafter(
            "http://localhost:11434", "qwen3.5:2b", transport=_FakeTransport(), **kwargs
        )


def test_drafter_sends_bounded_native_json_schema_request() -> None:
    transport = _FakeTransport()
    drafter = OllamaStrategyDrafter(
        "http://localhost:11434",
        "qwen3.5:2b",
        timeout_seconds=45,
        max_output_tokens=768,
        transport=transport,
    )

    result = drafter.draft([_hit()], max_specs=2)

    request = transport.calls[0]
    body = request["json"]
    assert request["url"] == "http://localhost:11434/api/chat"
    assert request["timeout"] == 45
    assert body["model"] == "qwen3.5:2b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["temperature"] == 0
    assert body["options"]["num_predict"] == 768
    assert body["options"]["num_ctx"] == 8192
    assert body["format"]["type"] == "object"
    assert body["format"]["additionalProperties"] is False
    assert isinstance(body["format"]["properties"]["specs"]["items"], dict)
    assert "tools" not in body
    assert result.ok is True
    assert len(result.specs) == 1
    assert isinstance(result.specs[0], StrategyDraft)


def test_drafter_sends_explicit_bounded_context_length() -> None:
    transport = _FakeTransport()
    drafter = OllamaStrategyDrafter(
        "http://localhost:11434",
        "qwen3.5:2b",
        context_tokens=4096,
        transport=transport,
    )

    drafter.draft([_hit()], max_specs=1)

    assert transport.calls[0]["json"]["options"]["num_ctx"] == 4096


def test_drafter_rejects_unbounded_or_invalid_context_length() -> None:
    for context_tokens in (True, 0, 255, 32769):
        with pytest.raises((TypeError, ValueError), match="context_tokens"):
            OllamaStrategyDrafter(
                "http://localhost:11434",
                "qwen3.5:2b",
                context_tokens=context_tokens,
                transport=_FakeTransport(),
            )


def test_drafter_strictly_rejects_malformed_or_schema_invalid_json() -> None:
    for content in (
        "not-json",
        json.dumps({"specs": [{"name": "missing required fields"}]}),
        json.dumps({"specs": [], "hidden_reasoning": PRIVATE_REASONING_MARKER}),
    ):
        transport = _FakeTransport(
            {
                "message": {"content": content},
                "done": True,
                "done_reason": "stop",
            }
        )
        result = OllamaStrategyDrafter(
            "http://localhost:11434", "qwen3.5:2b", transport=transport
        ).draft([_hit()], max_specs=2)

        assert result.ok is False
        assert result.error_code == "SCHEMA_INVALID"
        assert len(transport.calls) == 1


def test_drafter_fails_closed_on_provider_error_without_retry() -> None:
    transport = _FakeTransport(error=TimeoutError(PRIVATE_REASONING_MARKER))

    result = OllamaStrategyDrafter(
        "http://localhost:11434", "qwen3.5:2b", transport=transport
    ).draft([_hit()], max_specs=1)

    assert result.ok is False
    assert result.error_code == "PROVIDER_ERROR"
    assert result.provider == "ollama-local"
    assert result.model == "qwen3.5:2b"
    assert result.error_type == "TimeoutError"
    assert len(transport.calls) == 1


def test_drafter_result_contains_no_prompt_completion_or_reasoning() -> None:
    transport = _FakeTransport()
    result = OllamaStrategyDrafter(
        "http://localhost:11434", "qwen3.5:2b", transport=transport
    ).draft([_hit()], max_specs=1)

    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert SECRET_PROMPT_MARKER not in serialized
    assert PRIVATE_REASONING_MARKER not in serialized
    assert "prompt" not in result.to_dict()
    assert "completion" not in result.to_dict()
    assert "reasoning" not in result.to_dict()
    assert result.input_tokens == 137
    assert result.output_tokens == 81
    assert result.finish_reason == "stop"
    assert result.timeout_seconds == 120
    assert result.context_tokens == 8192
    assert result.max_output_tokens == 2048


def test_schema_failure_retains_safe_usage_metadata_not_raw_completion() -> None:
    transport = _FakeTransport(
        {
            "message": {"content": PRIVATE_REASONING_MARKER},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 19,
            "eval_count": 23,
        }
    )

    result = OllamaStrategyDrafter(
        "http://localhost:11434", "qwen3.5:2b", transport=transport
    ).draft([_hit()], max_specs=1)

    assert result.ok is False
    assert result.input_tokens == 19
    assert result.output_tokens == 23
    assert result.finish_reason == "stop"
    assert PRIVATE_REASONING_MARKER not in json.dumps(result.to_dict())


def test_drafter_rejects_output_exceeding_requested_spec_bound() -> None:
    body = _draft_payload()
    body["specs"].append(body["specs"][0])
    transport = _FakeTransport(
        {"message": {"content": json.dumps(body)}, "done": True, "done_reason": "stop"}
    )

    result = OllamaStrategyDrafter(
        "http://localhost:11434", "qwen3.5:2b", transport=transport
    ).draft([_hit()], max_specs=1)

    assert result.ok is False
    assert result.error_code == "SCHEMA_INVALID"


def test_draft_batch_uses_closed_strict_schema() -> None:
    schema = StrategyDraftBatch.model_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"specs"}
    assert schema["properties"]["specs"]["maxItems"] <= 10
    assert all(
        definition.get("additionalProperties") is False
        for definition in schema["$defs"].values()
        if definition.get("type") == "object"
    )


def test_horizon_field_is_rejected_on_non_horizon_claims() -> None:
    payload = _draft_payload()["specs"][0]["rule_claims"][0]
    payload["horizon_seconds"] = 5

    with pytest.raises(ValidationError, match="horizon_seconds"):
        StrategyDraftClaim.model_validate_json(json.dumps(payload))


def test_drafter_rejects_truncated_output_even_if_transport_succeeded() -> None:
    transport = _FakeTransport(
        {
            "message": {"content": json_draft_content},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 21,
            "eval_count": 768,
        }
    )

    result = OllamaStrategyDrafter(
        "http://localhost:11434", "qwen3.5:2b", transport=transport
    ).draft([_hit()], max_specs=1)

    assert result.ok is False
    assert result.error_code == "SCHEMA_INVALID"
    assert result.finish_reason == "length"
    assert result.output_tokens == 768
