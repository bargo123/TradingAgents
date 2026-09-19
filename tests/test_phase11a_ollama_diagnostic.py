import json
from types import SimpleNamespace

import pytest

from scripts.phase11a_ollama_diagnostic import _compact_validator, select_real_packets
from tests.fixtures.phase11a_knowledge import fixture_source
from tradingagents.distillation.ollama_diagnostic import (
    build_native_payload,
    classify_diagnostic_error,
    run_native_call,
    summarise_native_response,
)
from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner


def test_native_payload_uses_json_schema_and_bounded_native_controls():
    payload = build_native_payload(
        "qwen3.5:2b",
        [{"role": "user", "content": "answer"}],
        {"type": "object", "properties": {"answer": {"type": "string"}}},
        context_tokens=8192,
        max_tokens=256,
        keep_alive="5m",
    )

    assert payload["model"] == "qwen3.5:2b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"]["type"] == "object"
    assert payload["options"] == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 256}
    assert payload["keep_alive"] == "5m"
    assert "tools" not in payload


def test_native_response_summary_is_scalar_only_and_reports_rates():
    response = {
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": '{"answer":"ok"}'},
        "prompt_eval_count": 100,
        "prompt_eval_duration": 2_000_000_000,
        "eval_count": 50,
        "eval_duration": 5_000_000_000,
        "load_duration": 3_000_000_000,
        "total_duration": 10_000_000_000,
    }

    summary = summarise_native_response(
        response,
        test_name="tiny",
        model="qwen3.5:2b",
        request_chars=120,
        approximate_input_tokens=30,
        schema_valid=True,
        elapsed_seconds=10.0,
    )

    assert summary["status"] == "SUCCESS"
    assert summary["prompt_eval_count"] == 100
    assert summary["eval_count"] == 50
    assert summary["prompt_tokens_per_second"] == 50.0
    assert summary["generation_tokens_per_second"] == 10.0
    assert summary["done_reason"] == "stop"
    assert "content" not in json.dumps(summary)
    assert "answer" not in json.dumps(summary)


def test_native_response_summary_marks_invalid_json_without_persisting_it():
    response = {
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": "not-json"},
    }

    summary = summarise_native_response(
        response,
        test_name="tiny",
        model="qwen3.5:2b",
        request_chars=120,
        approximate_input_tokens=30,
        schema_valid=False,
        elapsed_seconds=1.0,
        parse_error="JSONDecodeError",
    )

    assert summary["status"] == "JSON_INVALID"
    assert summary["parse_error"] == "JSONDecodeError"
    assert "not-json" not in json.dumps(summary)


def test_native_response_classifies_length_as_output_truncated():
    response = {
        "done": True,
        "done_reason": "length",
        "message": {"role": "assistant", "content": "partial"},
    }

    summary = summarise_native_response(
        response,
        test_name="real",
        model="qwen3.5:2b",
        request_chars=120,
        approximate_input_tokens=30,
        schema_valid=False,
        elapsed_seconds=1.0,
        parse_error="JSONDecodeError",
    )

    assert summary["status"] == "OUTPUT_TRUNCATED"
    assert summary["output_truncated"] is True


def test_diagnostic_error_classification_is_type_only():
    assert classify_diagnostic_error(TimeoutError("private")) == "PROVIDER_TIMEOUT"
    assert classify_diagnostic_error(ValueError("private")) == "PROVIDER_ERROR"
    assert classify_diagnostic_error(SimpleNamespace(status_code=500)) == "HTTP_ERROR"


def test_native_call_preserves_validator_failure_category_without_response_text():
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "done": True,
                "done_reason": "stop",
                "message": {"content": '{"answer":"ok"}'},
            }

    class _Client:
        def post(self, path, *, json):
            assert path == "/api/chat"
            return _Response()

    result = run_native_call(
        _Client(),
        model="qwen3.5:2b",
        test_name="real",
        messages=[{"role": "user", "content": "source"}],
        schema={"type": "object"},
        validator=lambda value: "GROUNDING_FAILED",
        context_tokens=8192,
        max_tokens=256,
        keep_alive="5m",
    )

    assert result["status"] == "GROUNDING_FAILED"


def test_select_real_packets_is_deterministic_and_bounded():
    packets = [SimpleNamespace(packet_id=str(index)) for index in range(5)]

    selected = select_real_packets(packets, (4, 2, 3))

    assert [packet.packet_id for packet in selected] == ["4", "2", "3"]


def test_compact_validator_reports_only_safe_size_and_status_metadata():
    packet = SourcePacketPlanner.plan(
        fixture_source(), (), PlannerConfig(max_blocks=1, max_chars=2_000, max_estimated_tokens=400)
    ).packets[0]
    telemetry = {}
    validator = _compact_validator(packet, telemetry)

    result = validator(
        {
            "lesson_type": "DEFINITION",
            "difficulty": "FOUNDATIONAL",
            "topic": "order flow imbalance",
            "question": "What is the concept?",
            "answer": "The supplied evidence describes the concept.",
            "claims": [{"claim": "The supplied source describes the concept.", "refs": ["B0"]}],
        }
    )

    assert result is None
    assert telemetry["schema_status"] == "VALID"
    assert telemetry["grounding_status"] == "VALID"
    assert telemetry["quality_status"] == "VALID"
    assert telemetry["compact_response_chars"] > 0
    assert telemetry["hydrated_example_chars"] > telemetry["compact_response_chars"]


def test_compact_validator_reports_bounded_grounding_reason_only():
    packet = SourcePacketPlanner.plan(
        fixture_source(), (), PlannerConfig(max_blocks=1, max_chars=2_000, max_estimated_tokens=400)
    ).packets[0]
    telemetry = {}
    validator = _compact_validator(packet, telemetry)

    result = validator(
        {
            "lesson_type": "DEFINITION",
            "difficulty": "FOUNDATIONAL",
            "topic": "order flow imbalance",
            "question": "What is the concept?",
            "answer": "The supplied evidence describes the concept.",
            "claims": [{"claim": "The moon is made of cheese.", "refs": ["B0"]}],
        }
    )

    assert result == "GROUNDING_FAILED"
    assert telemetry["grounding_reason"] == "UNSUPPORTED_CLAIM"


@pytest.mark.parametrize("indexes", [(4, 4), (5,), ()])
def test_select_real_packets_rejects_invalid_indexes(indexes):
    packets = [SimpleNamespace(packet_id=str(index)) for index in range(5)]

    with pytest.raises(ValueError):
        select_real_packets(packets, indexes)
