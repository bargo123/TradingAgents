"""Deterministic contracts for the standalone Qwen prose benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.benchmark_qwen_prose as benchmark
from scripts.benchmark_qwen_prose import (
    BenchmarkRecord,
    build_synthetic_state,
    classify_visible_output,
    run_agent_call,
    run_benchmark,
    summarize_records,
)


class FakeResponse:
    content = "visible synthetic report"
    response_metadata = {
        "finish_reason": "stop",
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }
    usage_metadata = {
        "input_tokens": 10,
        "output_tokens": 8,
        "total_tokens": 18,
        "input_token_details": {},
        "output_token_details": {},
    }
    additional_kwargs = {"reasoning_content": "private reasoning must not be persisted"}


class FakeLLM:
    model_name = "qwen3.5:2b"

    def __init__(self, content: str = "visible synthetic report"):
        self.content = content
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        response = FakeResponse()
        response.content = self.content
        return response


def test_empty_visible_content_is_distinguished_from_label_only_wrapper():
    assert classify_visible_output("", "", "Bull Analyst") == "EMPTY"
    assert classify_visible_output("", "Bull Analyst: ", "Bull Analyst") == "LABEL_ONLY"


def test_substantive_visible_content_is_good():
    assert (
        classify_visible_output(
            "  visible report  ",
            "Bull Analyst: visible report",
            "Bull Analyst",
        )
        == "GOOD"
    )


def test_truncated_requires_explicit_finish_and_token_evidence():
    assert (
        classify_visible_output(
            "short",
            "Bull Analyst: short",
            "Bull Analyst",
            finish_reason="length",
            output_tokens=5,
            max_output_tokens=None,
        )
        == "GOOD"
    )
    assert (
        classify_visible_output(
            "short",
            "Bull Analyst: short",
            "Bull Analyst",
            finish_reason="length",
            output_tokens=1024,
            max_output_tokens=1024,
        )
        == "TRUNCATED"
    )


def test_numeric_string_max_tokens_is_available_for_truncation_evidence():
    assert benchmark._as_int("1024") == 1024


def test_summary_contains_counts_statistics_and_no_text():
    records = [
        BenchmarkRecord(
            agent="bull",
            iteration=1,
            model="qwen3.5:2b",
            thinking=False,
            content_len=24,
            content_empty=False,
            classification="GOOD",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=8,
            elapsed_seconds=1.0,
            exception_type=None,
            transport_ok=True,
            reasoning_present=False,
            reasoning_len=0,
        ),
        BenchmarkRecord(
            agent="bull",
            iteration=2,
            model="qwen3.5:2b",
            thinking=False,
            content_len=0,
            content_empty=True,
            classification="LABEL_ONLY",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=8,
            elapsed_seconds=2.0,
            exception_type=None,
            transport_ok=True,
            reasoning_present=True,
            reasoning_len=12,
        ),
    ]
    summary = summarize_records(records)
    assert summary["total_calls"] == 2
    assert summary["good_count"] == 1
    assert summary["label_only_count"] == 1
    assert summary["success_percentage"] == 50.0
    assert summary["median_content_chars"] == 12.0
    assert summary["p95_runtime_seconds"] == 1.95
    assert "prompt" not in summary
    assert "completion" not in summary


def test_fixture_is_fresh_deterministic_and_forex_safe():
    first = build_synthetic_state("bull", 1)
    second = build_synthetic_state("bull", 1)
    assert first == second
    assert first is not second
    assert first["asset_type"] == "forex"
    assert first["market_data_mode"] == "forex_mt5"
    assert "EURUSD" in first["market_context"]


def test_run_agent_call_uses_existing_factory_and_only_records_metadata():
    record = run_agent_call(FakeLLM(), "bull", 1)
    assert record.classification == "GOOD"
    assert record.content_len == len(FakeResponse.content)
    assert record.reasoning_present is True
    assert record.reasoning_len == len(FakeResponse.additional_kwargs["reasoning_content"])
    assert not hasattr(record, "content")
    assert "private reasoning" not in json.dumps(record.to_dict())


def test_run_benchmark_is_sequential_four_agent_order_and_text_free(tmp_path: Path):
    output_path = tmp_path / "rows.jsonl"
    summary_path = tmp_path / "summary.md"
    llm = FakeLLM()
    records = run_benchmark(
        model="qwen3.5:2b",
        calls_per_agent=1,
        output_path=output_path,
        summary_path=summary_path,
        llm=llm,
    )
    assert [record.agent for record in records] == ["bull", "bear", "aggressive", "neutral"]
    assert llm.calls == 4
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert all("prompt" not in row and "completion" not in row for row in rows)
    assert all("reasoning_content" not in row and "content" not in row for row in rows)
    assert "Bull Researcher" in summary_path.read_text(encoding="utf-8")


def test_quick_client_uses_existing_forex_provider_kwargs(monkeypatch):
    captured = {}

    class FakeClient:
        def get_llm(self):
            return FakeLLM()

    def fake_create_llm_client(**kwargs):
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(benchmark, "create_llm_client", fake_create_llm_client)
    llm, config, quick_kwargs = benchmark._quick_client(
        "qwen3.5:2b",
        {
            "llm_provider": "ollama",
            "backend_url": "http://localhost:11434/v1",
            "forex_quick_thinking": False,
            "temperature": 0.1,
            "max_tokens": 1024,
        },
    )
    assert isinstance(llm, FakeLLM)
    assert config["quick_think_llm"] == "qwen3.5:2b"
    assert quick_kwargs["extra_body"] == {"think": False}
    assert captured == {
        "provider": "ollama",
        "model": "qwen3.5:2b",
        "base_url": "http://localhost:11434/v1",
        "extra_body": {"think": False},
        "temperature": 0.1,
        "max_tokens": 1024,
    }
