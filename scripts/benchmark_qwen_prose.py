"""Standalone, metadata-only reliability benchmark for forex prose agents.

The module deliberately does not initialize or compile ``TradingAgentsGraph``.
It creates
the same quick LLM client used by the graph, then invokes the existing prose
agent factories with deterministic synthetic forex state.  Prompts,
completions, and private reasoning are never written to benchmark records.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients import create_llm_client

AGENT_ORDER = ("bull", "bear", "aggressive", "neutral")


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Metadata needed to invoke one existing prose-agent factory."""

    display_name: str
    response_label: str
    state_key: str
    current_response_key: str
    factory: Callable[[Any], Callable[[Mapping[str, Any]], Mapping[str, Any]]]


AGENT_SPECS: dict[str, AgentSpec] = {
    "bull": AgentSpec(
        "Bull Researcher",
        "Bull Analyst",
        "investment_debate_state",
        "current_response",
        create_bull_researcher,
    ),
    "bear": AgentSpec(
        "Bear Researcher",
        "Bear Analyst",
        "investment_debate_state",
        "current_response",
        create_bear_researcher,
    ),
    "aggressive": AgentSpec(
        "Aggressive Risk Analyst",
        "Aggressive Analyst",
        "risk_debate_state",
        "current_aggressive_response",
        create_aggressive_debator,
    ),
    "neutral": AgentSpec(
        "Neutral Risk Analyst",
        "Neutral Analyst",
        "risk_debate_state",
        "current_neutral_response",
        create_neutral_debator,
    ),
}


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """Scalar-only evidence for one prose-agent invocation."""

    agent: str
    iteration: int
    model: str
    thinking: bool
    content_len: int
    content_empty: bool
    classification: str
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_seconds: float
    exception_type: str | None
    transport_ok: bool
    reasoning_present: bool
    reasoning_len: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe scalar fields only."""
        return asdict(self)


_BASE_SYNTHETIC_STATE: dict[str, Any] = {
    "company_of_interest": "EURUSD",
    "asset_type": "forex",
    "market_data_mode": "forex_mt5",
    "forex_analysis_profile": "INTRADAY",
    "trade_date": "2026-09-09",
    "instrument_context": (
        "EURUSD synthetic MT5 context; bid/ask/spread and M1/M5/M15/H1 features."
    ),
    "market_context": (
        "EURUSD bid 1.10000 ask 1.10002 spread 0.00002; M1 trend positive; "
        "M5 range-bound; M15 support 1.0990; H1 resistance 1.1050."
    ),
    "market_report": (
        "Synthetic market report: deterministic EURUSD quote, momentum, "
        "volatility, and timeframe features."
    ),
    "sentiment_report": "Unavailable for forex; do not infer social sentiment.",
    "news_report": (
        "Synthetic broad macro report: central-bank policy, inflation, "
        "employment, and geopolitical risk context only."
    ),
    "fundamentals_report": (
        "Unavailable for forex; do not infer company fundamentals."
    ),
    "trader_investment_plan": (
        "Synthetic trader plan: observe only; hypothetical directional bias "
        "with explicit spread and invalidation controls."
    ),
    "investment_debate_state": {
        "history": "Bull and bear debate context.",
        "bull_history": "Bull context.",
        "bear_history": "Bear context.",
        "current_response": "Last debate response.",
        "judge_decision": "",
        "count": 2,
    },
    "risk_debate_state": {
        "history": "Aggressive, conservative, and neutral risk debate context.",
        "aggressive_history": "Aggressive context.",
        "conservative_history": "Conservative context.",
        "neutral_history": "Neutral context.",
        "latest_speaker": "Neutral",
        "current_aggressive_response": "Last aggressive response.",
        "current_conservative_response": "Last conservative response.",
        "current_neutral_response": "Last neutral response.",
        "judge_decision": "",
        "count": 3,
    },
}


def build_synthetic_state(agent: str, iteration: int) -> dict[str, Any]:
    """Return a fresh, deterministic forex state for one benchmark call."""
    del iteration  # Iteration identifies the record, not the prompt contents.
    if agent not in AGENT_SPECS:
        raise ValueError(f"unknown benchmark agent: {agent}")
    return copy.deepcopy(_BASE_SYNTHETIC_STATE)


def _content_text(value: object) -> str:
    """Extract visible text blocks without exposing or retaining metadata."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _scalar_text_length(value: object) -> int:
    """Count text length in a known response field, never return its text."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_scalar_text_length(item) for item in value)
    if isinstance(value, dict):
        return sum(_scalar_text_length(item) for item in value.values())
    return 0


def _truncation_proven(
    finish_reason: object,
    output_tokens: int | None,
    max_output_tokens: int | None,
) -> bool:
    """Require both an explicit limit finish and matching token evidence."""
    if not isinstance(finish_reason, str):
        return False
    if finish_reason.strip().lower() not in {"length", "max_tokens", "token_limit"}:
        return False
    return (
        output_tokens is not None
        and max_output_tokens is not None
        and output_tokens >= max_output_tokens
    )


def classify_visible_output(
    raw_content: object,
    wrapped_output: object,
    agent_label: str,
    *,
    finish_reason: object = None,
    output_tokens: int | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """Classify visible output using strict, non-guessing rules."""
    visible = _content_text(raw_content).strip()
    label = agent_label.strip().rstrip(":").casefold()
    if visible:
        if _truncation_proven(finish_reason, output_tokens, max_output_tokens):
            return "TRUNCATED"
        if visible.rstrip(":").casefold() == label:
            return "LABEL_ONLY"
        return "GOOD"

    wrapped = _content_text(wrapped_output).strip()
    if wrapped.rstrip(":").casefold() == label:
        return "LABEL_ONLY"
    return "EMPTY"


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _response_metadata(response: object) -> tuple[str | None, int | None, int | None, bool, int]:
    """Extract finish/usage/reasoning lengths as scalar metadata."""
    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = getattr(response, "usage_metadata", {}) or {}
    if not isinstance(response_metadata, Mapping):
        response_metadata = {}
    if not isinstance(usage_metadata, Mapping):
        usage_metadata = {}

    finish_reason = response_metadata.get("finish_reason")
    if not isinstance(finish_reason, str):
        finish_reason = None

    token_usage = response_metadata.get("token_usage")
    if not isinstance(token_usage, Mapping):
        token_usage = {}
    input_tokens = _as_int(usage_metadata.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _as_int(token_usage.get("prompt_tokens"))
    output_tokens = _as_int(usage_metadata.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _as_int(token_usage.get("completion_tokens"))

    reasoning_present = False
    reasoning_len = 0
    seen_values: set[tuple[str, int]] = set()
    for container in (
        getattr(response, "additional_kwargs", {}),
        response_metadata,
        usage_metadata,
    ):
        if not isinstance(container, Mapping):
            continue
        for key in ("reasoning_content", "reasoning", "thinking", "think"):
            if key not in container:
                continue
            value = container[key]
            length = _scalar_text_length(value)
            identity = (key, id(value))
            if identity in seen_values:
                continue
            seen_values.add(identity)
            reasoning_present = reasoning_present or value not in (None, "", [], {})
            reasoning_len = max(reasoning_len, length)
    return finish_reason, input_tokens, output_tokens, reasoning_present, reasoning_len


def _wrapped_output(result: object, spec: AgentSpec) -> str:
    if not isinstance(result, Mapping):
        return ""
    nested = result.get(spec.state_key)
    if not isinstance(nested, Mapping):
        return ""
    value = nested.get(spec.current_response_key)
    return value if isinstance(value, str) else ""


class _ResponseRecorder:
    """Transparent invoke wrapper that retains only the latest response object."""

    def __init__(self, llm: Any):
        self._llm = llm
        self.last_response: object | None = None

    def invoke(self, *args: Any, **kwargs: Any) -> object:
        self.last_response = None
        response = self._llm.invoke(*args, **kwargs)
        self.last_response = response
        return response


def _model_name(llm: object, fallback: str = "unknown") -> str:
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return fallback


def run_agent_call(
    llm: object,
    agent: str,
    iteration: int,
    *,
    model: str | None = None,
    thinking: bool = False,
    max_output_tokens: int | None = 1024,
) -> BenchmarkRecord:
    """Invoke one existing prose node and return metadata-only evidence."""
    spec = AGENT_SPECS.get(agent)
    if spec is None:
        raise ValueError(f"unknown benchmark agent: {agent}")
    recorder = _ResponseRecorder(llm)
    node = spec.factory(recorder)
    started = time.perf_counter()
    try:
        result = node(build_synthetic_state(agent, iteration))
    except Exception as exc:  # noqa: BLE001 - benchmark records exception type only
        return BenchmarkRecord(
            agent=agent,
            iteration=iteration,
            model=model or _model_name(llm),
            thinking=thinking,
            content_len=0,
            content_empty=True,
            classification="ERROR",
            finish_reason=None,
            input_tokens=None,
            output_tokens=None,
            elapsed_seconds=time.perf_counter() - started,
            exception_type=type(exc).__name__,
            transport_ok=False,
            reasoning_present=False,
            reasoning_len=0,
        )

    response = recorder.last_response
    if response is None:
        return BenchmarkRecord(
            agent=agent,
            iteration=iteration,
            model=model or _model_name(llm),
            thinking=thinking,
            content_len=0,
            content_empty=True,
            classification="ERROR",
            finish_reason=None,
            input_tokens=None,
            output_tokens=None,
            elapsed_seconds=time.perf_counter() - started,
            exception_type="MissingResponse",
            transport_ok=False,
            reasoning_present=False,
            reasoning_len=0,
        )

    raw_content = getattr(response, "content", None)
    visible = _content_text(raw_content)
    finish_reason, input_tokens, output_tokens, reasoning_present, reasoning_len = (
        _response_metadata(response)
    )
    wrapped = _wrapped_output(result, spec)
    classification = classify_visible_output(
        raw_content,
        wrapped,
        spec.response_label,
        finish_reason=finish_reason,
        output_tokens=output_tokens,
        max_output_tokens=max_output_tokens,
    )
    return BenchmarkRecord(
        agent=agent,
        iteration=iteration,
        model=model or _model_name(llm),
        thinking=thinking,
        content_len=len(visible),
        content_empty=not visible.strip(),
        classification=classification,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=time.perf_counter() - started,
        exception_type=None,
        transport_ok=True,
        reasoning_present=reasoning_present,
        reasoning_len=reasoning_len,
    )


def _mean(values: Iterable[int | float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.mean(usable), 3) if usable else None


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    return round(statistics.quantiles(values, n=20, method="inclusive")[18], 3)


def summarize_records(records: Sequence[BenchmarkRecord]) -> dict[str, object]:
    """Return scalar counts/statistics without retaining any response text."""
    total = len(records)
    content_lengths = [record.content_len for record in records]
    runtimes = [record.elapsed_seconds for record in records]
    counts = {name.lower() + "_count": 0 for name in ("GOOD", "EMPTY", "LABEL_ONLY", "ERROR", "TRUNCATED")}
    for record in records:
        counts[record.classification.lower() + "_count"] = (
            counts.get(record.classification.lower() + "_count", 0) + 1
        )
    return {
        "total_calls": total,
        **counts,
        "success_percentage": round(counts["good_count"] / total * 100, 2) if total else 0.0,
        "median_content_chars": round(statistics.median(content_lengths), 3) if content_lengths else None,
        "min_content_chars": min(content_lengths) if content_lengths else None,
        "max_content_chars": max(content_lengths) if content_lengths else None,
        "median_runtime_seconds": round(statistics.median(runtimes), 3) if runtimes else None,
        "p95_runtime_seconds": _p95(runtimes),
        "mean_input_tokens": _mean(record.input_tokens for record in records),
        "mean_output_tokens": _mean(record.output_tokens for record in records),
        "reasoning_present_count": sum(record.reasoning_present for record in records),
        "reasoning_total_chars": sum(record.reasoning_len for record in records),
        "transport_success_count": sum(record.transport_ok for record in records),
    }


def _safe_model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"


def _quick_client(model: str, config: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    merged["quick_think_llm"] = model
    graph = object.__new__(TradingAgentsGraph)
    graph.config = merged
    graph.market_data_mode = "forex_mt5"
    quick_kwargs = graph._get_provider_kwargs(role="quick")
    client = create_llm_client(
        provider=merged["llm_provider"],
        model=model,
        base_url=merged.get("backend_url"),
        **quick_kwargs,
    )
    return client.get_llm(), merged, quick_kwargs


def render_summary_markdown(
    records: Sequence[BenchmarkRecord],
    *,
    config: Mapping[str, Any],
    quick_kwargs: Mapping[str, Any],
    requested_agents: Sequence[str],
    command: str | None = None,
) -> str:
    """Render a scalar-only Markdown report for a benchmark run."""
    by_agent = {
        agent: summarize_records([record for record in records if record.agent == agent])
        for agent in requested_agents
    }
    overall = summarize_records(records)
    extra_body = quick_kwargs.get("extra_body")
    thinking = extra_body.get("think") if isinstance(extra_body, Mapping) else None
    lines = [
        "# Phase 6.3 Qwen3.5 Prose Reliability Benchmark",
        "",
        "Scope: standalone prose-node calls only; no MT5, graph, execution, or Phase 7.",
        "",
        "## Production-equivalent configuration",
        "",
        f"- Provider: `{config.get('llm_provider')}`",
        f"- Quick model: `{config.get('quick_think_llm')}`",
        f"- Backend URL: `{config.get('backend_url')}`",
        f"- Quick thinking control: `{thinking}`",
        f"- Temperature: `{config.get('temperature')}`",
        f"- Max tokens: `{config.get('max_tokens')}`",
    ]
    if command:
        lines.extend([f"- Command: `{command}`"])
    lines.extend([
        "",
        "## Per-agent results",
        "",
        "| Agent | Calls | GOOD | EMPTY | LABEL_ONLY | ERROR | TRUNCATED | Reliability | Median chars | Min/Max chars | Median sec | P95 sec | Mean input tokens | Mean output tokens | Reasoning present |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for agent in requested_agents:
        summary = by_agent[agent]
        display = AGENT_SPECS[agent].display_name
        lines.append(
            "| {display} | {total_calls} | {good_count} | {empty_count} | {label_only_count} | "
            "{error_count} | {truncated_count} | {success_percentage}% | {median_content_chars} | "
            "{min_content_chars}/{max_content_chars} | {median_runtime_seconds} | {p95_runtime_seconds} | "
            "{mean_input_tokens} | {mean_output_tokens} | {reasoning_present_count} |".format(
                display=display, **summary
            )
        )
    lines.extend([
        "",
        "## Overall",
        "",
        f"- Calls: `{overall['total_calls']}`",
        f"- GOOD: `{overall['good_count']}`",
        f"- EMPTY: `{overall['empty_count']}`",
        f"- LABEL_ONLY: `{overall['label_only_count']}`",
        f"- ERROR: `{overall['error_count']}`",
        f"- TRUNCATED (evidence-backed only): `{overall['truncated_count']}`",
        f"- Reliability: `{overall['success_percentage']}%`",
        f"- Median content chars: `{overall['median_content_chars']}`",
        f"- Content min/max chars: `{overall['min_content_chars']}/{overall['max_content_chars']}`",
        f"- Median/P95 runtime seconds: `{overall['median_runtime_seconds']}/{overall['p95_runtime_seconds']}`",
        f"- Mean input/output tokens: `{overall['mean_input_tokens']}/{overall['mean_output_tokens']}`",
        f"- Reasoning present count / total chars: `{overall['reasoning_present_count']}/{overall['reasoning_total_chars']}`",
        "",
        "Only scalar metadata is retained. Prompts, completions, and private reasoning text are not stored.",
    ])
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    model: str,
    calls_per_agent: int = 20,
    agents: Sequence[str] = AGENT_ORDER,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    llm: object | None = None,
    progress: bool = True,
) -> list[BenchmarkRecord]:
    """Run agents sequentially and optionally write metadata-only artifacts."""
    if calls_per_agent <= 0:
        raise ValueError("calls_per_agent must be positive")
    requested_agents = tuple(agents)
    if not requested_agents or any(agent not in AGENT_SPECS for agent in requested_agents):
        raise ValueError("agents must be a non-empty subset of bull,bear,aggressive,neutral")

    merged_config = dict(DEFAULT_CONFIG)
    if config:
        merged_config.update(config)
    if llm is None:
        llm, merged_config, quick_kwargs = _quick_client(model, merged_config)
    else:
        quick_kwargs = {"extra_body": {"think": False}}
    extra_body = quick_kwargs.get("extra_body")
    thinking = bool(extra_body.get("think")) if isinstance(extra_body, Mapping) else False
    max_output_tokens = merged_config.get("max_tokens")
    max_output_tokens = _as_int(max_output_tokens)

    output_file = None
    if output_path is not None:
        output_file_path = Path(output_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_file_path.open("w", encoding="utf-8")

    records: list[BenchmarkRecord] = []
    try:
        for agent in requested_agents:
            for iteration in range(1, calls_per_agent + 1):
                record = run_agent_call(
                    llm,
                    agent,
                    iteration,
                    model=model,
                    thinking=thinking,
                    max_output_tokens=max_output_tokens,
                )
                records.append(record)
                if output_file is not None:
                    output_file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                    output_file.flush()
                if progress:
                    print(
                        json.dumps(
                            {
                                "agent": record.agent,
                                "iteration": record.iteration,
                                "classification": record.classification,
                                "content_len": record.content_len,
                                "elapsed_seconds": round(record.elapsed_seconds, 3),
                                "transport_ok": record.transport_ok,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        if output_file is not None:
            output_file.close()

    if summary_path is not None:
        summary_file_path = Path(summary_path)
        summary_file_path.parent.mkdir(parents=True, exist_ok=True)
        summary_file_path.write_text(
            render_summary_markdown(
                records,
                config=merged_config,
                quick_kwargs=quick_kwargs,
                requested_agents=requested_agents,
                command=" ".join(sys.argv),
            ),
            encoding="utf-8",
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_qwen_prose",
        description="Sequential metadata-only benchmark for existing forex prose nodes.",
    )
    parser.add_argument("--calls-per-agent", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_CONFIG.get("quick_think_llm", "qwen3.5:2b"))
    parser.add_argument(
        "--agents",
        default=",".join(AGENT_ORDER),
        help="comma-separated subset: bull,bear,aggressive,neutral",
    )
    default_tag = _safe_model_tag(str(DEFAULT_CONFIG.get("quick_think_llm", "qwen3.5:2b")))
    parser.add_argument("--output", default=f"data_cache/phase6-3-{default_tag}.jsonl")
    parser.add_argument("--summary", default="docs/superpowers/reports/2026-09-09-phase-6-3-qwen-reliability.md")
    parser.add_argument("--quiet", action="store_true", help="suppress per-call scalar progress")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agents = tuple(part.strip() for part in args.agents.split(",") if part.strip())
    try:
        records = run_benchmark(
            model=args.model,
            calls_per_agent=args.calls_per_agent,
            agents=agents,
            output_path=args.output,
            summary_path=args.summary,
            progress=not args.quiet,
        )
    except Exception as exc:  # noqa: BLE001 - do not print provider response bodies
        print(
            f"ERROR: benchmark unavailable ({type(exc).__name__}); no decision was fabricated.",
            file=sys.stderr,
        )
        return 2
    summary = summarize_records(records)
    print(
        f"completed {summary['total_calls']} calls; GOOD={summary['good_count']} "
        f"EMPTY={summary['empty_count']} LABEL_ONLY={summary['label_only_count']} "
        f"ERROR={summary['error_count']} TRUNCATED={summary['truncated_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke
    raise SystemExit(main())
