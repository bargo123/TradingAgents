"""Callback telemetry used by both the stock CLI and forex shadow command.

The handler intentionally records only numeric usage/timing and model/agent
identifiers. Prompt text, completions, and reasoning content are never kept.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from tradingagents.forex.telemetry import current_agent_context


def _as_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _reasoning_tokens(source: Mapping[str, Any]) -> int:
    values: list[int] = []
    for key in ("reasoning_tokens", "reasoning_token_count"):
        values.append(_as_nonnegative_int(source.get(key)))
    for key in (
        "output_token_details",
        "completion_tokens_details",
        "output_tokens_details",
    ):
        details = _mapping(source.get(key))
        for nested_key in ("reasoning", "reasoning_tokens", "reasoning_token_count"):
            values.append(_as_nonnegative_int(details.get(nested_key)))
    return max(values, default=0)


def _token_usage(response: LLMResult) -> tuple[int, int, int]:
    """Extract input/output/reasoning counts across common LangChain shapes."""
    sources: list[Mapping[str, Any]] = []
    try:
        generation = response.generations[0][0]
    except (AttributeError, IndexError, TypeError):
        generation = None

    if generation is not None:
        message = getattr(generation, "message", None)
        sources.extend(
            [
                _mapping(getattr(message, "usage_metadata", None)),
                _mapping(getattr(message, "response_metadata", None)),
                _mapping(getattr(generation, "generation_info", None)),
            ]
        )
    sources.append(_mapping(getattr(response, "llm_output", None)))

    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    for source in sources:
        for key in ("input_tokens", "prompt_tokens", "input_token_count"):
            if key in source:
                input_tokens = max(input_tokens, _as_nonnegative_int(source[key]))
        for key in ("output_tokens", "completion_tokens", "output_token_count"):
            if key in source:
                output_tokens = max(output_tokens, _as_nonnegative_int(source[key]))
        reasoning_tokens = max(reasoning_tokens, _reasoning_tokens(source))
    return input_tokens, output_tokens, reasoning_tokens


def _run_key(run_id: Any) -> str | None:
    if run_id is None:
        return None
    return str(run_id)


class StatsCallbackHandler(BaseCallbackHandler):
    """Track aggregate and per-agent LLM/tool usage without private content."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.reasoning_tokens = 0
        self._active_runs: dict[str, dict[str, Any]] = {}
        self._completed_runs: set[str] = set()
        self._agents: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        """Start a fresh accounting window for one shadow analysis.

        A watcher process reuses one callback handler across scheduled
        opportunities.  Resetting at the runner boundary keeps persisted
        telemetry per-decision instead of reporting process-lifetime totals.
        """
        with self._lock:
            self.llm_calls = 0
            self.tool_calls = 0
            self.tokens_in = 0
            self.tokens_out = 0
            self.reasoning_tokens = 0
            self._active_runs.clear()
            self._completed_runs.clear()
            self._agents.clear()

    @staticmethod
    def _event_model(serialized: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> str:
        serialized = serialized or {}
        invocation = _mapping(kwargs.get("invocation_params"))
        for source in (invocation, serialized, kwargs):
            for key in ("model_name", "model", "model_id", "name"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        context = current_agent_context()
        return context.model if context is not None else "unknown"

    def _start_run(
        self,
        serialized: Mapping[str, Any] | None,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        key = _run_key(run_id)
        context = current_agent_context()
        agent = context.name if context is not None else "unattributed"
        model = self._event_model(serialized, kwargs)
        now = time.perf_counter()
        with self._lock:
            if key is not None:
                if key in self._active_runs or key in self._completed_runs:
                    return
                self._active_runs[key] = {
                    "started_at": now,
                    "agent": agent,
                    "model": model,
                }
            self.llm_calls += 1

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Increment an LLM call counter and begin a timing span."""
        self._start_run(serialized, run_id, **kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Increment a chat-model call counter and begin a timing span."""
        self._start_run(serialized, run_id, **kwargs)

    def on_llm_end(
        self,
        response: LLMResult,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record numeric token usage and elapsed time only."""
        input_tokens, output_tokens, reasoning_tokens = _token_usage(response)
        key = _run_key(run_id)
        now = time.perf_counter()
        context = current_agent_context()
        with self._lock:
            if key is not None and key in self._completed_runs:
                return
            span = self._active_runs.pop(key, None) if key is not None else None
            if key is not None:
                self._completed_runs.add(key)
            if span is None:
                span = {
                    "started_at": now,
                    "agent": context.name if context is not None else "unattributed",
                    "model": context.model if context is not None else "unknown",
                }
            elapsed = max(0.0, now - float(span["started_at"]))
            agent_name = str(span.get("agent") or "unattributed")
            model = str(span.get("model") or "unknown")
            self.tokens_in += input_tokens
            self.tokens_out += output_tokens
            self.reasoning_tokens += reasoning_tokens
            metrics = self._agents.setdefault(
                agent_name,
                {
                    "model": model,
                    "calls": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "reasoning_tokens": 0,
                    "elapsed_seconds": 0.0,
                },
            )
            if metrics["model"] == "unknown" and model != "unknown":
                metrics["model"] = model
            metrics["calls"] += 1
            metrics["tokens_in"] += input_tokens
            metrics["tokens_out"] += output_tokens
            metrics["reasoning_tokens"] += reasoning_tokens
            metrics["elapsed_seconds"] += elapsed

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment the tool-call counter without retaining tool input."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate and per-agent numeric statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "reasoning_tokens": self.reasoning_tokens,
                "agents": {
                    name: dict(values) for name, values in self._agents.items()
                },
            }
