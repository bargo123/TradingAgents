"""Small, context-local telemetry helpers for forex shadow runs.

The graph executes one agent node at a time.  A context variable lets the
callback handler attribute the LLM calls made inside that node without
changing LangGraph state or retaining prompt/reasoning content.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Any

from .context_integrity import state_artifact_metrics


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Identity attached to LLM calls made by one graph node."""

    name: str
    model: str


_CURRENT_AGENT: ContextVar[AgentContext | None] = ContextVar(
    "tradingagents_current_agent", default=None
)
_STATE_TRACE: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "tradingagents_forex_state_trace", default=None
)


def model_identifier(llm: Any) -> str:
    """Return a stable, non-secret model label for telemetry."""
    for attribute in ("model_name", "model", "model_id", "name"):
        value = getattr(llm, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if llm is None:
        return "unknown"
    return type(llm).__name__


@contextmanager
def agent_context(name: str, llm: Any = None) -> Iterator[AgentContext]:
    """Attribute nested callback events to ``name`` for the current task."""
    context = AgentContext(str(name), model_identifier(llm))
    token = _CURRENT_AGENT.set(context)
    try:
        yield context
    finally:
        _CURRENT_AGENT.reset(token)


def current_agent_context() -> AgentContext | None:
    """Return the active agent attribution, if a graph node is running."""
    return _CURRENT_AGENT.get()


@contextmanager
def capture_state_trace() -> Iterator[list[dict[str, Any]]]:
    """Capture metadata-only state boundaries for one graph invocation."""
    trace: list[dict[str, Any]] = []
    token = _STATE_TRACE.set(trace)
    try:
        yield trace
    finally:
        _STATE_TRACE.reset(token)


def record_state_boundary(
    node: str,
    phase: str,
    state: Mapping[str, Any] | None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Append artifact and evidence-context metadata without retaining contents.

    ``state_artifact_metrics`` includes only the evidence context hash, size,
    and selected/dropped counts; rendered evidence, prompts, and messages are
    never copied into the trace.
    """
    trace = _STATE_TRACE.get()
    if trace is None:
        return
    entry: dict[str, Any] = {
        "node": str(node),
        "phase": str(phase),
        "artifacts": state_artifact_metrics(state),
    }
    if metadata:
        # Only scalar metadata is accepted here.  This boundary is deliberately
        # not a transport for prompts, completions, or arbitrary state.
        entry.update(
            {
                str(key): value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
        )
    trace.append(entry)


def stage_timings_from_trace(
    trace: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, int | float]]:
    """Aggregate metadata-only node durations from an instrumented trace."""

    timings: dict[str, dict[str, int | float]] = {}
    for entry in trace or ():
        if not isinstance(entry, Mapping) or entry.get("phase") != "after":
            continue
        node = str(entry.get("node") or "").strip()
        duration = entry.get("duration_seconds")
        if not node or not isinstance(duration, (int, float)) or isinstance(duration, bool):
            continue
        try:
            normalized_duration = float(duration)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(normalized_duration):
            continue
        current = timings.setdefault(node, {"calls": 0, "elapsed_seconds": 0.0})
        current["calls"] = int(current["calls"]) + 1
        current["elapsed_seconds"] = float(current["elapsed_seconds"]) + max(
            0.0, normalized_duration
        )
    return timings


def extend_state_trace(entries: list[dict[str, Any]]) -> None:
    """Append metadata-only child-thread boundaries to the active trace."""
    trace = _STATE_TRACE.get()
    if trace is not None:
        trace.extend(entries)


def instrument_agent_node(node, agent_name: str, llm: Any):
    """Wrap an LLM-bearing graph node with context and state attribution."""
    if not callable(node):
        raise TypeError("node must be callable")

    @wraps(node)
    def wrapped(*args, **kwargs):
        state = args[0] if args else kwargs.get("state")
        record_state_boundary(agent_name, "before", state)
        started = perf_counter()
        with agent_context(agent_name, llm):
            result = node(*args, **kwargs)
        if isinstance(state, Mapping):
            merged_state = dict(state)
            if isinstance(result, Mapping):
                merged_state.update(result)
        else:
            merged_state = result if isinstance(result, Mapping) else None
        record_state_boundary(
            agent_name,
            "after",
            merged_state,
            metadata={"duration_seconds": max(0.0, perf_counter() - started)},
        )
        return result

    return wrapped


__all__ = [
    "AgentContext",
    "agent_context",
    "capture_state_trace",
    "extend_state_trace",
    "current_agent_context",
    "instrument_agent_node",
    "model_identifier",
    "record_state_boundary",
    "stage_timings_from_trace",
]
