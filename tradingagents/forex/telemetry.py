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
_TIMING_TRACE: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "tradingagents_forex_timing_trace", default=None
)

FOREX_DEPENDENCY_EDGES: tuple[tuple[str, str], ...] = (
    ("Market Analyst", "Bull Researcher"),
    ("News Analyst", "Bull Researcher"),
    ("Bull Researcher", "Bear Researcher"),
    ("Bear Researcher", "Research Manager"),
    ("Research Manager", "Trader"),
    ("Trader", "Aggressive Analyst"),
    ("Aggressive Analyst", "Conservative Analyst"),
    ("Conservative Analyst", "Neutral Analyst"),
    ("Neutral Analyst", "Portfolio Manager"),
)

PHASE12_STRATEGIC_DEPENDENCY_EDGES: tuple[tuple[str, str], ...] = (
    ("Market Analyst", "Bull Researcher"),
    ("News Analyst", "Bull Researcher"),
    ("Market Analyst", "Bear Researcher"),
    ("News Analyst", "Bear Researcher"),
    ("Bull Researcher", "Research Manager"),
    ("Bear Researcher", "Research Manager"),
    ("Research Manager", "Trader"),
    ("Trader", "Aggressive Analyst"),
    ("Trader", "Conservative Analyst"),
    ("Trader", "Neutral Analyst"),
    ("Aggressive Analyst", "Portfolio Manager"),
    ("Conservative Analyst", "Portfolio Manager"),
    ("Neutral Analyst", "Portfolio Manager"),
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


@contextmanager
def capture_timing_trace() -> Iterator[list[dict[str, Any]]]:
    """Capture monotonic node intervals without copying state or content."""

    trace: list[dict[str, Any]] = []
    token = _TIMING_TRACE.set(trace)
    try:
        yield trace
    finally:
        _TIMING_TRACE.reset(token)


def record_timing_interval(node: str, started: float, finished: float) -> None:
    """Record one bounded monotonic interval in the active benchmark trace."""

    trace = _TIMING_TRACE.get()
    if trace is None:
        return
    trace.append({"node": str(node), "started": started, "finished": finished})


def extend_timing_trace(entries: list[dict[str, Any]]) -> None:
    """Merge child-thread timing intervals into the parent benchmark trace."""

    trace = _TIMING_TRACE.get()
    if trace is not None:
        trace.extend(entries)


def critical_path_from_intervals(
    intervals: Sequence[Mapping[str, Any]] | None,
    dependencies: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Calculate wall, overlap, waits, and longest dependency path safely."""

    valid: list[dict[str, Any]] = []
    unknown = 0
    for item in intervals or ():
        if not isinstance(item, Mapping):
            unknown += 1
            continue
        node = item.get("node")
        started = item.get("started")
        finished = item.get("finished")
        if (
            not isinstance(node, str)
            or not node.strip()
            or isinstance(started, bool)
            or isinstance(finished, bool)
            or not isinstance(started, (int, float))
            or not isinstance(finished, (int, float))
            or not math.isfinite(float(started))
            or not math.isfinite(float(finished))
            or float(finished) < float(started)
        ):
            unknown += 1
            continue
        valid.append(
            {"node": node.strip(), "started": float(started), "finished": float(finished)}
        )
    if not valid:
        return {
            "wall_seconds": 0.0,
            "total_node_seconds": 0.0,
            "overlap_seconds": 0.0,
            "idle_wait_seconds": 0.0,
            "critical_path_seconds": 0.0,
            "critical_path_nodes": [],
            "unknown_intervals": unknown,
        }

    by_node: dict[str, dict[str, Any]] = {}
    for item in valid:
        node = item["node"]
        current = by_node.get(node)
        if current is None:
            by_node[node] = {
                "started": item["started"],
                "finished": item["finished"],
                "duration": item["finished"] - item["started"],
            }
        else:
            current["started"] = min(current["started"], item["started"])
            current["finished"] = max(current["finished"], item["finished"])
            current["duration"] += item["finished"] - item["started"]
    edges = tuple(dependencies or ())
    predecessors: dict[str, tuple[str, ...]] = {
        node: tuple(source for source, target in edges if target == node)
        for node in by_node
    }
    order = sorted(by_node, key=lambda node: (by_node[node]["finished"], node))
    path_end: dict[str, float] = {}
    path_start: dict[str, float] = {}
    path_parent: dict[str, str | None] = {}
    idle_wait = 0.0
    for node in order:
        current = by_node[node]
        prior = [source for source in predecessors.get(node, ()) if source in by_node]
        if prior:
            best = max(prior, key=lambda source: path_end.get(source, by_node[source]["finished"]))
            path_start[node] = path_start.get(best, by_node[best]["started"])
            path_parent[node] = best
            idle_wait += max(0.0, current["started"] - max(by_node[source]["finished"] for source in prior))
        else:
            path_start[node] = current["started"]
            path_parent[node] = None
        path_end[node] = current["finished"]
    end_node = max(order, key=lambda node: path_end[node] - path_start[node])
    critical_nodes: list[str] = []
    current_node: str | None = end_node
    while current_node is not None:
        critical_nodes.append(current_node)
        current_node = path_parent[current_node]
    critical_nodes.reverse()
    wall_seconds = max(item["finished"] for item in valid) - min(item["started"] for item in valid)
    total_seconds = sum(item["finished"] - item["started"] for item in valid)
    return {
        "wall_seconds": wall_seconds,
        "total_node_seconds": total_seconds,
        "overlap_seconds": max(0.0, total_seconds - wall_seconds),
        "idle_wait_seconds": idle_wait,
        "critical_path_seconds": path_end[end_node] - path_start[end_node],
        "critical_path_nodes": critical_nodes,
        "unknown_intervals": unknown,
    }


def instrument_agent_node(node, agent_name: str, llm: Any):
    """Wrap an LLM-bearing graph node with context and state attribution."""
    if not callable(node):
        raise TypeError("node must be callable")

    @wraps(node)
    def wrapped(*args, **kwargs):
        state = args[0] if args else kwargs.get("state")
        record_state_boundary(agent_name, "before", state)
        started = perf_counter()
        try:
            with agent_context(agent_name, llm):
                result = node(*args, **kwargs)
        finally:
            record_timing_interval(agent_name, started, perf_counter())
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
    "capture_timing_trace",
    "critical_path_from_intervals",
    "extend_state_trace",
    "extend_timing_trace",
    "current_agent_context",
    "instrument_agent_node",
    "model_identifier",
    "record_state_boundary",
    "record_timing_interval",
    "stage_timings_from_trace",
    "FOREX_DEPENDENCY_EDGES",
    "PHASE12_STRATEGIC_DEPENDENCY_EDGES",
]
