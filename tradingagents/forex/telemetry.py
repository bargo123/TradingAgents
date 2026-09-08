"""Small, context-local telemetry helpers for forex shadow runs.

The graph executes one agent node at a time.  A context variable lets the
callback handler attribute the LLM calls made inside that node without
changing LangGraph state or retaining prompt/reasoning content.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
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
) -> None:
    """Append presence/size metadata without retaining state contents."""
    trace = _STATE_TRACE.get()
    if trace is None:
        return
    trace.append(
        {
            "node": str(node),
            "phase": str(phase),
            "artifacts": state_artifact_metrics(state),
        }
    )


def instrument_agent_node(node, agent_name: str, llm: Any):
    """Wrap an LLM-bearing graph node with context and state attribution."""
    if not callable(node):
        raise TypeError("node must be callable")

    @wraps(node)
    def wrapped(*args, **kwargs):
        state = args[0] if args else kwargs.get("state")
        record_state_boundary(agent_name, "before", state)
        with agent_context(agent_name, llm):
            result = node(*args, **kwargs)
        if isinstance(state, Mapping):
            merged_state = dict(state)
            if isinstance(result, Mapping):
                merged_state.update(result)
        else:
            merged_state = result if isinstance(result, Mapping) else None
        record_state_boundary(agent_name, "after", merged_state)
        return result

    return wrapped


__all__ = [
    "AgentContext",
    "agent_context",
    "capture_state_trace",
    "current_agent_context",
    "instrument_agent_node",
    "model_identifier",
    "record_state_boundary",
]
