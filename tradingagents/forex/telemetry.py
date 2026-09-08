"""Small, context-local telemetry helpers for forex shadow runs.

The graph executes one agent node at a time.  A context variable lets the
callback handler attribute the LLM calls made inside that node without
changing LangGraph state or retaining prompt/reasoning content.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Identity attached to LLM calls made by one graph node."""

    name: str
    model: str


_CURRENT_AGENT: ContextVar[AgentContext | None] = ContextVar(
    "tradingagents_current_agent", default=None
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


def instrument_agent_node(node, agent_name: str, llm: Any):
    """Wrap an LLM-bearing graph node with context-local attribution."""
    if not callable(node):
        raise TypeError("node must be callable")

    @wraps(node)
    def wrapped(*args, **kwargs):
        with agent_context(agent_name, llm):
            return node(*args, **kwargs)

    return wrapped


__all__ = [
    "AgentContext",
    "agent_context",
    "current_agent_context",
    "instrument_agent_node",
    "model_identifier",
]
