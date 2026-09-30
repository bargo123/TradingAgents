"""Bounded parallel execution for independent forex analyst branches.

The normal graph keeps debate and risk speakers sequential because each speaker
reads the previous speaker's response.  Market and News are different: each
starts from the same initial snapshot and produces a separate report.  This
module executes only those independent analyst branches concurrently while
retaining each branch's existing tool-call loop and deterministic merge order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from tradingagents.forex.telemetry import (
    capture_state_trace,
    capture_timing_trace,
    extend_state_trace,
    extend_timing_trace,
)

try:  # pragma: no cover - fallback is for minimal test environments
    from langgraph.graph.message import add_messages
except ModuleNotFoundError:  # pragma: no cover
    add_messages = None


@dataclass(frozen=True)
class ParallelAnalystBranch:
    """One independent analyst branch and its existing graph routing seams."""

    key: str
    agent_node: str
    clear_node: str
    tool_node: str
    report_key: str
    node: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    clear: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    tool: Any
    route: Callable[[Mapping[str, Any]], str]


def _copy_branch_state(state: Mapping[str, Any]) -> dict[str, Any]:
    branch_state = dict(state)
    messages = branch_state.get("messages")
    if isinstance(messages, list):
        branch_state["messages"] = list(messages)
    return branch_state


def _invoke_tool(tool: Any, state: Mapping[str, Any], config: Any | None) -> Mapping[str, Any]:
    invoke = getattr(tool, "invoke", None)
    if callable(invoke):
        result = invoke(state, config=config) if config is not None else invoke(state)
    elif callable(tool):
        result = tool(state)
    else:
        raise TypeError("parallel analyst tool node must be callable")
    if not isinstance(result, Mapping):
        raise TypeError("parallel analyst tool node must return a mapping")
    return result


def _apply_update(state: dict[str, Any], update: Mapping[str, Any]) -> None:
    """Apply a branch update with LangGraph's message-reducer semantics."""
    for key, value in update.items():
        if key == "messages" and isinstance(value, list) and isinstance(
            state.get("messages"), list
        ):
            if add_messages is None:
                state[key] = [*state["messages"], *value]
            else:
                state[key] = add_messages(state["messages"], value)
        else:
            state[key] = value


def _run_branch(
    branch: ParallelAnalystBranch,
    initial_state: Mapping[str, Any],
    config: Any | None,
    max_steps: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    state = _copy_branch_state(initial_state)
    with capture_state_trace() as branch_trace, capture_timing_trace() as timing_trace:
        for _ in range(max_steps):
            update = branch.node(state)
            if not isinstance(update, Mapping):
                raise TypeError(f"{branch.agent_node} must return a mapping")
            _apply_update(state, update)
            target = branch.route(state)
            if target == branch.tool_node:
                _apply_update(state, _invoke_tool(branch.tool, state, config))
                continue
            if target == branch.clear_node:
                clear_update = branch.clear(state)
                if not isinstance(clear_update, Mapping):
                    raise TypeError(f"{branch.clear_node} must return a mapping")
                _apply_update(state, clear_update)
                return (
                    str(state.get(branch.report_key, "") or ""),
                    list(branch_trace),
                    list(timing_trace),
                )
            raise RuntimeError(
                f"{branch.agent_node} routed to unexpected target {target!r}"
            )
    raise RuntimeError(f"{branch.agent_node} exceeded {max_steps} tool steps")


def run_parallel_analysts(
    branches: tuple[ParallelAnalystBranch, ...],
    state: Mapping[str, Any],
    *,
    config: Any | None = None,
    max_steps: int = 100,
) -> dict[str, str]:
    """Run independent analyst branches concurrently and merge by declaration order.

    A branch failure propagates to the graph unchanged (fail closed).  The
    worker traces are appended only after all branches complete, in declared
    order, so metadata remains deterministic even though execution overlaps.
    """

    if len(branches) < 2:
        raise ValueError("parallel analyst execution requires at least two branches")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    with ThreadPoolExecutor(max_workers=len(branches), thread_name_prefix="forex-analyst") as pool:
        futures = [
            pool.submit(_run_branch, branch, state, config, max_steps)
            for branch in branches
        ]
        results = [future.result() for future in futures]

    merged: dict[str, str] = {}
    for branch, (report, branch_trace, timing_trace) in zip(branches, results, strict=True):
        extend_state_trace(branch_trace)
        extend_timing_trace(timing_trace)
        merged[branch.report_key] = report
    return merged


def _run_independent_node(
    name: str,
    node: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    state: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Invoke one already-instrumented node in an isolated worker context."""

    with capture_state_trace() as state_trace, capture_timing_trace() as timing_trace:
        result = node(_copy_branch_state(state))
    if not isinstance(result, Mapping):
        raise TypeError(f"{name} must return a mapping")
    return name, result, list(state_trace), list(timing_trace)


def _merge_independent_updates(
    state: Mapping[str, Any],
    results: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    nested_key: str,
) -> dict[str, Any]:
    """Merge independent speaker updates without letting last-write semantics erase siblings."""

    merged = dict(state)
    base_nested = state.get(nested_key)
    nested = dict(base_nested) if isinstance(base_nested, Mapping) else {}
    current_responses: list[str] = []
    for _name, update in results:
        for key, value in update.items():
            if key == nested_key and isinstance(value, Mapping):
                for nested_name, nested_value in value.items():
                    if nested_name == "history":
                        continue
                    if nested_name == "count":
                        continue
                    if (
                        (nested_name == "current_response" or nested_name.endswith("_response"))
                        and isinstance(nested_value, str)
                        and nested_value.strip()
                    ):
                        current_responses.append(nested_value)
                    # Each independent speaker returns the complete nested
                    # state shape, including empty copies of fields owned by
                    # its sibling.  Those empty placeholders must not erase
                    # a substantive value already merged from another
                    # speaker; independent joins are additive by ownership.
                    existing_value = nested.get(nested_name)
                    if (
                        isinstance(nested_value, str)
                        and not nested_value.strip()
                        and isinstance(existing_value, str)
                        and existing_value.strip()
                    ):
                        continue
                    nested[nested_name] = nested_value
            elif key != nested_key:
                merged[key] = value
    if results:
        base_history = nested.get("history")
        base_history = base_history if isinstance(base_history, str) else ""
        additions = [item for item in current_responses if item not in base_history]
        if additions:
            nested["history"] = base_history + "\n" + "\n".join(additions)
        base_count = state.get(nested_key, {}).get("count", 0) if isinstance(state.get(nested_key), Mapping) else 0
        if type(base_count) is int:
            nested["count"] = base_count + len(results)
    if nested_key in state or results:
        merged[nested_key] = nested
    return merged


def run_parallel_nodes(
    nodes: Sequence[tuple[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]],
    state: Mapping[str, Any],
    *,
    nested_key: str,
) -> dict[str, Any]:
    """Run causally independent agent nodes concurrently and join their state.

    The node factories, callbacks, strict schemas, and safety boundaries remain
    owned by the existing agents.  This helper only overlaps calls whose input
    state is identical and merges their separately-owned nested fields.
    """

    if not isinstance(nodes, Sequence) or len(nodes) < 2:
        raise ValueError("parallel node execution requires at least two nodes")
    if not isinstance(state, Mapping):
        raise TypeError("parallel node state must be a mapping")
    with ThreadPoolExecutor(max_workers=len(nodes), thread_name_prefix="forex-independent") as pool:
        futures = [pool.submit(_run_independent_node, name, node, state) for name, node in nodes]
        completed = [future.result() for future in futures]
    results = [(name, update) for name, update, _state_trace, _timing_trace in completed]
    for _name, _update, state_trace, timing_trace in completed:
        extend_state_trace(state_trace)
        extend_timing_trace(timing_trace)
    return _merge_independent_updates(state, results, nested_key=nested_key)


__all__ = ["ParallelAnalystBranch", "run_parallel_analysts", "run_parallel_nodes"]
