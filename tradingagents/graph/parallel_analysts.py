"""Bounded parallel execution for independent forex analyst branches.

The normal graph keeps debate and risk speakers sequential because each speaker
reads the previous speaker's response.  Market and News are different: each
starts from the same initial snapshot and produces a separate report.  This
module executes only those independent analyst branches concurrently while
retaining each branch's existing tool-call loop and deterministic merge order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from tradingagents.forex.telemetry import capture_state_trace, extend_state_trace

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
) -> tuple[str, list[dict[str, Any]]]:
    state = _copy_branch_state(initial_state)
    with capture_state_trace() as branch_trace:
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
                return str(state.get(branch.report_key, "") or ""), list(branch_trace)
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
    for branch, (report, branch_trace) in zip(branches, results, strict=True):
        extend_state_trace(branch_trace)
        merged[branch.report_key] = report
    return merged


__all__ = ["ParallelAnalystBranch", "run_parallel_analysts"]
