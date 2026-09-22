from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.graph.parallel_analysts import (
    ParallelAnalystBranch,
    run_parallel_analysts,
)


def _branch(key: str, node, report_key: str) -> ParallelAnalystBranch:
    return ParallelAnalystBranch(
        key=key,
        agent_node=f"{key.title()} Analyst",
        clear_node=f"clear_{key}",
        tool_node=f"tools_{key}",
        report_key=report_key,
        node=node,
        clear=lambda state: {},
        tool=SimpleNamespace(),
        route=lambda state: f"clear_{key}",
    )


@pytest.mark.unit
def test_parallel_analysts_share_prerequisites_and_merge_in_declared_order():
    barrier = threading.Barrier(2, timeout=2)
    seen: list[dict] = []
    calls = {"market": 0, "news": 0}

    def make_node(label: str):
        def node(state):
            calls[label] += 1
            seen.append(dict(state))
            barrier.wait()
            return {f"{label}_report": f"{label}-report"}

        return node

    branches = (
        _branch("market", make_node("market"), "market_report"),
        _branch("news", make_node("news"), "news_report"),
    )
    state = {"market_context": "snapshot", "messages": [("human", "EURUSD")]}

    started = time.perf_counter()
    result = run_parallel_analysts(branches, state)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.5
    assert [item["market_context"] for item in seen] == ["snapshot", "snapshot"]
    assert [item["messages"] for item in seen] == [[("human", "EURUSD")]] * 2
    assert list(result) == ["market_report", "news_report"]
    assert result == {"market_report": "market-report", "news_report": "news-report"}
    assert calls == {"market": 1, "news": 1}


@pytest.mark.unit
def test_parallel_analyst_failure_is_fail_closed():
    def fail(_state):
        raise RuntimeError("provider failure")

    branches = (
        _branch("market", fail, "market_report"),
        _branch("news", lambda _state: {"news_report": "ok"}, "news_report"),
    )

    with pytest.raises(RuntimeError, match="provider failure"):
        run_parallel_analysts(branches, {"messages": []})


@pytest.mark.unit
def test_parallel_branch_preserves_message_reducer_through_tool_loop():
    seen_lengths: list[int] = []

    def market_node(state):
        seen_lengths.append(len(state["messages"]))
        if len(seen_lengths) == 1:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "snapshot",
                                "args": {},
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }
        return {
            "messages": [AIMessage(content="final")],
            "market_report": "complete",
        }

    class SnapshotTool:
        def invoke(self, state, config=None):
            assert len(state["messages"]) == 2
            return {
                "messages": [
                    ToolMessage(content="snapshot", tool_call_id="call-1")
                ]
            }

    def market_route(state):
        return "tools_market" if state["messages"][-1].tool_calls else "clear_market"

    market = ParallelAnalystBranch(
        key="market",
        agent_node="Market Analyst",
        clear_node="clear_market",
        tool_node="tools_market",
        report_key="market_report",
        node=market_node,
        clear=lambda state: {},
        tool=SnapshotTool(),
        route=market_route,
    )
    news = _branch("news", lambda _state: {"news_report": "complete"}, "news_report")

    result = run_parallel_analysts(
        (market, news), {"messages": [HumanMessage(content="EURUSD")]}
    )

    assert result["market_report"] == "complete"
    assert seen_lengths == [1, 3]
