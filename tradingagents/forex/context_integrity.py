"""Metadata-only validation for the forex shadow graph hand-offs.

The validator intentionally inspects only the normal state artifacts that the
existing agents already exchange.  It records presence and size/count
metadata, never report text or model reasoning.  A separate status lets later
evaluation code exclude runs whose graph context was incomplete even when the
Portfolio Manager returned a syntactically valid rating.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

DecisionContextStatus = Literal["COMPLETE", "INCOMPLETE"]

EXPECTED_FOREX_NODES = (
    "Market Analyst",
    "News Analyst",
    "Bull Researcher",
    "Bear Researcher",
    "Research Manager",
    "Trader",
    "Aggressive Analyst",
    "Conservative Analyst",
    "Neutral Analyst",
    "Portfolio Manager",
)

_DEBATE_LABEL_RE = re.compile(
    r"^\s*(?:Bull Analyst|Bear Analyst|Aggressive Analyst|"
    r"Conservative Analyst|Neutral Analyst)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _text_metric(value: Any, *, debate_labels: bool = False) -> dict[str, Any]:
    raw = _text(value)
    visible = _DEBATE_LABEL_RE.sub("", raw) if debate_labels else raw
    visible = visible.strip()
    return {
        "present": bool(visible),
        "chars": len(raw),
        "content_chars": len(visible),
    }


def _mapping_metric(value: Any) -> dict[str, Any]:
    mapping = _mapping(value)
    return {
        "present": bool(mapping),
        "fields": len(mapping),
    }


def state_artifact_metrics(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return safe presence/size metadata for the required state artifacts."""
    state_map = _mapping(state)
    investment = _mapping(state_map.get("investment_debate_state"))
    risk = _mapping(state_map.get("risk_debate_state"))

    bull = _text_metric(investment.get("bull_history"), debate_labels=True)
    bear = _text_metric(investment.get("bear_history"), debate_labels=True)
    research_history = _text_metric(investment.get("history"), debate_labels=True)
    risk_history = _text_metric(risk.get("history"), debate_labels=True)
    aggressive = _text_metric(risk.get("aggressive_history"), debate_labels=True)
    conservative = _text_metric(risk.get("conservative_history"), debate_labels=True)
    neutral = _text_metric(risk.get("neutral_history"), debate_labels=True)

    raw_pm = state_map.get("portfolio_manager_raw_result")
    final_pm = state_map.get("final_trade_decision")
    raw_pm_metric = _mapping_metric(raw_pm)
    # A failed PM marker is evidence of an attempted node, not a complete
    # normal PM artifact.
    raw_pm_mapping = _mapping(raw_pm)
    raw_pm_valid = bool(raw_pm_mapping.get("rating"))
    final_pm_text = _text(final_pm).strip()
    final_pm_valid = bool(final_pm_text) and final_pm_text != "FOREX_PORTFOLIO_MANAGER_FAILED"

    return {
        "market": {
            "field": "market_report",
            **_text_metric(state_map.get("market_report")),
        },
        "news": {
            "field": "news_report",
            **_text_metric(state_map.get("news_report")),
        },
        "bull": {
            "field": "investment_debate_state.bull_history",
            **bull,
        },
        "bear": {
            "field": "investment_debate_state.bear_history",
            **bear,
        },
        "research_manager": {
            "field": "investment_plan",
            **_text_metric(state_map.get("investment_plan")),
            "debate_history": research_history,
        },
        "trader": {
            "field": "trader_investment_plan",
            **_text_metric(state_map.get("trader_investment_plan")),
        },
        "risk_debate": {
            "field": "risk_debate_state.history",
            "present": all(
                metric["present"]
                for metric in (risk_history, aggressive, conservative, neutral)
            ),
            "history": risk_history,
            "aggressive": aggressive,
            "conservative": conservative,
            "neutral": neutral,
        },
        "portfolio_manager": {
            "field": "portfolio_manager_raw_result/final_trade_decision",
            "present": raw_pm_valid or final_pm_valid,
            "raw": raw_pm_metric,
            "raw_rating_present": raw_pm_valid,
            "final": {
                "present": final_pm_valid,
                "chars": len(final_pm_text),
            },
        },
    }


def _nodes_seen(trace: Sequence[Mapping[str, Any]] | None) -> set[str]:
    if not trace:
        return set()
    return {
        str(item.get("node"))
        for item in trace
        if isinstance(item, Mapping) and item.get("phase") == "after"
    }


def evaluate_context_integrity(
    state: Mapping[str, Any] | None,
    *,
    trace: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate required forex context and return metadata-only evidence.

    ``trace`` is optional for compatibility with callers that provide a final
    state from a test/future graph wrapper.  When a non-empty instrumented
    trace is available, every expected LLM-bearing node must also have emitted
    an after-boundary; this prevents stale/injected state from masquerading as
    a full run.
    """
    artifacts = state_artifact_metrics(state)
    missing: list[str] = []
    for name in ("market", "news", "bull", "bear", "research_manager", "trader", "risk_debate", "portfolio_manager"):
        if not artifacts[name]["present"]:
            missing.append(name)

    seen = _nodes_seen(trace)
    missing_nodes = []
    if seen:
        missing_nodes = [node for node in EXPECTED_FOREX_NODES if node not in seen]
        if missing_nodes:
            missing.extend(f"node:{node}" for node in missing_nodes)

    status: DecisionContextStatus = "COMPLETE" if not missing else "INCOMPLETE"
    return {
        "status": status,
        "missing": missing,
        "artifacts": artifacts,
        "nodes_seen": sorted(seen),
        "missing_nodes": missing_nodes,
    }


__all__ = [
    "DecisionContextStatus",
    "EXPECTED_FOREX_NODES",
    "evaluate_context_integrity",
    "state_artifact_metrics",
]
