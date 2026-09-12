from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from tradingagents.forex.context_integrity import evaluate_context_integrity, state_artifact_metrics
from tradingagents.forex.evidence_context import EvidenceContext
from tradingagents.forex.telemetry import capture_state_trace, instrument_agent_node
from tradingagents.graph.propagation import Propagator


def _context() -> EvidenceContext:
    return EvidenceContext(
        as_of=datetime(2026, 9, 12, tzinfo=timezone.utc),
        rendered_context="bounded evidence",
        rendered_context_hash="hash-123",
        rendered_character_count=16,
        selected_knowledge_count=2,
        selected_experience_count=1,
        selected_statistics_count=0,
        dropped_knowledge_count=3,
        dropped_experience_count=4,
        dropped_statistics_count=5,
    )


def test_initial_forex_state_contains_one_context():
    context = _context()
    state = Propagator().create_initial_state(
        "EURUSD", "2026-09-12", asset_type="forex", evidence_context=context
    )
    assert state["evidence_context"] is context


def test_stock_initial_state_has_no_context():
    state = Propagator().create_initial_state("AAPL", "2026-09-12")
    assert "evidence_context" not in state


def test_every_forex_node_observes_same_context_hash():
    state = Propagator().create_initial_state(
        "EURUSD", "2026-09-12", asset_type="forex", evidence_context=_context()
    )

    def node(current_state):
        return {"news_report": "private report"}

    with capture_state_trace() as trace:
        instrument_agent_node(node, "News Analyst", SimpleNamespace(model="test"))(state)

    assert [entry["artifacts"]["evidence_context_hash"] for entry in trace] == [
        "hash-123",
        "hash-123",
    ]
    assert "private report" not in repr(trace)


def test_trace_contains_hash_counts_and_no_text():
    state = Propagator().create_initial_state(
        "EURUSD", "2026-09-12", asset_type="forex", evidence_context=_context()
    )
    metrics = state_artifact_metrics(state)
    assert {key: metrics[key] for key in (
        "evidence_context_present",
        "evidence_context_hash",
        "evidence_rendered_chars",
        "evidence_selected_counts",
        "evidence_dropped_counts",
    )} == {
        "evidence_context_present": True,
        "evidence_context_hash": "hash-123",
        "evidence_rendered_chars": 16,
        "evidence_selected_counts": {"knowledge": 2, "experience": 1, "statistics": 0},
        "evidence_dropped_counts": {"knowledge": 3, "experience": 4, "statistics": 5},
    }


def test_context_integrity_reports_missing_hash_observation():
    state = Propagator().create_initial_state(
        "EURUSD", "2026-09-12", asset_type="forex", evidence_context=_context()
    )
    trace = [{"node": "News Analyst", "phase": "after", "artifacts": {}}]
    result = evaluate_context_integrity(state, trace=trace)
    assert result["status"] == "INCOMPLETE"
    assert "context_hash:News Analyst" in result["missing"]


def test_context_integrity_reports_divergent_hash_observation():
    state = Propagator().create_initial_state(
        "EURUSD", "2026-09-12", asset_type="forex", evidence_context=_context()
    )
    trace = [
        {
            "node": "News Analyst",
            "phase": "after",
            "artifacts": {"evidence_context_hash": "other-hash"},
        }
    ]
    result = evaluate_context_integrity(state, trace=trace)
    assert result["status"] == "INCOMPLETE"
    assert result["divergent_context_hash_nodes"] == ["News Analyst"]
