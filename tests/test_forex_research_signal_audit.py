from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tradingagents.forex.research_signal_audit import (
    ResearchSignalAuditSchemaError,
    audit_research_signals,
)


def _init_db(tmp_path: Path) -> Path:
    path = tmp_path / "research-signal.db"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE shadow_decisions (
                decision_id TEXT PRIMARY KEY,
                action TEXT,
                research_manager_recommendation TEXT,
                trader_summary TEXT,
                decision_context_status TEXT,
                normalization_status TEXT
            );
            CREATE TABLE forex_watch_runs (
                run_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL,
                run_status TEXT NOT NULL,
                metrics_json TEXT
            );
            """
        )
    return path


def _metrics(*, macro: str = "MACRO/EVENT DATA UNAVAILABLE") -> str:
    return json.dumps(
        {
            "macro_event_status": macro,
            "evidence_integration_status": "DISABLED",
            "state_boundaries": [
                {
                    "node": "Market Analyst",
                    "phase": "after",
                    "artifacts": {"market": {"present": True, "content_chars": 100}},
                },
                {
                    "node": "News Analyst",
                    "phase": "after",
                    "artifacts": {"news": {"present": True, "content_chars": 80}},
                },
                {
                    "node": "Bull Researcher",
                    "phase": "after",
                    "artifacts": {"bull": {"present": True, "content_chars": 90}},
                },
                {
                    "node": "Bear Researcher",
                    "phase": "after",
                    "artifacts": {"bear": {"present": True, "content_chars": 90}},
                },
            ],
        }
    )


def test_research_signal_audit_reports_availability_and_path(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO shadow_decisions VALUES (?, ?, ?, ?, ?, ?)",
            ("d1", "HOLD", "HOLD", "FINAL TRANSACTION PROPOSAL: **HOLD**", "COMPLETE", "NORMALIZED"),
        )
        db.execute(
            "INSERT INTO forex_watch_runs VALUES (?, ?, ?, ?)",
            ("r1", "d1", "SUCCEEDED", _metrics()),
        )

    before = path.read_bytes()
    report = audit_research_signals(path)

    assert report.population_count == 1
    assert report.recommendation_counts["HOLD"] == 1
    assert report.pipeline_counts["HOLD->HOLD->HOLD"] == 1
    assert report.availability["macro_unavailable"]["samples"] == 1
    assert report.availability["macro_unavailable"]["holds"] == 1
    assert report.observations[0].bull_present is True
    assert report.observations[0].bear_present is True
    assert report.structured_upstream["bull"]["available"] is False
    assert path.read_bytes() == before


def test_research_signal_audit_does_not_treat_missing_recommendation_as_hold(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO shadow_decisions VALUES (?, ?, ?, ?, ?, ?)",
            ("d1", "HOLD", None, "FINAL TRANSACTION PROPOSAL: **HOLD**", "COMPLETE", "NORMALIZED"),
        )
        db.execute(
            "INSERT INTO forex_watch_runs VALUES (?, ?, ?, ?)",
            ("r1", "d1", "SUCCEEDED", _metrics()),
        )

    report = audit_research_signals(path)

    assert report.recommendation_counts["UNAVAILABLE"] == 1
    assert report.availability["macro_unavailable"]["samples"] == 0
    assert report.observations[0].research_recommendation is None


def test_research_signal_audit_requires_read_contract(tmp_path: Path) -> None:
    path = tmp_path / "invalid.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE shadow_decisions (decision_id TEXT)")

    with pytest.raises(ResearchSignalAuditSchemaError):
        audit_research_signals(path)


def test_research_signal_audit_cli_is_json_and_read_only(tmp_path: Path, capsys) -> None:
    path = _init_db(tmp_path)
    from cli.forex_research_signal_audit import main

    assert main(["--db-path", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["review_status"] == "DIAGNOSTIC ONLY — NO STRATEGY CHANGE PERFORMED"
    assert payload["population_count"] == 0
