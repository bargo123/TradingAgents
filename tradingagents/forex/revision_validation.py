"""Read-only revision-scoped validation for forex watcher runs."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.parse import quote

from .decision_path_audit import extract_trader_action


class RevisionValidationError(RuntimeError):
    """The requested revision cannot be validated from the source DB."""


def _connect(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RevisionValidationError(f"database does not exist: {resolved}")
    uri = "file:" + quote(resolved.as_posix(), safe="/:\\") + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        return conn
    except sqlite3.Error as exc:
        raise RevisionValidationError(f"read-only database open failed: {exc}") from exc


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def validate_revision(db_path: str | Path, commit: str) -> dict[str, Any]:
    """Return scalar-only rows attributable to one commit SHA."""

    if not isinstance(commit, str) or not commit.strip():
        raise ValueError("commit must be non-empty")
    with _connect(Path(db_path)) as conn:
        run_columns = _table_columns(conn, "forex_watch_runs")
        if "git_commit" not in run_columns:
            raise RevisionValidationError("forex_watch_runs.git_commit is unavailable")
        rows = conn.execute(
            """
            SELECT r.*, d.action AS persisted_action,
                   d.research_manager_recommendation,
                   d.trader_summary,
                   d.decision_context_status AS persisted_context_status,
                   d.normalization_status AS persisted_normalization_status
            FROM forex_watch_runs AS r
            LEFT JOIN shadow_decisions AS d ON d.decision_id = r.decision_id
            WHERE r.git_commit = ?
            ORDER BY r.started_at, r.run_id
            """,
            (commit.strip(),),
        ).fetchall()

    status_counts = Counter(str(row["run_status"]) for row in rows)
    reference_counts = Counter(str(row["decision_reference_status"] or "UNAVAILABLE") for row in rows)
    research_counts = Counter(
        str(row["research_manager_recommendation"] or "UNAVAILABLE") for row in rows
    )
    trader_counts = Counter(
        extract_trader_action(row["trader_summary"])
        if row["trader_summary"]
        else "UNAVAILABLE"
        for row in rows
    )
    pm_counts = Counter(str(row["persisted_action"] or "UNAVAILABLE") for row in rows)
    runtimes = [float(row["runtime_seconds"]) for row in rows if row["runtime_seconds"] is not None]
    transitions = Counter(
        f"{row['research_manager_recommendation'] or 'UNAVAILABLE'}"
        f"->{extract_trader_action(row['trader_summary']) if row['trader_summary'] else 'UNAVAILABLE'}"
        f"->{row['persisted_action'] or 'UNAVAILABLE'}"
        for row in rows
    )
    stage_timings: Counter[str] = Counter()
    for row in rows:
        try:
            metrics = json.loads(row["metrics_json"] or "{}")
        except (TypeError, ValueError):
            metrics = {}
        for stage, value in (metrics.get("stage_timings") or {}).items():
            if isinstance(value, dict) and isinstance(value.get("elapsed_seconds"), (int, float)):
                stage_timings[stage] += float(value["elapsed_seconds"])

    return {
        "commit": commit.strip(),
        "run_count": len(rows),
        "run_status_counts": dict(status_counts),
        "runtime_seconds": {
            "count": len(runtimes),
            "average": mean(runtimes) if runtimes else None,
            "median": median(runtimes) if runtimes else None,
            "maximum": max(runtimes) if runtimes else None,
        },
        "temporal_status_counts": dict(reference_counts),
        "research_recommendations": dict(research_counts),
        "trader_actions": dict(trader_counts),
        "portfolio_manager_actions": dict(pm_counts),
        "decision_path_transitions": dict(transitions),
        "stage_elapsed_seconds": dict(stage_timings),
        "upstream_stances": "NOT_PERSISTED_PROSE_BOUNDARY",
        "rows": [
            {
                "run_id": row["run_id"],
                "source_run_id": row["source_run_id"],
                "decision_id": row["decision_id"],
                "run_status": row["run_status"],
                "runtime_seconds": row["runtime_seconds"],
                "decision_context_status": row["decision_context_status"],
                "normalization_status": row["normalization_status"],
                "normalized_action": row["normalized_action"],
                "decision_reference_status": row["decision_reference_status"],
                "research_manager_recommendation": row["research_manager_recommendation"],
            }
            for row in rows
        ],
    }


__all__ = ["RevisionValidationError", "validate_revision"]
