"""Read-only revision-scoped validation for forex watcher runs."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.parse import quote

from .decision_path_audit import extract_trader_action


class RevisionValidationError(RuntimeError):
    """The requested revision cannot be validated from the source DB."""


def _safe_nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _safe_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(parsed, 0)


def _nearest_rank(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(percentile * len(ordered))) - 1
    return ordered[min(rank, len(ordered) - 1)]


def _numeric_series(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = sorted(values)
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "median": median(numbers) if numbers else None,
        "p95": _nearest_rank(numbers, 0.95),
        "max": max(numbers) if numbers else None,
    }


def aggregate_agent_metrics(metrics_documents: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate persisted per-agent numeric telemetry without private content.

    ``metrics_json`` is historical data and may contain malformed or legacy
    entries.  Invalid fields are ignored independently; no prompt, completion,
    reasoning text, or arbitrary metadata is copied into the report.
    """

    collected: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "run_count": 0,
            "models": set(),
            "calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "reasoning_tokens": 0,
            "elapsed_values": [],
        }
    )
    for document in metrics_documents:
        if not isinstance(document, Mapping):
            continue
        agents = document.get("agents")
        if not isinstance(agents, Mapping):
            continue
        for raw_name, raw_metrics in agents.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            if not isinstance(raw_metrics, Mapping):
                continue
            name = raw_name.strip()[:120]
            current = collected[name]
            current["run_count"] += 1
            model = raw_metrics.get("model")
            if isinstance(model, str) and model.strip():
                current["models"].add(model.strip()[:160])
            current["calls"] += _safe_nonnegative_int(raw_metrics.get("calls"))
            current["tokens_in"] += _safe_nonnegative_int(raw_metrics.get("tokens_in"))
            current["tokens_out"] += _safe_nonnegative_int(raw_metrics.get("tokens_out"))
            current["reasoning_tokens"] += _safe_nonnegative_int(
                raw_metrics.get("reasoning_tokens")
            )
            elapsed = _safe_nonnegative_float(raw_metrics.get("elapsed_seconds"))
            if elapsed is not None:
                current["elapsed_values"].append(elapsed)

    result: dict[str, dict[str, Any]] = {}
    for name in sorted(collected):
        current = collected[name]
        result[name] = {
            "run_count": current["run_count"],
            "calls": current["calls"],
            "models": sorted(current["models"]),
            "elapsed_seconds": _numeric_series(current["elapsed_values"]),
            "tokens_in": current["tokens_in"],
            "tokens_out": current["tokens_out"],
            "reasoning_tokens": current["reasoning_tokens"],
        }
    return result


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
    metrics_documents: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            metrics = json.loads(row["metrics_json"] or "{}")
        except (TypeError, ValueError):
            metrics = {}
        if not isinstance(metrics, Mapping):
            metrics = {}
        metrics_documents.append(metrics)
        stage_values = metrics.get("stage_timings")
        if not isinstance(stage_values, Mapping):
            stage_values = {}
        explicit_stages: set[str] = set()
        for stage, value in stage_values.items():
            if not isinstance(stage, str) or not isinstance(value, Mapping):
                continue
            elapsed = _safe_nonnegative_float(value.get("elapsed_seconds"))
            if elapsed is None:
                continue
            explicit_stages.add(stage)
            stage_timings[stage] += elapsed
        if not explicit_stages:
            boundaries = metrics.get("state_boundaries")
            if isinstance(boundaries, (list, tuple)):
                for boundary in boundaries:
                    if not isinstance(boundary, Mapping) or boundary.get("phase") != "after":
                        continue
                    stage = boundary.get("node")
                    elapsed = _safe_nonnegative_float(boundary.get("duration_seconds"))
                    if isinstance(stage, str) and stage.strip() and elapsed is not None:
                        stage_timings[stage.strip()] += elapsed
    agent_metrics = aggregate_agent_metrics(metrics_documents)
    latency_bottlenecks = sorted(
        (
            {
                "agent": agent,
                "run_count": int(values["run_count"]),
                "calls": int(values["calls"]),
                "mean_seconds": values["elapsed_seconds"]["mean"],
                "p95_seconds": values["elapsed_seconds"]["p95"],
                "max_seconds": values["elapsed_seconds"]["max"],
            }
            for agent, values in agent_metrics.items()
            if values["elapsed_seconds"]["count"]
        ),
        key=lambda value: (-float(value["p95_seconds"]), value["agent"]),
    )

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
        "agent_metrics": agent_metrics,
        "latency_bottlenecks": latency_bottlenecks,
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


__all__ = ["RevisionValidationError", "aggregate_agent_metrics", "validate_revision"]
