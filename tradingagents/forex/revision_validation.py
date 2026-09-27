"""Read-only revision-scoped validation for forex watcher runs."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from contextlib import closing
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import quote

from .decision_path_audit import (
    RESEARCH_RECOMMENDATIONS,
    TRADING_ACTIONS,
    extract_trader_action,
)


class RevisionValidationError(RuntimeError):
    """The requested revision cannot be validated from the source DB."""


def _canonical_or_unavailable(value: Any, allowed: tuple[str, ...]) -> str:
    """Report only exact persisted enum values; quarantine everything else."""

    return value if isinstance(value, str) and value in allowed else "UNAVAILABLE"


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


def _safe_db_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    return None


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


def _freshness_report(rows: Iterable[sqlite3.Row]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    stale_count = 0
    budgets: list[float] = []
    for row in rows:
        budget = _safe_nonnegative_float(row["freshness_budget_seconds"])
        if budget is not None:
            budgets.append(budget)
        raw_stale = row["stale_by_completion"]
        parsed_stale = _safe_db_bool(raw_stale)
        if raw_stale is not None and parsed_stale is None:
            status = "UNAVAILABLE"
            stale = False
        elif raw_stale is not None:
            stale = parsed_stale is True
            status = "STALE" if stale else "WITHIN_BUDGET"
        else:
            latency = _safe_nonnegative_float(row["analysis_latency_seconds"])
            stale = latency is not None and budget is not None and latency >= budget
            if latency is None or budget is None:
                status = "UNAVAILABLE"
            else:
                status = "STALE" if stale else "WITHIN_BUDGET"
        status_counts[status] += 1
        stale_count += int(status == "STALE")
    return {
        "status_counts": dict(status_counts),
        "stale_by_completion_count": stale_count,
        "budget_seconds": _numeric_series(budgets),
    }


def _evaluation_coverage(conn: sqlite3.Connection, rows: Iterable[sqlite3.Row]) -> dict[str, Any]:
    decision_ids = {str(row["decision_id"]) for row in rows if row["decision_id"]}
    result: dict[str, Any] = {
        "evaluations_total": 0,
        "decisions_with_evaluations": 0,
        "decision_count": len(decision_ids),
        "status_counts": {},
        "by_basis_horizon_status": {},
    }
    if not decision_ids:
        return result
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='shadow_decision_evaluations'"
    ).fetchone()
    if table_exists is None:
        return result
    status_counts: Counter[str] = Counter()
    by_key: Counter[str] = Counter()
    observed_decisions: set[str] = set()
    for evaluation in conn.execute(
        "SELECT decision_id, evaluation_basis, horizon_seconds, evaluation_status "
        "FROM shadow_decision_evaluations"
    ):
        decision_id = str(evaluation["decision_id"])
        if decision_id not in decision_ids:
            continue
        basis = str(evaluation["evaluation_basis"])
        horizon = _safe_nonnegative_int(evaluation["horizon_seconds"])
        status = str(evaluation["evaluation_status"])
        observed_decisions.add(decision_id)
        status_counts[status] += 1
        by_key[f"{basis}/{horizon}/{status}"] += 1
    result.update(
        evaluations_total=sum(status_counts.values()),
        decisions_with_evaluations=len(observed_decisions),
        status_counts=dict(status_counts),
        by_basis_horizon_status=dict(by_key),
    )
    return result


def _connect(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RevisionValidationError(f"database does not exist: {resolved}")
    uri = "file:" + quote(resolved.as_posix(), safe="/:\\") + "?mode=ro"
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        return conn
    except sqlite3.Error as exc:
        if conn is not None:
            conn.close()
        raise RevisionValidationError(f"read-only database open failed: {exc}") from exc


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def validate_revision(db_path: str | Path, commit: str) -> dict[str, Any]:
    """Return scalar-only rows attributable to one commit SHA."""

    if not isinstance(commit, str) or not commit.strip():
        raise ValueError("commit must be non-empty")
    with closing(_connect(Path(db_path))) as conn, conn:
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
        freshness = _freshness_report(rows)
        evaluation_coverage = _evaluation_coverage(conn, rows)

    status_counts = Counter(str(row["run_status"]) for row in rows)
    reference_counts = Counter(str(row["decision_reference_status"] or "UNAVAILABLE") for row in rows)
    research_counts = Counter(
        _canonical_or_unavailable(row["research_manager_recommendation"], RESEARCH_RECOMMENDATIONS)
        for row in rows
    )
    trader_counts = Counter(
        extract_trader_action(row["trader_summary"])
        if row["trader_summary"]
        else "UNAVAILABLE"
        for row in rows
    )
    pm_counts = Counter(
        _canonical_or_unavailable(row["persisted_action"], TRADING_ACTIONS) for row in rows
    )
    runtimes = [float(row["runtime_seconds"]) for row in rows if row["runtime_seconds"] is not None]
    transitions = Counter(
        f"{_canonical_or_unavailable(row['research_manager_recommendation'], RESEARCH_RECOMMENDATIONS)}"
        f"->{extract_trader_action(row['trader_summary']) if row['trader_summary'] else 'UNAVAILABLE'}"
        f"->{_canonical_or_unavailable(row['persisted_action'], TRADING_ACTIONS)}"
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
    runtime_summary = _numeric_series(runtimes)
    # Preserve the original report keys while exposing the shared p95 shape.
    runtime_summary["average"] = runtime_summary["mean"]
    runtime_summary["maximum"] = runtime_summary["max"]
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
        "runtime_seconds": runtime_summary,
        "freshness": freshness,
        "evaluation_coverage": evaluation_coverage,
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
