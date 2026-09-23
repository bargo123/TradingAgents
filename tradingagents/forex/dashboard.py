"""Read-only collection metrics for the forex shadow database.

This module deliberately does not use the writer-oriented watcher or shadow
stores.  It opens SQLite in ``mode=ro`` and derives an immutable monitoring
snapshot from the tables already maintained by the collector/evaluator.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
EVALUATION_HORIZONS = (300, 900, 1800, 3600)
EVALUATION_STATUSES = ("PENDING", "COMPLETE", "DATA_UNAVAILABLE", "INELIGIBLE")
RUN_STATUSES = ("RUNNING", "SUCCEEDED", "SUCCEEDED_SLOW", "FAILED", "ABANDONED")
VALID_RUN_STATUSES = frozenset({"SUCCEEDED", "SUCCEEDED_SLOW"})
ACTION_VALUES = ("BUY", "SELL", "HOLD")


class DashboardReadError(RuntimeError):
    """A bounded read failure that must not affect the collector writer."""


@dataclass(frozen=True)
class DashboardSnapshot:
    database_path: str
    observed_at: datetime
    database_size_bytes: int | None
    oldest_decision_timestamp: datetime | None
    newest_decision_timestamp: datetime | None
    collection_duration_seconds: float | None
    watcher: Mapping[str, Any]
    total_runs: int
    run_counts: Mapping[str, int]
    total_decisions: int
    decision_context_counts: Mapping[str, int]
    normalization_counts: Mapping[str, int]
    valid_collection_decisions: int
    action_counts: Mapping[str, int]
    reliability: Mapping[str, Any]
    latency: Mapping[str, Any]
    evaluation_horizons: Mapping[int, Mapping[str, int]]
    fully_evaluated_decisions: int
    fully_training_eligible_decisions: int
    outcome_performance: Mapping[int, Mapping[str, Any]]
    training_readiness: Mapping[str, Any]
    recent_decisions: tuple[Mapping[str, Any], ...]
    health: str
    health_reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _row_value(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    try:
        result = None if value is None else float(value)
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def _is_true(value: Any) -> bool:
    return value is True or value == 1 or value == "1" or value == "true"


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered))) - 1
    return ordered[min(rank, len(ordered) - 1)]


def _series(values: Iterable[Any]) -> dict[str, float | int | None]:
    numbers = [number for value in values if (number := _float(value)) is not None]
    return {
        "count": len(numbers),
        "latest": numbers[-1] if numbers else None,
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "p50": _percentile(numbers, 50),
        "p95": _percentile(numbers, 95),
        "max": max(numbers) if numbers else None,
    }


def _read_only_connection(path: Path, busy_timeout_seconds: float) -> sqlite3.Connection:
    if busy_timeout_seconds <= 0:
        raise ValueError("busy_timeout_seconds must be positive")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise DashboardReadError(f"database does not exist: {resolved}")
    uri = f"file:{resolved.as_posix()}?mode=ro"
    try:
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=busy_timeout_seconds,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {max(1, int(busy_timeout_seconds * 1000))}")
        connection.execute("PRAGMA query_only = ON")
        return connection
    except sqlite3.Error as exc:
        raise DashboardReadError(f"database read unavailable: {exc}") from exc


def _table_rows(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if exists is None:
        return []
    return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"').fetchall()]


def _preferred_run(runs: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not runs:
        return None
    return max(
        runs,
        key=lambda row: _parse_timestamp(
            _row_value(row, "completed_at", "started_at")
        )
        or datetime.min.replace(tzinfo=UTC),
    )


def _decision_run_map(runs: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        decision_id = _row_value(row, "decision_id")
        if decision_id:
            grouped[str(decision_id)].append(row)
    return {decision_id: _preferred_run(rows) for decision_id, rows in grouped.items()}


def _runtime_stale(decision: Mapping[str, Any], run: Mapping[str, Any] | None) -> int:
    return _int(_row_value(decision, "stale_by_completion", default=_row_value(run or {}, "stale_by_completion")))


def _decision_status(
    decision: Mapping[str, Any],
    run: Mapping[str, Any] | None,
    *names: str,
    default: Any = None,
) -> Any:
    return _row_value(decision, *names, default=_row_value(run or {}, *names, default=default))


def _valid_collection_decision(decision: Mapping[str, Any], run: Mapping[str, Any] | None) -> bool:
    if run is None:
        return False
    return (
        _row_value(run, "run_status", "status") in VALID_RUN_STATUSES
        and _decision_status(decision, run, "decision_context_status", "context_status") == "COMPLETE"
        and _decision_status(decision, run, "normalization_status") == "NORMALIZED"
        and _runtime_stale(decision, run) == 0
        and _decision_status(decision, run, "decision_reference_status", "reference_status")
        == "AVAILABLE"
    )


def _decision_time(row: Mapping[str, Any]) -> datetime | None:
    return _parse_timestamp(
        _row_value(row, "created_at", "decision_completed_timestamp", "completed_at", "snapshot_timestamp")
    )


def _build_evaluation_metrics(
    evaluations: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[int, dict[str, int]],
    int,
    int,
    dict[int, dict[str, Any]],
]:
    horizons = {
        horizon: dict.fromkeys(EVALUATION_STATUSES, 0)
        for horizon in EVALUATION_HORIZONS
    }
    grouped: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    eligible_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluations:
        if _row_value(row, "evaluation_basis") != "DECISION_REFERENCE":
            continue
        horizon = _int(_row_value(row, "horizon_seconds"), -1)
        status = str(_row_value(row, "evaluation_status", default="UNKNOWN"))
        if horizon in horizons and status in EVALUATION_STATUSES:
            horizons[horizon][status] += 1
        decision_id = str(_row_value(row, "decision_id", default=""))
        if decision_id and horizon in EVALUATION_HORIZONS:
            grouped[decision_id][horizon] = row
        if status == "COMPLETE" and _is_true(_row_value(row, "training_eligible")):
            eligible_rows[horizon].append(row)

    fully_evaluated = 0
    fully_training_eligible = 0
    for rows in grouped.values():
        if all(
            horizon in rows and _row_value(rows[horizon], "evaluation_status") == "COMPLETE"
            for horizon in EVALUATION_HORIZONS
        ):
            fully_evaluated += 1
            if all(_is_true(_row_value(rows[horizon], "training_eligible")) for horizon in EVALUATION_HORIZONS):
                fully_training_eligible += 1

    performance: dict[int, dict[str, Any]] = {}
    for horizon in EVALUATION_HORIZONS:
        rows = eligible_rows[horizon]
        values = [
            number
            for row in rows
            if (number := _float(_row_value(row, "selected_action_net_points"))) is not None
        ]
        action_counts = Counter(
            str(_row_value(row, "selected_action", default="UNKNOWN")) for row in rows
        )
        denominator = len(values)
        performance[horizon] = {
            "sample_count": len(rows),
            "average_selected_action_net_points": sum(values) / denominator if denominator else None,
            "median_selected_action_net_points": (
                _percentile(values, 50) if values else None
            ),
            "positive_percent": (
                sum(value > 0 for value in values) * 100.0 / denominator if denominator else None
            ),
            "negative_percent": (
                sum(value < 0 for value in values) * 100.0 / denominator if denominator else None
            ),
            "flat_percent": (
                sum(value == 0 for value in values) * 100.0 / denominator if denominator else None
            ),
            "action_counts": {action: int(action_counts.get(action, 0)) for action in ACTION_VALUES},
        }
    return horizons, fully_evaluated, fully_training_eligible, performance


def derive_health(snapshot: DashboardSnapshot) -> tuple[str, tuple[str, ...]]:
    """Derive operational health without considering direction or returns."""
    degraded: list[str] = []
    warnings: list[str] = []
    watcher = snapshot.watcher
    if watcher.get("circuit_reason"):
        degraded.append(f"circuit: {watcher['circuit_reason']}")
    latest = snapshot.recent_decisions[0] if snapshot.recent_decisions else None
    if latest is not None:
        if latest.get("normalization") == "FAILED":
            degraded.append("latest normalization failed")
        if latest.get("stale"):
            degraded.append("latest decision exceeded freshness budget")
        if latest.get("reference_status") == "INVALID_TEMPORAL":
            degraded.append("latest decision reference is temporally invalid")
    recent_statuses = [
        row.get("run_status")
        for row in snapshot.recent_decisions
        if row.get("run_status") is not None
    ]
    if len(recent_statuses) >= 3 and all(status == "FAILED" for status in recent_statuses[:3]):
        degraded.append("three most recent runs failed")
    if snapshot.valid_collection_decisions == 0:
        warnings.append("no valid collection decisions")
    if sum(snapshot.evaluation_horizons[3600].values()) == 0:
        warnings.append("no completed 60m outcomes")
    if snapshot.reliability.get("data_unavailable_evaluations", 0):
        warnings.append("some outcomes are data unavailable")
    p95 = snapshot.latency.get("runtime_p95_seconds")
    budget = snapshot.reliability.get("freshness_budget_seconds")
    if p95 is not None and budget and p95 >= float(budget) * 0.8:
        warnings.append("runtime p95 is close to freshness budget")
    if degraded:
        return "DEGRADED", tuple(degraded + warnings)
    if warnings:
        return "WARNING", tuple(warnings)
    return "HEALTHY", ()


def _read_snapshot(connection: sqlite3.Connection, path: Path, observed_at: datetime) -> DashboardSnapshot:
    state_rows = _table_rows(connection, "forex_watcher_state")
    state = state_rows[0] if state_rows else {}
    runs = _table_rows(connection, "forex_watch_runs")
    decisions = _table_rows(connection, "shadow_decisions")
    evaluations = _table_rows(connection, "shadow_decision_evaluations")
    run_map = _decision_run_map(runs)

    run_counts = {status: sum(_row_value(row, "run_status", "status") == status for row in runs) for status in RUN_STATUSES}
    run_counts["UNKNOWN"] = sum(
        _row_value(row, "run_status", "status") not in RUN_STATUSES for row in runs
    )
    decision_context_counts = Counter(
        str(_row_value(row, "decision_context_status", "context_status", default="UNKNOWN"))
        for row in decisions
    )
    normalization_counts = Counter(
        str(_row_value(row, "normalization_status", default="UNKNOWN")) for row in decisions
    )
    valid_decisions = [
        row for row in decisions if _valid_collection_decision(row, run_map.get(str(row.get("decision_id"))))
    ]
    action_counts = {
        action: sum(
            _row_value(row, "action", default=_row_value(run_map.get(str(row.get("decision_id"))) or {}, "normalized_action"))
            == action
            for row in valid_decisions
        )
        for action in ACTION_VALUES
    }
    reference_statuses = Counter(
        str(_decision_status(row, run_map.get(str(row.get("decision_id"))), "decision_reference_status", "reference_status", default="UNKNOWN"))
        for row in decisions
    )
    stale_count = sum(
        _runtime_stale(row, run_map.get(str(row.get("decision_id")))) == 1 for row in decisions
    )
    failure_codes = Counter(
        str(_row_value(row, "failure_code", default="UNKNOWN"))
        for row in runs
        if _row_value(row, "failure_code")
    )
    failure_codes.update(
        {"NORMALIZATION_FAILED": int(normalization_counts.get("FAILED", 0))}
        if normalization_counts.get("FAILED", 0)
        else {}
    )
    runtime_exceeded = sum(
        (_float(_row_value(row, "runtime_seconds")) or 0.0)
        >= (_float(_row_value(row, "freshness_budget_seconds")) or 900.0)
        for row in runs
        if _row_value(row, "runtime_seconds") is not None
    )
    data_unavailable = sum(
        _row_value(row, "evaluation_status") == "DATA_UNAVAILABLE"
        and _row_value(row, "evaluation_basis") == "DECISION_REFERENCE"
        for row in evaluations
    )
    budgets = [
        _float(_row_value(row, "freshness_budget_seconds"))
        for row in runs
        if _row_value(row, "freshness_budget_seconds") is not None
    ]
    reliability = {
        "stale_by_completion_count": stale_count,
        "decision_reference_status_counts": dict(reference_statuses),
        "decision_context_counts": dict(decision_context_counts),
        "normalization_counts": dict(normalization_counts),
        "failure_codes": dict(failure_codes),
        "pm_normalization_failure_count": int(normalization_counts.get("FAILED", 0)),
        "runtime_at_or_above_freshness_budget_count": runtime_exceeded,
        "freshness_budget_seconds": max(budgets) if budgets else 900.0,
        "data_unavailable_evaluations": data_unavailable,
    }

    successful_runs = [
        row
        for row in runs
        if _row_value(row, "run_status", "status") in VALID_RUN_STATUSES
        and _row_value(row, "completed_at", "started_at") is not None
    ]
    successful_runs.sort(key=lambda row: _parse_timestamp(_row_value(row, "completed_at", "started_at")) or datetime.min.replace(tzinfo=UTC))
    latency = {
        "runtime_latest_seconds": _float(_row_value(successful_runs[-1], "runtime_seconds")) if successful_runs else None,
        "runtime_mean_seconds": _series(_row_value(row, "runtime_seconds") for row in successful_runs)["mean"],
        "runtime_p50_seconds": _series(_row_value(row, "runtime_seconds") for row in successful_runs)["p50"],
        "runtime_p95_seconds": _series(_row_value(row, "runtime_seconds") for row in successful_runs)["p95"],
        "runtime_max_seconds": _series(_row_value(row, "runtime_seconds") for row in successful_runs)["max"],
        "analysis_latency_latest_seconds": _float(_row_value(successful_runs[-1], "analysis_latency_seconds")) if successful_runs else None,
        "analysis_latency_p50_seconds": _series(_row_value(row, "analysis_latency_seconds") for row in successful_runs)["p50"],
        "analysis_latency_p95_seconds": _series(_row_value(row, "analysis_latency_seconds") for row in successful_runs)["p95"],
        "llm_calls_latest": _int(_row_value(successful_runs[-1], "llm_calls"), 0) if successful_runs else None,
        "llm_calls_mean": _series(_row_value(row, "llm_calls") for row in successful_runs)["mean"],
        "llm_calls_p95": _series(_row_value(row, "llm_calls") for row in successful_runs)["p95"],
        "tokens_in_latest": _int(_row_value(successful_runs[-1], "tokens_in"), 0) if successful_runs else None,
        "tokens_out_latest": _int(_row_value(successful_runs[-1], "tokens_out"), 0) if successful_runs else None,
        "tokens_in_mean": _series(_row_value(row, "tokens_in") for row in successful_runs)["mean"],
        "tokens_out_mean": _series(_row_value(row, "tokens_out") for row in successful_runs)["mean"],
    }
    horizon_counts, fully_evaluated, fully_training, performance = _build_evaluation_metrics(evaluations)

    decision_times = [time for row in decisions if (time := _decision_time(row)) is not None]
    oldest = min(decision_times) if decision_times else None
    newest = max(decision_times) if decision_times else None
    collection_duration = (newest - oldest).total_seconds() if oldest and newest else None
    recent_rows = sorted(decisions, key=lambda row: _decision_time(row) or datetime.min.replace(tzinfo=UTC), reverse=True)[:10]
    recent = tuple(
        {
            "timestamp": _decision_time(row),
            "symbol": _row_value(row, "resolved_symbol", "requested_symbol", default="?"),
            "action": _row_value(row, "action", default=_row_value(run_map.get(str(row.get("decision_id"))) or {}, "normalized_action")),
            "context_status": _row_value(row, "decision_context_status", "context_status", default="UNKNOWN"),
            "normalization": _row_value(row, "normalization_status", default="UNKNOWN"),
            "runtime": _float(_row_value(run_map.get(str(row.get("decision_id"))) or {}, "runtime_seconds")),
            "stale": _runtime_stale(row, run_map.get(str(row.get("decision_id")))),
            "reference_status": _decision_status(row, run_map.get(str(row.get("decision_id"))), "decision_reference_status", "reference_status", default="UNKNOWN"),
            "reference_delay": _float(_decision_status(row, run_map.get(str(row.get("decision_id"))), "decision_reference_delay_seconds", "reference_delay")),
            "llm_calls": _int(_row_value(run_map.get(str(row.get("decision_id"))) or {}, "llm_calls"), 0),
            "run_status": _row_value(run_map.get(str(row.get("decision_id"))) or {}, "run_status", "status"),
        }
        for row in recent_rows
    )
    watcher = dict(state)
    current_run_id = watcher.get("current_run_id")
    if current_run_id:
        active = next((row for row in runs if row.get("run_id") == current_run_id), None)
        if active:
            watcher["active_run_status"] = _row_value(active, "run_status", "status")
            watcher["active_symbol"] = _row_value(active, "resolved_symbol", "requested_symbol")
            watcher["active_run_started_at"] = _row_value(active, "started_at")
    successful_decision_times = [
        _decision_time(row)
        for row in decisions
        if _row_value(run_map.get(str(row.get("decision_id"))) or {}, "run_status", "status")
        in VALID_RUN_STATUSES
    ]
    if successful_decision_times:
        watcher.setdefault("last_successful_decision_at", max(successful_decision_times).isoformat())
    readiness = {
        "valid_collection_decisions": len(valid_decisions),
        "fully_evaluated_decisions": fully_evaluated,
        "fully_training_eligible_decisions": fully_training,
        "pending_60m": horizon_counts[3600]["PENDING"],
        "data_unavailable_evaluations": data_unavailable,
        "review_target": 100,
        "stronger_sample_target": 300,
    }
    draft = DashboardSnapshot(
        database_path=str(path),
        observed_at=observed_at,
        database_size_bytes=path.stat().st_size if path.exists() else None,
        oldest_decision_timestamp=oldest,
        newest_decision_timestamp=newest,
        collection_duration_seconds=collection_duration,
        watcher=watcher,
        total_runs=len(runs),
        run_counts=run_counts,
        total_decisions=len(decisions),
        decision_context_counts=dict(decision_context_counts),
        normalization_counts=dict(normalization_counts),
        valid_collection_decisions=len(valid_decisions),
        action_counts=action_counts,
        reliability=reliability,
        latency=latency,
        evaluation_horizons=horizon_counts,
        fully_evaluated_decisions=fully_evaluated,
        fully_training_eligible_decisions=fully_training,
        outcome_performance=performance,
        training_readiness=readiness,
        recent_decisions=recent,
        health="WARNING",
        health_reasons=(),
    )
    health, reasons = derive_health(draft)
    return DashboardSnapshot(**{**draft.__dict__, "health": health, "health_reasons": reasons})


def read_dashboard_snapshot(
    db_path: str | Path,
    *,
    busy_timeout_seconds: float = 0.25,
) -> DashboardSnapshot:
    """Read one consistent snapshot without creating or modifying the DB."""
    path = Path(db_path)
    observed_at = datetime.now(UTC)
    try:
        connection = _read_only_connection(path, busy_timeout_seconds)
        try:
            return _read_snapshot(connection, path, observed_at)
        finally:
            connection.close()
    except DashboardReadError:
        raise
    except sqlite3.Error as exc:
        message = str(exc)
        if "locked" in message.lower() or "busy" in message.lower():
            raise DashboardReadError(f"database busy or locked: {message}") from exc
        raise DashboardReadError(f"database read failed: {message}") from exc


__all__ = [
    "ACTION_VALUES",
    "EVALUATION_HORIZONS",
    "EVALUATION_STATUSES",
    "DashboardReadError",
    "DashboardSnapshot",
    "derive_health",
    "read_dashboard_snapshot",
]
