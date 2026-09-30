"""Read-only scalar dashboard metrics for the Phase 12 ledger."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .dataset import read_tick_dataset


@dataclass(frozen=True, slots=True)
class HftDashboardSnapshot:
    path: str
    status: str
    runs: int
    ticks: int
    actions: int
    entries: int
    exits: int
    open_positions: int
    p50_processing_ms: float | None
    p95_processing_ms: float | None
    p99_processing_ms: float | None = None
    max_processing_ms: float | None = None
    ticks_per_second: float | None = None
    trades: int = 0
    balance: float | None = None
    equity: float | None = None
    drawdown: float | None = None
    compound_return: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    expectancy: float | None = None
    benchmark_target: float = 0.10
    benchmark_difference: float | None = None
    benchmark_achieved: bool | None = None
    active_plan_id: str | None = None
    active_plan_direction: str | None = None
    active_plan_strategy_family: str | None = None
    active_plan_created_at: str | None = None
    active_plan_expires_at: str | None = None
    hft_lease_status: str = "INACTIVE"
    dropped_ticks: int = 0
    stale_ticks: int = 0
    out_of_order_ticks: int = 0
    last_error_code: str | None = None
    runtime_status: str = "NOT_INITIALIZED"
    last_disconnect_at: str | None = None
    last_recovery_attempt_at: str | None = None
    recovery_count: int = 0
    last_recovery_result: str | None = None
    executed: bool = False
    unique_ticks: int = 0
    tick_quality_status: str = "NOT_INITIALIZED"
    dataset_first_timestamp: str | None = None
    dataset_last_timestamp: str | None = None
    dataset_duration_seconds: float | None = None
    dataset_days: tuple[str, ...] = ()
    dataset_sessions: tuple[str, ...] = ()
    dataset_invalid_ticks: int = 0
    dataset_duplicate_ticks: int = 0
    dataset_large_gap_count: int = 0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(len(values) * fraction) - 1))
    return values[index]


def read_hft_dashboard(path: str | Path) -> HftDashboardSnapshot:
    source = Path(path)
    empty = {
        "path": str(source),
        "status": "NOT_INITIALIZED",
        "runs": 0,
        "ticks": 0,
        "actions": 0,
        "entries": 0,
        "exits": 0,
        "open_positions": 0,
        "p50_processing_ms": None,
        "p95_processing_ms": None,
        "p99_processing_ms": None,
        "max_processing_ms": None,
        "ticks_per_second": None,
        "trades": 0,
        "balance": None,
        "equity": None,
        "drawdown": None,
        "compound_return": None,
        "win_rate": None,
        "profit_factor": None,
        "expectancy": None,
        "benchmark_target": 0.10,
        "benchmark_difference": None,
        "benchmark_achieved": None,
        "active_plan_id": None,
        "active_plan_direction": None,
        "active_plan_strategy_family": None,
        "active_plan_created_at": None,
        "active_plan_expires_at": None,
        "hft_lease_status": "INACTIVE",
        "dropped_ticks": 0,
        "stale_ticks": 0,
        "out_of_order_ticks": 0,
        "last_error_code": None,
        "runtime_status": "NOT_INITIALIZED",
        "last_disconnect_at": None,
        "last_recovery_attempt_at": None,
        "recovery_count": 0,
        "last_recovery_result": None,
        "executed": False,
        "unique_ticks": 0,
        "tick_quality_status": "NOT_INITIALIZED",
        "dataset_first_timestamp": None,
        "dataset_last_timestamp": None,
        "dataset_duration_seconds": None,
        "dataset_days": (),
        "dataset_sessions": (),
        "dataset_invalid_ticks": 0,
        "dataset_duplicate_ticks": 0,
        "dataset_large_gap_count": 0,
    }
    if not source.is_file():
        return HftDashboardSnapshot(**empty)
    normalized = str(source.resolve()).replace("\\", "/")
    uri = f"file:{quote(normalized, safe='/:')}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=0) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "hft_runs" not in tables:
                return HftDashboardSnapshot(**empty)
            runtime_state = {
                "status": "NOT_INITIALIZED",
                "error_code": None,
                "last_disconnect_at": None,
                "last_recovery_attempt_at": None,
                "recovery_count": 0,
                "last_recovery_result": None,
            }
            if "hft_runtime_state" in tables:
                runtime_row = db.execute(
                    """
                    SELECT status,error_code,last_disconnect_at,last_recovery_attempt_at,
                           recovery_count,last_recovery_result
                      FROM hft_runtime_state
                     WHERE singleton_id=1
                    """
                ).fetchone()
                if runtime_row is not None:
                    runtime_state = {
                        "status": str(runtime_row[0]),
                        "error_code": runtime_row[1],
                        "last_disconnect_at": runtime_row[2],
                        "last_recovery_attempt_at": runtime_row[3],
                        "recovery_count": int(runtime_row[4]),
                        "last_recovery_result": runtime_row[5],
                    }
            counts = {name: int(db.execute(f"SELECT COUNT(*) FROM hft_{name}").fetchone()[0]) for name in ("runs", "ticks", "actions")}
            entries = int(db.execute("SELECT COUNT(*) FROM hft_actions WHERE action IN ('ENTER_LONG','ENTER_SHORT')").fetchone()[0])
            exits = int(db.execute("SELECT COUNT(*) FROM hft_actions WHERE action IN ('EXIT','REDUCE')").fetchone()[0])
            open_positions = int(db.execute("SELECT COUNT(*) FROM hft_positions WHERE state IN ('LONG','SHORT','PENDING_ENTRY','PENDING_EXIT')").fetchone()[0])
            timings = [float(row[0]) for row in db.execute("SELECT processing_ms FROM hft_actions").fetchall() if isinstance(row[0], (int, float)) and math.isfinite(float(row[0]))]
            balance = equity = drawdown = compound_return = win_rate = profit_factor = expectancy = None
            if "hft_account" in tables:
                row = db.execute("SELECT payload_json FROM hft_account ORDER BY snapshot_id DESC LIMIT 1").fetchone()
                if row:
                    payload = json.loads(row[0])
                    balance, equity, drawdown = payload.get("balance"), payload.get("equity"), payload.get("max_drawdown")
                    compound_return = payload.get("compound_return")
                    win_rate = payload.get("win_rate")
                    profit_factor = payload.get("profit_factor")
                    expectancy = payload.get("expectancy")
            plan = db.execute(
                "SELECT plan_id, payload_json FROM hft_plans ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            plan_payload = {}
            if plan:
                try:
                    plan_payload = json.loads(plan[1])
                except (TypeError, ValueError, json.JSONDecodeError):
                    plan_payload = {}
            lease_status = "INACTIVE"
            if "hft_lease" in tables:
                lease = db.execute("SELECT lease_expires_at FROM hft_lease WHERE singleton_id=1").fetchone()
                if lease:
                    try:
                        expiry = datetime.fromisoformat(str(lease[0]).replace("Z", "+00:00"))
                        if expiry.tzinfo is not None and expiry > datetime.now(timezone.utc):
                            lease_status = "ACTIVE"
                    except ValueError:
                        lease_status = "UNKNOWN"
            durations: list[float] = []
            for started, ended in db.execute("SELECT started_at, ended_at FROM hft_runs").fetchall():
                if not ended:
                    continue
                try:
                    start_dt = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(str(ended).replace("Z", "+00:00"))
                    seconds = (end_dt - start_dt).total_seconds()
                except (TypeError, ValueError):
                    continue
                if seconds > 0 and math.isfinite(seconds):
                    durations.append(seconds)
            total_seconds = sum(durations)
            dropped_ticks = stale_ticks = out_of_order_ticks = 0
            last_error_code = None
            if "hft_run_health" in tables:
                quality = db.execute(
                    "SELECT COALESCE(SUM(dropped_ticks),0), COALESCE(SUM(stale_ticks),0), COALESCE(SUM(out_of_order_ticks),0) FROM hft_run_health"
                ).fetchone()
                dropped_ticks, stale_ticks, out_of_order_ticks = (
                    int(value or 0) for value in quality
                )
                error_row = db.execute(
                    """
                    SELECT health.error_code
                    FROM hft_run_health AS health
                    JOIN hft_runs AS runs ON runs.run_id = health.run_id
                    WHERE health.error_code IS NOT NULL
                      AND runs.run_id = (
                          SELECT run_id FROM hft_runs ORDER BY started_at DESC LIMIT 1
                      )
                    """
                ).fetchone()
                last_error_code = None if error_row is None else str(error_row[0])
            if runtime_state["error_code"]:
                last_error_code = str(runtime_state["error_code"])
            runtime_status = str(runtime_state["status"])
            if runtime_status == "OPERATOR_REVIEW_REQUIRED":
                dashboard_status = runtime_status
            elif runtime_status in {"DEGRADED", "RECOVERING", "STARTING"}:
                dashboard_status = "DEGRADED"
            elif lease_status == "ACTIVE" and runtime_status in {"RUNNING", "NOT_INITIALIZED"}:
                dashboard_status = "RUNNING"
            elif runtime_status == "RUNNING":
                # A worker claiming to be alive without its lease is not
                # healthy; the lease is the durable single-owner proof.
                dashboard_status = "DEGRADED"
            elif runtime_status == "STOPPED":
                dashboard_status = "INACTIVE"
            else:
                # Preserve legacy artifacts that predate runtime-state rows.
                dashboard_status = "RUNNING" if lease_status == "ACTIVE" else "HEALTHY"
            tick_report = read_tick_dataset(source)
            actual = None if compound_return is None else float(compound_return)
            benchmark_difference = None if actual is None else actual - 0.10
            return HftDashboardSnapshot(
                path=str(source), status=dashboard_status,
                runs=counts["runs"], ticks=counts["ticks"], actions=counts["actions"], entries=entries,
                exits=exits, open_positions=open_positions,
                p50_processing_ms=_percentile(timings, 0.5), p95_processing_ms=_percentile(timings, 0.95),
                p99_processing_ms=_percentile(timings, 0.99), max_processing_ms=max(timings) if timings else None,
                ticks_per_second=(counts["ticks"] / total_seconds) if total_seconds else None,
                trades=int(payload.get("trades") or 0) if "hft_account" in tables and row else 0,
                balance=balance, equity=equity, drawdown=drawdown, compound_return=compound_return,
                win_rate=win_rate, profit_factor=profit_factor, expectancy=expectancy,
                benchmark_target=0.10, benchmark_difference=benchmark_difference,
                benchmark_achieved=None if actual is None else actual >= 0.10,
                active_plan_id=plan_payload.get("plan_id") or (plan[0] if plan else None),
                active_plan_direction=plan_payload.get("primary_direction") or plan_payload.get("direction"),
                active_plan_strategy_family=plan_payload.get("strategy_family"),
                active_plan_created_at=plan_payload.get("created_at"), active_plan_expires_at=plan_payload.get("expires_at"),
                hft_lease_status=lease_status,
                dropped_ticks=dropped_ticks,
                stale_ticks=stale_ticks,
                out_of_order_ticks=out_of_order_ticks,
                last_error_code=last_error_code,
                runtime_status=runtime_status,
                last_disconnect_at=runtime_state["last_disconnect_at"],
                last_recovery_attempt_at=runtime_state["last_recovery_attempt_at"],
                recovery_count=int(runtime_state["recovery_count"]),
                last_recovery_result=runtime_state["last_recovery_result"],
                executed=False,
                unique_ticks=tick_report.unique_ticks,
                tick_quality_status=tick_report.quality_status,
                dataset_first_timestamp=tick_report.first_timestamp,
                dataset_last_timestamp=tick_report.last_timestamp,
                dataset_duration_seconds=tick_report.duration_seconds,
                dataset_days=tick_report.days,
                dataset_sessions=tick_report.sessions,
                dataset_invalid_ticks=tick_report.invalid_ticks,
                dataset_duplicate_ticks=tick_report.duplicate_ticks,
                dataset_large_gap_count=tick_report.large_gap_count,
            )
    except sqlite3.Error:
        return HftDashboardSnapshot(**{**empty, "status": "UNAVAILABLE"})


__all__ = ["HftDashboardSnapshot", "read_hft_dashboard"]
