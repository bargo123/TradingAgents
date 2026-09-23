"""Read-only Rich dashboard for persisted forex shadow collection."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tradingagents.forex.dashboard import (
    EVALUATION_HORIZONS,
    DashboardReadError,
    DashboardSnapshot,
    read_dashboard_snapshot,
)

UTC = timezone.utc


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex_dashboard",
        description="Read-only Rich dashboard for the forex shadow collector.",
    )
    parser.add_argument("--db-path", default="data_cache/shadow_decisions.db")
    parser.add_argument(
        "--refresh-seconds",
        type=_positive_float,
        default=10.0,
        help="refresh interval for the live dashboard (default: 10)",
    )
    parser.add_argument("--once", action="store_true", help="render one snapshot and exit")
    return parser


def _fmt(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _fmt_timestamp(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(value)


def _age(observed_at: datetime, value: Any) -> str:
    if value is None:
        return "-"
    if not isinstance(value, datetime):
        try:
            value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return "-"
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    seconds = max(0.0, (observed_at - value.astimezone(UTC)).total_seconds())
    return f"{seconds:.0f} s"


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def _bar(current: int, target: int, width: int = 20) -> str:
    if target <= 0:
        return ""
    filled = min(width, round(width * current / target))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _health_text(snapshot: DashboardSnapshot) -> Text:
    colour = {"HEALTHY": "green", "WARNING": "yellow", "DEGRADED": "red"}.get(
        snapshot.health, "white"
    )
    return Text(snapshot.health, style=f"bold {colour}")


def _header(snapshot: DashboardSnapshot, warning: str | None) -> RenderableType:
    watcher = snapshot.watcher
    lifecycle = watcher.get("lifecycle_status", "UNKNOWN")
    active_status = watcher.get("active_run_status", "none")
    last_success = watcher.get("last_analysis_completed_at") or watcher.get("last_successful_decision_at")
    lines = [
        Text("FOREX SHADOW COLLECTION DASHBOARD", style="bold cyan"),
        "READ-ONLY OBSERVABILITY - NO ORDER WILL BE SENT",
        f"Database: {snapshot.database_path}",
        f"Observed: {_fmt_timestamp(snapshot.observed_at)}    Health: ",
        f"Watcher lifecycle: {lifecycle}    Active run: {active_status}    Active symbol: {watcher.get('active_symbol', '-')}",
        f"Active run start: {_fmt_timestamp(watcher.get('active_run_started_at'))}",
        f"Lease expiry: {_fmt_timestamp(watcher.get('lease_expires_at'))}",
        f"Last successful decision: {_fmt_timestamp(last_success)} ({_age(snapshot.observed_at, last_success)})    Evaluation due: {bool(watcher.get('evaluation_due_pending'))}",
        f"DB size: {_human_bytes(snapshot.database_size_bytes)}    Collection duration: {_fmt(snapshot.collection_duration_seconds, suffix=' s')}",
    ]
    group: list[RenderableType] = [lines[0], Text("\n".join(str(line) for line in lines[1:]))]
    group.insert(1, _health_text(snapshot))
    if warning:
        group.append(Text(f"READ WARNING: {warning}", style="yellow"))
    return Panel(Group(*group), title="STATUS", border_style="cyan")


def _collection_table(snapshot: DashboardSnapshot) -> Table:
    table = Table(title="COLLECTION PROGRESS", expand=True)
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for label, value in (
        ("Total watcher runs", snapshot.total_runs),
        ("Succeeded", snapshot.run_counts.get("SUCCEEDED", 0)),
        ("Succeeded slow", snapshot.run_counts.get("SUCCEEDED_SLOW", 0)),
        ("Failed", snapshot.run_counts.get("FAILED", 0)),
        ("Abandoned", snapshot.run_counts.get("ABANDONED", 0)),
        ("Currently running", snapshot.run_counts.get("RUNNING", 0)),
        ("Total shadow decisions", snapshot.total_decisions),
        ("Decision context COMPLETE", snapshot.decision_context_counts.get("COMPLETE", 0)),
        ("Decision context INCOMPLETE", snapshot.decision_context_counts.get("INCOMPLETE", 0)),
        ("Normalization NORMALIZED", snapshot.normalization_counts.get("NORMALIZED", 0)),
        ("Normalization FAILED", snapshot.normalization_counts.get("FAILED", 0)),
        ("VALID COLLECTION DECISIONS", snapshot.valid_collection_decisions),
    ):
        table.add_row(label, str(value))
    return table


def _action_reliability_table(snapshot: DashboardSnapshot) -> Table:
    reliability = snapshot.reliability
    table = Table(title="ACTION / RELIABILITY", expand=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    total = sum(snapshot.action_counts.values())
    for action in ("BUY", "SELL", "HOLD"):
        count = snapshot.action_counts.get(action, 0)
        percentage = count * 100.0 / total if total else None
        table.add_row(action, f"{count} ({_fmt(percentage, suffix='%')})")
    table.add_row("Stale by completion", str(reliability.get("stale_by_completion_count", 0)))
    ref_counts = reliability.get("decision_reference_status_counts", {})
    for status in ("AVAILABLE", "INVALID_TEMPORAL", "UNAVAILABLE"):
        table.add_row(f"Reference {status}", str(ref_counts.get(status, 0)))
    table.add_row("PM/normalization failures", str(reliability.get("pm_normalization_failure_count", 0)))
    table.add_row(">= freshness budget", str(reliability.get("runtime_at_or_above_freshness_budget_count", 0)))
    failure_codes = reliability.get("failure_codes", {})
    table.add_row("Failure codes", ", ".join(f"{k}:{v}" for k, v in sorted(failure_codes.items())) or "-")
    return table


def _latency_table(snapshot: DashboardSnapshot) -> Table:
    table = Table(title="PERFORMANCE / LATENCY", expand=True)
    table.add_column("Metric")
    table.add_column("Latest", justify="right")
    table.add_column("Mean / p50", justify="right")
    table.add_column("p95 / max", justify="right")
    latency = snapshot.latency
    table.add_row(
        "Runtime (s)",
        _fmt(latency.get("runtime_latest_seconds")),
        f"{_fmt(latency.get('runtime_mean_seconds'))} / {_fmt(latency.get('runtime_p50_seconds'))}",
        f"{_fmt(latency.get('runtime_p95_seconds'))} / {_fmt(latency.get('runtime_max_seconds'))}",
    )
    table.add_row(
        "Analysis latency (s)",
        _fmt(latency.get("analysis_latency_latest_seconds")),
        f"- / {_fmt(latency.get('analysis_latency_p50_seconds'))}",
        f"{_fmt(latency.get('analysis_latency_p95_seconds'))} / -",
    )
    table.add_row(
        "LLM calls",
        _fmt(latency.get("llm_calls_latest"), digits=0),
        f"{_fmt(latency.get('llm_calls_mean'))} / -",
        f"{_fmt(latency.get('llm_calls_p95'))} / -",
    )
    table.add_row(
        "Tokens in / out (latest)",
        f"{_fmt(latency.get('tokens_in_latest'), digits=0)} / {_fmt(latency.get('tokens_out_latest'), digits=0)}",
        f"{_fmt(latency.get('tokens_in_mean'))} / {_fmt(latency.get('tokens_out_mean'))}",
        "-",
    )
    return table


def _evaluation_table(snapshot: DashboardSnapshot) -> Table:
    table = Table(title="DECISION_REFERENCE OUTCOMES", expand=True)
    table.add_column("Horizon")
    for status in ("COMPLETE", "PENDING", "DATA_UNAVAILABLE", "INELIGIBLE"):
        table.add_column(status, justify="right")
    for horizon in EVALUATION_HORIZONS:
        counts = snapshot.evaluation_horizons[horizon]
        table.add_row(
            f"{horizon // 60}m",
            *(str(counts.get(status, 0)) for status in ("COMPLETE", "PENDING", "DATA_UNAVAILABLE", "INELIGIBLE")),
        )
    table.add_row("All 4 complete", str(snapshot.fully_evaluated_decisions), "", "", "")
    table.add_row("Fully training-eligible", str(snapshot.fully_training_eligible_decisions), "", "", "")
    return table


def _outcome_table(snapshot: DashboardSnapshot) -> Table:
    table = Table(title="OUTCOME OBSERVATIONS (TRAINING-ELIGIBLE ONLY)", expand=True)
    table.add_column("Horizon")
    table.add_column("N", justify="right")
    table.add_column("Avg net pts", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("+ / - / flat", justify="right")
    for horizon in EVALUATION_HORIZONS:
        row = snapshot.outcome_performance[horizon]
        table.add_row(
            f"{horizon // 60}m",
            str(row["sample_count"]),
            _fmt(row["average_selected_action_net_points"]),
            _fmt(row["median_selected_action_net_points"]),
            f"{_fmt(row['positive_percent'], suffix='%')} / {_fmt(row['negative_percent'], suffix='%')} / {_fmt(row['flat_percent'], suffix='%')}",
        )
    return table


def _training_panel(snapshot: DashboardSnapshot) -> Panel:
    readiness = snapshot.training_readiness
    valid = int(readiness["valid_collection_decisions"])
    evaluated = int(readiness["fully_evaluated_decisions"])
    eligible = int(readiness["fully_training_eligible_decisions"])
    lines = [
        "TRAINING READINESS (informational only)",
        f"Status: {'ENOUGH_SAMPLES_FOR_REVIEW' if eligible >= readiness['review_target'] else 'COLLECTING'}",
        f"Valid collection decisions: {valid}",
        f"Fully evaluated decisions: {evaluated}",
        f"Fully training-eligible decisions: {eligible}",
        f"Pending 60m outcomes: {readiness['pending_60m']}",
        f"Data unavailable evaluations: {readiness['data_unavailable_evaluations']}",
        f"Review target 100: {_bar(eligible, readiness['review_target'])} {eligible}/100",
        f"Stronger target 300: {_bar(eligible, readiness['stronger_sample_target'])} {eligible}/300",
    ]
    return Panel("\n".join(lines), border_style="magenta")


def _recent_table(snapshot: DashboardSnapshot) -> Table:
    table = Table(title="RECENT DECISIONS (LAST 10)", expand=True)
    for column in ("UTC", "Symbol", "Action", "Ctx", "Norm", "Run(s)", "Stale", "Ref", "Delay", "LLM"):
        table.add_column(column, overflow="crop", no_wrap=True)
    for row in snapshot.recent_decisions:
        table.add_row(
            _fmt_timestamp(row["timestamp"]),
            str(row["symbol"]),
            str(row["action"] or "-"),
            str(row["context_status"]),
            str(row["normalization"]),
            _fmt(row["runtime"], suffix="s"),
            "yes" if row["stale"] else "no",
            str(row["reference_status"]),
            _fmt(row["reference_delay"], suffix="s"),
            str(row["llm_calls"]),
        )
    if not snapshot.recent_decisions:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
    return table


def render_dashboard(snapshot: DashboardSnapshot, *, warning: str | None = None) -> RenderableType:
    """Build the terminal view without performing any I/O."""
    alerts = (
        Panel("\n".join(snapshot.health_reasons), title="ALERTS", border_style="red")
        if snapshot.health_reasons
        else Panel("No operational alerts", title="ALERTS", border_style="green")
    )
    columns = Table.grid(padding=(0, 1), expand=True)
    columns.add_row(_collection_table(snapshot), _action_reliability_table(snapshot))
    return Group(
        _header(snapshot, warning),
        columns,
        _latency_table(snapshot),
        _evaluation_table(snapshot),
        _outcome_table(snapshot),
        _training_panel(snapshot),
        _recent_table(snapshot),
        alerts,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    snapshot_reader: Callable[..., DashboardSnapshot] | None = None,
    console: Console | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = build_parser().parse_args(argv)
    reader = snapshot_reader or read_dashboard_snapshot
    output = console or Console(legacy_windows=False)
    last_snapshot: DashboardSnapshot | None = None
    warning: str | None = None

    def refresh() -> None:
        nonlocal last_snapshot, warning
        try:
            last_snapshot = reader(args.db_path)
            warning = None
        except DashboardReadError as exc:
            warning = str(exc)
            if last_snapshot is None:
                raise

    if args.once:
        try:
            refresh()
        except DashboardReadError as exc:
            print(f"FOREX DASHBOARD ERROR: {exc}", file=sys.stderr)
            return 1
        output.print(render_dashboard(last_snapshot, warning=warning))
        return 0

    try:
        with Live(Panel("Waiting for a readable shadow database..."), console=output, refresh_per_second=4, screen=False) as live:
            while True:
                try:
                    refresh()
                except DashboardReadError:
                    if last_snapshot is None:
                        live.update(Text(f"FOREX DASHBOARD: {warning}", style="yellow"))
                    else:
                        live.update(render_dashboard(last_snapshot, warning=warning))
                else:
                    live.update(render_dashboard(last_snapshot, warning=warning))
                sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
