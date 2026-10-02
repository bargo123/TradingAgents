"""Metrics, chronological walk-forward, and cost sensitivity reports."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Any

from tradingagents.forex.hft.models import Tick

from .models import CandidateSpec
from .replay import ReplayEvaluator, ReplayMetrics, chronological_splits


@dataclass(frozen=True, slots=True)
class MetricReport:
    trades: int
    wins: int
    losses: int
    win_rate: float | None
    average_win: float | None
    average_loss: float | None
    expectancy: float | None
    profit_factor: float | None
    max_drawdown: float
    total_return: float
    average_holding_seconds: float | None
    median_holding_seconds: float | None
    mfe_points: float
    mae_points: float
    mfe_capture_ratio: float | None
    profit_to_loss_flips: int
    spread_cost_points: float
    slippage_points: float
    long_trades: int
    short_trades: int
    exit_reason_distribution: dict[str, int]
    session_metrics: dict[str, int]
    regime_metrics: dict[str, int]
    trades_per_hour: float | None = None
    sharpe_like: float | None = None
    commission_known: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    stage_reports: dict[str, ReplayMetrics]
    guard_gap_seconds: float
    dataset_fingerprint: str
    window_reports: tuple[dict[str, ReplayMetrics], ...] = ()


@dataclass(frozen=True, slots=True)
class CostSensitivityReport:
    scenarios: tuple[float, ...]
    metrics_by_scenario: dict[float, ReplayMetrics]


def summarize_trades(trades: Iterable[dict[str, Any]]) -> MetricReport:
    rows = tuple(dict(row) for row in trades)
    pnls = [float(row["pnl"]) for row in rows]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in pnls:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    durations = [float(row["duration_seconds"]) for row in rows]
    mfe = [float(row["mfe_points"]) for row in rows]
    timestamps = []
    for row in rows:
        try:
            timestamps.append(datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00")))
        except (KeyError, TypeError, ValueError):
            timestamps = []
            break
    standard_deviation = math.sqrt(sum((value - (sum(pnls) / len(pnls))) ** 2 for value in pnls) / len(pnls)) if pnls else 0.0
    return MetricReport(
        trades=len(rows),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(rows) if rows else None,
        average_win=sum(wins) / len(wins) if wins else None,
        average_loss=sum(losses) / len(losses) if losses else None,
        expectancy=sum(pnls) / len(rows) if rows else None,
        profit_factor=sum(wins) / abs(sum(losses)) if losses else None,
        max_drawdown=max_drawdown,
        total_return=sum(pnls),
        average_holding_seconds=sum(durations) / len(durations) if durations else None,
        median_holding_seconds=median(durations) if durations else None,
        mfe_points=sum(mfe) / len(mfe) if mfe else 0.0,
        mae_points=sum(float(row["mae_points"]) for row in rows) / len(rows) if rows else 0.0,
        mfe_capture_ratio=(sum(pnls) / sum(mfe) if mfe and sum(mfe) > 0 else None),
        profit_to_loss_flips=sum(1 for left, right in zip(pnls, pnls[1:], strict=False) if left > 0 > right),
        spread_cost_points=sum(float(row.get("spread_cost_points", 0.0)) for row in rows),
        slippage_points=sum(float(row.get("slippage_points", 0.0)) for row in rows),
        long_trades=sum(str(row.get("direction", "")).upper() == "LONG" for row in rows),
        short_trades=sum(str(row.get("direction", "")).upper() == "SHORT" for row in rows),
        exit_reason_distribution=_counts(rows, "exit_reason"),
        session_metrics=_counts(rows, "session"),
        regime_metrics=_counts(rows, "regime"),
        trades_per_hour=(len(rows) * 3600.0 / (timestamps[-1] - timestamps[0]).total_seconds()) if len(timestamps) > 1 and (timestamps[-1] - timestamps[0]).total_seconds() > 0 else None,
        sharpe_like=((sum(pnls) / len(pnls)) / standard_deviation * math.sqrt(len(pnls))) if len(pnls) > 1 and standard_deviation > 0 else None,
        commission_known=all(bool(row.get("commission_known", False)) for row in rows),
    )


def _counts(rows: tuple[dict[str, Any], ...], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "UNKNOWN"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def walk_forward(
    evaluator: ReplayEvaluator,
    ticks: Iterable[Tick],
    candidate: CandidateSpec,
    *,
    guard_gap_seconds: float = 1.0,
    windows: int = 1,
) -> WalkForwardReport:
    if isinstance(windows, bool) or not isinstance(windows, int) or windows <= 0:
        raise ValueError("windows must be a positive integer")
    values = tuple(ticks)
    if windows == 1:
        split_values = (chronological_splits(values, guard_gap_seconds=guard_gap_seconds),)
    else:
        if len(values) < windows * 8:
            raise ValueError("rolling walk-forward requires at least eight ticks per window")
        split_values = tuple(
            chronological_splits(
                values[int(index * len(values) / windows) : int((index + 1) * len(values) / windows)],
                guard_gap_seconds=guard_gap_seconds,
            )
            for index in range(windows)
        )
    window_reports = tuple(
        {
            "DEVELOPMENT": evaluator.evaluate(split.development, candidate, split="DEVELOPMENT"),
            "VALIDATION": evaluator.evaluate(split.validation, candidate, split="VALIDATION"),
            "UNSEEN_HOLDOUT": evaluator.evaluate(split.unseen, candidate, split="UNSEEN_HOLDOUT"),
        }
        for split in split_values
    )
    stage_reports = {
        stage: _aggregate_replay_metrics(tuple(window[stage] for window in window_reports))
        for stage in ("DEVELOPMENT", "VALIDATION", "UNSEEN_HOLDOUT")
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            [(tick.timestamp.isoformat(), tick.bid, tick.ask) for tick in values],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return WalkForwardReport(stage_reports, float(guard_gap_seconds), fingerprint, window_reports)


def _aggregate_replay_metrics(reports: tuple[ReplayMetrics, ...]) -> ReplayMetrics:
    """Aggregate independent chronological windows without mixing observations."""

    if not reports:
        raise ValueError("at least one replay report is required")
    trades = sum(report.trades for report in reports)
    wins = sum(report.wins for report in reports)
    losses = sum(report.losses for report in reports)
    durations = [
        (report.average_holding_seconds, report.trades)
        for report in reports
        if report.average_holding_seconds is not None and report.trades
    ]
    mfe_weight = sum(report.mfe_points * report.trades for report in reports)
    mae_weight = sum(report.mae_points * report.trades for report in reports)
    capture_values = [report.mfe_capture_ratio for report in reports if report.mfe_capture_ratio is not None]
    exit_reasons: dict[str, int] = {}
    for report in reports:
        for reason, count in report.exit_reasons.items():
            exit_reasons[reason] = exit_reasons.get(reason, 0) + int(count)
    return ReplayMetrics(
        candidate_id=reports[0].candidate_id,
        strategy_id=reports[0].strategy_id,
        ticks_processed=sum(report.ticks_processed for report in reports),
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=wins / trades if trades else None,
        expectancy=(sum((report.expectancy or 0.0) * report.trades for report in reports) / trades) if trades else None,
        # Per-window gross win/loss totals are not retained by ReplayMetrics;
        # use the conservative minimum rather than inventing a pooled value.
        profit_factor=min(
            (report.profit_factor for report in reports if report.profit_factor is not None),
            default=None,
        ),
        total_return=sum(report.total_return for report in reports),
        max_drawdown=max(report.max_drawdown for report in reports),
        average_holding_seconds=(sum(value * count for value, count in durations) / sum(count for _, count in durations)) if durations else None,
        median_holding_seconds=sorted(
            value for report in reports if (value := report.median_holding_seconds) is not None
        )[len([report for report in reports if report.median_holding_seconds is not None]) // 2]
        if any(report.median_holding_seconds is not None for report in reports)
        else None,
        mfe_points=mfe_weight / trades if trades else 0.0,
        mae_points=mae_weight / trades if trades else 0.0,
        mfe_capture_ratio=sum(capture_values) / len(capture_values) if capture_values else None,
        profit_to_loss_flips=sum(report.profit_to_loss_flips for report in reports),
        spread_cost_points=sum(report.spread_cost_points for report in reports),
        slippage_points=sum(report.slippage_points for report in reports),
        broker_rejections=sum(report.broker_rejections for report in reports),
        exit_reasons=exit_reasons,
        long_trades=sum(report.long_trades for report in reports),
        short_trades=sum(report.short_trades for report in reports),
        future_leak_detected=any(report.future_leak_detected for report in reports),
        trades_per_hour=(sum(report.trades_per_hour or 0.0 for report in reports) / len(reports)),
        sharpe_like=(sum(report.sharpe_like or 0.0 for report in reports) / len(reports)) if any(report.sharpe_like is not None for report in reports) else None,
        commission_known=all(report.commission_known for report in reports),
    )


def cost_sensitivity(
    evaluator: ReplayEvaluator,
    ticks: Iterable[Tick],
    candidate: CandidateSpec,
    scenarios: Iterable[float],
) -> CostSensitivityReport:
    values = tuple(float(item) for item in scenarios)
    if not values or any(not math.isfinite(item) or item < 0 for item in values):
        raise ValueError("cost scenarios must be finite and non-negative")
    values_ticks = tuple(ticks)
    reports = {
        scenario: ReplayEvaluator(slippage_points=scenario, latency_ms=evaluator.latency_ms).evaluate(values_ticks, candidate)
        for scenario in values
    }
    return CostSensitivityReport(values, reports)


__all__ = ["CostSensitivityReport", "MetricReport", "WalkForwardReport", "cost_sensitivity", "summarize_trades", "walk_forward"]
