from datetime import datetime, timedelta, timezone

from tradingagents.forex.hft.models import Tick
from tradingagents.self_enhancement.evaluation import (
    cost_sensitivity,
    summarize_trades,
    walk_forward,
)
from tradingagents.self_enhancement.models import CandidateSpec, ExitPolicyConfig, StrategyVersion
from tradingagents.self_enhancement.replay import ReplayEvaluator

UTC = timezone.utc


def _ticks(count=36):
    start = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    return tuple(
        Tick("EURUSD", start + timedelta(seconds=i), 1.1 + (i % 5) * 0.00003, 1.10001 + (i % 5) * 0.00003, 0.00001, i)
        for i in range(count)
    )


def _candidate():
    parent = StrategyVersion("range_rejection", "v1", "cfg1", ExitPolicyConfig().to_dict(), "abc")
    return CandidateSpec("cand-1", parent, "range_rejection", ExitPolicyConfig(), "test")


def test_summarize_trades_is_descriptive_and_handles_long_short_metrics():
    report = summarize_trades(
        [
            {"pnl": 2.0, "direction": "LONG", "duration_seconds": 5, "mfe_points": 4, "mae_points": -1, "spread_cost_points": 1, "slippage_points": 0, "exit_reason": "TAKE_PROFIT", "session": "LONDON", "regime": "NEUTRAL"},
            {"pnl": -1.0, "direction": "SHORT", "duration_seconds": 7, "mfe_points": 1, "mae_points": -2, "spread_cost_points": 1, "slippage_points": 1, "exit_reason": "MFE_GIVEBACK", "session": "NEW_YORK", "regime": "NEUTRAL"},
        ]
    )
    assert report.trades == 2
    assert report.win_rate == 0.5
    assert report.profit_factor == 2.0
    assert report.long_trades == 1
    assert report.short_trades == 1


def test_walk_forward_keeps_development_validation_and_unseen_separate():
    report = walk_forward(ReplayEvaluator(), _ticks(), _candidate())
    assert set(report.stage_reports) == {"DEVELOPMENT", "VALIDATION", "UNSEEN_HOLDOUT"}
    assert all(item.future_leak_detected is False for item in report.stage_reports.values())


def test_cost_sensitivity_never_assumes_zero_cost_only():
    report = cost_sensitivity(ReplayEvaluator(), _ticks(), _candidate(), (0.0, 1.0))
    assert report.scenarios == (0.0, 1.0)
    assert report.metrics_by_scenario[1.0].slippage_points >= report.metrics_by_scenario[0.0].slippage_points
