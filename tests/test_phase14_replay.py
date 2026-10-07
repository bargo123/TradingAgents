from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.forex.hft.models import Tick
from tradingagents.self_enhancement.evaluation import walk_forward
from tradingagents.self_enhancement.models import CandidateSpec, ExitPolicyConfig, StrategyVersion
from tradingagents.self_enhancement.replay import ReplayError, ReplayEvaluator, chronological_splits

UTC = timezone.utc


def _ticks(count=32):
    start = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    values = []
    for index in range(count):
        phase = index % 8
        mid = 1.1 + (phase * 0.00003 if phase < 4 else (7 - phase) * 0.00003)
        values.append(Tick("EURUSD", start + timedelta(seconds=index), mid - 0.00001, mid + 0.00001, 0.00001, index))
    return tuple(values)


def _candidate():
    parent = StrategyVersion("range_rejection", "v1", "cfg1", ExitPolicyConfig().to_dict(), "abc")
    return CandidateSpec("cand-1", parent, "range_rejection", ExitPolicyConfig(), "test")


def test_chronological_splits_have_guarded_ordered_boundaries():
    splits = chronological_splits(_ticks(), guard_gap_seconds=1)
    assert splits.development
    assert splits.validation
    assert splits.unseen
    assert max(item.timestamp for item in splits.development) < min(item.timestamp for item in splits.validation)
    assert max(item.timestamp for item in splits.validation) < min(item.timestamp for item in splits.unseen)


def test_replay_evaluator_is_causal_and_reports_metrics():
    report = ReplayEvaluator().evaluate(_ticks(), _candidate())
    assert report.future_leak_detected is False
    assert report.ticks_processed == 32
    assert report.real_money is False
    assert report.execution_mode == "REPLAY"


def test_replay_rejects_non_monotonic_or_future_ticks():
    ticks = list(_ticks())
    ticks[4] = ticks[2]
    with pytest.raises(ReplayError, match="monotonic"):
        ReplayEvaluator().evaluate(ticks, _candidate())
    future = datetime.now(UTC) + timedelta(minutes=1)
    with pytest.raises(ReplayError, match="future"):
        ReplayEvaluator().evaluate([Tick("EURUSD", future, 1.1, 1.1001)], _candidate())


def test_walk_forward_fingerprint_is_stable_and_direction_fields_are_not_pnl_labels():
    first = walk_forward(ReplayEvaluator(), _ticks(), _candidate())
    second = walk_forward(ReplayEvaluator(), _ticks(), _candidate())
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert len(first.dataset_fingerprint) == 64
    for report in first.stage_reports.values():
        assert report.long_trades >= 0
        assert report.short_trades >= 0


def test_walk_forward_supports_multiple_rolling_windows():
    report = walk_forward(ReplayEvaluator(), _ticks(80), _candidate(), windows=2)
    assert len(report.window_reports) == 2
    assert set(report.stage_reports) == {"DEVELOPMENT", "VALIDATION", "UNSEEN_HOLDOUT"}


def _range_ticks(mids):
    start = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    return tuple(
        Tick("EURUSD", start + timedelta(seconds=index), mid - 0.00001,
             mid + 0.00001, 0.00001, index)
        for index, mid in enumerate(mids)
    )


def test_mfe_capture_uses_matching_price_units():
    # Entry ask 1.10011; best exit bid 1.10049; actual exit bid 1.10029.
    # The captured move is 18 / 38 points, independent of quote scale.
    ticks = _range_ticks((1.1010, 1.1005, 1.1000, 1.1001, 1.1004, 1.1005, 1.1003))
    report = ReplayEvaluator().evaluate(ticks, _candidate())
    assert report.trades == 1
    assert report.mfe_capture_ratio == pytest.approx(18 / 38)


def test_flip_counts_a_profitable_excursion_inside_the_losing_trade():
    ticks = _range_ticks((1.1010, 1.1005, 1.1000, 1.1001, 1.1003, 1.0999))
    report = ReplayEvaluator().evaluate(ticks, _candidate())
    assert report.trades == 1
    assert report.losses == 1
    assert report.mfe_points > 0
    assert report.profit_to_loss_flips == 1


def test_winning_trade_followed_by_unprofitable_trade_is_not_a_flip():
    ticks = _range_ticks((1.1010, 1.1005, 1.1000, 1.1001, 1.1004, 1.1005,
                          1.1003, 1.0990, 1.0991, 1.0989, 1.0988))
    report = ReplayEvaluator().evaluate(ticks, _candidate())
    assert report.trades == 2
    assert report.wins == 1
    assert report.losses == 1
    assert report.profit_to_loss_flips == 0


def test_incumbent_replay_keeps_strategy_specific_live_no_progress_profile():
    # Current range engine caps no-progress at 10 seconds; the generic
    # ExitPolicyConfig default of 30 must not overwrite an unchanged control.
    ticks = _range_ticks((1.1010, 1.1005, 1.1000, 1.1001) + (1.1001,) * 12)
    report = ReplayEvaluator().evaluate(ticks, _candidate())
    assert report.trades == 1
    assert report.exit_reasons == {"NO_PROGRESS": 1}


def test_replay_stricter_cost_margin_blocks_insufficient_edge():
    ticks = _range_ticks((1.1010, 1.1005, 1.1000, 1.1001, 1.1004, 1.1005, 1.1003))
    assert ReplayEvaluator().evaluate(ticks, _candidate()).trades == 1
    assert ReplayEvaluator(cost_safety_margin_points=50).evaluate(ticks, _candidate()).trades == 0


@pytest.mark.parametrize("margin", [0, -1, float("nan"), float("inf")])
def test_replay_cost_margin_cannot_weaken_the_existing_cost_gate(margin):
    with pytest.raises(ValueError):
        ReplayEvaluator(cost_safety_margin_points=margin)


def test_cost_sensitivity_preserves_the_selected_entry_filter():
    from tradingagents.self_enhancement.evaluation import cost_sensitivity
    ticks = _range_ticks((1.1010, 1.1005, 1.1000, 1.1001, 1.1004, 1.1005, 1.1003))
    report = cost_sensitivity(ReplayEvaluator(cost_safety_margin_points=50), ticks, _candidate(), (0, 1))
    assert all(metrics.trades == 0 for metrics in report.metrics_by_scenario.values())


def test_replay_reports_boundary_position_instead_of_hiding_unrealized_pnl():
    ticks = _range_ticks((1.1010, 1.1005, 1.1000, 1.1001, 1.1004, 1.1005))
    report = ReplayEvaluator().evaluate(ticks, _candidate()).to_dict()
    assert report["trades"] == 0
    assert report.get("open_positions") == 1
    # Still open: entry 1.10011, terminal liquidation bid 1.10049.
    assert report.get("unrealized_pnl") == pytest.approx(0.00038)
