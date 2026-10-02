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
