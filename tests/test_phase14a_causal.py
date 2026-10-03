import sqlite3
from datetime import datetime, timedelta, timezone

from tradingagents.forex.hft.models import Tick
from tradingagents.self_enhancement import evaluation
from tradingagents.self_enhancement.causal import (
    CausalSegment,
    CausalTick,
    load_causal_tick_dataset,
)
from tradingagents.self_enhancement.models import CandidateSpec, ExitPolicyConfig, StrategyVersion
from tradingagents.self_enhancement.replay import ReplayMetrics

UTC = timezone.utc


def test_causal_dataset_sorts_persistence_order_and_splits_large_gaps(tmp_path):
    path = tmp_path / "hft.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE hft_ticks(run_id TEXT,tick_key TEXT,symbol TEXT,timestamp TEXT,bid REAL,ask REAL,features_json TEXT,executed INTEGER)")
        db.executemany("INSERT INTO hft_ticks VALUES(?,?,?,?,?,?,?,?)", [
            ("run", "late", "EURUSD", "2026-01-01T10:00:02Z", 1.1, 1.1001, "{}", 0),
            ("run", "first", "EURUSD", "2026-01-01T10:00:00Z", 1.1, 1.1001, "{}", 0),
            ("run", "gap", "EURUSD", "2026-01-01T10:10:00Z", 1.1, 1.1001, "{}", 0),
        ])

    dataset = load_causal_tick_dataset(path, gap_seconds=300)

    assert dataset.out_of_order_rows == 1
    assert len(dataset.segments) == 2
    assert [item.timestamp.second for item in dataset.segments[0].ticks] == [0, 2]
    assert all(item.source_tick_key for segment in dataset.segments for item in segment.ticks)


def test_causal_dataset_preserves_zero_spread_and_raw_source(tmp_path):
    path = tmp_path / "hft.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE hft_ticks(run_id TEXT,tick_key TEXT,symbol TEXT,timestamp TEXT,bid REAL,ask REAL,features_json TEXT,executed INTEGER)")
        db.execute("INSERT INTO hft_ticks VALUES(?,?,?,?,?,?,?,?)", ("run", "zero", "EURUSD", "2026-01-01T10:00:00Z", 1.1, 1.1, "{}", 0))
    before = path.read_bytes()

    dataset = load_causal_tick_dataset(path)

    assert dataset.zero_spread_ticks == 1
    assert path.read_bytes() == before


def test_segmented_evaluation_never_crosses_a_large_gap(monkeypatch):
    candidate = CandidateSpec(
        candidate_id="candidate",
        parent=StrategyVersion("range_rejection", "v1", "cfg1", {}, "commit"),
        strategy_id="range_rejection",
        exit_policy=ExitPolicyConfig(),
        hypothesis="test",
    )
    starts = (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC))
    segments = []
    for segment_index, start in enumerate(starts):
        ticks = tuple(
            CausalTick(
                source_run_id=f"run-{segment_index}",
                source_tick_key=f"tick-{segment_index}-{index}",
                source_row_id=index,
                tick=Tick("EURUSD", start + timedelta(seconds=index), 1.1, 1.1001, 0.00001, index),
                features={},
            )
            for index in range(16)
        )
        segments.append(CausalSegment(f"segment-{segment_index}", f"run-{segment_index}", ticks))

    observed: list[tuple[datetime, ...]] = []

    def fake_walk_forward(_evaluator, ticks, _candidate, **_kwargs):
        values = tuple(ticks)
        observed.append(tuple(item.timestamp for item in values))
        metric = ReplayMetrics(
            candidate_id="candidate",
            strategy_id="range_rejection",
            ticks_processed=len(values),
            trades=0,
            wins=0,
            losses=0,
            win_rate=None,
            expectancy=None,
            profit_factor=None,
            total_return=0.0,
            max_drawdown=0.0,
            average_holding_seconds=None,
            median_holding_seconds=None,
            mfe_points=0.0,
            mae_points=0.0,
            mfe_capture_ratio=None,
            profit_to_loss_flips=0,
            spread_cost_points=0.0,
            slippage_points=0.0,
            broker_rejections=0,
            exit_reasons={},
            long_trades=0,
            short_trades=0,
        )
        return evaluation.WalkForwardReport(
            stage_reports={"DEVELOPMENT": metric, "VALIDATION": metric, "UNSEEN_HOLDOUT": metric},
            guard_gap_seconds=1.0,
            dataset_fingerprint=f"segment-{len(observed)}",
        )

    monkeypatch.setattr(evaluation, "walk_forward", fake_walk_forward)
    evaluation.walk_forward_segments(object(), tuple(segments), candidate)

    assert len(observed) == 2
    assert observed[0][-1] < observed[1][0]
