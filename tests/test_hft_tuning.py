from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.forex.hft.models import Tick
from tradingagents.self_enhancement.causal import CausalSegment, CausalTick


def _segment(day, name):
    start = datetime(2026, 1, day, 10, tzinfo=timezone.utc)
    ticks = tuple(
        CausalTick(name, str(i), i, Tick("EURUSD", start + timedelta(seconds=i),
                                      1.1, 1.10002, 0.00001, i), {})
        for i in range(30)
    )
    return CausalSegment(name, name, ticks)


def test_tuning_partitions_are_globally_chronological_across_runtime_segments():
    from tradingagents.self_enhancement.tuning import partition_segments
    stages = partition_segments((_segment(3, "a"), _segment(1, "z"), _segment(2, "m")), guard_seconds=1)
    times = {stage: [item.timestamp for segment in segments for item in segment.ticks]
             for stage, segments in stages.items()}
    assert max(times["DEVELOPMENT"]) < min(times["VALIDATION"])
    assert max(times["VALIDATION"]) < min(times["UNSEEN_HOLDOUT"])
    assert min(times["VALIDATION"]) - max(times["DEVELOPMENT"]) >= timedelta(seconds=2)
    assert times["DEVELOPMENT"][0].day == 1
    assert times["UNSEEN_HOLDOUT"][-1].day == 3


def test_tuning_selection_accounts_for_boundary_losses_and_requires_trades():
    from tradingagents.self_enhancement.tuning import select_margin
    reports = {
        1.0: {"trades": 20, "expectancy": 0.00001, "open_positions": 1, "unrealized_pnl": -0.01},
        3.0: {"trades": 20, "expectancy": 0.000005, "open_positions": 0, "unrealized_pnl": 0.0},
        8.0: {"trades": 0, "expectancy": None, "open_positions": 0, "unrealized_pnl": 0.0},
    }
    assert select_margin(reports, minimum_trades=20) == 3.0


def test_tuning_does_not_select_an_empty_or_insufficient_sample():
    from tradingagents.self_enhancement.tuning import select_margin
    assert select_margin({1.0: {"trades": 2, "expectancy": 0.1}}, minimum_trades=20) is None


def test_tuning_rejects_empty_chronological_partitions():
    from tradingagents.self_enhancement.tuning import partition_segments
    with pytest.raises(ValueError):
        partition_segments((_segment(1, "a"),), guard_seconds=60)


@pytest.mark.parametrize("zero_spread", [False, True])
def test_tuning_runs_on_a_frozen_database_without_mutating_the_source(tmp_path, zero_spread):
    import hashlib
    import json
    import sqlite3

    from tradingagents.self_enhancement.tuning import run_tuning

    source = tmp_path / "capture.sqlite3"
    start = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    mids = (1.1010, 1.1005, 1.1000, 1.1001, 1.1004, 1.1005, 1.1003)
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE hft_ticks(run_id,tick_key,symbol,timestamp,bid,ask,features_json)")
        db.executemany("INSERT INTO hft_ticks VALUES(?,?,?,?,?,?,?)", [
            ("run", str(i), "EURUSD", (start + timedelta(seconds=10*i)).isoformat(),
             mids[i % 7] - 0.00001, mids[i % 7] + (-0.00001 if zero_spread else 0.00001), json.dumps({"point": 0.00001}))
            for i in range(280)
        ])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    report = run_tuning(source, minimum_trades=1)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert report["source_sha256"] == before
    assert set(report["families"]) == {"range_rejection", "momentum_continuation"}
    assert report["assumptions"]["live_configuration_applied"] is False
    assert report["real_money"] is False
    assert report["data_quality_eligible"] is (not zero_spread)
    if zero_spread:
        assert "ZERO_SPREAD_PROVENANCE_UNVERIFIED" in report["quality_reasons"]
        assert all(family["status"] != "RESEARCH_VALIDATED" for family in report["families"].values())
    for family in report["families"].values():
        assert set(family["development"]) == {1.0, 3.0, 5.0, 8.0}
        if "later_period_checks" in family:
            assert set(family["later_period_checks"]) == {"VALIDATION", "UNSEEN_HOLDOUT"}
