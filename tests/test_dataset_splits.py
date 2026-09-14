from datetime import datetime, timezone

import pytest

from tradingagents.datasets.models import CanonicalExampleV1
from tradingagents.datasets.splits import (
    assign_splits,
    validate_split_assignments,
)

UTC = timezone.utc


def example(i, stamp, *, run=None, decision=None, basis="ANALYSIS_SNAPSHOT", horizon=300):
    return CanonicalExampleV1(
        example_id=f"e{i}",
        decision={
            "decision_id": decision or f"d{i}",
            "source_run_id": run or f"r{i}",
            "analysis_snapshot_timestamp": datetime(2026, 1, 1, 0, stamp, tzinfo=UTC),
        },
        outcome={"evaluation_basis": basis, "horizon_seconds": horizon},
    )


def test_assigns_chronological_grouped_largest_remainder_boundaries():
    rows = [example(i, i, run=f"r{i}") for i in range(10)]
    result = assign_splits(rows)
    assert result.status == "COMPLETE"
    assert [a.split for a in result.assignments] == ["train"] * 7 + ["validation"] * 2 + ["test"]
    validate_split_assignments(result, rows)


def test_input_order_and_timestamp_ties_are_deterministic():
    rows = [example(2, 1, run="r2", decision="d2"), example(1, 1, run="r1", decision="d1"), example(3, 2, run="r3")]
    a = assign_splits(rows)
    b = assign_splits(list(reversed(rows)))
    assert a == b


def test_same_run_and_same_decision_horizons_never_cross_splits():
    rows = [example(1, 1, run="run", decision="decision", horizon=300), example(2, 2, run="run", decision="decision", horizon=900)]
    rows += [example(i, i + 2, run=f"r{i}") for i in range(3, 9)]
    result = assign_splits(rows)
    by_group = {}
    for assignment in result.assignments:
        by_group.setdefault(assignment.group_id, set()).add(assignment.split)
    assert by_group["run:run"] == {"train"}


def test_zero_and_small_populations_are_explicitly_insufficient():
    assert assign_splits([]).status == "INSUFFICIENT_DATA"
    assert assign_splits([example(1, 1)]).status == "INSUFFICIENT_DATA"
    assert assign_splits([example(i, i) for i in range(2)]).status == "INSUFFICIENT_DATA"


def test_validator_rejects_cross_split_groups_and_duplicate_keys():
    rows = [example(1, 1, run="same"), example(2, 2, run="same", horizon=900)]
    rows += [example(i, i, run=f"r{i}") for i in range(3, 7)]
    result = assign_splits(rows)
    bad = list(result.assignments)
    bad[1] = type(bad[1])(bad[1].example_id, "test", bad[1].group_id)
    with pytest.raises(ValueError, match="cross-split"):
        validate_split_assignments(bad, rows)
    with pytest.raises(ValueError, match="duplicate"):
        validate_split_assignments(list(result.assignments) + [result.assignments[0]], rows)


def test_validator_rejects_malformed_status():
    result = assign_splits([example(i, i) for i in range(6)])
    with pytest.raises(ValueError, match="status"):
        validate_split_assignments(result.assignments, [example(i, i) for i in range(6)], status="BOGUS")
