from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.forex.hold_audit import (
    CATEGORY_BUY_BETTER,
    CATEGORY_DIRECTIONAL_TIE,
    CATEGORY_HOLD_BEST,
    CATEGORY_SELL_BETTER,
    audit_hold_outcomes,
    classify_hold_outcome,
    percentile,
)

UTC = timezone.utc
HORIZONS = (300, 900, 1800, 3600)


def _ts(hours: float = 0.0) -> str:
    return (datetime.now(UTC) + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def _init_db(tmp_path: Path) -> Path:
    path = tmp_path / "hold-audit.db"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE shadow_decisions (
                decision_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                decision_completed_timestamp TEXT,
                resolved_symbol TEXT NOT NULL,
                action TEXT
            );
            CREATE TABLE shadow_decision_evaluations (
                decision_id TEXT NOT NULL,
                evaluation_basis TEXT NOT NULL,
                horizon_seconds INTEGER NOT NULL,
                resolved_symbol TEXT NOT NULL,
                source_context_eligible INTEGER NOT NULL,
                training_eligible INTEGER,
                evaluation_status TEXT NOT NULL,
                unavailable_reason TEXT,
                selected_action TEXT,
                buy_net_points REAL,
                sell_net_points REAL,
                best_counterfactual_action TEXT,
                hold_opportunity_cost_points REAL,
                created_at TEXT NOT NULL
            );
            """
        )
    return path


def _decision(path: Path, decision_id: str, *, hours: float = 0.0, action: str = "HOLD") -> None:
    timestamp = _ts(hours)
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO shadow_decisions VALUES (?, ?, ?, ?, ?)",
            (decision_id, timestamp, timestamp, "EURUSD", action),
        )


def _evaluation(
    path: Path,
    decision_id: str,
    horizon: int,
    *,
    buy: float = -2.0,
    sell: float = -1.0,
    status: str = "COMPLETE",
    source_context_eligible: bool = True,
    action: str | None = "HOLD",
    opportunity_cost: float | None = None,
    best_counterfactual_action: str | None = "SELL",
    unavailable_reason: str | None = None,
) -> None:
    if opportunity_cost is None:
        opportunity_cost = max(0.0, buy, sell)
    with sqlite3.connect(path) as db:
        db.execute(
            """
            INSERT INTO shadow_decision_evaluations (
                decision_id, evaluation_basis, horizon_seconds, resolved_symbol,
                source_context_eligible, training_eligible, evaluation_status,
                unavailable_reason, selected_action, buy_net_points, sell_net_points,
                best_counterfactual_action, hold_opportunity_cost_points, created_at
            ) VALUES (?, 'DECISION_REFERENCE', ?, 'EURUSD', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                horizon,
                int(source_context_eligible),
                status,
                unavailable_reason,
                action,
                buy,
                sell,
                best_counterfactual_action,
                opportunity_cost,
                _ts(),
            ),
        )


def _full_hold(
    path: Path,
    decision_id: str,
    opportunity_costs: tuple[float, float, float, float],
    *,
    buys: tuple[float, float, float, float] | None = None,
    sells: tuple[float, float, float, float] | None = None,
    hours: float = 0.0,
) -> None:
    _decision(path, decision_id, hours=hours)
    buys = buys or tuple(-value if value == 0 else value for value in (-2.0, -2.0, -2.0, -2.0))
    sells = sells or (-1.0, -1.0, -1.0, -1.0)
    for horizon, opportunity_cost, buy, sell in zip(
        HORIZONS, opportunity_costs, buys, sells, strict=True
    ):
        _evaluation(
            path,
            decision_id,
            horizon,
            buy=buy,
            sell=sell,
            opportunity_cost=opportunity_cost,
            best_counterfactual_action="BUY" if buy > sell else "SELL",
        )


def test_hold_classification_categories() -> None:
    assert classify_hold_outcome(-2, -1, 0) == CATEGORY_HOLD_BEST
    assert classify_hold_outcome(5, -3, 5) == CATEGORY_BUY_BETTER
    assert classify_hold_outcome(-2, 7, 7) == CATEGORY_SELL_BETTER
    assert classify_hold_outcome(4, 4, 4) == CATEGORY_DIRECTIONAL_TIE


def test_negative_counterfactuals_are_hold_best_even_if_best_action_says_sell() -> None:
    assert classify_hold_outcome(-2, -1, 0, best_counterfactual_action="SELL") == CATEGORY_HOLD_BEST


def test_threshold_counts_use_strictly_greater_than_points(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    for index, value in enumerate((0, 1, 2, 5, 10, 20, 50, 50.1), start=1):
        decision_id = f"threshold-{index}"
        _decision(path, decision_id)
        _evaluation(path, decision_id, 300, buy=value, sell=-1, opportunity_cost=value)

    report = audit_hold_outcomes(path)
    thresholds = report.all_population.horizons[300].thresholds

    assert [thresholds[value]["count"] for value in (0, 1, 2, 5, 10, 20, 50)] == [7, 6, 5, 4, 3, 2, 1]


def test_percentiles_are_deterministic_nearest_rank() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 75) == 3
    assert percentile([1, 2, 3, 4], 90) == 4
    assert percentile([1, 2, 3, 4], 95) == 4


@pytest.mark.parametrize(
    ("opportunity_costs", "expected"),
    [
        ((0, 0, 0, 0), "CONSISTENT_HOLD"),
        ((1, 0, 0, 0), "TRANSIENT_MISS"),
        ((1, 1, 1, 0), "PERSISTENT_DIRECTIONAL_MISS"),
        ((5, 5, 5, 0), "STRONG_PERSISTENT_MISS"),
    ],
)
def test_cross_horizon_classification(tmp_path: Path, opportunity_costs, expected: str) -> None:
    path = _init_db(tmp_path)
    _full_hold(path, "cross", opportunity_costs)

    report = audit_hold_outcomes(path, strong_miss_points=5)

    assert report.all_population.cross_horizon.classification_counts[expected] == 1


def test_direction_consistency_and_mixed_direction(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _full_hold(
        path,
        "buy-consistent",
        (4, 4, 4, 0),
        buys=(4, 4, 4, -1),
        sells=(-1, -1, -1, -1),
    )
    _full_hold(
        path,
        "sell-consistent",
        (4, 4, 4, 0),
        buys=(-1, -1, -1, -1),
        sells=(4, 4, 4, -1),
    )
    _full_hold(
        path,
        "mixed",
        (4, 4, 0, 0),
        buys=(4, -1, -1, -1),
        sells=(-1, 4, -1, -1),
    )

    directions = audit_hold_outcomes(path).all_population.cross_horizon.direction_counts

    assert directions["CONSISTENT_BUY_BETTER"] == 1
    assert directions["CONSISTENT_SELL_BETTER"] == 1
    assert directions["MIXED_DIRECTION"] == 1


def test_pending_horizon_excludes_decision_from_cross_horizon_audit(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _decision(path, "pending")
    for horizon in HORIZONS:
        _evaluation(path, "pending", horizon, status="PENDING" if horizon == 3600 else "COMPLETE", action=None if horizon == 3600 else "HOLD")

    report = audit_hold_outcomes(path)

    assert report.all_population.cross_horizon.fully_evaluated_hold_decisions == 0


def test_source_context_ineligible_is_excluded_and_training_null_is_allowed(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _full_hold(path, "eligible", (0, 0, 0, 0))
    _decision(path, "ineligible")
    for horizon in HORIZONS:
        _evaluation(path, "ineligible", horizon, source_context_eligible=False)

    report = audit_hold_outcomes(path)

    assert report.fully_evaluated_source_eligible_decisions == 1
    assert report.fully_evaluated_source_eligible_hold_decisions == 1


def test_data_unavailable_is_counted_but_not_a_hold_outcome(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _decision(path, "unavailable")
    _evaluation(path, "unavailable", 300, status="DATA_UNAVAILABLE", action=None, unavailable_reason="NO_TICK")
    report = audit_hold_outcomes(path)

    assert report.data_quality["DATA_UNAVAILABLE"] == 1
    assert report.data_quality_unavailable_reasons == {"NO_TICK": 1}
    assert report.all_population.horizons[300].complete_hold_samples == 0


def test_top_missed_opportunity_uses_largest_horizon_and_direction(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _full_hold(path, "decision-with-long-id", (1, 2, 10, 3), buys=(-1, -1, 10, -1), sells=(-1, -1, -1, -1))

    report = audit_hold_outcomes(path)
    top = report.top_missed_opportunities[0]

    assert top["decision_id_short"] == "decision"
    assert top["largest_missed_direction"] == "BUY"
    assert top["max_missed_points"] == 10
    assert top["30m_opportunity_cost_points"] == 10


def test_recent_population_is_utc_bounded(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _full_hold(path, "recent", (1, 0, 0, 0), hours=-1)
    _full_hold(path, "old", (1, 0, 0, 0), hours=-48)

    report = audit_hold_outcomes(path, recent_hours=24)

    assert report.all_population.horizons[300].complete_hold_samples == 2
    assert report.recent_population.horizons[300].complete_hold_samples == 1


def test_read_is_strictly_read_only_and_empty_dataset_is_safe(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    before = path.read_bytes()

    report = audit_hold_outcomes(path)

    assert path.read_bytes() == before
    assert report.all_population.horizons[300].complete_hold_samples == 0
    assert report.review_status == "OBSERVATION ONLY"


def test_json_cli_output_is_machine_readable(tmp_path: Path, capsys) -> None:
    path = _init_db(tmp_path)
    from cli.forex_hold_audit import main

    assert main(["--db-path", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["review_status"] == "OBSERVATION ONLY"
    assert payload["data_quality"]["COMPLETE"] == 0


def test_read_succeeds_with_concurrent_writer_transaction(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    writer = sqlite3.connect(path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO shadow_decisions VALUES (?, ?, ?, ?, ?)",
            ("writer", _ts(), _ts(), "EURUSD", "HOLD"),
        )
        # The reader sees the last committed snapshot and never waits on or mutates the writer.
        report = audit_hold_outcomes(path)
        assert report.review_status == "OBSERVATION ONLY"
    finally:
        writer.rollback()
        writer.close()
