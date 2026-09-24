from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.forex.decision_path_audit import (
    TRADER_ACTION_UNAVAILABLE,
    DecisionPathAuditSchemaError,
    audit_decision_path,
    extract_trader_action,
)

UTC = timezone.utc
HORIZONS = (300, 900, 1800, 3600)


def _ts(hours: float = 0.0) -> str:
    return (datetime.now(UTC) + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def _init_db(tmp_path: Path) -> Path:
    path = tmp_path / "decision-path.db"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE shadow_decisions (
                decision_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                decision_completed_timestamp TEXT,
                resolved_symbol TEXT NOT NULL,
                action TEXT,
                decision_context_status TEXT NOT NULL,
                normalization_status TEXT NOT NULL,
                decision_reference_status TEXT NOT NULL,
                raw_portfolio_manager_result TEXT,
                trader_summary TEXT,
                research_manager_recommendation TEXT,
                training_eligible INTEGER
            );
            CREATE TABLE forex_watch_runs (
                run_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL,
                run_status TEXT NOT NULL,
                stale_by_completion INTEGER
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
                hold_opportunity_cost_points REAL,
                created_at TEXT NOT NULL
            );
            """
        )
    return path


def _decision(
    path: Path,
    decision_id: str,
    *,
    trader_action: str | None = "HOLD",
    final_action: str = "HOLD",
    hours: float = 0.0,
    training_eligible: int | None = None,
    raw_pm: str | None = None,
    research_recommendation: str | None = None,
    context_status: str = "COMPLETE",
    normalization_status: str = "NORMALIZED",
    reference_status: str = "AVAILABLE",
    run_status: str = "SUCCEEDED",
    stale_by_completion: int = 0,
) -> None:
    timestamp = _ts(hours)
    marker = (
        f"Report\nFINAL TRANSACTION PROPOSAL: **{trader_action}**\n"
        if trader_action is not None
        else "Report without a bounded proposal marker"
    )
    with sqlite3.connect(path) as db:
        db.execute(
            """
            INSERT INTO shadow_decisions (
                decision_id, created_at, decision_completed_timestamp, resolved_symbol, action,
                decision_context_status, normalization_status, decision_reference_status,
                raw_portfolio_manager_result, trader_summary, research_manager_recommendation,
                training_eligible
            ) VALUES (?, ?, ?, 'EURUSD', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                timestamp,
                timestamp,
                final_action,
                context_status,
                normalization_status,
                reference_status,
                raw_pm,
                marker,
                research_recommendation,
                training_eligible,
            ),
        )
        db.execute(
            "INSERT INTO forex_watch_runs VALUES (?, ?, ?, ?)",
            (f"run-{decision_id}", decision_id, run_status, stale_by_completion),
        )


def _evaluation(
    path: Path,
    decision_id: str,
    horizon: int,
    *,
    buy: float = -2.0,
    sell: float = -1.0,
    opportunity_cost: float | None = None,
    status: str = "COMPLETE",
    source_context_eligible: bool = True,
    selected_action: str | None = "HOLD",
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
                hold_opportunity_cost_points, created_at
            ) VALUES (?, 'DECISION_REFERENCE', ?, 'EURUSD', ?, NULL, ?, NULL, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                horizon,
                int(source_context_eligible),
                status,
                selected_action,
                buy,
                sell,
                opportunity_cost,
                _ts(),
            ),
        )


def _full_eval(
    path: Path,
    decision_id: str,
    values: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    *,
    buy: tuple[float, float, float, float] | None = None,
    sell: tuple[float, float, float, float] | None = None,
) -> None:
    buys = buy or (-2.0, -2.0, -2.0, -2.0)
    sells = sell or (-1.0, -1.0, -1.0, -1.0)
    for horizon, opportunity, buy_points, sell_points in zip(
        HORIZONS, values, buys, sells, strict=True
    ):
        _evaluation(
            path,
            decision_id,
            horizon,
            buy=buy_points,
            sell=sell_points,
            opportunity_cost=opportunity,
        )


def test_exact_trader_markers_are_the_only_actions() -> None:
    assert extract_trader_action("FINAL TRANSACTION PROPOSAL: **BUY**") == "BUY"
    assert extract_trader_action("FINAL TRANSACTION PROPOSAL: **HOLD**") == "HOLD"
    assert extract_trader_action("FINAL TRANSACTION PROPOSAL: **SELL**") == "SELL"
    assert extract_trader_action("The trader says buy if momentum improves") == TRADER_ACTION_UNAVAILABLE
    assert extract_trader_action("No recommendation") == TRADER_ACTION_UNAVAILABLE


def test_transition_matrix_records_buy_sell_hold_paths_and_unavailable(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _decision(path, "buy-hold", trader_action="BUY")
    _decision(path, "sell-hold", trader_action="SELL")
    _decision(path, "hold-hold", trader_action="HOLD")
    _decision(path, "unavailable", trader_action=None)

    report = audit_decision_path(path)
    matrix = report.all_population.transition_matrix

    assert matrix["BUY->HOLD"]["count"] == 1
    assert matrix["SELL->HOLD"]["count"] == 1
    assert matrix["HOLD->HOLD"]["count"] == 1
    assert matrix[f"{TRADER_ACTION_UNAVAILABLE}->HOLD"]["count"] == 1
    assert matrix["BUY->BUY"]["count"] == 0
    assert sum(row["count"] for row in matrix.values()) == 4


def test_hold_origins_and_outcomes_are_broken_down_by_transition(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _decision(path, "buy-hold", trader_action="BUY")
    _decision(path, "sell-hold", trader_action="SELL")
    _decision(path, "hold-hold", trader_action="HOLD")
    _full_eval(path, "buy-hold", (5.0, 0.0, 0.0, 0.0), buy=(5.0, -1.0, -1.0, -1.0))
    _full_eval(path, "sell-hold", (5.0, 0.0, 0.0, 0.0), sell=(5.0, -1.0, -1.0, -1.0))
    _full_eval(path, "hold-hold", (2.0, 0.0, 0.0, 0.0), buy=(2.0, -1.0, -1.0, -1.0))

    report = audit_decision_path(path)

    assert report.all_population.final_hold_origins == {
        "BUY->HOLD": 1,
        "HOLD->HOLD": 1,
        "SELL->HOLD": 1,
    }
    assert report.all_population.outcome_by_transition["BUY->HOLD"]["buy_better_count"] == 1
    assert report.all_population.outcome_by_transition["SELL->HOLD"]["sell_better_count"] == 1
    assert report.all_population.outcome_by_transition["HOLD->HOLD"]["positive_directional_count"] == 1


def test_research_to_trader_suppression_and_outcome_join(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _decision(path, "research-buy-trader-hold", trader_action="HOLD", research_recommendation="BUY")
    _full_eval(
        path,
        "research-buy-trader-hold",
        (5.0, 5.0, 5.0, 0.0),
        buy=(5.0, 5.0, 5.0, -1.0),
    )

    report = audit_decision_path(path)

    assert report.all_population.research_to_trader_matrix["BUY->HOLD"]["count"] == 1
    assert report.all_population.research_to_trader_pm_matrix["BUY->HOLD->HOLD"]["count"] == 1
    assert report.all_population.research_to_trader_suppression["BUY->HOLD"] == 1
    assert report.all_population.research_to_trader_outcomes["BUY->HOLD"]["samples"] == 4
    assert report.all_population.research_to_trader_persistent["BUY->HOLD"]["STRONG_PERSISTENT_MISS"] == 1


def test_research_manager_audit_counts_only_canonical_persisted_values(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    for index, recommendation in enumerate(("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL")):
        _decision(path, f"research-{index}", research_recommendation=recommendation)
    _decision(path, "legacy-null")

    report = audit_decision_path(path)

    assert report.research_manager_audit["status"] == "AVAILABLE"
    assert report.research_manager_audit["counts"] == {
        "BUY": 1,
        "OVERWEIGHT": 1,
        "HOLD": 1,
        "UNDERWEIGHT": 1,
        "SELL": 1,
    }


def test_persistent_breakdown_matches_hold_audit_definition(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _decision(path, "buy-hold", trader_action="BUY")
    _full_eval(path, "buy-hold", (5.0, 5.0, 5.0, 0.0), buy=(5.0, 5.0, 5.0, -1.0))

    report = audit_decision_path(path, strong_miss_points=5.0)
    counts = report.all_population.persistent_by_transition["BUY->HOLD"]

    assert counts["total"] == 1
    assert counts["persistent_directional_miss_total"] == 1
    assert counts["STRONG_PERSISTENT_MISS"] == 1


def test_training_eligible_null_does_not_exclude_source_context_data(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _decision(path, "eligible", training_eligible=None)
    _full_eval(path, "eligible")

    report = audit_decision_path(path)

    assert report.all_population.population_count == 1
    assert report.all_population.fully_evaluated_source_eligible_count == 1


def test_stale_and_invalid_temporal_decisions_are_excluded(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _decision(path, "stale", stale_by_completion=1)
    _decision(path, "invalid", reference_status="INVALID_TEMPORAL")

    report = audit_decision_path(path)

    assert report.all_population.population_count == 0


def test_pm_rating_mismatch_and_malformed_json_are_safe(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    _decision(path, "mismatch", final_action="HOLD", raw_pm=json.dumps({"rating": "Buy"}))
    _decision(path, "malformed", final_action="HOLD", raw_pm="not-json")

    report = audit_decision_path(path)
    consistency = report.pm_rating_consistency

    assert consistency["mismatch_count"] == 1
    assert consistency["malformed_count"] == 1


def test_read_only_json_output_and_empty_dataset_are_safe(tmp_path: Path, capsys) -> None:
    path = _init_db(tmp_path)
    before = path.read_bytes()
    report = audit_decision_path(path)
    assert report.all_population.population_count == 0
    assert path.read_bytes() == before

    from cli.forex_decision_path_audit import main

    assert main(["--db-path", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["all"]["population_count"] == 0
    assert payload["review_status"] == "DIAGNOSTIC ONLY — NO STRATEGY CHANGE PERFORMED"


def test_schema_requires_watch_runs_and_evaluation_contract(tmp_path: Path) -> None:
    path = tmp_path / "invalid.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE shadow_decisions (decision_id TEXT)")
    with pytest.raises(DecisionPathAuditSchemaError):
        audit_decision_path(path)
