from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.dataflows.mt5.models import Mt5Tick
from tradingagents.forex.evaluation import (
    EvaluationBasis,
    EvaluationConfig,
    EvaluationStatus,
    ShadowEvaluationStore,
    build_horizon_evaluation,
    decision_source_eligibility,
    evaluate_directional_outcomes,
    evaluate_excursions,
    evaluation_target,
    first_valid_tick,
)
from tradingagents.forex.shadow import ShadowTradeDecision


ANCHOR = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
COMPLETED = ANCHOR + timedelta(seconds=41)
REFERENCE = COMPLETED + timedelta(seconds=1)


def _decision(**overrides) -> ShadowTradeDecision:
    base = {
        "decision_id": "decision-eval-001",
        "created_at": ANCHOR,
        "snapshot_timestamp": ANCHOR,
        "analysis_date": date(2026, 9, 8),
        "requested_symbol": "EURUSD",
        "resolved_symbol": "EURUSDm",
        "action": "BUY",
        "raw_portfolio_manager_result": {"rating": "Overweight"},
        "normalization_status": "NORMALIZED",
        "normalization_error": None,
        "confidence": 0.7,
        "reference_bid": 1.1000,
        "reference_ask": 1.1002,
        "reference_mid": 1.1001,
        "spread": 0.0002,
        "spread_points": 20.0,
        "analysis_timeframe": "M15",
        "trader_summary": "trader",
        "portfolio_manager_summary": "pm",
        "snapshot_json": {
            "symbol_metadata": {"digits": 5, "point": 0.00001},
        },
        "future_evaluation_status": "PENDING",
        "source_run_id": "run-eval-001",
        "executed": False,
        "decision_context_status": "COMPLETE",
        "analysis_snapshot_timestamp": ANCHOR,
        "analysis_snapshot_bid": 1.1000,
        "analysis_snapshot_ask": 1.1002,
        "analysis_snapshot_spread": 0.0002,
        "analysis_snapshot_spread_points": 20.0,
        "decision_completed_timestamp": COMPLETED,
        "analysis_latency_seconds": 41.0,
        "decision_reference_timestamp": REFERENCE,
        "decision_reference_bid": 1.1003,
        "decision_reference_ask": 1.1005,
        "decision_reference_spread": 0.0002,
        "decision_reference_spread_points": 20.0,
        "decision_reference_status": "AVAILABLE",
        "decision_reference_delay_seconds": 1.0,
    }
    base.update(overrides)
    return ShadowTradeDecision(**base)


def _tick(timestamp: datetime, bid: float, ask: float) -> Mt5Tick:
    return Mt5Tick(symbol="EURUSDm", timestamp=timestamp, bid=bid, ask=ask)


def test_evaluation_config_defaults_and_validation() -> None:
    config = EvaluationConfig()
    assert config.horizons_seconds == (300, 900, 1800, 3600)
    assert config.observation_tolerance_seconds == 30

    with pytest.raises(ValueError, match="strictly increasing"):
        EvaluationConfig(horizons_seconds=(300, 300))
    with pytest.raises(ValueError, match="positive"):
        EvaluationConfig(horizons_seconds=(0, 300))
    with pytest.raises(ValueError, match="tolerance"):
        EvaluationConfig(observation_tolerance_seconds=-1)


def test_source_context_eligibility_is_separate_from_training() -> None:
    eligible = decision_source_eligibility(_decision())
    assert eligible.eligible is True
    assert eligible.reason == "ELIGIBLE_SOURCE_CONTEXT"

    incomplete = decision_source_eligibility(
        _decision(decision_context_status="INCOMPLETE")
    )
    assert incomplete.eligible is False
    assert incomplete.reason == "DECISION_CONTEXT_INCOMPLETE"

    failed = decision_source_eligibility(
        _decision(action=None, normalization_status="FAILED", normalization_error="bad")
    )
    assert failed.eligible is False
    assert failed.reason == "NORMALIZATION_FAILED"


def test_evaluation_target_and_first_tick_respect_tolerance_deadline() -> None:
    target, deadline = evaluation_target(ANCHOR, 300, tolerance_seconds=30)
    assert target == ANCHOR + timedelta(seconds=300)
    assert deadline == ANCHOR + timedelta(seconds=330)

    ticks = (
        _tick(target - timedelta(seconds=1), 1.0, 1.1),
        _tick(target + timedelta(seconds=1), 1.1001, 1.1003),
        _tick(deadline + timedelta(seconds=1), 1.2, 1.3),
    )
    assert first_valid_tick(ticks, target, deadline) == ticks[1]
    assert first_valid_tick(ticks, target, target - timedelta(seconds=1)) is None


def test_pending_until_deadline_then_complete_with_first_valid_tick() -> None:
    config = EvaluationConfig(horizons_seconds=(300,), observation_tolerance_seconds=30)
    decision = _decision()
    target = ANCHOR + timedelta(seconds=300)
    ticks = (_tick(target + timedelta(seconds=1), 1.1004, 1.1006),)

    pending = build_horizon_evaluation(
        decision,
        evaluation_basis="ANALYSIS_SNAPSHOT",
        horizon_seconds=300,
        now=target + timedelta(seconds=29),
        ticks=ticks,
        config=config,
    )
    assert pending.evaluation_status == "PENDING"
    assert pending.target_timestamp == target

    complete = build_horizon_evaluation(
        decision,
        evaluation_basis="ANALYSIS_SNAPSHOT",
        horizon_seconds=300,
        now=target + timedelta(seconds=30),
        ticks=ticks,
        config=config,
    )
    assert complete.evaluation_status == "COMPLETE"
    assert complete.observation_timestamp == ticks[0].timestamp
    assert complete.observation_lag_ms == 1000


def test_directional_formulas_preserve_spread_and_hold_opportunity_cost() -> None:
    outcome = evaluate_directional_outcomes(
        entry_bid=1.1000,
        entry_ask=1.1002,
        future_bid=1.1005,
        future_ask=1.1007,
        point=0.00001,
        selected_action="BUY",
    )
    assert outcome.buy_net_price == pytest.approx(0.0003)
    assert outcome.buy_net_points == pytest.approx(30.0)
    assert outcome.sell_net_price == pytest.approx(-0.0007)
    assert outcome.sell_net_points == pytest.approx(-70.0)
    assert outcome.selected_action_net_points == pytest.approx(30.0)
    assert outcome.hold_opportunity_cost_points == pytest.approx(30.0)

    both_negative = evaluate_directional_outcomes(
        entry_bid=1.1000,
        entry_ask=1.1002,
        future_bid=1.0999,
        future_ask=1.1001,
        point=0.00001,
        selected_action="HOLD",
    )
    assert both_negative.buy_net_points < 0
    assert both_negative.sell_net_points < 0
    assert both_negative.selected_action_net_points == 0
    assert both_negative.hold_opportunity_cost_points == 0
    assert both_negative.best_counterfactual_action == "SELL"


def test_mfe_mae_is_cost_aware_and_excludes_post_target_tolerance_ticks() -> None:
    target = ANCHOR + timedelta(seconds=300)
    metrics = evaluate_excursions(
        entry_bid=1.1000,
        entry_ask=1.1002,
        point=0.00001,
        ticks=(
            _tick(ANCHOR + timedelta(seconds=10), 1.1001, 1.1003),
            # This tick can be selected as the terminal quote but is outside
            # the exact MFE/MAE window and must not improve cost-aware MFE.
            _tick(target + timedelta(seconds=1), 1.1010, 1.1012),
        ),
        anchor_timestamp=ANCHOR,
        target_timestamp=target,
    )

    assert metrics.buy_mfe_price == pytest.approx(-0.0001)
    assert metrics.buy_mfe_points == pytest.approx(-10.0)
    assert metrics.sell_mfe_price == pytest.approx(-0.0002)
    assert metrics.sell_mfe_points == pytest.approx(-20.0)


def test_basis_reference_requires_actual_reference_and_never_falls_back() -> None:
    config = EvaluationConfig(horizons_seconds=(300,), observation_tolerance_seconds=30)
    decision = _decision(
        decision_reference_timestamp=None,
        decision_reference_bid=None,
        decision_reference_ask=None,
        decision_reference_spread=None,
        decision_reference_spread_points=None,
        decision_reference_status="UNAVAILABLE",
        decision_reference_delay_seconds=None,
    )
    result = build_horizon_evaluation(
        decision,
        evaluation_basis="DECISION_REFERENCE",
        horizon_seconds=300,
        now=ANCHOR + timedelta(hours=1),
        ticks=(),
        config=config,
    )
    assert result.evaluation_status == "DATA_UNAVAILABLE"
    assert "reference" in (result.unavailable_reason or "").lower()


def test_public_status_literals_remain_closed() -> None:
    assert EvaluationBasis.__args__ == ("ANALYSIS_SNAPSHOT", "DECISION_REFERENCE")
    assert EvaluationStatus.__args__ == (
        "PENDING",
        "COMPLETE",
        "DATA_UNAVAILABLE",
        "INELIGIBLE",
    )


def _record(
    decision: ShadowTradeDecision,
    *,
    now: datetime,
    ticks: tuple[Mt5Tick, ...] = (),
) -> object:
    return build_horizon_evaluation(
        decision,
        evaluation_basis="ANALYSIS_SNAPSHOT",
        horizon_seconds=300,
        now=now,
        ticks=ticks,
        config=EvaluationConfig(horizons_seconds=(300,), observation_tolerance_seconds=30),
    )


def test_evaluation_store_creates_basis_schema_and_triple_key(tmp_path: Path) -> None:
    store = ShadowEvaluationStore(tmp_path / "evaluations.db")
    store.initialize()

    with sqlite3.connect(store.path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(shadow_decision_evaluations)")}
        primary_key = [
            row[1]
            for row in sorted(
                conn.execute("PRAGMA table_info(shadow_decision_evaluations)").fetchall(),
                key=lambda row: row[5],
            )
            if row[5]
        ]
    assert {"evaluation_basis", "source_context_eligible", "training_eligible"}.issubset(columns)
    assert {"evaluated_at", "recovered_from_unavailable_at", "previous_unavailable_reason"}.issubset(columns)
    assert primary_key == ["decision_id", "evaluation_basis", "horizon_seconds"]


def test_evaluation_store_allows_recovery_but_preserves_complete_and_ineligible(
    tmp_path: Path,
) -> None:
    store = ShadowEvaluationStore(tmp_path / "transitions.db")
    decision = _decision()
    target = ANCHOR + timedelta(seconds=300)
    pending = _record(decision, now=target)
    store.upsert([pending])
    assert store.get(decision.decision_id, "ANALYSIS_SNAPSHOT", 300).evaluation_status == "PENDING"

    unavailable = _record(decision, now=target + timedelta(seconds=30))
    store.upsert([unavailable])
    unavailable_row = store.get(decision.decision_id, "ANALYSIS_SNAPSHOT", 300)
    assert unavailable_row.evaluation_status == "DATA_UNAVAILABLE"

    observed = _tick(target + timedelta(seconds=1), 1.1004, 1.1006)
    complete = _record(
        decision,
        now=target + timedelta(seconds=30),
        ticks=(observed,),
    )
    store.upsert([complete])
    recovered = store.get(decision.decision_id, "ANALYSIS_SNAPSHOT", 300)
    assert recovered.evaluation_status == "COMPLETE"
    assert recovered.created_at == unavailable_row.created_at
    assert recovered.evaluated_at == complete.evaluated_at
    assert recovered.recovered_from_unavailable_at == complete.evaluated_at
    assert recovered.previous_unavailable_reason == unavailable_row.unavailable_reason
    assert recovered.market_data_source == "MT5"
    assert recovered.training_eligible is None

    changed = replace(complete, future_bid=1.2, evaluated_at=target + timedelta(minutes=1))
    store.upsert([changed])
    still_complete = store.get(decision.decision_id, "ANALYSIS_SNAPSHOT", 300)
    assert still_complete.future_bid == complete.future_bid
    assert still_complete.evaluated_at == complete.evaluated_at

    ineligible_decision = _decision(
        decision_id="decision-ineligible",
        decision_context_status="INCOMPLETE",
    )
    ineligible = _record(ineligible_decision, now=target)
    store.upsert([ineligible])
    attempted_complete = replace(
        ineligible,
        evaluation_status="COMPLETE",
        observation_timestamp=target,
        entry_timestamp=ANCHOR,
    )
    store.upsert([attempted_complete])
    assert (
        store.get("decision-ineligible", "ANALYSIS_SNAPSHOT", 300).evaluation_status
        == "INELIGIBLE"
    )


def test_evaluation_store_migrates_legacy_single_key_table(tmp_path: Path) -> None:
    path = tmp_path / "legacy-evaluations.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE shadow_decision_evaluations (
                decision_id TEXT NOT NULL,
                horizon_seconds INTEGER NOT NULL,
                resolved_symbol TEXT NOT NULL,
                evaluation_version TEXT NOT NULL,
                market_data_source TEXT NOT NULL,
                training_eligible INTEGER NOT NULL,
                target_timestamp TEXT,
                evaluation_status TEXT NOT NULL,
                unavailable_reason TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (decision_id, horizon_seconds)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO shadow_decision_evaluations
                (decision_id, horizon_seconds, resolved_symbol, evaluation_version,
                 market_data_source, training_eligible, target_timestamp,
                 evaluation_status, unavailable_reason, created_at)
            VALUES ('legacy', 300, 'EURUSDm', 'old', 'MT5', 1, NULL,
                    'DATA_UNAVAILABLE', 'old gap', '2026-09-08T00:00:00Z')
            """
        )

    store = ShadowEvaluationStore(path)
    store.initialize()
    rows = store.list_for_decision("legacy")

    assert len(rows) == 1
    assert rows[0].evaluation_basis == "ANALYSIS_SNAPSHOT"
    assert rows[0].training_eligible is None
    assert rows[0].training_eligibility_reason == "SOURCE_CONTEXT_INELIGIBLE"
    assert rows[0].evaluation_status == "DATA_UNAVAILABLE"
    assert rows[0].previous_unavailable_reason is None
