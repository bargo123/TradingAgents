import sqlite3
from datetime import date, datetime, timedelta, timezone

from tradingagents.forex.hft.models import Direction
from tradingagents.forex.hft.plan_feed import build_plan_from_shadow_decision
from tradingagents.forex.shadow import ShadowDecisionStore, ShadowTradeDecision

UTC = timezone.utc


def _decision(
    *,
    action="BUY",
    status="NORMALIZED",
    context="COMPLETE",
    reference_status="AVAILABLE",
):
    completed = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    effective_action = action if status == "NORMALIZED" else None
    values = {
        "decision_id": "decision-1",
        "created_at": completed,
        "snapshot_timestamp": completed - timedelta(minutes=1),
        "analysis_date": date(2026, 1, 1),
        "requested_symbol": "EURUSD",
        "resolved_symbol": "EURUSD",
        "action": effective_action,
        "raw_portfolio_manager_result": {"rating": action or "HOLD"},
        "normalization_status": status,
        "normalization_error": None if status == "NORMALIZED" else "failed",
        "confidence": 0.8,
        "reference_bid": 1.1,
        "reference_ask": 1.1001,
        "reference_mid": 1.10005,
        "spread": 0.0001,
        "spread_points": 10,
        "analysis_timeframe": "M15",
        "trader_summary": "summary",
        "portfolio_manager_summary": "summary",
        "source_run_id": "run-1",
        "analysis_profile": "INTRADAY",
        "valid_for_seconds": 300,
        "valid_until": completed + timedelta(seconds=240),
        "executed": False,
        "decision_context_status": context,
        "decision_completed_timestamp": completed,
        "decision_reference_status": reference_status,
    }
    if reference_status != "UNAVAILABLE":
        values.update(
            {
                "decision_reference_timestamp": (
                    completed + timedelta(seconds=1)
                    if reference_status == "AVAILABLE"
                    else completed - timedelta(seconds=1)
                ),
                "decision_reference_bid": 1.1,
                "decision_reference_ask": 1.1001,
                "decision_reference_spread": 0.0001,
                "decision_reference_spread_points": 10,
            }
        )
    return ShadowTradeDecision(**values)


def test_build_plan_binds_runtime_provenance_and_buy_direction():
    plan = build_plan_from_shadow_decision(_decision(), git_commit="abc123")
    assert plan is not None
    assert plan.primary_direction is Direction.LONG
    assert plan.source_decision_id == "decision-1"
    assert plan.source_run_id == "run-1"
    assert plan.git_commit == "abc123"
    assert plan.analysis_profile == "INTRADAY"
    assert plan.plan_schema_version == "phase12.plan.v1"
    assert plan.provenance_valid is True


def test_hold_decision_maps_to_explicit_none_direction():
    plan = build_plan_from_shadow_decision(_decision(action="HOLD"), git_commit="abc123")
    assert plan is not None
    assert plan.primary_direction is Direction.NONE


def test_fresh_sell_decision_maps_to_short_direction():
    plan = build_plan_from_shadow_decision(_decision(action="SELL"), git_commit="abc123")
    assert plan is not None
    assert plan.primary_direction is Direction.SHORT


def test_invalid_temporal_directional_decisions_fail_closed():
    for action in ("BUY", "SELL"):
        assert (
            build_plan_from_shadow_decision(
                _decision(action=action, reference_status="INVALID_TEMPORAL"),
                git_commit="abc123",
            )
            is None
        )


def test_failed_or_incomplete_decision_does_not_produce_a_live_plan():
    assert build_plan_from_shadow_decision(_decision(status="FAILED"), git_commit="abc123") is None
    assert build_plan_from_shadow_decision(_decision(context="INCOMPLETE"), git_commit="abc123") is None


def test_latest_eligible_rejects_stale_directional_decision(tmp_path):
    path = tmp_path / "stale.db"
    store = ShadowDecisionStore(path)
    store.record(_decision())
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE forex_watch_runs (
                decision_id TEXT,
                source_run_id TEXT,
                run_status TEXT,
                decision_context_status TEXT,
                normalization_status TEXT,
                decision_reference_status TEXT,
                stale_by_completion INTEGER,
                freshness_budget_seconds INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO forex_watch_runs VALUES (?, ?, 'SUCCEEDED', 'COMPLETE', 'NORMALIZED', 'AVAILABLE', 1, 900)",
            ("decision-1", "run-1"),
        )
    assert store.latest_eligible("EURUSD") is None


def test_latest_eligible_rejects_invalid_temporal_directional_decision(tmp_path):
    path = tmp_path / "invalid-temporal.db"
    store = ShadowDecisionStore(path)
    store.record(_decision(reference_status="INVALID_TEMPORAL"))
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE forex_watch_runs (
                decision_id TEXT,
                source_run_id TEXT,
                run_status TEXT,
                decision_context_status TEXT,
                normalization_status TEXT,
                decision_reference_status TEXT,
                stale_by_completion INTEGER,
                freshness_budget_seconds INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO forex_watch_runs VALUES (?, ?, 'SUCCEEDED', 'COMPLETE', 'NORMALIZED', 'AVAILABLE', 0, 900)",
            ("decision-1", "run-1"),
        )
    assert store.latest_eligible("EURUSD") is None


def test_latest_eligible_accepts_fresh_directional_decision(tmp_path):
    path = tmp_path / "fresh.db"
    store = ShadowDecisionStore(path)
    decision = _decision()
    store.record(decision)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE forex_watch_runs (
                decision_id TEXT,
                source_run_id TEXT,
                run_status TEXT,
                decision_context_status TEXT,
                normalization_status TEXT,
                decision_reference_status TEXT,
                stale_by_completion INTEGER,
                freshness_budget_seconds INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO forex_watch_runs VALUES (?, ?, 'SUCCEEDED', 'COMPLETE', 'NORMALIZED', 'AVAILABLE', 0, 900)",
            (decision.decision_id, decision.source_run_id),
        )
    assert store.latest_eligible("EURUSD").decision_id == decision.decision_id


def test_source_run_freshness_validator_rejects_stale_and_accepts_fresh(tmp_path):
    path = tmp_path / "validator.db"
    store = ShadowDecisionStore(path)
    decision = _decision()
    store.record(decision)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE forex_watch_runs (
                decision_id TEXT,
                source_run_id TEXT,
                run_status TEXT,
                decision_context_status TEXT,
                normalization_status TEXT,
                decision_reference_status TEXT,
                stale_by_completion INTEGER,
                freshness_budget_seconds INTEGER,
                started_at TEXT,
                completed_at TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO forex_watch_runs VALUES (?, ?, 'SUCCEEDED', 'COMPLETE', 'NORMALIZED', 'AVAILABLE', 0, 900, ?, ?)",
            (
                decision.decision_id,
                decision.source_run_id,
                "2026-01-01T11:59:00Z",
                "2026-01-01T12:00:01Z",
            ),
        )
    assert store.is_execution_eligible(decision) is True
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE forex_watch_runs SET stale_by_completion=1 WHERE decision_id=?",
            (decision.decision_id,),
        )
    assert store.is_execution_eligible(decision) is False
