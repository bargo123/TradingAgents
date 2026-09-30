from datetime import date, datetime, timedelta, timezone

from tradingagents.forex.hft.models import Direction
from tradingagents.forex.hft.plan_feed import build_plan_from_shadow_decision
from tradingagents.forex.shadow import ShadowTradeDecision

UTC = timezone.utc


def _decision(*, action="BUY", status="NORMALIZED", context="COMPLETE"):
    completed = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    effective_action = action if status == "NORMALIZED" else None
    return ShadowTradeDecision(
        decision_id="decision-1",
        created_at=completed,
        snapshot_timestamp=completed - timedelta(minutes=1),
        analysis_date=date(2026, 1, 1),
        requested_symbol="EURUSD",
        resolved_symbol="EURUSD",
        action=effective_action,
        raw_portfolio_manager_result={"rating": action or "HOLD"},
        normalization_status=status,
        normalization_error=None if status == "NORMALIZED" else "failed",
        confidence=0.8,
        reference_bid=1.1,
        reference_ask=1.1001,
        reference_mid=1.10005,
        spread=0.0001,
        spread_points=10,
        analysis_timeframe="M15",
        trader_summary="summary",
        portfolio_manager_summary="summary",
        source_run_id="run-1",
        analysis_profile="INTRADAY",
        valid_for_seconds=300,
        valid_until=completed + timedelta(seconds=240),
        executed=False,
        decision_context_status=context,
        decision_completed_timestamp=completed,
        decision_reference_status="UNAVAILABLE",
    )


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


def test_failed_or_incomplete_decision_does_not_produce_a_live_plan():
    assert build_plan_from_shadow_decision(_decision(status="FAILED"), git_commit="abc123") is None
    assert build_plan_from_shadow_decision(_decision(context="INCOMPLETE"), git_commit="abc123") is None
