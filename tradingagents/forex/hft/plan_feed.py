"""Fail-closed translation from completed shadow decisions to HFT plans."""

from __future__ import annotations

from datetime import timedelta, timezone

from tradingagents.forex.shadow import ShadowTradeDecision

from .models import (
    Direction,
    EntryConstraints,
    RiskPosture,
    StopPolicy,
    StrategicExecutionPlan,
)

UTC = timezone.utc


def build_plan_from_shadow_decision(
    decision: ShadowTradeDecision,
    *,
    git_commit: str,
    strategy_family: str = "strategic_shadow",
    stop_distance_points: float = 20.0,
    take_profit_distance_points: float = 30.0,
    time_stop_seconds: int = 300,
) -> StrategicExecutionPlan | None:
    """Build a provenance-bound plan only from a complete normalized decision.

    A HOLD is intentionally represented as ``Direction.NONE``.  Failed or
    incomplete decisions do not replace an existing plan.
    """

    if not isinstance(decision, ShadowTradeDecision):
        raise TypeError("decision must be ShadowTradeDecision")
    if decision.executed is not False:
        return None
    if decision.decision_context_status != "COMPLETE" or decision.normalization_status != "NORMALIZED":
        return None
    if decision.action not in {"BUY", "SELL", "HOLD"}:
        return None
    # Directional plans require a broker quote captured at/after decision
    # completion. UNAVAILABLE and INVALID_TEMPORAL are valid Phase 5 states,
    # but they are never sufficient to activate LONG/SHORT.
    if decision.action in {"BUY", "SELL"} and decision.decision_reference_status != "AVAILABLE":
        return None
    if not isinstance(git_commit, str) or not git_commit.strip():
        return None
    completed = decision.decision_completed_timestamp or decision.created_at
    completed = completed.astimezone(UTC)
    expires = decision.valid_until
    if expires is None:
        if decision.valid_for_seconds is None:
            return None
        expires = completed + timedelta(seconds=decision.valid_for_seconds)
    expires = expires.astimezone(UTC)
    if expires <= completed:
        return None
    direction = {
        "BUY": Direction.LONG,
        "SELL": Direction.SHORT,
        "HOLD": Direction.NONE,
    }[decision.action]
    confidence = 0.0 if decision.confidence is None else float(decision.confidence)
    return StrategicExecutionPlan(
        symbol=decision.resolved_symbol,
        created_at=completed,
        valid_from=completed,
        expires_at=expires,
        allowed_until=expires,
        timeframe=decision.analysis_timeframe,
        regime="STRATEGIC_SHADOW",
        primary_direction=direction,
        confidence=confidence,
        strategy_family=strategy_family,
        entry_constraints=EntryConstraints(max_spread_points=20.0, minimum_momentum=0.0),
        risk_posture=RiskPosture.NORMAL,
        stop_policy=StopPolicy(
            stop_distance_points=stop_distance_points,
            take_profit_distance_points=take_profit_distance_points,
            time_stop_seconds=time_stop_seconds,
        ),
        plan_id=f"plan-{decision.decision_id}",
        source_decision_id=decision.decision_id,
        source_run_id=decision.source_run_id,
        git_commit=git_commit.strip(),
        analysis_profile=decision.analysis_profile,
        plan_schema_version="phase12.plan.v1",
        test_only=False,
    )


__all__ = ["build_plan_from_shadow_decision"]
