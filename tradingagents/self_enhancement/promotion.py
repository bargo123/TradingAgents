"""Deterministic candidate gates and rollback controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import CandidateSpec
from .store import SelfEnhancementStore


@dataclass(frozen=True, slots=True)
class ReplayGateReports:
    stage_reports: dict[str, dict[str, Any]]
    cost_reports: dict[float, dict[str, Any]] | None = None
    safety_regression: bool = False


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    decision: str
    can_promote: bool
    candidate_id: str
    reasons: tuple[str, ...]


class CandidatePromotionGate:
    """Require evidence across independent stages before shadow promotion."""

    def __init__(self, *, minimum_trades: int = 20, minimum_expectancy_improvement: float = 0.0, max_drawdown_tolerance: float = 0.0) -> None:
        if minimum_trades <= 0 or minimum_expectancy_improvement < 0 or max_drawdown_tolerance < 0:
            raise ValueError("promotion thresholds are invalid")
        self.minimum_trades = int(minimum_trades)
        self.minimum_expectancy_improvement = float(minimum_expectancy_improvement)
        self.max_drawdown_tolerance = float(max_drawdown_tolerance)

    def evaluate(
        self,
        incumbent_metrics: dict[str, Any],
        candidate: CandidateSpec,
        reports: ReplayGateReports,
    ) -> PromotionDecision:
        reasons: list[str] = []
        required = ("DEVELOPMENT", "VALIDATION", "UNSEEN_HOLDOUT")
        missing = [stage for stage in required if stage not in reports.stage_reports]
        if missing:
            reasons.append("MISSING_STAGE:" + ",".join(missing))
        for stage in required:
            report = reports.stage_reports.get(stage, {})
            if int(report.get("trades", 0)) < self.minimum_trades:
                reasons.append(f"INSUFFICIENT_EVIDENCE:{stage}")
            if report.get("future_leak_detected") is True or report.get("real_money") is not False:
                reasons.append(f"SAFETY_REGRESSION:{stage}")
        if reports.safety_regression:
            reasons.append("SAFETY_REGRESSION")
        if reasons:
            return PromotionDecision("INSUFFICIENT_EVIDENCE" if any(item.startswith("INSUFFICIENT") for item in reasons) else "REJECTED", False, candidate.candidate_id, tuple(reasons))
        validation = reports.stage_reports["VALIDATION"]
        unseen = reports.stage_reports["UNSEEN_HOLDOUT"]
        incumbent_expectancy = float(incumbent_metrics.get("expectancy", 0.0) or 0.0)
        candidate_expectancy = min(float(validation.get("expectancy", 0.0) or 0.0), float(unseen.get("expectancy", 0.0) or 0.0))
        if candidate_expectancy <= incumbent_expectancy + self.minimum_expectancy_improvement:
            reasons.append("EXPECTANCY_NOT_IMPROVED")
        incumbent_pf = incumbent_metrics.get("profit_factor")
        candidate_pf = min(float(validation.get("profit_factor", 0.0) or 0.0), float(unseen.get("profit_factor", 0.0) or 0.0))
        if incumbent_pf is not None and candidate_pf < float(incumbent_pf):
            reasons.append("PROFIT_FACTOR_REGRESSION")
        incumbent_dd = float(incumbent_metrics.get("max_drawdown", 0.0) or 0.0)
        candidate_dd = max(float(validation.get("max_drawdown", 0.0) or 0.0), float(unseen.get("max_drawdown", 0.0) or 0.0))
        if candidate_dd > incumbent_dd + self.max_drawdown_tolerance:
            reasons.append("DRAWDOWN_REGRESSION")
        for scenario, report in (reports.cost_reports or {}).items():
            if int(report.get("trades", 0)) < self.minimum_trades or float(report.get("expectancy", 0.0) or 0.0) <= 0:
                reasons.append(f"COST_SENSITIVITY_FAILURE:{scenario}")
        if reasons:
            return PromotionDecision("REJECTED", False, candidate.candidate_id, tuple(reasons))
        return PromotionDecision("PROMOTE_TO_SHADOW", True, candidate.candidate_id, ())


class RollbackController:
    @staticmethod
    def rollback(store: SelfEnhancementStore, deployment_id: str, reason: str) -> None:
        store.rollback(deployment_id, reason)


__all__ = ["CandidatePromotionGate", "PromotionDecision", "ReplayGateReports", "RollbackController"]
