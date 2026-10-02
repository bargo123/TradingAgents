from tradingagents.self_enhancement.models import CandidateSpec, ExitPolicyConfig, StrategyVersion
from tradingagents.self_enhancement.promotion import (
    CandidatePromotionGate,
    ReplayGateReports,
)


def _candidate():
    parent = StrategyVersion("range_rejection", "v1", "cfg1", ExitPolicyConfig().to_dict(), "abc")
    return CandidateSpec("cand-1", parent, "range_rejection", ExitPolicyConfig(), "test")


def _report(trades=3, expectancy=1.0, profit_factor=2.0, drawdown=0.01):
    return {"trades": trades, "expectancy": expectancy, "profit_factor": profit_factor, "max_drawdown": drawdown, "future_leak_detected": False, "real_money": False}


def test_promotion_rejects_insufficient_evidence_before_metric_comparison():
    result = CandidatePromotionGate(minimum_trades=5).evaluate(
        _report(trades=10, expectancy=1, profit_factor=2, drawdown=.02),
        _candidate(),
        ReplayGateReports({"DEVELOPMENT": _report(4), "VALIDATION": _report(4), "UNSEEN_HOLDOUT": _report(4)}),
    )
    assert result.decision == "INSUFFICIENT_EVIDENCE"
    assert result.can_promote is False


def test_promotion_requires_unseen_and_cost_safe_improvement():
    result = CandidatePromotionGate(minimum_trades=2).evaluate(
        _report(trades=6, expectancy=1, profit_factor=2, drawdown=.04),
        _candidate(),
        ReplayGateReports(
            {"DEVELOPMENT": _report(4, 1.3, 2.5, .02), "VALIDATION": _report(3, 1.2, 2.2, .02), "UNSEEN_HOLDOUT": _report(3, 1.1, 2.1, .03)},
            cost_reports={0.0: _report(3, 1.2, 2.2, .02), 1.0: _report(3, 1.0, 1.8, .03)},
        ),
    )
    assert result.decision == "PROMOTE_TO_SHADOW"
    assert result.can_promote is True


def test_promotion_rejects_future_leak_or_drawdown_regression():
    result = CandidatePromotionGate(minimum_trades=2).evaluate(
        _report(trades=6),
        _candidate(),
        ReplayGateReports({"DEVELOPMENT": _report(3), "VALIDATION": _report(3, 2, 3, .20), "UNSEEN_HOLDOUT": _report(3, 2, 3, .02,)}, safety_regression=True),
    )
    assert result.decision == "REJECTED"
    assert result.can_promote is False
