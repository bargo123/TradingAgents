import inspect

import tradingagents.self_enhancement as package
from tradingagents.self_enhancement.models import (
    CandidateSpec,
    ExecutionMode,
    ExitPolicyConfig,
    StrategyVersion,
)


def test_phase14_package_has_no_broker_mutation_api():
    source = "\n".join(inspect.getsource(module) for module in [package])
    assert "order_send" not in source
    assert "positions_get" not in source


def test_phase14_candidates_are_shadow_only():
    candidate = CandidateSpec("c", StrategyVersion("s", "v", "cfg", {}, "abc"), "s", ExitPolicyConfig(), "h")
    assert candidate.execution_mode is ExecutionMode.SHADOW
    assert candidate.real_money is False
