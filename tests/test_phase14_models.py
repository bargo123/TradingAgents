from datetime import datetime, timezone

import pytest

from tradingagents.self_enhancement.models import (
    CandidateSpec,
    CandidateState,
    ExecutionMode,
    ExitPolicyConfig,
    ExperienceTrade,
    FindingKind,
    LearningFinding,
    StrategyVersion,
)

UTC = timezone.utc


def _trade(**overrides):
    values = {
        "experience_id": "exp-1",
        "source_database_id": "demo-db",
        "source_position_id": "ticket-1",
        "strategy_id": "range_rejection",
        "strategy_version": "v1",
        "config_version": "cfg-1",
        "symbol": "EURUSD",
        "direction": "LONG",
        "entry_timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "exit_timestamp": datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC),
        "entry_bid": 1.1000,
        "entry_ask": 1.1001,
        "entry_fill": 1.1001,
        "exit_bid": 1.1003,
        "exit_ask": 1.1004,
        "exit_fill": 1.1003,
        "spread_points": 1.0,
        "slippage_points": 0.0,
        "feature_snapshot": {"momentum": 2.0},
        "regime": "NEUTRAL",
        "expected_move_points": 8.0,
        "volume": 0.01,
        "risk": 0.005,
        "mfe_points": 20.0,
        "mae_points": -2.0,
        "exit_reason": "TAKE_PROFIT",
        "broker_execution_latency_ms": 2.0,
        "gross_result": 2.0,
        "net_known_result": 2.0,
        "commission_known": False,
        "profit_to_loss_flip": False,
        "session": "LONDON",
        "volatility_state": "NORMAL",
        "data_quality_state": "VALID",
        "source_git_commit": "abc123",
        "execution_mode": ExecutionMode.DEMO,
        "real_money": False,
        "synthetic": False,
    }
    values.update(overrides)
    return ExperienceTrade(**values)


def test_exit_policy_rejects_unknown_or_out_of_range_mutations():
    with pytest.raises(ValueError, match="unknown parameter"):
        ExitPolicyConfig.from_mapping({"made_up": 1.0})
    with pytest.raises(ValueError, match="profit_target_fraction"):
        ExitPolicyConfig(profit_target_fraction=99.0)


def test_strategy_version_hash_is_deterministic_and_immutable():
    left = StrategyVersion("range_rejection", "v1", "cfg1", {"x": 1.0}, "abc")
    right = StrategyVersion("range_rejection", "v1", "cfg1", {"x": 1.0}, "abc")
    assert left.config_hash == right.config_hash
    with pytest.raises(TypeError):
        left.parameters["x"] = 2.0


def test_experience_trade_rejects_real_money_and_test_only_records():
    with pytest.raises(ValueError, match="real_money"):
        _trade(real_money=True)
    with pytest.raises(ValueError, match="synthetic"):
        _trade(synthetic=True)
    with pytest.raises(ValueError, match="execution_mode"):
        _trade(execution_mode=ExecutionMode.SHADOW)


def test_candidate_requires_bounded_exit_policy_and_is_experimental():
    candidate = CandidateSpec(
        candidate_id="cand-1",
        parent=StrategyVersion("range_rejection", "v1", "cfg1", {}, "abc"),
        strategy_id="range_rejection",
        exit_policy=ExitPolicyConfig(),
        hypothesis="bounded exit policy",
        state=CandidateState.EXPERIMENTAL,
    )
    assert candidate.state is CandidateState.EXPERIMENTAL
    assert candidate.execution_mode is ExecutionMode.SHADOW


def test_experience_findings_separate_observation_from_verified_causal_conclusion():
    observation = LearningFinding("obs-1", FindingKind.OBSERVATION, "losses cluster in ASIA", ("exp-1",))
    assert observation.verified is False
    with pytest.raises(ValueError, match="verified experiment"):
        LearningFinding("cause-1", FindingKind.CAUSAL_CONCLUSION, "exit caused loss")
    conclusion = LearningFinding(
        "cause-2",
        FindingKind.CAUSAL_CONCLUSION,
        "bounded exit reduced giveback in replay",
        ("exp-1",),
        experiment_id="experiment-1",
        verified=True,
    )
    assert conclusion.kind is FindingKind.CAUSAL_CONCLUSION
