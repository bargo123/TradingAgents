from datetime import datetime, timezone

from tradingagents.self_enhancement.models import (
    CandidateSpec,
    ExecutionMode,
    ExitPolicyConfig,
    ExperienceTrade,
    FindingKind,
    LearningFinding,
    StrategyVersion,
    TriggerKind,
)
from tradingagents.self_enhancement.store import SelfEnhancementStore

UTC = timezone.utc


def _trade(identifier="exp-1"):
    return ExperienceTrade(
        experience_id=identifier,
        source_database_id="demo-db",
        source_position_id=identifier,
        strategy_id="range_rejection",
        strategy_version="v1",
        config_version="cfg1",
        symbol="EURUSD",
        direction="LONG",
        entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        exit_timestamp=datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC),
        entry_bid=1.1,
        entry_ask=1.1001,
        entry_fill=1.1001,
        exit_bid=1.1003,
        exit_ask=1.1004,
        exit_fill=1.1003,
        spread_points=1,
        slippage_points=0,
        feature_snapshot={"momentum": 2},
        regime="NEUTRAL",
        expected_move_points=8,
        volume=0.01,
        risk=0.005,
        mfe_points=20,
        mae_points=-2,
        exit_reason="TAKE_PROFIT",
        broker_execution_latency_ms=2,
        gross_result=2,
        net_known_result=2,
        commission_known=False,
        profit_to_loss_flip=False,
        session="LONDON",
        volatility_state="NORMAL",
        data_quality_state="VALID",
        source_git_commit="abc",
        execution_mode=ExecutionMode.DEMO,
    )


def _candidate():
    parent = StrategyVersion("range_rejection", "v1", "cfg1", {}, "abc")
    return CandidateSpec("cand-1", parent, "range_rejection", ExitPolicyConfig(), "test")


def test_store_is_idempotent_and_records_lifecycle(tmp_path):
    store = SelfEnhancementStore(tmp_path / "phase14.sqlite3")
    store.initialize()
    store.record_experience(_trade())
    store.record_experience(_trade())
    experiment_id = store.create_experiment("exp-1", parent_version="v1", dataset_fingerprint="data1")
    store.record_candidate(experiment_id, _candidate())
    store.record_evaluation(experiment_id, "cand-1", "DEVELOPMENT", {"trades": 3}, passed=True)
    assert store.snapshot()["experience"] == 1
    assert store.snapshot()["candidates"] == 1
    assert store.snapshot()["evaluations"] == 1


def test_store_recovers_running_experiments_without_promoting(tmp_path):
    store = SelfEnhancementStore(tmp_path / "phase14.sqlite3")
    store.initialize()
    store.create_experiment("exp-1", parent_version="v1", dataset_fingerprint="data1")
    assert store.recover_incomplete_experiments() == ("exp-1",)
    assert store.experiment_status("exp-1") == "ABORTED_RECOVERABLE"


def test_promotion_and_rollback_are_transactional_and_bounded(tmp_path):
    store = SelfEnhancementStore(tmp_path / "phase14.sqlite3")
    store.initialize()
    experiment_id = store.create_experiment("exp-1", parent_version="v1", dataset_fingerprint="data1")
    store.record_candidate(experiment_id, _candidate())
    deployment_id = store.record_promotion(
        experiment_id,
        "cand-1",
        decision="REJECTED",
        reason="INSUFFICIENT_EVIDENCE",
        rollback_package={"previous_version": "v1", "config_hash": "abc"},
    )
    assert deployment_id is None
    assert store.snapshot()["deployments"] == 0
    assert store.snapshot()["promotions"] == 1


def test_registry_transitions_triggers_and_finding_maturity_are_auditable(tmp_path):
    store = SelfEnhancementStore(tmp_path / "phase14.sqlite3")
    store.initialize()
    experiment_id = store.create_experiment("exp-1", parent_version="v1", dataset_fingerprint="data1")
    store.record_candidate(experiment_id, _candidate())
    store.transition_candidate(experiment_id, "cand-1", "REPLAYED", "replayed")
    store.record_trigger(TriggerKind.NEW_VERIFIED_TRADES, {"count": 20})
    store.record_finding(
        LearningFinding("finding-1", FindingKind.OBSERVATION, "range exit gave back MFE", ("exp-1",))
    )
    assert store.snapshot()["candidate_transitions"] == 2
    assert store.snapshot()["triggers"] == 1
    assert store.snapshot()["findings"] == 1
