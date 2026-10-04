from __future__ import annotations

import builtins
import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradingagents.self_enhancement import orchestrator
from tradingagents.self_enhancement.models import CandidateState, ExperienceTrade
from tradingagents.self_enhancement.phase14c_models import (
    CandidateEvaluationRecord,
    Phase14CCandidateReason,
    Phase14CCandidateStatus,
    Phase14CReplayPreflight,
    Phase14CSourcePaths,
)
from tradingagents.self_enhancement.phase14c_runner import (
    evaluate_phase14c_candidates,
    preflight_phase14_sources,
)
from tradingagents.self_enhancement.strategy_specs import (
    EXECUTABLE_REQUIRED_STAGES,
    EvidenceSpan,
    RuleDirection,
    RuleOperator,
    RuleOrigin,
    RuleStage,
    RuleValidationResult,
    RuleValidationStatus,
    StrategyRuleClaim,
    StrategySpec,
    StrategySuitability,
)


def _strategy_spec(spec_id: str = "spec-1", family: str = "MOMENTUM_CONTINUATION") -> StrategySpec:
    values = {
        RuleStage.ENTRY: (RuleOperator.GREATER_THAN, RuleDirection.LONG, 2, "points", "after confirmation", None, "LONG when momentum > 2 points after confirmation"),
        RuleStage.CONFIRMATION: (RuleOperator.GREATER_OR_EQUAL, RuleDirection.LONG, 0.6, "fraction", "after three ticks", None, "CONFIRMATION: LONG when direction_persistence >= 0.6 fraction after three ticks"),
        RuleStage.INVALIDATION: (RuleOperator.LESS_OR_EQUAL, RuleDirection.LONG, 0, "points", "after reversal", None, "INVALIDATION: LONG when momentum <= 0 points after reversal"),
        RuleStage.EXPECTED_MOVE: (RuleOperator.GREATER_OR_EQUAL, RuleDirection.LONG, 5, "points", "after entry", 30, "EXPECTED_MOVE: LONG target 5 points within 30 seconds after entry"),
        RuleStage.EXIT: (RuleOperator.LESS_OR_EQUAL, RuleDirection.LONG, 0, "points", "after reversal", None, "EXIT: LONG when momentum <= 0 points after reversal"),
        RuleStage.PROFIT_PROTECTION: (RuleOperator.LESS_OR_EQUAL, RuleDirection.LONG, 0, "points", "after target retracement", None, "PROFIT_PROTECTION: LONG when momentum <= 0 points after target retracement"),
        RuleStage.STOP_BEHAVIOR: (RuleOperator.LESS_OR_EQUAL, RuleDirection.LONG, -3, "points", "after adverse move", None, "STOP_BEHAVIOR: LONG when momentum <= -3 points after adverse move"),
        RuleStage.HORIZON: (RuleOperator.LESS_OR_EQUAL, RuleDirection.BOTH, 30, "seconds", "after entry", 30, "HORIZON: hold no longer than 30 seconds after entry"),
    }
    claims = []
    for stage in EXECUTABLE_REQUIRED_STAGES:
        operator, direction, value, unit, condition, horizon, quote = values[stage]
        claims.append(
            StrategyRuleClaim(
                stage, operator, direction, value, unit, condition, horizon,
                RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
                EvidenceSpan("gen-1", "doc-1", f"chunk-{stage.value}", "a" * 64, 0, len(quote), quote),
            )
        )
    validations = tuple(
        RuleValidationResult(item.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE")
        for item in claims
    )
    return StrategySpec(
        spec_id, "Book-supported momentum", family, ("momentum", "direction_persistence"), tuple(claims),
        "gen-1", "b" * 64, "ollama-local", "qwen3.5:2b", "phase14b-draft-v1", "strategy-spec-v1",
        datetime(2026, 10, 3, tzinfo=UTC), 0.8, StrategySuitability.HFT_SUITABLE, validations,
    )


def _source_databases(root: Path) -> tuple[Path, Path, Path]:
    phase14a, hft, demo = root / "phase14a.sqlite3", root / "hft.sqlite3", root / "demo.sqlite3"
    start = datetime(2026, 1, 1, 9, tzinfo=UTC)
    trade = ExperienceTrade(
        experience_id="verified-1", source_database_id="demo:fixture", source_position_id="ticket-1",
        strategy_id="momentum_continuation", strategy_version="v1", config_version="c1", symbol="EURUSD",
        direction="LONG", entry_timestamp=start, exit_timestamp=start + timedelta(seconds=5),
        entry_bid=1.1, entry_ask=1.10002, entry_fill=1.10002, exit_bid=1.1001, exit_ask=1.10012,
        exit_fill=1.1001, spread_points=2, slippage_points=0, feature_snapshot={"momentum": 3}, regime="NEUTRAL",
        expected_move_points=5, volume=0.01, risk=0.001, mfe_points=8, mae_points=-1, exit_reason="TEST_FIXTURE",
        broker_execution_latency_ms=1, gross_result=8, net_known_result=8, commission_known=False,
        profit_to_loss_flip=False, session="LONDON", volatility_state="LOW", data_quality_state="VALID",
        source_git_commit="fixture-commit", execution_mode="DEMO", source_fingerprint="source-exp-1",
    )
    with sqlite3.connect(phase14a) as db:
        db.execute("CREATE TABLE experience_trades(experience_id TEXT PRIMARY KEY,source_fingerprint TEXT,payload_json TEXT,recorded_at TEXT)")
        db.execute("CREATE TABLE experience_quarantine(quarantine_id INTEGER PRIMARY KEY,source_id TEXT,reason TEXT,detail_json TEXT,recorded_at TEXT)")
        db.execute("INSERT INTO experience_trades VALUES(?,?,?,?)", (trade.experience_id, trade.source_fingerprint, json.dumps(trade.to_dict()), start.isoformat()))
        for index in range(3):
            db.execute("INSERT INTO experience_quarantine(source_id,reason,detail_json,recorded_at) VALUES(?,?,?,?)", (str(index), "FIXTURE_QUARANTINE", "{}", start.isoformat()))
    with sqlite3.connect(hft) as db:
        db.execute("CREATE TABLE hft_ticks(run_id TEXT,tick_key TEXT,symbol TEXT,timestamp TEXT,bid REAL,ask REAL,features_json TEXT,executed INTEGER)")
        mid = 1.1
        for index in range(96):
            phase = index % 20
            mid += 0.00003 if phase < 10 else -0.00003
            timestamp = datetime(2026, 1, 1, 10, tzinfo=UTC) + timedelta(seconds=index)
            db.execute(
                "INSERT INTO hft_ticks VALUES(?,?,?,?,?,?,?,?)",
                ("run-1", str(index), "EURUSD", timestamp.isoformat(), mid - 0.00001, mid + 0.00001, json.dumps({"point": 0.00001}), 0),
            )
    with sqlite3.connect(demo) as db:
        db.execute("CREATE TABLE demo_positions(ticket INTEGER,owned INTEGER,payload_json TEXT,execution_mode TEXT,real_money INTEGER,symbol TEXT,direction TEXT,volume REAL,price_open REAL,state TEXT,updated_at TEXT)")
        db.execute("CREATE TABLE demo_exits(ticket INTEGER,payload_json TEXT,observed_at TEXT,realized_pnl REAL)")
    return phase14a, hft, demo


def _paths(phase14a: Path, hft: Path, demo: Path) -> Phase14CSourcePaths:
    return Phase14CSourcePaths(phase14a, hft, demo)


def _fingerprints(paths: Phase14CSourcePaths) -> dict[str, str]:
    return {
        "phase14a": hashlib.sha256(paths.phase14a_path.read_bytes()).hexdigest(),
        "hft": hashlib.sha256(paths.hft_path.read_bytes()).hexdigest(),
        "demo": hashlib.sha256(paths.demo_path.read_bytes()).hexdigest(),
    }


def _fake_report(candidate, artifact_path: Path, paths: Phase14CSourcePaths):
    return SimpleNamespace(
        status="COMPLETED",
        candidate_id=candidate.candidate_id,
        candidate_state=CandidateState.INSUFFICIENT_EVIDENCE.value,
        gate_decision="INSUFFICIENT_EVIDENCE",
        gate_reasons=("fixture sample is intentionally below production floors",),
        source_fingerprints=_fingerprints(paths),
        source_unchanged=True,
        artifact_root=artifact_path,
    )


def test_preflight_reports_real_source_counts_paths_fingerprints_and_unknown_commission(tmp_path):
    paths = _paths(*_source_databases(tmp_path))

    report = preflight_phase14_sources(paths.phase14a_path, paths.hft_path, paths.demo_path)

    assert report.ready is True
    assert report.reason_code == "READY"
    assert report.source_paths == paths
    assert dict(report.source_fingerprints) == _fingerprints(paths)
    assert report.verified_experience_count == 1
    assert report.quarantined_experience_count == 3
    assert report.causal_tick_count == 96
    assert report.causal_segment_count == 1
    assert report.commission_status == "UNKNOWN"


def test_ready_preflight_requires_fingerprints_for_all_three_sources(tmp_path):
    paths = _paths(*_source_databases(tmp_path))
    with pytest.raises(ValueError, match="all source fingerprints"):
        Phase14CReplayPreflight(
            source_paths=paths,
            source_fingerprints={"phase14a": "a" * 64},
            verified_experience_count=1,
            quarantined_experience_count=0,
            causal_tick_count=1,
            causal_segment_count=1,
            commission_status="UNKNOWN",
            ready=True,
            reason_code="READY",
        )


def test_preflight_rejects_phase14a_without_verified_experience(tmp_path):
    paths = _paths(*_source_databases(tmp_path))
    with sqlite3.connect(paths.phase14a_path) as db:
        db.execute("DELETE FROM experience_trades")

    report = preflight_phase14_sources(paths.phase14a_path, paths.hft_path, paths.demo_path)

    assert report.ready is False
    assert report.reason_code == "NO_VERIFIED_PHASE14A_EXPERIENCE"
    assert report.verified_experience_count == 0
    assert report.quarantined_experience_count == 3


def test_preflight_rejects_unresolved_demo_reconciliation(tmp_path):
    paths = _paths(*_source_databases(tmp_path))
    with sqlite3.connect(paths.demo_path) as db:
        db.execute(
            "CREATE TABLE demo_reconciliation(status TEXT,details_json TEXT,observed_at TEXT)"
        )
        db.execute(
            "INSERT INTO demo_reconciliation VALUES(?,?,?)",
            (
                "RECONCILIATION_REQUIRED",
                json.dumps({"ticket": 152717255467, "symbol": "EURUSD"}),
                "2026-10-04T10:00:00+00:00",
            ),
        )

    report = preflight_phase14_sources(paths.phase14a_path, paths.hft_path, paths.demo_path)

    assert report.ready is False
    assert report.reason_code == "DEMO_RECONCILIATION_UNCERTAIN"


def test_preflight_rejects_invalid_causal_hft_rows(tmp_path):
    paths = _paths(*_source_databases(tmp_path))
    with sqlite3.connect(paths.hft_path) as db:
        db.execute("UPDATE hft_ticks SET bid=-1")

    report = preflight_phase14_sources(paths.phase14a_path, paths.hft_path, paths.demo_path)

    assert report.ready is False
    assert report.reason_code == "INVALID_CAUSAL_HFT_DATA"
    assert report.causal_tick_count == 0


def test_candidate_run_is_rejected_when_any_readonly_source_changes(tmp_path, monkeypatch):
    paths = _paths(*_source_databases(tmp_path))

    class MutatingExperiment:
        def __init__(self, artifact_root):
            self.artifact_root = artifact_root

        def run(self, **kwargs):
            Path(kwargs["hft_path"]).write_bytes(Path(kwargs["hft_path"]).read_bytes() + b"mutation")
            return _fake_report(kwargs["candidate"], self.artifact_root, paths)

    monkeypatch.setattr("tradingagents.self_enhancement.phase14c_runner.BookCandidateExperiment", MutatingExperiment)

    (record,) = evaluate_phase14c_candidates(
        (_strategy_spec(),),
        artifact_root=tmp_path / "mutated-artifacts",
        source_paths=paths,
        source_commit="fixture-commit",
    )

    assert record.status == "FAILED"
    assert record.reason_code == "SOURCE_MUTATED_DURING_EVALUATION"
    assert record.candidate_id is not None


def test_successful_candidate_run_preserves_all_source_fingerprints(tmp_path, monkeypatch):
    paths = _paths(*_source_databases(tmp_path))
    before = _fingerprints(paths)
    actual = orchestrator.BookCandidateExperiment

    class CapturingExperiment:
        def __init__(self, artifact_path):
            self._inner = actual(artifact_path)

        def run(self, **kwargs):
            return self._inner.run(**kwargs)

    monkeypatch.setattr("tradingagents.self_enhancement.phase14c_runner.BookCandidateExperiment", CapturingExperiment)

    (record,) = evaluate_phase14c_candidates(
        (_strategy_spec(),),
        artifact_root=tmp_path / "successful-artifacts",
        source_paths=paths,
        source_commit="fixture-commit",
    )

    assert record.status == "COMPLETED"
    assert record.gate_decision in {"INSUFFICIENT_EVIDENCE", "REJECTED"}
    assert record.candidate_state in {
        CandidateState.INSUFFICIENT_EVIDENCE.value,
        CandidateState.REJECTED.value,
    }
    assert dict(record.source_fingerprints) == before
    assert _fingerprints(paths) == before


def test_incomplete_or_unimplemented_specs_never_reach_replay(tmp_path, monkeypatch):
    paths = _paths(*_source_databases(tmp_path))
    calls = []

    class CountingExperiment:
        def __init__(self, artifact_path):
            self.artifact_path = artifact_path

        def run(self, **kwargs):
            calls.append(kwargs)
            return _fake_report(kwargs["candidate"], self.artifact_path, paths)

    monkeypatch.setattr("tradingagents.self_enhancement.phase14c_runner.BookCandidateExperiment", CountingExperiment)
    unsuitable = replace(_strategy_spec(), suitability=StrategySuitability.UNAVAILABLE_DATA)
    incomplete = replace(_strategy_spec(), spec_id="missing-validation", validation_results=())
    conflicted = replace(
        _strategy_spec(),
        spec_id="unsupported-evidence",
        validation_results=tuple(
            RuleValidationResult(item.fingerprint, RuleValidationStatus.UNSUPPORTED, "CONFLICTING_SOURCE_EVIDENCE")
            for item in _strategy_spec().rule_claims
        ),
    )
    unimplemented = _strategy_spec("unknown-family", "UNREVIEWED_FAMILY")

    records = evaluate_phase14c_candidates(
        (unsuitable, incomplete, conflicted, unimplemented),
        artifact_root=tmp_path / "filtered-artifacts",
        source_paths=paths,
        source_commit="fixture-commit",
    )

    assert [record.status for record in records] == ["NOT_RUN"] * 4
    assert {record.reason_code for record in records} == {"SPEC_NOT_EXECUTABLE", "UNIMPLEMENTED_SPEC"}
    assert calls == []


@pytest.mark.parametrize(
    ("requested_max", "expected_runs"),
    ((None, 5), (10, 10)),
)
def test_candidate_execution_respects_default_five_and_absolute_ten_run_caps(
    tmp_path, monkeypatch, requested_max, expected_runs
):
    paths = _paths(*_source_databases(tmp_path))
    calls = []

    class CountingExperiment:
        def __init__(self, artifact_path):
            self.artifact_path = artifact_path

        def run(self, **kwargs):
            calls.append(kwargs["candidate"])
            return _fake_report(kwargs["candidate"], self.artifact_path, paths)

    monkeypatch.setattr("tradingagents.self_enhancement.phase14c_runner.BookCandidateExperiment", CountingExperiment)
    specs = tuple(replace(_strategy_spec(), spec_id=f"spec-{index}") for index in range(12))
    kwargs = {} if requested_max is None else {"max_candidate_runs": requested_max}

    records = evaluate_phase14c_candidates(
        specs,
        artifact_root=tmp_path / f"capped-{requested_max or 'default'}",
        source_paths=paths,
        source_commit="fixture-commit",
        **kwargs,
    )

    assert len(calls) == expected_runs
    assert sum(record.status == "COMPLETED" for record in records) == expected_runs
    assert sum(record.reason_code == "MAX_CANDIDATE_RUNS_REACHED" for record in records) == 12 - expected_runs
    assert all(record.candidate_state != CandidateState.DEMO_CHALLENGER.value for record in records)


def test_candidate_execution_rejects_a_requested_run_cap_above_ten(tmp_path):
    paths = _paths(*_source_databases(tmp_path))

    with pytest.raises(ValueError, match="through 10"):
        evaluate_phase14c_candidates(
            (_strategy_spec(),),
            artifact_root=tmp_path / "over-cap",
            source_paths=paths,
            source_commit="fixture-commit",
            max_candidate_runs=11,
        )


def test_replay_adapter_does_not_import_mt5_and_caps_state_at_shadow_challenger(tmp_path, monkeypatch):
    paths = _paths(*_source_databases(tmp_path))
    original_import = builtins.__import__
    blocked_imports = []

    def guarded_import(name, *args, **kwargs):
        if name == "MetaTrader5" or name.startswith("tradingagents.forex.mt5"):
            blocked_imports.append(name)
            raise AssertionError("offline Phase 14C replay must not import MT5")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    (record,) = evaluate_phase14c_candidates(
        (_strategy_spec(),),
        artifact_root=tmp_path / "offline-artifacts",
        source_paths=paths,
        source_commit="fixture-commit",
    )

    assert blocked_imports == []
    assert record.status == "COMPLETED"
    assert record.candidate_state in {
        CandidateState.INSUFFICIENT_EVIDENCE.value,
        CandidateState.REJECTED.value,
        CandidateState.SHADOW_CHALLENGER.value,
    }
    assert record.candidate_state != CandidateState.DEMO_CHALLENGER.value


def test_candidate_evaluation_record_cannot_claim_state_above_shadow_ceiling(tmp_path):
    with pytest.raises(ValueError, match="shadow ceiling"):
        CandidateEvaluationRecord(
            spec_id="spec-1",
            candidate_id="candidate-1",
            status=Phase14CCandidateStatus.COMPLETED,
            reason_code=Phase14CCandidateReason.REPLAY_COMPLETED,
            artifact_path=tmp_path / "artifact",
            candidate_state=CandidateState.DEMO_CHALLENGER.value,
        )
