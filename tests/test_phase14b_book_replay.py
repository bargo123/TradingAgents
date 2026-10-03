from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.forex.hft.models import Tick
from tradingagents.self_enhancement import orchestrator
from tradingagents.self_enhancement.book_factory import BookStrategyFactory
from tradingagents.self_enhancement.causal import CausalSegment, CausalTick
from tradingagents.self_enhancement.evaluation import walk_forward_segments
from tradingagents.self_enhancement.models import (
    CandidateSpec,
    CandidateState,
    ExitPolicyConfig,
    ExperienceTrade,
    StrategyVersion,
)
from tradingagents.self_enhancement.promotion import PromotionDecision
from tradingagents.self_enhancement.replay import ReplayError, ReplayEvaluator
from tradingagents.self_enhancement.store import SelfEnhancementStore
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


def _strategy_spec() -> StrategySpec:
    values = {
        RuleStage.ENTRY: (RuleOperator.GREATER_THAN, RuleDirection.LONG, 2, "points", "after confirmation", None,
                          "LONG when momentum > 2 points after confirmation"),
        RuleStage.CONFIRMATION: (RuleOperator.GREATER_OR_EQUAL, RuleDirection.LONG, 0.6, "fraction", "after three ticks", None,
                                 "CONFIRMATION: LONG when direction_persistence >= 0.6 fraction after three ticks"),
        RuleStage.INVALIDATION: (RuleOperator.LESS_OR_EQUAL, RuleDirection.LONG, 0, "points", "after reversal", None,
                                 "INVALIDATION: LONG when momentum <= 0 points after reversal"),
        RuleStage.EXPECTED_MOVE: (RuleOperator.GREATER_OR_EQUAL, RuleDirection.LONG, 5, "points", "after entry", 30,
                                  "EXPECTED_MOVE: LONG target 5 points within 30 seconds after entry"),
        RuleStage.EXIT: (RuleOperator.LESS_OR_EQUAL, RuleDirection.LONG, 0, "points", "after reversal", None,
                         "EXIT: LONG when momentum <= 0 points after reversal"),
        RuleStage.PROFIT_PROTECTION: (RuleOperator.LESS_OR_EQUAL, RuleDirection.LONG, 0, "points", "after target retracement", None,
                                      "PROFIT_PROTECTION: LONG when momentum <= 0 points after target retracement"),
        RuleStage.STOP_BEHAVIOR: (RuleOperator.LESS_OR_EQUAL, RuleDirection.LONG, -3, "points", "after adverse move", None,
                                  "STOP_BEHAVIOR: LONG when momentum <= -3 points after adverse move"),
        RuleStage.HORIZON: (RuleOperator.LESS_OR_EQUAL, RuleDirection.BOTH, 30, "seconds", "after entry", 30,
                            "HORIZON: hold no longer than 30 seconds after entry"),
    }
    claims = []
    for stage in EXECUTABLE_REQUIRED_STAGES:
        operator, direction, value, unit, condition, horizon, quote = values[stage]
        claims.append(
            StrategyRuleClaim(
                stage,
                operator,
                direction,
                value,
                unit,
                condition,
                horizon,
                RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
                EvidenceSpan("gen-1", "doc-1", f"chunk-{stage.value}", "a" * 64, 0, len(quote), quote),
            )
        )
    validations = tuple(
        RuleValidationResult(item.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE")
        for item in claims
    )
    return StrategySpec(
        "spec-1", "Book-supported momentum", "MOMENTUM_CONTINUATION", ("momentum", "direction_persistence"),
        tuple(claims), "gen-1", "b" * 64, "ollama-local", "qwen3.5:2b", "phase14b-draft-v1",
        "strategy-spec-v1", datetime(2026, 10, 3, tzinfo=UTC), 0.8, StrategySuitability.HFT_SUITABLE,
        validations,
    )


def _parent() -> StrategyVersion:
    return StrategyVersion("range_rejection", "incumbent-v1", "cfg-v1", ExitPolicyConfig().to_dict(), "commit-1")


def _candidate(spec: StrategySpec | None = None) -> CandidateSpec:
    return BookStrategyFactory().from_validated_spec(spec or _strategy_spec(), parent=_parent())


def _ticks(count: int = 72, *, offset_seconds: int = 0) -> tuple[Tick, ...]:
    start = datetime(2026, 1, 1, 10, tzinfo=UTC) + timedelta(seconds=offset_seconds)
    mid = 1.1
    values = []
    for index in range(count):
        phase = index % 20
        mid += 0.00003 if phase < 10 else -0.00003
        values.append(Tick("EURUSD", start + timedelta(seconds=index), mid - 0.00001, mid + 0.00001, 0.00001, index))
    return tuple(values)


def _experience() -> ExperienceTrade:
    start = datetime(2026, 1, 1, 9, tzinfo=UTC)
    return ExperienceTrade(
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


def _source_databases(tmp_path):
    phase14a = tmp_path / "phase14a.sqlite3"
    trade = _experience()
    with sqlite3.connect(phase14a) as db:
        db.execute("CREATE TABLE experience_trades(experience_id TEXT PRIMARY KEY,source_fingerprint TEXT,payload_json TEXT,recorded_at TEXT)")
        db.execute("CREATE TABLE experience_quarantine(quarantine_id INTEGER PRIMARY KEY,source_id TEXT,reason TEXT,detail_json TEXT,recorded_at TEXT)")
        db.execute("INSERT INTO experience_trades VALUES(?,?,?,?)", (trade.experience_id, trade.source_fingerprint, json.dumps(trade.to_dict()), "2026-01-01T00:00:00+00:00"))
        for index in range(3):
            db.execute("INSERT INTO experience_quarantine(source_id,reason,detail_json,recorded_at) VALUES(?,?,?,?)", (str(index), "FIXTURE_QUARANTINE", "{}", "2026-01-01T00:00:00+00:00"))
    hft = tmp_path / "hft.sqlite3"
    with sqlite3.connect(hft) as db:
        db.execute("CREATE TABLE hft_ticks(run_id TEXT,tick_key TEXT,symbol TEXT,timestamp TEXT,bid REAL,ask REAL,features_json TEXT,executed INTEGER)")
        for tick in _ticks(96):
            db.execute("INSERT INTO hft_ticks VALUES(?,?,?,?,?,?,?,?)", ("run-1", str(tick.sequence), tick.symbol, tick.timestamp.isoformat(), tick.bid, tick.ask, json.dumps({"point": tick.point}), 0))
    demo = tmp_path / "demo.sqlite3"
    with sqlite3.connect(demo) as db:
        db.execute("CREATE TABLE demo_positions(ticket INTEGER,owned INTEGER,payload_json TEXT,execution_mode TEXT,real_money INTEGER,symbol TEXT,direction TEXT,volume REAL,price_open REAL,state TEXT,updated_at TEXT)")
        db.execute("CREATE TABLE demo_exits(ticket INTEGER,payload_json TEXT,observed_at TEXT,realized_pnl REAL)")
    return phase14a, hft, demo


def _fingerprint(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validated_candidate_retains_the_exact_serialized_strategy_spec() -> None:
    spec = _strategy_spec()

    candidate = _candidate(spec)

    assert StrategySpec.from_dict(candidate.strategy_spec).content_hash == spec.content_hash
    assert candidate.strategy_id.startswith("book-spec-")


def test_serialized_strategy_spec_round_trip_checks_hash_and_unknown_fields() -> None:
    spec = _strategy_spec()

    restored = StrategySpec.from_dict(spec.to_dict())
    altered = spec.to_dict()
    altered["family"] = "RANGE_REJECTION"
    unknown = spec.to_dict()
    unknown["model_reasoning"] = "must not be persisted"

    assert restored.content_hash == spec.content_hash
    with pytest.raises(ValueError, match="content_hash"):
        StrategySpec.from_dict(altered)
    with pytest.raises(ValueError, match="unknown"):
        StrategySpec.from_dict(unknown)


def test_candidate_store_persists_strategy_spec_without_schema_migration(tmp_path) -> None:
    candidate = _candidate()
    store = SelfEnhancementStore(tmp_path / "phase14.sqlite3")
    store.initialize()
    store.create_experiment("exp-1", parent_version="incumbent-v1", dataset_fingerprint="f" * 64)

    store.record_candidate("exp-1", candidate)

    with sqlite3.connect(store.path) as db:
        payload = json.loads(db.execute("SELECT payload_json FROM candidates").fetchone()[0])
        columns = {row[1] for row in db.execute("PRAGMA table_info(candidates)")}
    assert payload["strategy_spec"]["content_hash"] == candidate.strategy_spec["content_hash"]
    assert "strategy_spec" not in columns


def test_book_replay_uses_validated_registry_and_reports_complete_shadow_metrics() -> None:
    candidate = _candidate()

    result = ReplayEvaluator().evaluate(_ticks(), candidate)

    assert result.ticks_processed == 72
    assert result.trades > 0
    assert sum(result.exit_reasons.values()) == result.trades
    assert result.commission_known is False
    assert result.real_money is False
    assert result.execution_mode == "REPLAY"


def test_invalid_or_tampered_book_spec_fails_closed_in_replay() -> None:
    spec_payload = _strategy_spec().to_dict()
    spec_payload["family"] = "ARBITRARY_GENERATED_STRATEGY"
    candidate = CandidateSpec(
        "book-spec-tampered", _parent(), "book-spec-tampered", ExitPolicyConfig(), "tampered spec",
        strategy_spec=spec_payload,
    )

    with pytest.raises(ReplayError, match="StrategySpec"):
        ReplayEvaluator().evaluate(_ticks(), candidate)


def test_segmented_book_replay_resets_candidate_state_and_keeps_chronological_stages() -> None:
    segments = []
    for index in range(18):
        ticks = _ticks(offset_seconds=index * 600)
        causal_ticks = tuple(
            CausalTick("run", str(tick.sequence), position, tick, {})
            for position, tick in enumerate(ticks)
        )
        segments.append(CausalSegment(f"seg-{index}", "run", causal_ticks))

    report = walk_forward_segments(ReplayEvaluator(), segments, _candidate())

    assert len(report.window_reports) == 18
    assert all(set(window) == {"DEVELOPMENT", "VALIDATION", "UNSEEN_HOLDOUT"} for window in report.window_reports)
    assert set(report.stage_reports) == {"DEVELOPMENT", "VALIDATION", "UNSEEN_HOLDOUT"}
    assert all(item.commission_known is False for item in report.stage_reports.values())


def test_book_experiment_is_a_separate_entrypoint_and_does_not_change_legacy_run_once() -> None:
    assert hasattr(orchestrator, "BookCandidateExperiment")
    assert callable(getattr(orchestrator.BookCandidateExperiment, "run", None))
    assert callable(getattr(orchestrator.SelfEnhancementOrchestrator, "run_once", None))


def test_book_experiment_copies_verified_experience_and_reads_all_sources_read_only(tmp_path) -> None:
    phase14a, hft, demo = _source_databases(tmp_path)
    fingerprints = {path: _fingerprint(path) for path in (phase14a, hft, demo)}
    root = tmp_path / "phase14b-new"

    report = orchestrator.BookCandidateExperiment(root).run(
        phase14a_path=phase14a,
        hft_path=hft,
        demo_path=demo,
        candidate=_candidate(),
        source_commit="phase14a-source-commit",
    )

    snapshot = SelfEnhancementStore(root / "phase14.sqlite3").snapshot()
    assert report.status == "COMPLETED"
    assert report.verified_experience_copied == 1
    assert report.quarantined_experience_excluded == 3
    assert report.commission_known is False
    assert set(report.incumbent_stage_reports) == {"DEVELOPMENT", "VALIDATION", "UNSEEN_HOLDOUT"}
    assert report.gate_decision == "INSUFFICIENT_EVIDENCE"
    assert report.promotion_count == 0
    assert snapshot["experience"] == 1
    assert snapshot["promotions"] == 0
    assert all(_fingerprint(path) == fingerprints[path] for path in fingerprints)


def test_book_experiment_refuses_a_preexisting_artifact_root_without_overwrite(tmp_path) -> None:
    phase14a, hft, demo = _source_databases(tmp_path)
    root = tmp_path / "existing"
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="fresh"):
        orchestrator.BookCandidateExperiment(root).run(
            phase14a_path=phase14a, hft_path=hft, demo_path=demo, candidate=_candidate(), source_commit="commit"
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_book_experiment_cannot_replay_or_promote_a_previously_rejected_candidate(tmp_path) -> None:
    phase14a, hft, demo = _source_databases(tmp_path)
    root = tmp_path / "rejected-run"
    rejected = _candidate().with_state(CandidateState.REJECTED)

    with pytest.raises(ValueError, match="unpromoted EXTRACTED"):
        orchestrator.BookCandidateExperiment(root).run(
            phase14a_path=phase14a, hft_path=hft, demo_path=demo, candidate=rejected, source_commit="commit"
        )

    assert not root.exists()


def test_qualified_book_candidate_is_capped_at_shadow_challenger(monkeypatch, tmp_path) -> None:
    phase14a, hft, demo = _source_databases(tmp_path)
    candidate = _candidate()
    monkeypatch.setattr(
        orchestrator.CandidatePromotionGate,
        "evaluate",
        lambda self, incumbent, actual_candidate, reports: PromotionDecision(
            "PROMOTE_TO_SHADOW", True, actual_candidate.candidate_id, ()
        ),
    )

    report = orchestrator.BookCandidateExperiment(tmp_path / "qualified").run(
        phase14a_path=phase14a, hft_path=hft, demo_path=demo, candidate=candidate, source_commit="commit"
    )

    with sqlite3.connect(tmp_path / "qualified" / "phase14.sqlite3") as db:
        decisions = [row[0] for row in db.execute("SELECT decision FROM promotions")]
        final_state = db.execute("SELECT state FROM candidates").fetchone()[0]
    assert report.promotion_count == 1
    assert report.candidate_state == CandidateState.SHADOW_CHALLENGER.value
    assert decisions == ["SHADOW_CHALLENGER"]
    assert final_state == CandidateState.SHADOW_CHALLENGER.value
