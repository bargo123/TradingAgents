"""One-shot asynchronous Phase 14 experiment orchestrator."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .candidates import CandidateGenerator
from .causal import CausalDatasetError, load_causal_tick_dataset
from .evaluation import cost_sensitivity_segments, walk_forward_segments
from .importer import ImportReport, import_verified_demo
from .models import (
    CandidateSpec,
    CandidateState,
    ExitPolicyConfig,
    ExperienceEvidenceTier,
    ExperienceTrade,
    StrategyVersion,
)
from .promotion import CandidatePromotionGate, ReplayGateReports
from .replay import ReplayError, ReplayEvaluator
from .store import SelfEnhancementStore


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    experiment_id: str
    status: str
    accepted_experience: int
    quarantined_experience: int
    candidate_count: int
    decisions: tuple[dict[str, Any], ...]
    llm_calls: int = 0
    mt5_calls: int = 0
    source_unchanged: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BookCandidateExperimentReport:
    experiment_id: str
    status: str
    candidate_id: str
    candidate_state: str
    verified_experience_copied: int
    quarantined_experience_excluded: int
    tick_count: int
    segment_count: int
    dataset_fingerprint: str
    incumbent_stage_reports: dict[str, dict[str, Any]]
    stage_reports: dict[str, dict[str, Any]]
    cost_reports: dict[float, dict[str, Any]]
    gate_decision: str
    gate_reasons: tuple[str, ...]
    promotion_count: int
    commission_known: bool
    llm_calls: int
    mt5_calls: int
    source_fingerprints: dict[str, str]
    source_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _source_file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_verified_phase14a(path: Path) -> tuple[tuple[ExperienceTrade, ...], int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "experience_trades" not in tables:
            raise ValueError("Phase 14A source is missing experience_trades")
        quarantined = (
            int(connection.execute("SELECT COUNT(*) FROM experience_quarantine").fetchone()[0])
            if "experience_quarantine" in tables
            else 0
        )
        rows = connection.execute(
            "SELECT experience_id,source_fingerprint,payload_json FROM experience_trades ORDER BY recorded_at,experience_id"
        ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError("Phase 14A experience catalog could not be read") from exc
    finally:
        connection.close()

    trades: list[ExperienceTrade] = []
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
            if not isinstance(payload, dict):
                raise ValueError("experience payload must be an object")
            for name in ("entry_timestamp", "exit_timestamp"):
                value = payload.get(name)
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be an ISO timestamp")
                payload[name] = datetime.fromisoformat(value.replace("Z", "+00:00"))
            trade = ExperienceTrade(**payload)
            if trade.experience_id != str(row["experience_id"]):
                raise ValueError("experience row identity does not match payload")
            if trade.evidence_tier not in {ExperienceEvidenceTier.VERIFIED_EXECUTION, ExperienceEvidenceTier.FULLY_VERIFIED}:
                continue
            stored_fingerprint = row["source_fingerprint"]
            if stored_fingerprint and trade.source_fingerprint and str(stored_fingerprint) != trade.source_fingerprint:
                raise ValueError("experience source fingerprint does not match payload")
            trades.append(trade)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("Phase 14A contains an invalid verified experience row") from exc
    return tuple(trades), quarantined


def _validate_demo_source_readonly(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"demo_positions", "demo_exits"}.issubset(tables):
            raise ValueError("DEMO source is missing its position/exit ledger tables")
        if "demo_reconciliation" in tables:
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(demo_reconciliation)")}
            if "status" in columns:
                resolved_statuses = {"RECONCILED", "CLEAN", "OK"}
                if not {"details_json", "observed_at"}.issubset(columns):
                    statuses = {
                        str(row[0] or "").upper()
                        for row in connection.execute("SELECT status FROM demo_reconciliation")
                    }
                    if statuses - resolved_statuses:
                        raise ValueError("DEMO source has unresolved reconciliation evidence")
                    return

                latest_by_identity: dict[tuple[str, int], tuple[datetime, str]] = {}
                unscoped_unresolved = False
                for event in connection.execute(
                    "SELECT status,details_json,observed_at FROM demo_reconciliation"
                ):
                    status = str(event[0] or "").upper()
                    try:
                        details = json.loads(str(event[1] or "{}"))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        if status not in resolved_statuses:
                            unscoped_unresolved = True
                        continue
                    if not isinstance(details, dict):
                        if status not in resolved_statuses:
                            unscoped_unresolved = True
                        continue
                    raw_ticket = details.get("ticket")
                    raw_symbol = details.get("symbol")
                    if (
                        isinstance(raw_ticket, bool)
                        or not isinstance(raw_ticket, int)
                        or raw_ticket <= 0
                        or not isinstance(raw_symbol, str)
                        or not raw_symbol.strip()
                    ):
                        if status not in resolved_statuses:
                            unscoped_unresolved = True
                        continue
                    raw_timestamp = event[2]
                    try:
                        timestamp = datetime.fromisoformat(
                            str(raw_timestamp).replace("Z", "+00:00")
                        )
                    except (TypeError, ValueError):
                        raise ValueError(
                            "DEMO source has invalid reconciliation chronology"
                        ) from None
                    if timestamp.tzinfo is None:
                        raise ValueError(
                            "DEMO source has invalid reconciliation chronology"
                        )
                    timestamp = timestamp.astimezone(timezone.utc)
                    identity = (raw_symbol.strip().upper(), raw_ticket)
                    previous = latest_by_identity.get(identity)
                    if previous is None or timestamp > previous[0]:
                        latest_by_identity[identity] = (timestamp, status)
                    elif timestamp == previous[0] and status != previous[1]:
                        raise ValueError(
                            "DEMO source has ambiguous reconciliation chronology"
                        )
                if unscoped_unresolved or any(
                    status not in resolved_statuses
                    for _timestamp, status in latest_by_identity.values()
                ):
                    raise ValueError("DEMO source has unresolved reconciliation evidence")
    except sqlite3.Error as exc:
        raise ValueError("DEMO source could not be read") from exc
    finally:
        connection.close()


class BookCandidateExperiment:
    """Replay one validated book candidate in a fresh, isolated Phase 14B catalog."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root).expanduser().resolve()

    def run(
        self,
        *,
        phase14a_path: str | Path,
        hft_path: str | Path,
        demo_path: str | Path,
        candidate: CandidateSpec,
        source_commit: str,
        symbol: str = "EURUSD",
        cost_scenarios: tuple[float, ...] = (0.0, 1.0),
    ) -> BookCandidateExperimentReport:
        if self.artifact_root.exists():
            raise FileExistsError("Phase 14B artifact root must be fresh; existing data is preserved")
        if not isinstance(candidate, CandidateSpec) or candidate.strategy_spec is None:
            raise ValueError("book experiment requires a validated StrategySpec candidate")
        if candidate.state is not CandidateState.EXTRACTED:
            raise ValueError("book experiment requires an unpromoted EXTRACTED candidate")
        if not str(source_commit).strip():
            raise ValueError("source_commit is required")
        phase14a = Path(phase14a_path).expanduser().resolve()
        hft = Path(hft_path).expanduser().resolve()
        demo = Path(demo_path).expanduser().resolve()
        for path in (phase14a, hft, demo):
            if not path.is_file():
                raise FileNotFoundError(path)
        sources = {"phase14a": phase14a, "hft": hft, "demo": demo}
        before = {name: _source_file_fingerprint(path) for name, path in sources.items()}
        verified_experience, quarantined = _read_verified_phase14a(phase14a)
        if not verified_experience:
            raise ValueError("Phase 14A has no verified execution experience to copy")
        _validate_demo_source_readonly(demo)
        try:
            dataset = load_causal_tick_dataset(hft, symbol=symbol)
        except (ReplayError, CausalDatasetError) as exc:
            raise ValueError("HFT causal source is not replayable") from exc
        if dataset.valid_rows == 0 or dataset.invalid_rows:
            raise ValueError("HFT causal source is empty or contains invalid rows")

        experiment_id = "exp14b-" + uuid.uuid4().hex[:24]
        parent = candidate.parent
        incumbent = CandidateSpec(
            candidate_id=f"incumbent-{experiment_id}",
            parent=parent,
            strategy_id=parent.strategy_id,
            exit_policy=candidate.exit_policy,
            hypothesis="Existing Phase 14 strategy control; unchanged and replay-only.",
            state=CandidateState.EXPERIMENTAL,
        )
        evaluator = ReplayEvaluator()
        baseline_walk = walk_forward_segments(evaluator, dataset.segments, incumbent)
        candidate_walk = walk_forward_segments(evaluator, dataset.segments, candidate)
        cost_report = cost_sensitivity_segments(evaluator, dataset.segments, candidate, cost_scenarios)
        incumbent_stages = {stage: report.to_dict() for stage, report in baseline_walk.stage_reports.items()}
        candidate_stages = {stage: report.to_dict() for stage, report in candidate_walk.stage_reports.items()}
        candidate_costs = {scenario: report.to_dict() for scenario, report in cost_report.metrics_by_scenario.items()}
        gate = CandidatePromotionGate()
        gate_result = gate.evaluate(
            baseline_walk.stage_reports["VALIDATION"].to_dict(),
            candidate,
            ReplayGateReports(candidate_stages, candidate_costs),
        )
        after = {name: _source_file_fingerprint(path) for name, path in sources.items()}
        unchanged = before == after
        if not unchanged:
            raise ReplayError("a read-only research source changed during the experiment")

        self.artifact_root.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(exist_ok=False)
        store = SelfEnhancementStore(self.artifact_root / "phase14.sqlite3")
        store.initialize()
        for trade in verified_experience:
            store.record_experience(trade)
        store.create_experiment(
            experiment_id,
            parent_version=parent.strategy_version,
            dataset_fingerprint=dataset.source_fingerprint,
            metadata={
                "candidate_kind": "BOOK_STRATEGY_SPEC",
                "source_commit": str(source_commit),
                "source_fingerprints": before,
                "tick_count": dataset.valid_rows,
                "segment_count": len(dataset.segments),
                "incumbent_stage_reports": incumbent_stages,
                "candidate_stage_reports": candidate_stages,
                "commission_status": "UNKNOWN",
                "llm_calls": 0,
                "mt5_calls": 0,
            },
        )
        store.record_candidate(experiment_id, candidate)
        store.transition_candidate(experiment_id, candidate.candidate_id, CandidateState.REPLAYED, "causal segmented replay completed")
        for stage, metrics in candidate_stages.items():
            store.record_evaluation(experiment_id, candidate.candidate_id, stage, metrics, passed=True)
        for scenario, metrics in candidate_costs.items():
            store.record_evaluation(experiment_id, candidate.candidate_id, f"COST_{scenario:g}", metrics, passed=True)
        promotion_count = 0
        if gate_result.can_promote:
            store.transition_candidate(experiment_id, candidate.candidate_id, CandidateState.VALIDATED, "existing Phase 14 gates passed")
            store.transition_candidate(experiment_id, candidate.candidate_id, CandidateState.UNSEEN_PASSED, "frozen unseen holdout passed")
            store.record_promotion(
                experiment_id,
                candidate.candidate_id,
                decision="SHADOW_CHALLENGER",
                reason="existing deterministic Phase 14 promotion gate passed",
                rollback_package={
                    "previous_version": f"{parent.strategy_id}:{parent.strategy_version}",
                    "parent_config_hash": parent.config_hash,
                },
            )
            state = CandidateState.SHADOW_CHALLENGER.value
            promotion_count = 1
        else:
            state = (
                CandidateState.INSUFFICIENT_EVIDENCE.value
                if gate_result.decision == "INSUFFICIENT_EVIDENCE"
                else CandidateState.REJECTED.value
            )
            store.transition_candidate(experiment_id, candidate.candidate_id, state, ";".join(gate_result.reasons))
        store.update_experiment_status(
            experiment_id,
            "COMPLETED",
            metadata={"gate_decision": gate_result.decision, "gate_reasons": gate_result.reasons, "candidate_state": state},
        )
        return BookCandidateExperimentReport(
            experiment_id=experiment_id,
            status="COMPLETED",
            candidate_id=candidate.candidate_id,
            candidate_state=state,
            verified_experience_copied=len(verified_experience),
            quarantined_experience_excluded=quarantined,
            tick_count=dataset.valid_rows,
            segment_count=len(dataset.segments),
            dataset_fingerprint=dataset.source_fingerprint,
            incumbent_stage_reports=incumbent_stages,
            stage_reports=candidate_stages,
            cost_reports=candidate_costs,
            gate_decision=gate_result.decision,
            gate_reasons=gate_result.reasons,
            promotion_count=promotion_count,
            commission_known=False,
            llm_calls=0,
            mt5_calls=0,
            source_fingerprints=before,
            source_unchanged=unchanged,
        )


class SelfEnhancementOrchestrator:
    """Run bounded research exactly once; it never owns or constructs MT5."""

    def __init__(self, artifact_root: str | Path):
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.store = SelfEnhancementStore(self.artifact_root / "phase14.sqlite3")

    def run_once(
        self,
        *,
        hft_path: str | Path,
        demo_path: str | Path,
        source_commit: str,
        symbol: str = "EURUSD",
        minimum_verified_trades: int = 20,
        minimum_ticks: int = 100,
        max_experiments_per_dataset: int = 3,
    ) -> ExperimentReport:
        if minimum_verified_trades <= 0 or minimum_ticks <= 0 or max_experiments_per_dataset <= 0:
            raise ValueError("minimum evidence thresholds and experiment budget must be positive")
        self.store.initialize()
        self.store.recover_incomplete_experiments()
        imported: ImportReport = import_verified_demo(hft_path, demo_path, self.store)
        data_quality_error: str | None = None
        try:
            causal_dataset = load_causal_tick_dataset(str(hft_path), symbol=symbol)
            ticks = causal_dataset.ticks
            dataset_fingerprint = causal_dataset.source_fingerprint
        except (ReplayError, CausalDatasetError) as exc:
            causal_dataset = None
            ticks = ()
            data_quality_error = f"{type(exc).__name__}:{exc}"
            dataset_fingerprint = str(imported.quality.get("source_fingerprint", "UNKNOWN"))
        experiment_id = "exp14-" + uuid.uuid4().hex[:24]
        dataset_selection_count_before = self.store.dataset_experiment_count(dataset_fingerprint)
        self.store.create_experiment(
            experiment_id,
            parent_version="incumbent-v1",
            dataset_fingerprint=dataset_fingerprint,
            metadata={
                "source_commit": source_commit,
                "source_quality": imported.quality,
                "causal_quality": None if causal_dataset is None else causal_dataset.to_dict(),
                "minimum_verified_trades": minimum_verified_trades,
                "minimum_ticks": minimum_ticks,
                "accepted_experience": imported.accepted,
                "llm_calls": 0,
                "mt5_calls": 0,
                "reconciliation_uncertainty": imported.reconciliation_uncertainty,
                "dataset_selection_count_before": dataset_selection_count_before,
                "max_experiments_per_dataset": max_experiments_per_dataset,
            },
        )
        if data_quality_error is not None:
            self.store.quarantine("hft_ticks", "REPLAY_INPUT_INVALID", {"error": data_quality_error})
            self.store.update_experiment_status(experiment_id, "NO_EXPERIMENT", metadata={"reason": data_quality_error})
            return ExperimentReport(experiment_id, "NO_EXPERIMENT", imported.accepted, imported.quarantined, 0, (), source_unchanged=imported.source_unchanged)
        if dataset_selection_count_before >= max_experiments_per_dataset:
            reason = "dataset experiment budget exhausted"
            self.store.update_experiment_status(experiment_id, "NO_EXPERIMENT", metadata={"reason": reason})
            return ExperimentReport(experiment_id, "NO_EXPERIMENT", imported.accepted, imported.quarantined, 0, (), source_unchanged=imported.source_unchanged)
        if imported.reconciliation_uncertainty:
            reason = "source contains unresolved DEMO reconciliation uncertainty"
            self.store.update_experiment_status(experiment_id, "NO_EXPERIMENT", metadata={"reason": reason})
            return ExperimentReport(experiment_id, "NO_EXPERIMENT", imported.accepted, imported.quarantined, 0, (), source_unchanged=imported.source_unchanged)
        if causal_dataset is None or causal_dataset.valid_rows == 0 or causal_dataset.invalid_rows:
            reason = "causal dataset is incomplete or has invalid rows"
            self.store.update_experiment_status(experiment_id, "NO_EXPERIMENT", metadata={"reason": reason})
            return ExperimentReport(experiment_id, "NO_EXPERIMENT", imported.accepted, imported.quarantined, 0, (), source_unchanged=imported.source_unchanged)
        if imported.accepted < minimum_verified_trades:
            reason = f"verified experience {imported.accepted} < minimum {minimum_verified_trades}"
            self.store.update_experiment_status(experiment_id, "NO_EXPERIMENT", metadata={"reason": reason})
            return ExperimentReport(experiment_id, "NO_EXPERIMENT", imported.accepted, imported.quarantined, 0, (), source_unchanged=imported.source_unchanged)
        if len(ticks) < minimum_ticks:
            reason = f"ticks {len(ticks)} < minimum {minimum_ticks}"
            self.store.update_experiment_status(experiment_id, "NO_EXPERIMENT", metadata={"reason": reason})
            return ExperimentReport(experiment_id, "NO_EXPERIMENT", imported.accepted, imported.quarantined, 0, (), source_unchanged=imported.source_unchanged)

        parent = StrategyVersion(
            "range_rejection",
            "incumbent-v1",
            "config-v1",
            ExitPolicyConfig().to_dict(),
            source_commit,
        )
        candidates = CandidateGenerator().generate(parent)
        for candidate in candidates:
            self.store.record_candidate(experiment_id, candidate)
        evaluator = ReplayEvaluator()
        baseline_report = walk_forward_segments(evaluator, causal_dataset.segments, candidates[0])
        baseline_metrics = baseline_report.stage_reports["VALIDATION"].to_dict()
        gate = CandidatePromotionGate(minimum_trades=minimum_verified_trades)
        decisions: list[dict[str, Any]] = []
        for candidate, is_baseline in zip(candidates, (True, *([False] * (len(candidates) - 1))), strict=True):
            self.store.transition_candidate(
                experiment_id,
                candidate.candidate_id,
                CandidateState.REPLAYED,
                "causal replay completed",
            )
            walk = baseline_report if is_baseline else walk_forward_segments(evaluator, causal_dataset.segments, candidate)
            cost = cost_sensitivity_segments(evaluator, causal_dataset.segments, candidate, (0.0, 1.0))
            for stage, metrics in walk.stage_reports.items():
                self.store.record_evaluation(experiment_id, candidate.candidate_id, stage, metrics.to_dict(), passed=True)
            for scenario, metrics in cost.metrics_by_scenario.items():
                self.store.record_evaluation(experiment_id, candidate.candidate_id, f"COST_{scenario}", metrics.to_dict(), passed=True)
            if is_baseline:
                self.store.transition_candidate(
                    experiment_id,
                    candidate.candidate_id,
                    CandidateState.INCUMBENT,
                    "incumbent control retained; no live mutation",
                )
                decision = {"candidate_id": candidate.candidate_id, "decision": "INCUMBENT_CONTROL", "reasons": ()}
            else:
                gate_result = gate.evaluate(
                    baseline_metrics,
                    candidate,
                    ReplayGateReports(
                        {stage: metrics.to_dict() for stage, metrics in walk.stage_reports.items()},
                        {scenario: metrics.to_dict() for scenario, metrics in cost.metrics_by_scenario.items()},
                    ),
                )
                if gate_result.can_promote:
                    self.store.transition_candidate(
                        experiment_id,
                        candidate.candidate_id,
                        CandidateState.VALIDATED,
                        "validation and cost gates passed",
                    )
                    self.store.transition_candidate(
                        experiment_id,
                        candidate.candidate_id,
                        CandidateState.UNSEEN_PASSED,
                        "unseen holdout gate passed",
                    )
                    self.store.record_promotion(
                        experiment_id,
                        candidate.candidate_id,
                        decision="SHADOW_CHALLENGER",
                        reason="all deterministic Phase 14 gates passed",
                        rollback_package={"previous_version": parent.strategy_version, "config_hash": parent.config_hash},
                    )
                else:
                    self.store.record_promotion(
                        experiment_id,
                        candidate.candidate_id,
                        decision="REJECTED",
                        reason=";".join(gate_result.reasons),
                        rollback_package={"previous_version": parent.strategy_version, "config_hash": parent.config_hash},
                    )
                decision = {"candidate_id": candidate.candidate_id, "decision": gate_result.decision, "reasons": gate_result.reasons}
            decisions.append(decision)
        self.store.update_experiment_status(experiment_id, "COMPLETED", metadata={"decisions": decisions})
        return ExperimentReport(experiment_id, "COMPLETED", imported.accepted, imported.quarantined, len(candidates), tuple(decisions), source_unchanged=imported.source_unchanged)


__all__ = ["BookCandidateExperiment", "BookCandidateExperimentReport", "ExperimentReport", "SelfEnhancementOrchestrator"]
