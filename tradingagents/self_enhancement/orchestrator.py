"""One-shot asynchronous Phase 14 experiment orchestrator."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidates import CandidateGenerator
from .evaluation import cost_sensitivity, walk_forward
from .importer import ImportReport, import_verified_demo
from .models import CandidateState, ExitPolicyConfig, StrategyVersion
from .promotion import CandidatePromotionGate, ReplayGateReports
from .replay import ReplayError, ReplayEvaluator, load_hft_ticks
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
            ticks = load_hft_ticks(str(hft_path), symbol=symbol)
            dataset_fingerprint = hashlib.sha256(
                json.dumps(
                    [(tick.timestamp.isoformat(), tick.bid, tick.ask) for tick in ticks],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        except ReplayError as exc:
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
        if imported.quality.get("quality_status") != "VALID":
            reason = "source quality warnings block autonomous promotion"
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
        baseline_report = walk_forward(evaluator, ticks, candidates[0])
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
            walk = baseline_report if is_baseline else walk_forward(evaluator, ticks, candidate)
            cost = cost_sensitivity(evaluator, ticks, candidate, (0.0, 1.0))
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


__all__ = ["ExperimentReport", "SelfEnhancementOrchestrator"]
