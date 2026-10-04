"""Read-only preflight and bounded offline replay adapter for Phase 14C."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from .book_factory import BookStrategyFactory
from .book_strategies import BookStrategyRegistry
from .causal import CausalDatasetError, load_causal_tick_dataset
from .models import CandidateState, ExitPolicyConfig, StrategyVersion
from .orchestrator import (
    BookCandidateExperiment,
    _read_verified_phase14a,
    _source_file_fingerprint,
    _validate_demo_source_readonly,
)
from .phase14c_models import (
    CandidateEvaluationRecord,
    Phase14CCandidateReason,
    Phase14CCandidateStatus,
    Phase14CPreflightReason,
    Phase14CReplayPreflight,
    Phase14CSourcePaths,
)
from .replay import ReplayError
from .strategy_specs import StrategySpec

_SHADOW_STATES = frozenset(
    {
        CandidateState.INSUFFICIENT_EVIDENCE.value,
        CandidateState.REJECTED.value,
        CandidateState.SHADOW_CHALLENGER.value,
    }
)


def _source_path_map(paths: Phase14CSourcePaths) -> dict[str, Path]:
    return {
        "phase14a": paths.phase14a_path,
        "hft": paths.hft_path,
        "demo": paths.demo_path,
    }


def _fingerprint_sources(paths: Phase14CSourcePaths) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for name, path in _source_path_map(paths).items():
        try:
            if path.is_file():
                fingerprints[name] = _source_file_fingerprint(path)
        except OSError:
            continue
    return fingerprints


def _demo_failure_reason(exc: Exception) -> Phase14CPreflightReason:
    message = str(exc).lower()
    if "reconciliation" in message or "chronology" in message:
        return Phase14CPreflightReason.DEMO_RECONCILIATION_UNCERTAIN
    return Phase14CPreflightReason.INVALID_DEMO_SOURCE


def preflight_phase14_sources(
    phase14a_path: Path,
    hft_path: Path,
    demo_path: Path,
) -> Phase14CReplayPreflight:
    """Inspect three explicit source databases without modifying any of them."""

    paths = Phase14CSourcePaths(phase14a_path, hft_path, demo_path)
    before = _fingerprint_sources(paths)
    reason: Phase14CPreflightReason | None = None
    if len(before) != 3:
        reason = Phase14CPreflightReason.SOURCE_UNAVAILABLE

    verified_count = quarantined_count = tick_count = segment_count = 0
    path_map = _source_path_map(paths)

    if "phase14a" in before:
        try:
            experiences, quarantined_count = _read_verified_phase14a(path_map["phase14a"])
            verified_count = len(experiences)
            if not experiences and reason is None:
                reason = Phase14CPreflightReason.NO_VERIFIED_PHASE14A_EXPERIENCE
        except (OSError, ValueError, TypeError, sqlite3.Error):
            if reason is None:
                reason = Phase14CPreflightReason.INVALID_PHASE14A

    if "demo" in before:
        try:
            _validate_demo_source_readonly(path_map["demo"])
        except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
            if reason is None:
                reason = _demo_failure_reason(exc)

    if "hft" in before:
        try:
            dataset = load_causal_tick_dataset(path_map["hft"], symbol="EURUSD")
            tick_count = dataset.valid_rows
            segment_count = len(dataset.segments)
            if (dataset.valid_rows == 0 or dataset.invalid_rows > 0 or not dataset.segments) and reason is None:
                reason = Phase14CPreflightReason.INVALID_CAUSAL_HFT_DATA
        except (OSError, ValueError, TypeError, sqlite3.Error, ReplayError, CausalDatasetError):
            if reason is None:
                reason = Phase14CPreflightReason.INVALID_CAUSAL_HFT_DATA

    after = _fingerprint_sources(paths)
    if before != after:
        reason = Phase14CPreflightReason.SOURCE_MUTATED_DURING_PREFLIGHT
    if reason is None:
        reason = Phase14CPreflightReason.READY

    return Phase14CReplayPreflight(
        source_paths=paths,
        source_fingerprints=before,
        verified_experience_count=verified_count,
        quarantined_experience_count=quarantined_count,
        causal_tick_count=tick_count,
        causal_segment_count=segment_count,
        commission_status="UNKNOWN",
        ready=reason is Phase14CPreflightReason.READY,
        reason_code=reason,
    )


def _record_not_run(
    spec: StrategySpec,
    reason: Phase14CCandidateReason | Phase14CPreflightReason,
    source_fingerprints: dict[str, str],
    *,
    candidate_id: str | None = None,
) -> CandidateEvaluationRecord:
    return CandidateEvaluationRecord(
        spec_id=spec.spec_id,
        candidate_id=candidate_id,
        status=Phase14CCandidateStatus.NOT_RUN,
        reason_code=reason,
        artifact_path=None,
        source_fingerprints=source_fingerprints,
    )


def _record_failed(
    spec: StrategySpec,
    candidate_id: str,
    reason: Phase14CCandidateReason,
    artifact_path: Path,
    source_fingerprints: dict[str, str],
) -> CandidateEvaluationRecord:
    return CandidateEvaluationRecord(
        spec_id=spec.spec_id,
        candidate_id=candidate_id,
        status=Phase14CCandidateStatus.FAILED,
        reason_code=reason,
        artifact_path=artifact_path,
        source_fingerprints=source_fingerprints,
    )


def evaluate_phase14c_candidates(
    specs: Sequence[StrategySpec],
    *,
    artifact_root: Path,
    source_paths: Phase14CSourcePaths,
    source_commit: str,
    max_candidate_runs: int = 5,
) -> tuple[CandidateEvaluationRecord, ...]:
    """Run a bounded batch through the existing read-only causal experiment."""

    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise TypeError("specs must be a sequence of StrategySpec values")
    requested = tuple(specs)
    if any(not isinstance(spec, StrategySpec) for spec in requested):
        raise TypeError("specs must contain only StrategySpec values")
    if not isinstance(source_paths, Phase14CSourcePaths):
        raise TypeError("source_paths must be Phase14CSourcePaths")
    if type(max_candidate_runs) is not int or not 1 <= max_candidate_runs <= 10:
        raise ValueError("max_candidate_runs must be an integer from 1 through 10")
    commit = str(source_commit).strip()
    if not commit:
        raise ValueError("source_commit is required")
    root = Path(artifact_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError("Phase 14C candidate artifact root must be fresh")
    if not requested:
        return ()

    preflight = preflight_phase14_sources(
        source_paths.phase14a_path,
        source_paths.hft_path,
        source_paths.demo_path,
    )
    fingerprints = dict(preflight.source_fingerprints)
    if not preflight.ready:
        return tuple(
            _record_not_run(spec, preflight.reason_code, fingerprints)
            for spec in requested
        )

    records: list[CandidateEvaluationRecord] = []
    runs = 0
    seen_spec_hashes: set[str] = set()
    stop_for_source_mutation = False
    registry = BookStrategyRegistry()
    factory = BookStrategyFactory()

    for spec in requested:
        if stop_for_source_mutation:
            records.append(
                _record_not_run(
                    spec,
                    Phase14CCandidateReason.SOURCE_MUTATED_DURING_EVALUATION,
                    fingerprints,
                )
            )
            continue
        if not spec.is_executable:
            records.append(
                _record_not_run(spec, Phase14CCandidateReason.SPEC_NOT_EXECUTABLE, fingerprints)
            )
            continue
        if spec.content_hash in seen_spec_hashes:
            records.append(
                _record_not_run(spec, Phase14CCandidateReason.DUPLICATE_SPEC, fingerprints)
            )
            continue
        seen_spec_hashes.add(spec.content_hash)

        try:
            implementation = registry.create(spec)
        except (TypeError, ValueError):
            records.append(
                _record_not_run(spec, Phase14CCandidateReason.UNIMPLEMENTED_SPEC, fingerprints)
            )
            continue

        parent = StrategyVersion(
            strategy_id=implementation.strategy_id,
            strategy_version="phase14-reviewed-control-v1",
            config_version="phase14-default-exits-v1",
            parameters=ExitPolicyConfig().to_dict(),
            source_commit=commit,
        )
        candidate = factory.from_validated_spec(spec, parent=parent)
        if runs >= max_candidate_runs:
            records.append(
                _record_not_run(
                    spec,
                    Phase14CCandidateReason.MAX_CANDIDATE_RUNS_REACHED,
                    fingerprints,
                    candidate_id=candidate.candidate_id,
                )
            )
            continue

        if _fingerprint_sources(source_paths) != fingerprints:
            records.append(
                _record_failed(
                    spec,
                    candidate.candidate_id,
                    Phase14CCandidateReason.SOURCE_MUTATED_DURING_EVALUATION,
                    root / candidate.candidate_id,
                    _fingerprint_sources(source_paths),
                )
            )
            stop_for_source_mutation = True
            continue

        run_root = root / candidate.candidate_id
        runs += 1
        try:
            report = BookCandidateExperiment(run_root).run(
                phase14a_path=source_paths.phase14a_path,
                hft_path=source_paths.hft_path,
                demo_path=source_paths.demo_path,
                candidate=candidate,
                source_commit=commit,
            )
        except (OSError, ValueError, TypeError, ReplayError):
            current = _fingerprint_sources(source_paths)
            changed = current != fingerprints
            records.append(
                _record_failed(
                    spec,
                    candidate.candidate_id,
                    Phase14CCandidateReason.SOURCE_MUTATED_DURING_EVALUATION
                    if changed
                    else Phase14CCandidateReason.REPLAY_FAILED,
                    run_root,
                    current,
                )
            )
            stop_for_source_mutation = changed
            continue

        current = _fingerprint_sources(source_paths)
        report_fingerprints = dict(getattr(report, "source_fingerprints", {}))
        if current != fingerprints or report_fingerprints != fingerprints or not report.source_unchanged:
            records.append(
                _record_failed(
                    spec,
                    candidate.candidate_id,
                    Phase14CCandidateReason.SOURCE_MUTATED_DURING_EVALUATION,
                    run_root,
                    current,
                )
            )
            stop_for_source_mutation = True
            continue

        candidate_state = str(report.candidate_state)
        if report.status != "COMPLETED":
            records.append(
                _record_failed(
                    spec,
                    candidate.candidate_id,
                    Phase14CCandidateReason.REPLAY_FAILED,
                    run_root,
                    current,
                )
            )
            continue
        if candidate_state not in _SHADOW_STATES:
            records.append(
                _record_failed(
                    spec,
                    candidate.candidate_id,
                    Phase14CCandidateReason.STATE_EXCEEDS_SHADOW_CEILING,
                    run_root,
                    current,
                )
            )
            continue
        records.append(
            CandidateEvaluationRecord(
                spec_id=spec.spec_id,
                candidate_id=candidate.candidate_id,
                status=Phase14CCandidateStatus.COMPLETED,
                reason_code=Phase14CCandidateReason.REPLAY_COMPLETED,
                artifact_path=run_root,
                gate_decision=report.gate_decision,
                gate_reasons=tuple(report.gate_reasons),
                source_fingerprints=current,
                candidate_state=candidate_state,
            )
        )

    return tuple(records)


__all__ = ["evaluate_phase14c_candidates", "preflight_phase14_sources"]
