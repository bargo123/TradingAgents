"""Pure, read-only orchestration for Phase 10 dataset generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .canonical import canonicalize
from .eligibility import classify_observation, join_observations
from .errors import SourceReadError
from .models import (
    BuildReport,
    DatasetConfig,
    DatasetExclusion,
    DatasetExclusionReason,
    DatasetManifest,
    EvaluationObservation,
    JoinedObservation,
    SourceFingerprint,
)
from .sources import (
    ReadonlyExperienceSource,
    ReadonlyPhase9AuditSource,
    ReadonlyPhase56Source,
    SourceReadResult,
)
from .splits import assign_splits
from .writer import (
    GenerationExistsError,
    GenerationValidationError,
    validate_generation,
    write_generation,
)

_SAFETY = {"network_attempts": 0, "llm_calls": 0, "tool_calls": 0, "mt5_calls": 0}


def _empty_result(kind: str, reason: str) -> SourceReadResult:
    return SourceReadResult(
        available=False,
        unavailable=type("Unavailable", (), {"source": kind, "reason": reason})(),
    )


def _fingerprint(result: SourceReadResult | Any, kind: str) -> dict[str, Any]:
    value = getattr(result, "fingerprint", None)
    if value is not None:
        return value.to_dict() if hasattr(value, "to_dict") else dict(value)
    digest = hashlib.sha256(f"phase10-unavailable:{kind}".encode()).hexdigest()
    return SourceFingerprint(
        f"{kind}-unavailable", f"<unavailable:{kind}>", digest, digest, digest
    ).to_dict()


def _manifest(path: Path) -> DatasetManifest:
    data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    counts = data.get("counts", {})
    candidate_summary = data.get("candidate_summary", {})
    return DatasetManifest(
        dataset_id=str(data["dataset_id"]),
        examples=int(data.get("examples", 0)),
        exclusions=int(data.get("exclusions", 0)),
        split_status=str(data.get("split_status", "INSUFFICIENT_DATA")),
        safety=data.get("safety", _SAFETY),
        source_fingerprints=data.get("source_fingerprints", {}),
        status=str(data.get("status", "EMPTY_ELIGIBLE_SET")),
        candidate_count=int(candidate_summary.get("candidates", data.get("examples", 0) + data.get("exclusions", 0))),
        eligible_count=int(candidate_summary.get("eligible", data.get("examples", 0))),
        excluded_count=int(candidate_summary.get("excluded", data.get("exclusions", 0))),
        reason_counts=counts.get("reason_counts", {}),
    )


def _coalesce_fingerprints(results: list[SourceReadResult]) -> dict[str, Any]:
    """Represent a multi-file source set as one deterministic source identity."""
    values = [_fingerprint(result, "phase56") for result in results]
    if not values:
        return _fingerprint(_empty_result("phase56", "SOURCE_MISSING"), "phase56")
    if len(values) == 1:
        return values[0]
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return SourceFingerprint(
        "phase56-set-" + digest[:24],
        "<phase56-source-set>",
        hashlib.sha256(b"phase56-source-set").hexdigest(),
        digest,
        digest,
    ).to_dict()


def _config_metadata(config: DatasetConfig) -> dict[str, Any]:
    """Return only configuration that changes dataset meaning or identity."""
    return {
        "filters": json.loads(json.dumps(dict(config.filters), sort_keys=True, default=str)),
        "evaluation_basis": config.evaluation_basis,
        "horizon_seconds": config.horizon_seconds,
        "allow_empty": config.allow_empty,
    }


def _source_inventory(
    results: list[SourceReadResult], phase8: SourceReadResult, phase9: SourceReadResult
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for kind, result in (("phase56", item) for item in results):
        fingerprint = _fingerprint(result, kind)
        decisions = getattr(result, "decisions", ())
        inventory.append({
            "kind": kind,
            "canonical_path": fingerprint["canonical_path"],
            "source_id": fingerprint["source_id"],
            "file_sha256": fingerprint["file_sha256"],
            "snapshot_fingerprint": fingerprint["snapshot_fingerprint"],
            "decision_ids": sorted(
                str(item.decision_id) for item in decisions
                if getattr(item, "decision_id", None)
            ),
            "decision_fingerprints": sorted(
                str(item.fields.get("source_decision_fingerprint"))
                for item in decisions
                if getattr(item, "fields", {}).get("source_decision_fingerprint")
            ),
            "decision_identities": sorted(
                (
                    {"decision_id": str(item.decision_id),
                     "fingerprint": str(item.fields.get("source_decision_fingerprint"))}
                    for item in decisions
                    if getattr(item, "decision_id", None)
                    and getattr(item, "fields", {}).get("source_decision_fingerprint")
                ),
                key=lambda item: (item["decision_id"], item["fingerprint"]),
            ),
        })
    for kind, result in (("phase8", phase8), ("phase9", phase9)):
        fingerprint = _fingerprint(result, kind)
        inventory.append({
            "kind": kind,
            "canonical_path": fingerprint["canonical_path"],
            "source_id": fingerprint["source_id"],
            "file_sha256": fingerprint["file_sha256"],
            "snapshot_fingerprint": fingerprint["snapshot_fingerprint"],
        })
    return sorted(inventory, key=lambda item: (item["kind"], item["canonical_path"]))


def _prior_inventory(output_root: Path, config_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Read prior published factory metadata without trusting it as valid data."""
    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    if not output_root.is_dir():
        return []
    for candidate in sorted(output_root.iterdir(), key=lambda item: item.name):
        manifest_path = candidate / "manifest.json"
        if not candidate.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = manifest.get("metadata", {})
            if metadata.get("factory_config") != config_metadata:
                continue
            inventory = metadata.get("source_inventory")
            if isinstance(inventory, list):
                candidates.append((candidate.name, inventory))
        except (OSError, ValueError, TypeError):
            continue
    return candidates[-1][1] if candidates else []


def _inventory_diagnostics(
    current: list[dict[str, Any]], previous: list[dict[str, Any]]
) -> dict[str, list[str]]:
    diagnostics = {
        "changed_sources": [], "removed_sources": [], "removed_rows": [],
        "added_rows": [], "changed_rows": [],
    }
    current_by_path = {item["canonical_path"]: item for item in current}
    previous_by_path = {item["canonical_path"]: item for item in previous}
    for path, prior in previous_by_path.items():
        now = current_by_path.get(path)
        if now is None:
            diagnostics["removed_sources"].append(path)
            continue
        identity_fields = ("source_id", "file_sha256", "snapshot_fingerprint")
        if any(now.get(field) != prior.get(field) for field in identity_fields):
            diagnostics["changed_sources"].append(path)
        prior_rows = set(prior.get("decision_ids", ()))
        current_rows = set(now.get("decision_ids", ()))
        diagnostics["removed_rows"].extend(f"{path}:{row}" for row in sorted(prior_rows - current_rows))
        diagnostics["added_rows"].extend(f"{path}:{row}" for row in sorted(current_rows - prior_rows))
        prior_identity = {
            item.get("decision_id"): item.get("fingerprint")
            for item in prior.get("decision_identities", ())
            if isinstance(item, dict)
        }
        current_identity = {
            item.get("decision_id"): item.get("fingerprint")
            for item in now.get("decision_identities", ())
            if isinstance(item, dict)
        }
        diagnostics["changed_rows"].extend(
            f"{path}:{row}" for row in sorted(
                key for key in prior_identity.keys() & current_identity.keys()
                if prior_identity[key] != current_identity[key]
            )
        )
    return {key: sorted(set(value)) for key, value in diagnostics.items()}


def _source_error_exclusion(path: Path, exc: Exception) -> DatasetExclusion:
    return DatasetExclusion(
        f"source:{hashlib.sha256(str(path).encode()).hexdigest()[:24]}",
        (DatasetExclusionReason.SOURCE_INTEGRITY_FAILED,),
        {"source_path": str(path), "error_type": type(exc).__name__},
    )


def _effective_observation(observation: JoinedObservation, eligibility: Any, config: DatasetConfig) -> JoinedObservation:
    """Pair canonical output with the exact basis/horizon classified as eligible."""
    details = getattr(eligibility, "details", {}) or {}
    basis = details.get("effective_evaluation_basis", config.evaluation_basis)
    horizon = int(details.get("effective_horizon_seconds", config.horizon_seconds))
    current = observation.evaluation
    if (
        current is not None
        and current.evaluation_basis == basis
        and current.horizon_seconds == horizon
    ):
        return observation
    for item in observation.fields.get("evaluations", ()):
        if not isinstance(item, Mapping):
            continue
        if item.get("evaluation_basis") != basis or int(item.get("horizon_seconds", 0) or 0) != horizon:
            continue
        selected = EvaluationObservation(
            str(item.get("decision_id")),
            str(item.get("evaluation_basis")),
            int(item.get("horizon_seconds")),
            str(item.get("evaluation_status")),
            bool(item.get("source_context_eligible")),
            item.get("fields", {}),
        )
        return replace(observation, evaluation=selected)
    return observation


def _bind_sqlite_marks(result: SourceReadResult, marks_digest: str) -> SourceReadResult:
    """Bind the adapter result to both the SQLite main file and WAL marks."""
    fingerprint = getattr(result, "fingerprint", None)
    if fingerprint is None:
        return result
    decision_fingerprints = sorted(
        str(item.fields.get("source_decision_fingerprint"))
        for item in getattr(result, "decisions", ())
        if getattr(item, "fields", {}).get("source_decision_fingerprint")
    )
    snapshot = hashlib.sha256(json.dumps({
        "adapter": fingerprint.snapshot_fingerprint,
        "sqlite_marks": marks_digest,
        "decision_fingerprints": decision_fingerprints,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    bound = replace(
        fingerprint,
        source_id=f"{fingerprint.source_id}-{marks_digest[:16]}",
        file_sha256=marks_digest,
        snapshot_fingerprint=snapshot,
    )
    return replace(result, fingerprint=bound)


class DatasetFactory:
    """Build a generation without importing any trading/runtime components."""

    @staticmethod
    def safety_counters() -> dict[str, int]:
        return dict(_SAFETY)

    def build(self, config: DatasetConfig) -> BuildReport:
        if not isinstance(config, DatasetConfig):
            raise TypeError("config must be DatasetConfig")

        errors: list[str] = []
        diagnostics: dict[str, list[str]] = {
            "broken_sources": [], "duplicate_sources": [], "removed_sources": [],
            "changed_sources": [], "removed_rows": [], "added_rows": [], "changed_rows": [],
        }
        source_exclusions: list[DatasetExclusion] = []

        # These catalog/audit reads are deliberately performed once per build.
        try:
            phase8 = ReadonlyExperienceSource(config.phase8_root / "catalog.sqlite3").read()
        except Exception as exc:  # isolate a broken optional source
            phase8 = _empty_result("phase8", type(exc).__name__)
            errors.append(f"phase8: {type(exc).__name__}")
            source_exclusions.append(
                _source_error_exclusion(config.phase8_root / "catalog.sqlite3", exc)
            )
        try:
            phase9 = (
                ReadonlyPhase9AuditSource(config.phase9_audit_path).read()
                if config.phase9_audit_path
                else _empty_result("phase9", "SOURCE_MISSING")
            )
        except Exception as exc:
            phase9 = _empty_result("phase9", type(exc).__name__)
            errors.append(f"phase9: {type(exc).__name__}")
            if config.phase9_audit_path is not None:
                source_exclusions.append(
                    _source_error_exclusion(config.phase9_audit_path, exc)
                )

        source_results = []
        seen_bytes: set[str] = set()
        for path in sorted(config.source_db_paths, key=lambda item: str(item).casefold()):
            try:
                marks = []
                for marked_path in (path, Path(str(path) + "-wal")):
                    if marked_path.is_file():
                        marks.append(hashlib.sha256(marked_path.read_bytes()).hexdigest())
                    else:
                        marks.append("missing")
                file_hash = hashlib.sha256(json.dumps(marks).encode()).hexdigest()
            except OSError as exc:
                diagnostics["broken_sources"].append(str(path))
                errors.append(f"{path}: {type(exc).__name__}")
                source_exclusions.append(_source_error_exclusion(path, exc))
                continue
            if file_hash in seen_bytes:
                diagnostics["duplicate_sources"].append(str(path))
                source_exclusions.append(
                    DatasetExclusion(
                        f"source:{hashlib.sha256(str(path).encode()).hexdigest()[:24]}",
                        (DatasetExclusionReason.DUPLICATE,),
                        {"source_path": str(path)},
                    )
                )
                continue
            seen_bytes.add(file_hash)
            try:
                source_results.append(_bind_sqlite_marks(
                    ReadonlyPhase56Source(path).read(), file_hash
                ))
            except SourceReadError as exc:
                diagnostics["broken_sources"].append(str(path))
                errors.append(f"{path}: {type(exc).__name__}")
                source_exclusions.append(_source_error_exclusion(path, exc))
            except Exception as exc:
                diagnostics["broken_sources"].append(str(path))
                errors.append(f"{path}: {type(exc).__name__}")
                source_exclusions.append(_source_error_exclusion(path, exc))

        examples = []
        exclusions: list[DatasetExclusion] = list(source_exclusions)
        seen_keys: set[tuple[str, str, int]] = set()
        for result in source_results:
            try:
                joined = join_observations(result, phase8, phase9)
            except Exception as exc:
                errors.append(f"join: {type(exc).__name__}")
                reason = (
                    DatasetExclusionReason.SCHEMA_UNSUPPORTED
                    if isinstance(exc, (TypeError, ValueError))
                    else DatasetExclusionReason.SOURCE_INTEGRITY_FAILED
                )
                exclusions.extend(
                    DatasetExclusion(
                        decision.decision_id,
                        (reason,),
                        {"error_type": type(exc).__name__, "stage": "join"},
                    )
                    for decision in result.decisions
                )
                continue
            for observation in joined:
                eligibility = classify_observation(observation, config)
                decision_id = observation.decision.decision_id
                evaluation = observation.evaluation
                decision_fingerprint = str(
                    getattr(observation.decision, "fields", {}).get("source_decision_fingerprint")
                    or decision_id
                )
                key = (
                    decision_fingerprint,
                    str(getattr(evaluation, "evaluation_basis", config.evaluation_basis)),
                    int(getattr(evaluation, "horizon_seconds", config.horizon_seconds)),
                )
                if key in seen_keys:
                    exclusions.append(
                        DatasetExclusion(decision_id, (DatasetExclusionReason.DUPLICATE,))
                    )
                    continue
                seen_keys.add(key)
                if not eligibility.eligible:
                    exclusions.append(eligibility.exclusion)
                    continue
                try:
                    examples.append(canonicalize(_effective_observation(observation, eligibility, config), eligibility))
                except Exception as exc:
                    exclusions.append(
                        DatasetExclusion(
                            decision_id,
                            (DatasetExclusionReason.SCHEMA_UNSUPPORTED,),
                            {"error_type": type(exc).__name__},
                        )
                    )

        # A missing/changed row remains visible in the generated diagnostics;
        # no source is ever altered to manufacture a replacement observation.
        fingerprints = {
            "phase56": _coalesce_fingerprints(source_results),
            "phase8": _fingerprint(phase8, "phase8"),
            "phase9": _fingerprint(phase9, "phase9"),
        }
        inventory = _source_inventory(source_results, phase8, phase9)
        diagnostics.update(_inventory_diagnostics(inventory, _prior_inventory(
            config.output_root, _config_metadata(config)
        )))
        if len(source_results) > 1:
            # The writer has one phase56 fingerprint slot. Preserve each
            # source's decision identity while binding rows to the set-level
            # immutable input identity used by the generation.
            examples = [
                replace(
                    row,
                    provenance={
                        **dict(row.provenance),
                        "phase56_sources": tuple(
                            _fingerprint(item, "phase56") for item in source_results
                        ),
                        "phase56": fingerprints["phase56"],
                    },
                )
                for row in examples
            ]
        split_result = assign_splits(examples)
        metadata = {
            "factory_config": _config_metadata(config),
            "source_inventory": inventory,
            "diagnostics": diagnostics,
            "safety": _SAFETY,
        }
        try:
            destination = write_generation(
                config.output_root,
                examples,
                exclusions,
                split_result,
                source_fingerprints=fingerprints,
                metadata=metadata,
            )
        except GenerationExistsError:
            # Content-derived generations are immutable; an unchanged repeat
            # is successful and returns the already-published generation.
            matches = []
            for candidate in config.output_root.iterdir() if config.output_root.exists() else ():
                if candidate.is_dir() and (candidate / "manifest.json").is_file():
                    try:
                        item = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
                        if (
                            item.get("source_fingerprints") == fingerprints
                            and item.get("metadata", {}).get("factory_config")
                            == _config_metadata(config)
                            and item.get("metadata", {}).get("source_inventory") == inventory
                            and item.get("metadata", {}).get("diagnostics") == diagnostics
                        ):
                            validation = validate_generation(candidate)
                            if not validation.valid:
                                raise GenerationValidationError(
                                    "existing generation failed validation: "
                                    + "; ".join(validation.errors[:3])
                                )
                            matches.append(candidate)
                    except (OSError, ValueError):
                        continue
            if not matches:
                raise
            destination = sorted(matches, key=lambda item: item.name)[0]
        except Exception as exc:
            return BuildReport("FAILED", None, tuple(exclusions), tuple(errors + [type(exc).__name__]))

        manifest = _manifest(destination)
        status = "PUBLISHED" if examples else "EMPTY_ELIGIBLE_SET"
        return BuildReport(status, manifest, tuple(exclusions), tuple(errors))


__all__ = ["DatasetFactory"]
