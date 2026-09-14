"""Pure, read-only orchestration for Phase 10 dataset generation."""

from __future__ import annotations

import hashlib
import json
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
    SourceFingerprint,
)
from .sources import (
    ReadonlyExperienceSource,
    ReadonlyPhase9AuditSource,
    ReadonlyPhase56Source,
    SourceReadResult,
)
from .splits import assign_splits
from .writer import GenerationExistsError, write_generation

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
    return DatasetManifest(
        dataset_id=str(data["dataset_id"]),
        examples=int(data.get("examples", 0)),
        exclusions=int(data.get("exclusions", 0)),
        split_status=str(data.get("split_status", "INSUFFICIENT_DATA")),
        safety=data.get("safety", _SAFETY),
        status=str(data.get("status", "EMPTY_ELIGIBLE_SET")),
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
            "broken_sources": [], "duplicate_sources": [], "removed_sources": []
        }

        # These catalog/audit reads are deliberately performed once per build.
        try:
            phase8 = ReadonlyExperienceSource(config.phase8_root / "catalog.sqlite3").read()
        except Exception as exc:  # isolate a broken optional source
            phase8 = _empty_result("phase8", type(exc).__name__)
            errors.append(f"phase8: {type(exc).__name__}")
        try:
            phase9 = (
                ReadonlyPhase9AuditSource(config.phase9_audit_path).read()
                if config.phase9_audit_path
                else _empty_result("phase9", "SOURCE_MISSING")
            )
        except Exception as exc:
            phase9 = _empty_result("phase9", type(exc).__name__)
            errors.append(f"phase9: {type(exc).__name__}")

        source_results = []
        seen_bytes: set[str] = set()
        for path in sorted(config.source_db_paths, key=lambda item: str(item).casefold()):
            try:
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                diagnostics["broken_sources"].append(str(path))
                errors.append(f"{path}: {type(exc).__name__}")
                continue
            if file_hash in seen_bytes:
                diagnostics["duplicate_sources"].append(str(path))
                continue
            seen_bytes.add(file_hash)
            try:
                source_results.append(ReadonlyPhase56Source(path).read())
            except SourceReadError as exc:
                diagnostics["broken_sources"].append(str(path))
                errors.append(f"{path}: {type(exc).__name__}")
            except Exception as exc:
                diagnostics["broken_sources"].append(str(path))
                errors.append(f"{path}: {type(exc).__name__}")

        examples = []
        exclusions: list[DatasetExclusion] = []
        seen_keys: set[tuple[str, str, int]] = set()
        for result in source_results:
            try:
                joined = join_observations(result, phase8, phase9)
            except Exception as exc:
                errors.append(f"join: {type(exc).__name__}")
                continue
            for observation in joined:
                eligibility = classify_observation(observation, config)
                decision_id = observation.decision.decision_id
                evaluation = observation.evaluation
                key = (
                    decision_id,
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
                    examples.append(canonicalize(observation, eligibility))
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
        metadata = {"diagnostics": diagnostics, "safety": _SAFETY}
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
                        if item.get("source_fingerprints") == fingerprints:
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
