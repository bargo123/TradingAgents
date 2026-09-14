"""Deterministic, append-only dataset generation writer and validator."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    CANONICALIZATION_VERSION,
    DATASET_SCHEMA_VERSION,
    ELIGIBILITY_POLICY_VERSION,
    EXAMPLE_SCHEMA_VERSION,
    SOURCE_ADAPTER_VERSION,
    SPLIT_POLICY_VERSION,
    CanonicalExampleV1,
    DatasetExclusion,
    SplitAssignment,
)
from .splits import SplitResult, validate_split_assignments

_FILES = ("examples.jsonl", "excluded.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl")
_PHASES = ("phase56", "phase8", "phase9")
_FINGERPRINT_FIELDS = (
    "source_id", "canonical_path", "schema_fingerprint", "file_sha256",
    "snapshot_fingerprint", "contract_version",
)
_POLICY_DEFAULTS = {
    "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
    "canonicalization_version": CANONICALIZATION_VERSION,
    "split_policy_version": SPLIT_POLICY_VERSION,
    "source_adapter_version": SOURCE_ADAPTER_VERSION,
}
_FORBIDDEN = re.compile(
    r"prompt|completion|reasoning|chain[ _-]?of[ _-]?thought|\bcot\b|secret|password|credential|api[ _-]?key|token",
    re.I,
)


class GenerationWriteError(RuntimeError):
    pass


class GenerationExistsError(GenerationWriteError):
    pass


class GenerationValidationError(GenerationWriteError):
    pass


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value") and isinstance(value.value, (str, int, float)):
        return value.value
    return value


def _safe(value: Any, key: str = "") -> None:
    if key and _FORBIDDEN.search(key):
        raise GenerationValidationError(f"forbidden field: {key}")
    if isinstance(value, Mapping):
        for k, child in value.items():
            _safe(child, str(k))
    elif isinstance(value, list):
        for child in value:
            _safe(child)
    elif isinstance(value, str) and _FORBIDDEN.search(value):
        raise GenerationValidationError("forbidden sensitive value")


def _line(value: Any) -> bytes:
    _safe(value)
    return (
        json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _validated_fingerprints(value: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(_PHASES):
        raise GenerationValidationError("source_fingerprints must contain phase56, phase8, phase9")
    result: dict[str, dict[str, str]] = {}
    for phase in _PHASES:
        fingerprint = value[phase]
        if hasattr(fingerprint, "to_dict") and callable(fingerprint.to_dict):
            fingerprint = fingerprint.to_dict()
        if not isinstance(fingerprint, Mapping) or set(fingerprint) != set(_FINGERPRINT_FIELDS):
            raise GenerationValidationError(f"invalid {phase} source fingerprint")
        normalized = {}
        for field in _FINGERPRINT_FIELDS:
            item = fingerprint[field]
            if not isinstance(item, str) or not item or len(item) > 512:
                raise GenerationValidationError(f"invalid {phase} source fingerprint value")
            normalized[field] = item
        result[phase] = normalized
    return result


def _validated_policies(value: Mapping[str, str] | None) -> dict[str, str]:
    result = dict(_POLICY_DEFAULTS)
    if value is not None:
        if not isinstance(value, Mapping):
            raise GenerationValidationError("policy_versions must be a mapping")
        result.update(value)
    for name in _POLICY_DEFAULTS:
        if not isinstance(result.get(name), str) or not result[name]:
            raise GenerationValidationError(f"missing policy version: {name}")
        if result[name] != _POLICY_DEFAULTS[name]:
            raise GenerationValidationError(f"unsupported policy version: {name}")
    return {str(k): str(v) for k, v in sorted(result.items())}


def _canonical_row(value: Any) -> CanonicalExampleV1:
    if not isinstance(value, Mapping):
        raise GenerationValidationError("example row must be an object")
    try:
        row = CanonicalExampleV1(**dict(value))
    except (TypeError, ValueError) as exc:
        raise GenerationValidationError(f"canonical schema invalid: {exc}") from exc
    if row.schema_version != EXAMPLE_SCHEMA_VERSION:
        raise GenerationValidationError("unsupported example schema version")
    return row


def _example_key(row: CanonicalExampleV1) -> tuple[str, str, str, int, str]:
    decision = row.decision
    outcome = row.outcome
    stamp = decision.get("analysis_snapshot_timestamp", "")
    return (
        str(stamp), str(decision.get("decision_id", "")),
        str(outcome.get("evaluation_basis", "")), int(outcome.get("horizon_seconds", 0)),
        row.example_id,
    )


def _group_id_for_row(row: CanonicalExampleV1) -> str:
    run_id = row.decision.get("source_run_id", "")
    decision_id = row.decision.get("decision_id", "")
    return f"run:{run_id}" if run_id else f"decision:{decision_id}"


def _provenance_fingerprints(row: CanonicalExampleV1) -> dict[str, Any]:
    provenance = row.provenance
    phase8 = provenance.get("phase8", {})
    phase9 = provenance.get("phase9", {})
    return {
        "phase56": provenance.get("phase56"),
        "phase8": phase8.get("source_fingerprint") if isinstance(phase8, Mapping) else None,
        "phase9": phase9.get("source_fingerprint") if isinstance(phase9, Mapping) else None,
    }


def _write_jsonl(path: Path, rows: Iterable[Any]) -> dict[str, Any]:
    data = b"".join(_line(row) for row in rows)
    path.write_bytes(data)
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "count": data.count(b"\n"),
    }


def _assignment_map(split_result: SplitResult | Iterable[SplitAssignment]) -> dict[str, str]:
    assignments = (
        split_result.assignments if isinstance(split_result, SplitResult) else tuple(split_result)
    )
    return {item.example_id: item.split for item in assignments}


def _default_id(examples: tuple[Any, ...], exclusions: tuple[Any, ...]) -> str:
    payload = json.dumps(
        _plain({"examples": examples, "excluded": exclusions}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return "generation-" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def write_generation(
    output_root: str | Path,
    examples: Iterable[CanonicalExampleV1],
    exclusions: Iterable[DatasetExclusion],
    split_result: SplitResult | Iterable[SplitAssignment],
    *,
    dataset_id: str | None = None,
    source_fingerprints: Mapping[str, Any] | None = None,
    policy_versions: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Stage and atomically publish one generation; never overwrite a generation."""
    rows = tuple(sorted(examples, key=lambda row: row.example_id))
    excluded = tuple(sorted(exclusions, key=lambda row: row.decision_id))
    generation_id = dataset_id or _default_id(rows, excluded)
    fingerprints = _validated_fingerprints(source_fingerprints)
    policies = _validated_policies(policy_versions)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / generation_id
    if destination.exists():
        raise GenerationExistsError(f"generation already exists: {generation_id}")
    stage = root / f".staging-{generation_id}-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        assignment_items = tuple(
            split_result.assignments if isinstance(split_result, SplitResult) else split_result
        )
        split_status = (
            split_result.status
            if isinstance(split_result, SplitResult)
            else ("COMPLETE" if assignment_items else "INSUFFICIENT_DATA")
        )
        split_contract = SplitResult(assignment_items, split_status)
        validate_split_assignments(split_contract, rows)
        assignments = {}
        for assignment in assignment_items:
            if assignment.example_id in assignments:
                raise GenerationWriteError("duplicate split assignment")
            assignments[assignment.example_id] = assignment.split
        by_id = {row.example_id: row for row in rows}
        if len(by_id) != len(rows):
            raise GenerationWriteError("duplicate example id")
        for row in rows:
            actual = _provenance_fingerprints(row)
            if any(actual.get(phase) != fingerprints[phase] for phase in _PHASES):
                raise GenerationValidationError("example/source fingerprint mismatch")
        file_info = {
            "examples.jsonl": _write_jsonl(stage / "examples.jsonl", rows),
            "excluded.jsonl": _write_jsonl(stage / "excluded.jsonl", excluded),
            "train.jsonl": _write_jsonl(
                stage / "train.jsonl", (by_id[i] for i, s in assignments.items() if s == "train")
            ),
            "validation.jsonl": _write_jsonl(
                stage / "validation.jsonl",
                (by_id[i] for i, s in assignments.items() if s == "validation"),
            ),
            "test.jsonl": _write_jsonl(
                stage / "test.jsonl", (by_id[i] for i, s in assignments.items() if s == "test")
            ),
        }
        timestamps = [
            row.decision.get("analysis_snapshot_timestamp")
            for row in rows
            if isinstance(row.decision, Mapping)
        ]
        timestamps = [
            x.isoformat() if isinstance(x, datetime) else x for x in timestamps if x is not None
        ]
        reason_counts = Counter(reason.value for item in excluded for reason in item.reasons)
        action_counts = Counter(str(row.decision.get("action", "")) for row in rows)
        profile_counts = Counter(str(row.decision.get("analysis_profile", "")) for row in rows)
        symbol_counts = Counter(str(row.decision.get("resolved_symbol", "")) for row in rows)
        outcome_counts = Counter(
            f"{row.outcome.get('evaluation_basis', '')}:{row.outcome.get('horizon_seconds', '')}"
            for row in rows
        )
        candidate_count = len(rows) + len(excluded)
        manifest = {
            "dataset_id": generation_id,
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "example_schema_version": EXAMPLE_SCHEMA_VERSION,
            "eligibility_policy_version": policies["eligibility_policy_version"],
            "canonicalization_version": CANONICALIZATION_VERSION,
            "split_policy_version": policies["split_policy_version"],
            "source_adapter_version": policies["source_adapter_version"],
            "status": "PUBLISHED" if rows else "EMPTY_ELIGIBLE_SET",
            "split_status": split_status,
            "examples": len(rows),
            "exclusions": len(excluded),
            "counts": {
                "examples": len(rows),
                "exclusions": len(excluded),
                "reason_counts": dict(sorted(reason_counts.items())),
            },
            "candidate_summary": {
                "candidates": candidate_count,
                "eligible": len(rows),
                "excluded": len(excluded),
            },
            "class_counts": dict(sorted(action_counts.items())),
            "profile_counts": dict(sorted(profile_counts.items())),
            "symbol_counts": dict(sorted(symbol_counts.items())),
            "evaluation_counts": dict(sorted(outcome_counts.items())),
            "split_counts": dict(sorted(Counter(assignments.values()).items())),
            "group_count": len(
                {
                    assignment.group_id
                    for assignment in assignment_items
                }
            ),
            "time_range": {
                "first": min(timestamps) if timestamps else None,
                "last": max(timestamps) if timestamps else None,
            },
            "source_fingerprints": fingerprints,
            "policy_versions": policies,
            "reproducibility": {
                "writer_version": "phase10.writer.v1",
                "serialization": "json-sort-keys-compact-utf8-newline",
            },
            "safety": {"network_attempts": 0, "llm_calls": 0, "tool_calls": 0, "mt5_calls": 0},
            "files": file_info,
        }
        if metadata:
            manifest["metadata"] = _plain(metadata)
        (stage / "manifest.json").write_bytes(
            (
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
        )
        try:
            os.replace(stage, destination)
        except FileExistsError as exc:
            raise GenerationExistsError(f"generation already exists: {generation_id}") from exc
        return destination
    except Exception:
        # Keep staging directories for recovery/diagnostics, never publish partial data.
        raise


class GenerationWriter:
    """Small object facade useful to orchestration code and dependency injection."""

    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root)

    def write(self, examples, exclusions, split_result, **kwargs) -> Path:
        return write_generation(self.output_root, examples, exclusions, split_result, **kwargs)


def validate_generation(path: str | Path):
    """Validate a published generation without writing to it."""
    from .models import ValidationReport

    root = Path(path)
    errors: list[str] = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise GenerationValidationError("manifest must be an object")
        _safe(manifest)
        expected_versions = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "example_schema_version": EXAMPLE_SCHEMA_VERSION,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "source_adapter_version": SOURCE_ADAPTER_VERSION,
        }
        for name, expected in expected_versions.items():
            if manifest.get(name) != expected:
                errors.append(f"manifest version mismatch: {name}")
        try:
            fingerprints = _validated_fingerprints(manifest.get("source_fingerprints"))
            manifest_policies = manifest.get("policy_versions")
            if (
                not isinstance(manifest_policies, Mapping)
                or set(manifest_policies) != set(_POLICY_DEFAULTS)
            ):
                raise GenerationValidationError("manifest policy_versions are incomplete")
            _validated_policies(manifest_policies)
        except GenerationValidationError as exc:
            errors.append(str(exc))
            fingerprints = {}
        for name in _FILES:
            info = manifest.get("files", {}).get(name)
            if not isinstance(info, Mapping):
                errors.append(f"missing file metadata: {name}")
                continue
            data = (root / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != info["sha256"]:
                errors.append(f"hash mismatch: {name}")
            if len(data) != info["size"] or data.count(b"\n") != info["count"]:
                errors.append(f"file metadata mismatch: {name}")
        def read_rows(name: str) -> list[dict[str, Any]]:
            output = []
            for line in (root / name).read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                try:
                    value = json.loads(line)
                    _safe(value)
                except Exception as exc:
                    errors.append(f"invalid JSON row: {name}: {exc}")
                    continue
                if not isinstance(value, Mapping):
                    errors.append(f"row is not an object: {name}")
                    continue
                output.append(dict(value))
            return output

        raw_rows = read_rows("examples.jsonl")
        rows: list[CanonicalExampleV1] = []
        for value in raw_rows:
            try:
                rows.append(_canonical_row(value))
            except GenerationValidationError as exc:
                errors.append(str(exc))
        ids = [row.example_id for row in rows]
        if len(ids) != len(set(ids)):
            errors.append("duplicate or invalid example IDs")
        if manifest.get("examples") != len(rows):
            errors.append("example count mismatch")
        example_by_id = {item.example_id: item for item in rows}
        # Source fingerprints are part of the trust boundary: a generation
        # cannot claim one source while its canonical rows identify another.
        if fingerprints:
            for row in rows:
                actual = _provenance_fingerprints(row)
                for phase in _PHASES:
                    if actual.get(phase) != fingerprints.get(phase):
                        errors.append(f"source fingerprint mismatch: {phase}")
        partition_assignments: list[SplitAssignment] = []
        seen: dict[str, str] = {}
        for split in ("train", "validation", "test"):
            split_rows = read_rows(f"{split}.jsonl")
            for item in split_rows:
                try:
                    split_row = _canonical_row(item)
                except GenerationValidationError as exc:
                    errors.append(f"{split}: {exc}")
                    continue
                item_id = split_row.example_id
                if item_id not in example_by_id:
                    errors.append(f"{split} contains unknown example: {item_id}")
                if item_id in seen and seen[item_id] != split:
                    errors.append(f"example crosses splits: {item_id}")
                seen[item_id] = split
                if item != _plain(example_by_id.get(item_id)):
                    errors.append(f"{split} row differs from examples: {item_id}")
                group_id = _group_id_for_row(split_row)
                partition_assignments.append(SplitAssignment(item_id, split, group_id))
        if len(seen) != sum(manifest.get("split_counts", {}).values()):
            errors.append("split count mismatch")
        try:
            status = manifest.get("split_status", "INSUFFICIENT_DATA")
            ordered = tuple(sorted(partition_assignments, key=lambda a: _example_key(example_by_id[a.example_id])))
            validate_split_assignments(SplitResult(ordered, status), rows)
        except Exception as exc:
            errors.append(f"split validation failed: {exc}")
        for field, expected in (
            ("class_counts", Counter(str(row.decision.get("action", "")) for row in rows)),
            ("profile_counts", Counter(str(row.decision.get("analysis_profile", "")) for row in rows)),
            ("symbol_counts", Counter(str(row.decision.get("resolved_symbol", "")) for row in rows)),
        ):
            if manifest.get(field) != dict(sorted(expected.items())):
                errors.append(f"manifest summary mismatch: {field}")
    except Exception as exc:
        errors.append(str(exc))
    return ValidationReport(not errors, tuple(errors))


__all__ = [
    "GenerationWriter",
    "GenerationWriteError",
    "GenerationExistsError",
    "GenerationValidationError",
    "write_generation",
    "validate_generation",
]
