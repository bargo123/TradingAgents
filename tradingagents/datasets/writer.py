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
    CanonicalExampleV1,
    DatasetExclusion,
    SplitAssignment,
)
from .splits import SplitResult, validate_split_assignments

_FILES = ("examples.jsonl", "excluded.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl")
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
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / generation_id
    if destination.exists():
        raise GenerationExistsError(f"generation already exists: {generation_id}")
    stage = root / f".staging-{generation_id}-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        assignments = _assignment_map(split_result)
        by_id = {row.example_id: row for row in rows}
        if len(by_id) != len(rows):
            raise GenerationWriteError("duplicate example id")
        if isinstance(split_result, SplitResult):
            validate_split_assignments(split_result, rows)
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
        split_status = (
            split_result.status
            if isinstance(split_result, SplitResult)
            else ("COMPLETE" if assignments else "INSUFFICIENT_DATA")
        )
        reason_counts = Counter(reason.value for item in excluded for reason in item.reasons)
        manifest = {
            "dataset_id": generation_id,
            "schema_version": DATASET_SCHEMA_VERSION,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "status": "PUBLISHED" if rows else "EMPTY_ELIGIBLE_SET",
            "split_status": split_status,
            "examples": len(rows),
            "exclusions": len(excluded),
            "counts": {
                "examples": len(rows),
                "exclusions": len(excluded),
                "reason_counts": dict(sorted(reason_counts.items())),
            },
            "split_counts": dict(sorted(Counter(assignments.values()).items())),
            "group_count": len(
                {
                    assignment.group_id
                    for assignment in (
                        split_result.assignments
                        if isinstance(split_result, SplitResult)
                        else tuple(split_result)
                    )
                }
            ),
            "time_range": {
                "first": min(timestamps) if timestamps else None,
                "last": max(timestamps) if timestamps else None,
            },
            "source_fingerprints": _plain(source_fingerprints or {}),
            "policy_versions": _plain(policy_versions or {}),
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
        for name in _FILES:
            info = manifest["files"][name]
            data = (root / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != info["sha256"]:
                errors.append(f"hash mismatch: {name}")
            if len(data) != info["size"] or data.count(b"\n") != info["count"]:
                errors.append(f"file metadata mismatch: {name}")
            for line in data.splitlines():
                if line:
                    _safe(json.loads(line))
        rows = [
            json.loads(line)
            for line in (root / "examples.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        ids = [row.get("example_id") for row in rows]
        if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
            errors.append("duplicate or invalid example IDs")
        if manifest.get("examples") != len(rows):
            errors.append("example count mismatch")
        # Every partition is a projection of examples.jsonl, never an
        # independent population.  This catches tampered IDs and leakage.
        example_by_id = {item.get("example_id"): item for item in rows}
        seen: dict[str, str] = {}
        for split in ("train", "validation", "test"):
            split_rows = [
                json.loads(line)
                for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            for item in split_rows:
                item_id = item.get("example_id")
                if item_id not in example_by_id:
                    errors.append(f"{split} contains unknown example: {item_id}")
                if item_id in seen and seen[item_id] != split:
                    errors.append(f"example crosses splits: {item_id}")
                seen[item_id] = split
                if item != example_by_id.get(item_id):
                    errors.append(f"{split} row differs from examples: {item_id}")
        if len(seen) != sum(manifest.get("split_counts", {}).values()):
            errors.append("split count mismatch")
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
