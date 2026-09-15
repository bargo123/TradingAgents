"""Read-only binding for published Phase 10 dataset generations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tradingagents.datasets import writer
from tradingagents.datasets.models import (
    CANONICALIZATION_VERSION,
    DATASET_SCHEMA_VERSION,
    ELIGIBILITY_POLICY_VERSION,
    EXAMPLE_SCHEMA_VERSION,
    SOURCE_ADAPTER_VERSION,
    SPLIT_POLICY_VERSION,
    CanonicalExampleV1,
)

from .errors import EmptyEligibleSetError, Phase10InvalidError
from .models import DatasetBinding


def _reason(value: Any) -> str:
    text = str(value).replace("\x00", " ").strip()
    return text[:512] or "generation validation failed"


def _rows(path: Path, expected: str) -> tuple[CanonicalExampleV1, ...]:
    result: list[CanonicalExampleV1] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Phase10InvalidError(f"cannot read {expected}: {_reason(exc)}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line:
            continue
        try:
            value = json.loads(line)
            row = CanonicalExampleV1(**value) if isinstance(value, dict) else None
            if row is None:
                raise ValueError("row must be an object")
        except Exception as exc:
            raise Phase10InvalidError(f"invalid {expected} row {line_no}: {_reason(exc)}") from exc
        result.append(row)
    return tuple(result)


def _validate_fingerprints(value: Any) -> dict[str, dict[str, str]]:
    """Return a defensive copy of the closed Phase 10 fingerprint contract."""
    phases = {"phase56", "phase8", "phase9"}
    fields = {
        "source_id", "canonical_path", "schema_fingerprint", "file_sha256",
        "snapshot_fingerprint", "contract_version",
    }
    if not isinstance(value, Mapping) or set(value) != phases:
        raise Phase10InvalidError("source fingerprints are incomplete")
    result: dict[str, dict[str, str]] = {}
    for phase in sorted(phases):
        fingerprint = value[phase]
        if not isinstance(fingerprint, Mapping) or set(fingerprint) != fields:
            raise Phase10InvalidError(f"invalid {phase} source fingerprint")
        if any(not isinstance(item, str) or not item or len(item) > 512 for item in fingerprint.values()):
            raise Phase10InvalidError(f"invalid {phase} source fingerprint value")
        result[phase] = dict(fingerprint)
    return result


def _row_fingerprints(row: CanonicalExampleV1) -> dict[str, Any]:
    provenance = row.provenance
    phase8 = provenance.get("phase8", {})
    phase9 = provenance.get("phase9", {})
    return {
        "phase56": provenance.get("phase56"),
        "phase8": phase8.get("source_fingerprint") if isinstance(phase8, Mapping) else None,
        "phase9": phase9.get("source_fingerprint") if isinstance(phase9, Mapping) else None,
    }


@dataclass(frozen=True, slots=True)
class Phase10Generation:
    """Validated immutable generation data used by training."""

    path: Path
    generation_id: str
    manifest: dict[str, Any]
    train: tuple[CanonicalExampleV1, ...]
    validation: tuple[CanonicalExampleV1, ...]
    manifest_hash: str
    file_hashes: dict[str, str]
    policy_versions: dict[str, str]
    source_fingerprints: dict[str, dict[str, str]]

    @property
    def train_rows(self) -> tuple[CanonicalExampleV1, ...]:
        return self.train

    @property
    def validation_rows(self) -> tuple[CanonicalExampleV1, ...]:
        return self.validation

    @property
    def binding(self) -> DatasetBinding:
        return DatasetBinding(
            generation_id=self.generation_id,
            generation_path=self.path,
            manifest_hash=self.manifest_hash,
            train_count=len(self.train),
            validation_count=len(self.validation),
        )

    @classmethod
    def open(cls, path: str | Path) -> Phase10Generation:
        root = Path(path).resolve()
        try:
            report = writer.validate_generation(root)
            valid = bool(getattr(report, "valid", False))
            errors = getattr(report, "errors", ())
        except Exception as exc:
            raise Phase10InvalidError(f"validation failed: {_reason(exc)}") from exc
        if not valid:
            bounded = [_reason(item) for item in tuple(errors)[:8]]
            raise Phase10InvalidError("; ".join(bounded) or "generation validation failed")

        try:
            manifest_bytes = (root / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as exc:
            raise Phase10InvalidError(f"invalid manifest: {_reason(exc)}") from exc
        if not isinstance(manifest, dict):
            raise Phase10InvalidError("manifest must be an object")
        generation_id = manifest.get("dataset_id")
        if generation_id != root.name:
            raise Phase10InvalidError("generation identity does not match directory")
        expected = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "example_schema_version": EXAMPLE_SCHEMA_VERSION,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "source_adapter_version": SOURCE_ADAPTER_VERSION,
        }
        mismatches = [name for name, value in expected.items() if manifest.get(name) != value]
        if mismatches:
            raise Phase10InvalidError("incompatible manifest: " + ", ".join(mismatches[:8]))
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise Phase10InvalidError("manifest files metadata is missing")
        hashes: dict[str, str] = {}
        for name in ("train.jsonl", "validation.jsonl"):
            info = files.get(name)
            if not isinstance(info, dict) or not isinstance(info.get("sha256"), str):
                raise Phase10InvalidError(f"missing file metadata: {name}")
            try:
                data = (root / name).read_bytes()
            except OSError as exc:
                raise Phase10InvalidError(f"cannot read {name}: {_reason(exc)}") from exc
            digest = hashlib.sha256(data).hexdigest()
            if digest != info["sha256"]:
                raise Phase10InvalidError(f"hash mismatch: {name}")
            hashes[name] = digest
        policies = manifest.get("policy_versions")
        expected_policies = {
            "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "source_adapter_version": SOURCE_ADAPTER_VERSION,
        }
        if policies != expected_policies:
            raise Phase10InvalidError("incompatible policy versions")
        fingerprints = _validate_fingerprints(manifest.get("source_fingerprints"))
        train = _rows(root / "train.jsonl", "train.jsonl")
        validation = _rows(root / "validation.jsonl", "validation.jsonl")
        for row in train + validation:
            if _row_fingerprints(row) != fingerprints:
                raise Phase10InvalidError(
                    f"source fingerprint mismatch: {row.example_id}"
                )
        if manifest.get("status") == "EMPTY_ELIGIBLE_SET" or not train and not validation:
            raise EmptyEligibleSetError("generation has no eligible training examples")
        split_counts = manifest.get("split_counts")
        if not isinstance(split_counts, dict) or any(
            split_counts.get(name) != count
            for name, count in (("train", len(train)), ("validation", len(validation)))
        ):
            raise Phase10InvalidError("split counts do not match bound rows")
        return cls(root, generation_id, manifest, train, validation,
                   hashlib.sha256(manifest_bytes).hexdigest(), hashes,
                   dict(policies), fingerprints)

    def audit_test_count(self) -> int:
        """Read test metadata/rows only for an explicit audit, returning count."""
        try:
            data = (self.path / "test.jsonl").read_bytes()
        except OSError as exc:
            raise Phase10InvalidError(f"cannot read test.jsonl: {_reason(exc)}") from exc
        info = self.manifest.get("files", {}).get("test.jsonl", {})
        if hashlib.sha256(data).hexdigest() != info.get("sha256"):
            raise Phase10InvalidError("hash mismatch: test.jsonl")
        return sum(1 for line in data.splitlines() if line.strip())


__all__ = ["Phase10Generation"]
