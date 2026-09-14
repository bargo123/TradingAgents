from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

import tradingagents.datasets.writer as writer_module
from tradingagents.datasets.models import (
    CanonicalExampleV1,
    DatasetExclusion,
    DatasetExclusionReason,
)
from tradingagents.datasets.splits import SPLIT_STATUS_COMPLETE, SplitAssignment, SplitResult
from tradingagents.datasets.writer import (
    GenerationExistsError,
    GenerationValidationError,
    validate_generation,
    write_generation,
)

FINGERPRINTS = {
    phase: {
        "source_id": phase,
        "canonical_path": f"{phase}.sqlite3",
        "schema_fingerprint": f"schema-{phase}",
        "file_sha256": f"file-{phase}",
        "snapshot_fingerprint": f"snapshot-{phase}",
        "contract_version": "phase10.source-adapter.v1",
    }
    for phase in ("phase56", "phase8", "phase9")
}
POLICIES = {
    "eligibility_policy_version": "phase10.eligibility.v1",
    "canonicalization_version": "phase10.canonicalization.v1",
    "split_policy_version": "phase10.split.v1",
    "source_adapter_version": "phase10.source-adapter.v1",
}


def example(example_id: str = "ex-1", ts: str = "2026-01-01T00:00:00+00:00"):
    return CanonicalExampleV1(
        example_id=example_id,
        decision={
            "decision_id": example_id,
            "source_run_id": "run-1",
            "analysis_snapshot_timestamp": datetime.fromisoformat(ts),
            "action": "BUY",
        },
        market={"snapshot": {"symbol": "EURUSD"}},
        research={"context_integrity": "COMPLETE"},
        outcome={"evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300},
        trust={"tier": "A"},
        provenance={
            "phase56": FINGERPRINTS["phase56"],
            "phase8": {"source_fingerprint": FINGERPRINTS["phase8"]},
            "phase9": {"source_fingerprint": FINGERPRINTS["phase9"]},
        },
    )


def splits(rows):
    return SplitResult(
        tuple(
            SplitAssignment(row.example_id, split, "run-1")
            for row, split in zip(rows, ("train", "validation"), strict=False)
        ),
        SPLIT_STATUS_COMPLETE,
    )


def test_write_is_canonical_and_validates(tmp_path: Path):
    rows = (example(), example("ex-2", "2026-01-02T00:00:00+00:00"))
    root = write_generation(
        tmp_path, rows, (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="dataset-a",
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    assert validate_generation(root).valid
    assert (root / "examples.jsonl").read_bytes().endswith(b"\n")
    assert (root / "examples.jsonl").read_bytes() == (root / "examples.jsonl").read_bytes()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["safety"] == {
        "llm_calls": 0,
        "mt5_calls": 0,
        "network_attempts": 0,
        "tool_calls": 0,
    }
    assert manifest["files"]["examples.jsonl"]["count"] == 2


def test_existing_generation_is_never_overwritten(tmp_path: Path):
    rows = (example(),)
    root = write_generation(
        tmp_path, rows, (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="same",
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    with pytest.raises(GenerationExistsError):
        write_generation(
            tmp_path, rows, (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="same",
            source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
        )
    assert root.exists()


def test_tampering_is_detected(tmp_path: Path):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="tamper",
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    (root / "examples.jsonl").write_text("{}\n", encoding="utf-8", newline="\n")
    report = validate_generation(root)
    assert not report.valid
    assert any("hash" in error for error in report.errors)


def test_empty_population_is_explicit(tmp_path: Path):
    root = write_generation(
        tmp_path,
        (),
        (DatasetExclusion("d1", (DatasetExclusionReason.OUTCOME_UNAVAILABLE,)),),
        SplitResult((), "INSUFFICIENT_DATA"),
        dataset_id="empty",
        source_fingerprints=FINGERPRINTS,
        policy_versions=POLICIES,
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "EMPTY_ELIGIBLE_SET"
    assert json.loads((root / "excluded.jsonl").read_text())["reasons"] == ["OUTCOME_UNAVAILABLE"]


def test_validation_rejects_tampered_canonical_schema(tmp_path: Path):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="schema",
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    path = root / "examples.jsonl"
    row = json.loads(path.read_text())
    row["decision"] = "not-a-mapping"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
    report = validate_generation(root)
    assert not report.valid
    assert any("schema" in error or "decision" in error for error in report.errors)


def test_validation_rejects_source_run_split_leakage(tmp_path: Path):
    rows = (example("ex-1"), example("ex-2", "2026-01-02T00:00:00+00:00"))
    split = SplitResult(
        (SplitAssignment("ex-1", "train", "run-1"), SplitAssignment("ex-2", "test", "run-1")),
        "COMPLETE",
    )
    with pytest.raises(ValueError, match="cross-split"):
        write_generation(tmp_path, rows, (), split, dataset_id="leak",
                         source_fingerprints=FINGERPRINTS, policy_versions=POLICIES)


def test_validation_rejects_manifest_fingerprint_or_version_tampering(tmp_path: Path):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="fingerprint",
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_fingerprints"]["phase56"]["file_sha256"] = "tampered"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8", newline="\n")
    report = validate_generation(root)
    assert not report.valid
    assert any("fingerprint" in error for error in report.errors)


def test_generator_split_assignments_are_materialized_once(tmp_path: Path):
    rows = (example("ex-1"), example("ex-2", "2026-01-02T00:00:00+00:00"))
    assignments = (item for item in (
        SplitAssignment("ex-1", "train", "run:run-1"),
        SplitAssignment("ex-2", "train", "run:run-1"),
    ))
    root = write_generation(tmp_path, rows, (), assignments, dataset_id="generator",
                            source_fingerprints=FINGERPRINTS, policy_versions=POLICIES)
    first = json.loads((root / "train.jsonl").read_text().splitlines()[0])
    assert first["example_id"] == "ex-1"


def test_repeated_builds_in_distinct_roots_have_identical_bytes(tmp_path: Path):
    kwargs = {
        "examples": (example(),), "exclusions": (),
        "split_result": SplitResult((), "INSUFFICIENT_DATA"),
        "source_fingerprints": FINGERPRINTS, "policy_versions": POLICIES,
    }
    first = write_generation(tmp_path / "one", **kwargs)
    second = write_generation(tmp_path / "two", **kwargs)
    assert {
        name: (first / name).read_bytes() for name in (
            "manifest.json", "examples.jsonl", "excluded.jsonl", "train.jsonl",
            "validation.jsonl", "test.jsonl",
        )
    } == {
        name: (second / name).read_bytes() for name in (
            "manifest.json", "examples.jsonl", "excluded.jsonl", "train.jsonl",
            "validation.jsonl", "test.jsonl",
        )
    }


def test_interrupted_publish_leaves_unpublished_staging(tmp_path: Path, monkeypatch):
    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(writer_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        write_generation(tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"),
                         dataset_id="interrupted", source_fingerprints=FINGERPRINTS,
                         policy_versions=POLICIES)
    assert not (tmp_path / "interrupted").exists()
    assert list(tmp_path.glob(".staging-interrupted-*"))


def test_sensitive_manifest_metadata_is_rejected_before_publish(tmp_path: Path):
    with pytest.raises(GenerationValidationError, match="forbidden"):
        write_generation(
            tmp_path,
            (example(),),
            (),
            SplitResult((), "INSUFFICIENT_DATA"),
            dataset_id="sensitive-metadata",
            source_fingerprints=FINGERPRINTS,
            policy_versions=POLICIES,
            metadata={"nested": {"private_reasoning": "must not persist"}},
        )
    assert not (tmp_path / "sensitive-metadata").exists()


def test_validate_generation_rejects_nonzero_safety_counter(tmp_path: Path):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"),
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["safety"]["network_attempts"] = 1
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8", newline="\n",
    )

    report = validate_generation(root)

    assert not report.valid
    assert any("safety" in error for error in report.errors)


@pytest.mark.parametrize("value", [False, 0.0])
def test_validate_generation_rejects_non_integer_zero_safety_counter(
    tmp_path: Path, value,
):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"),
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["safety"]["network_attempts"] = value
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8", newline="\n",
    )

    report = validate_generation(root)

    assert not report.valid
    assert any("safety" in error for error in report.errors)


def test_validate_generation_requires_exact_safety_counter_keys(tmp_path: Path):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"),
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["safety"]["mt5_calls"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8", newline="\n")

    report = validate_generation(root)

    assert not report.valid
    assert any("safety" in error for error in report.errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "UNKNOWN"),
        ("generation_id_mode", "UNKNOWN"),
        ("reproducibility", None),
        ("time_range", {"first": "2026-01-02T00:00:00", "last": None}),
    ],
)
def test_validate_generation_rejects_invalid_manifest_contract_fields(
    tmp_path: Path, field, value,
):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"),
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8", newline="\n")

    report = validate_generation(root)

    assert not report.valid
    assert any(field.replace("_", " ") in error or field in error for error in report.errors)


def test_validate_generation_rejects_status_inconsistent_with_population(tmp_path: Path):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"),
        dataset_id="population-status", source_fingerprints=FINGERPRINTS,
        policy_versions=POLICIES,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "EMPTY_ELIGIBLE_SET"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8", newline="\n")

    report = validate_generation(root)

    assert not report.valid
    assert any("status" in error and "population" in error for error in report.errors)


def test_validate_generation_rejects_tampered_explicit_dataset_id(tmp_path: Path):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"),
        dataset_id="explicit-id", source_fingerprints=FINGERPRINTS,
        policy_versions=POLICIES,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_id"] = "different-id"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8", newline="\n")

    report = validate_generation(root)

    assert not report.valid
    assert any("dataset id" in error for error in report.errors)


def test_validate_generation_requires_canonical_reproducibility_values(tmp_path: Path):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"),
        dataset_id="reproducibility", source_fingerprints=FINGERPRINTS,
        policy_versions=POLICIES,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reproducibility"]["serialization"] = "non-canonical"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8", newline="\n")

    report = validate_generation(root)

    assert not report.valid
    assert any("reproducibility" in error for error in report.errors)


def test_validate_generation_requires_actual_utc_time_range(tmp_path: Path):
    root = write_generation(
        tmp_path,
        (example("ex-1", "2026-01-01T00:00:00+00:00"),
         example("ex-2", "2026-01-02T00:00:00+00:00")),
        (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="time-range",
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["time_range"]["first"] = "2025-12-31T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8", newline="\n")

    report = validate_generation(root)

    assert not report.valid
    assert any("time_range" in error and "summary" in error for error in report.errors)


def test_write_generation_normalizes_time_range_to_utc(tmp_path: Path):
    root = write_generation(
        tmp_path,
        (example("ex-1", "2026-01-01T02:00:00+02:00"),
         example("ex-2", "2026-01-01T01:00:00+00:00")),
        (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="utc-range",
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["time_range"] == {
        "first": "2026-01-01T00:00:00Z",
        "last": "2026-01-01T01:00:00Z",
    }


def test_default_generation_id_includes_source_fingerprints(tmp_path: Path):
    first = write_generation(
        tmp_path / "first", (),
        (DatasetExclusion("ex-1", (DatasetExclusionReason.OUTCOME_UNAVAILABLE,)),),
        SplitResult((), "INSUFFICIENT_DATA"),
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    changed = {phase: dict(value) for phase, value in FINGERPRINTS.items()}
    changed["phase56"]["file_sha256"] = "changed-file-sha"
    second = write_generation(
        tmp_path / "second", (),
        (DatasetExclusion("ex-1", (DatasetExclusionReason.OUTCOME_UNAVAILABLE,)),),
        SplitResult((), "INSUFFICIENT_DATA"),
        source_fingerprints=changed, policy_versions=POLICIES,
    )

    assert first.name != second.name


def test_default_generation_id_includes_split_inputs(tmp_path: Path):
    first = write_generation(
        tmp_path / "first", (example(),), (), SplitResult((), "INSUFFICIENT_DATA"),
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )
    second = write_generation(
        tmp_path / "second", (example(),), (),
        SplitResult((SplitAssignment("ex-1", "train", "run:run-1"),), "COMPLETE"),
        source_fingerprints=FINGERPRINTS, policy_versions=POLICIES,
    )

    assert first.name != second.name


def test_unknown_policy_keys_are_rejected_during_write(tmp_path: Path):
    with pytest.raises(GenerationValidationError, match="unsupported policy key"):
        write_generation(
            tmp_path,
            (example(),),
            (),
            SplitResult((), "INSUFFICIENT_DATA"),
            dataset_id="unknown-policy",
            source_fingerprints=FINGERPRINTS,
            policy_versions={**POLICIES, "future_policy": "v2"},
        )


def test_validate_generation_validates_exclusion_schema_after_hash_refresh(tmp_path: Path):
    root = write_generation(
        tmp_path,
        (),
        (DatasetExclusion("d1", (DatasetExclusionReason.OUTCOME_UNAVAILABLE,)),),
        SplitResult((), "INSUFFICIENT_DATA"),
        dataset_id="bad-exclusion",
        source_fingerprints=FINGERPRINTS,
        policy_versions=POLICIES,
    )
    path = root / "excluded.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["reasons"] = ["NOT_A_CLOSED_REASON"]
    data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(data)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["excluded.jsonl"] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "count": 1,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    report = validate_generation(root)
    assert not report.valid
    assert any("excluded.jsonl" in error and "schema invalid" in error for error in report.errors)


def test_validate_generation_cross_checks_exact_split_counts(tmp_path: Path):
    root = write_generation(
        tmp_path,
        (example(),),
        (),
        SplitResult((), "INSUFFICIENT_DATA"),
        dataset_id="bad-split-counts",
        source_fingerprints=FINGERPRINTS,
        policy_versions=POLICIES,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_counts"] = {"train": 0}
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    report = validate_generation(root)
    assert not report.valid
    assert any("split count mismatch" in error for error in report.errors)
