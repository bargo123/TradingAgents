from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from tradingagents.datasets.models import (
    CanonicalExampleV1,
    DatasetExclusion,
    DatasetExclusionReason,
)
from tradingagents.datasets.splits import SPLIT_STATUS_COMPLETE, SplitAssignment, SplitResult
from tradingagents.datasets.writer import (
    GenerationExistsError,
    validate_generation,
    write_generation,
)


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
        provenance={"phase56": {"file_sha256": "abc"}},
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
        tmp_path, rows, (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="dataset-a"
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
        tmp_path, rows, (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="same"
    )
    with pytest.raises(GenerationExistsError):
        write_generation(
            tmp_path, rows, (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="same"
        )
    assert root.exists()


def test_tampering_is_detected(tmp_path: Path):
    root = write_generation(
        tmp_path, (example(),), (), SplitResult((), "INSUFFICIENT_DATA"), dataset_id="tamper"
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
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "EMPTY_ELIGIBLE_SET"
    assert json.loads((root / "excluded.jsonl").read_text())["reasons"] == ["OUTCOME_UNAVAILABLE"]
