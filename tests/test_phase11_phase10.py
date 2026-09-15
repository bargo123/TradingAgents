from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.datasets.models import CanonicalExampleV1, SplitAssignment
from tradingagents.datasets.splits import SplitResult
from tradingagents.datasets.writer import write_generation
from tradingagents.finetuning.errors import EmptyEligibleSetError, Phase10InvalidError
from tradingagents.finetuning.phase10 import Phase10Generation

FP = {p: {"source_id": p, "canonical_path": f"/{p}", "schema_fingerprint": "s",
          "file_sha256": "f", "snapshot_fingerprint": "t", "contract_version": "v"}
      for p in ("phase56", "phase8", "phase9")}


def _row(i: str) -> CanonicalExampleV1:
    return CanonicalExampleV1(i, decision={"decision_id": i, "analysis_snapshot_timestamp": "2026-01-01T00:00:00Z"},
                              provenance={"phase56": FP["phase56"], "phase8": {"source_fingerprint": FP["phase8"]},
                                          "phase9": {"source_fingerprint": FP["phase9"]}})


def _generation(tmp_path: Path, *, empty: bool = False) -> Path:
    rows = () if empty else (_row("a"), _row("b"))
    splits = SplitResult((), "INSUFFICIENT_DATA") if empty else SplitResult((
        SplitAssignment("a", "train", "decision:a"),
        SplitAssignment("b", "validation", "decision:b"),
    ), "COMPLETE")
    return write_generation(tmp_path, rows, (), splits, dataset_id="generation-a", source_fingerprints=FP)


def test_open_binds_rows_hashes_policies_and_fingerprints(tmp_path: Path) -> None:
    generation = Phase10Generation.open(_generation(tmp_path))
    assert generation.generation_id == "generation-a"
    assert [r.example_id for r in generation.train] == ["a"]
    assert [r.example_id for r in generation.validation] == ["b"]
    assert generation.policy_versions["split_policy_version"] == "phase10.split.v1"
    assert generation.source_fingerprints["phase56"] == FP["phase56"]
    assert generation.manifest_hash
    assert set(generation.file_hashes) >= {"train.jsonl", "validation.jsonl"}


def test_invalid_or_empty_generation_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(EmptyEligibleSetError):
        Phase10Generation.open(_generation(tmp_path, empty=True))


def test_validator_is_called_and_test_rows_are_not_consumed(tmp_path: Path, monkeypatch) -> None:
    path = _generation(tmp_path)
    called = []
    import tradingagents.datasets.writer as writer
    real = writer.validate_generation
    monkeypatch.setattr(writer, "validate_generation", lambda p: (called.append(p) or real(p)))
    generation = Phase10Generation.open(path)
    assert called == [path]
    assert generation.audit_test_count() == 0


def test_tampered_manifest_or_row_hash_fails(tmp_path: Path) -> None:
    path = _generation(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["policy_versions"]["split_policy_version"] = "bad"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(Phase10InvalidError):
        Phase10Generation.open(path)
