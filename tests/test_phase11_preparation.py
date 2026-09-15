from __future__ import annotations

import json
from pathlib import Path

from tests.test_phase11_phase10 import FP
from tradingagents.datasets.models import CanonicalExampleV1, SplitAssignment
from tradingagents.datasets.splits import SplitResult
from tradingagents.datasets.writer import write_generation
from tradingagents.finetuning.formatting import SFTFormatter
from tradingagents.finetuning.phase10 import Phase10Generation
from tradingagents.finetuning.preparation import prepare_generation, validate_prepared
from tradingagents.finetuning.tokenization import TokenizationPolicy


class TinyTokenizer:
    model_max_length = 1000
    name_or_path = "tiny"
    chat_template = "native"
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        value = "".join(m["role"] + m["content"] for m in messages)
        return self.encode(value) if tokenize else value
    def encode(self, value, **kwargs):
        return list(range(len(value)))


def generation(tmp_path: Path) -> Path:
    rows = tuple(CanonicalExampleV1(i, decision={"action": "BUY", "resolved_symbol": "EURUSD", "decision_id": i, "analysis_snapshot_timestamp": f"2026-01-01T00:0{n}:00+00:00"}, outcome={"evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300 + n}, provenance={"phase56": FP["phase56"], "phase8": {"source_fingerprint": FP["phase8"]}, "phase9": {"source_fingerprint": FP["phase9"]}}) for n, i in enumerate(("b", "a")))
    split = SplitResult((SplitAssignment("b", "train", "decision:b"), SplitAssignment("a", "validation", "decision:a")), "COMPLETE")
    return write_generation(tmp_path, rows, (), split, dataset_id="generation-a", source_fingerprints=FP)


def test_preparation_is_sorted_and_idempotent(tmp_path: Path) -> None:
    gen = Phase10Generation.open(generation(tmp_path / "gen"))
    policy = TokenizationPolicy(tokenizer=TinyTokenizer(), max_length=1000)
    prepare_generation(gen, tmp_path / "out", SFTFormatter(), policy)
    train = (tmp_path / "out" / "prepared" / "train.sft.jsonl").read_bytes()
    second = prepare_generation(gen, tmp_path / "out", SFTFormatter(), policy)
    assert train == (tmp_path / "out" / "prepared" / "train.sft.jsonl").read_bytes()
    assert second.manifest["train_count"] == 1
    assert json.loads(train)["example_id"] == "b"


def test_failed_publication_keeps_staging_for_recovery_diagnostics(tmp_path: Path, monkeypatch) -> None:
    from tradingagents.finetuning import preparation

    gen = Phase10Generation.open(generation(tmp_path / "gen"))
    policy = TokenizationPolicy(tokenizer=TinyTokenizer(), max_length=1000)

    def fail_rename(*_args, **_kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(preparation.os, "rename", fail_rename)
    try:
        prepare_generation(gen, tmp_path / "out", SFTFormatter(), policy)
    except OSError as exc:
        assert "simulated" in str(exc)
    else:  # pragma: no cover - assertion keeps the RED test explicit
        raise AssertionError("publication interruption should fail")
    assert list((tmp_path / "out").glob(".prepared-staging-*"))


def test_prepared_manifest_rejects_unexpected_files(tmp_path: Path) -> None:
    gen = Phase10Generation.open(generation(tmp_path / "gen"))
    policy = TokenizationPolicy(tokenizer=TinyTokenizer(), max_length=1000)
    result = prepare_generation(gen, tmp_path / "out", SFTFormatter(), policy)
    (result.output_root / "prepared" / "unexpected.json").write_text("{}", encoding="utf-8")
    import pytest

    from tradingagents.finetuning.errors import Phase10InvalidError

    with pytest.raises(Phase10InvalidError, match="unexpected"):
        validate_prepared(result.output_root)


def test_prepared_manifest_revalidates_its_phase10_generation_binding(tmp_path: Path) -> None:
    gen = Phase10Generation.open(generation(tmp_path / "gen"))
    policy = TokenizationPolicy(tokenizer=TinyTokenizer(), max_length=1000)
    result = prepare_generation(gen, tmp_path / "out", SFTFormatter(), policy)
    manifest_path = result.output_root / "prepared" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_manifest_hash"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    import pytest

    from tradingagents.finetuning.errors import Phase10InvalidError

    with pytest.raises(Phase10InvalidError, match="generation"):
        validate_prepared(result.output_root)
