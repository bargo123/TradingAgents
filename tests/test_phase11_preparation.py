from __future__ import annotations

import json
from pathlib import Path

from tradingagents.datasets.models import CanonicalExampleV1, SplitAssignment
from tradingagents.datasets.splits import SplitResult
from tradingagents.datasets.writer import write_generation
from tradingagents.finetuning.phase10 import Phase10Generation
from tradingagents.finetuning.formatting import SFTFormatter
from tradingagents.finetuning.preparation import prepare_generation
from tradingagents.finetuning.tokenization import TokenizationPolicy
from tests.test_phase11_phase10 import FP


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
    first = prepare_generation(gen, tmp_path / "out", SFTFormatter(), policy)
    train = (tmp_path / "out" / "prepared" / "train.sft.jsonl").read_bytes()
    second = prepare_generation(gen, tmp_path / "out", SFTFormatter(), policy)
    assert train == (tmp_path / "out" / "prepared" / "train.sft.jsonl").read_bytes()
    assert second.manifest["train_count"] == 1
    assert json.loads(train)["example_id"] == "b"
