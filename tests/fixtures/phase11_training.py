"""Small, deterministic, test-only Phase 11 training fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tradingagents.datasets.models import CanonicalExampleV1, SplitAssignment
from tradingagents.datasets.splits import SplitResult
from tradingagents.datasets.writer import write_generation

FIXTURE_FINGERPRINTS = {
    phase: {
        "source_id": f"phase11-fixture-{phase}",
        "canonical_path": f"/phase11/fixture/{phase}",
        "schema_fingerprint": f"schema-{phase}",
        "file_sha256": f"file-{phase}",
        "snapshot_fingerprint": f"snapshot-{phase}",
        "contract_version": "phase10.source-adapter.v1",
    }
    for phase in ("phase56", "phase8", "phase9")
}


def fixture_rows() -> tuple[CanonicalExampleV1, ...]:
    """Return six rows: BUY/SELL/HOLD train data, validation, and one test row."""
    actions = ("BUY", "SELL", "HOLD", "BUY", "SELL", "HOLD")
    rows: list[CanonicalExampleV1] = []
    for index, action in enumerate(actions):
        decision_id = f"phase11-fixture-decision-{index}"
        rows.append(
            CanonicalExampleV1(
                example_id=f"phase11-fixture-{index}",
                decision={
                    "action": action,
                    "decision_id": decision_id,
                    "resolved_symbol": "EURUSD",
                    "requested_symbol": "EURUSD",
                    "analysis_profile": "fixture",
                    "analysis_timeframe": "M5",
                    "decision_context_status": "COMPLETE",
                    "analysis_snapshot_timestamp": f"2026-01-01T00:0{index}:00+00:00",
                },
                market={"symbol": "EURUSD", "point": "1.1000", "digits": 5},
                research={"evidence_refs": [f"fixture-evidence-{index}"]},
                outcome={"evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300},
                provenance={
                    "phase56": FIXTURE_FINGERPRINTS["phase56"],
                    "phase8": {"source_fingerprint": FIXTURE_FINGERPRINTS["phase8"]},
                    "phase9": {"source_fingerprint": FIXTURE_FINGERPRINTS["phase9"]},
                },
            )
        )
    return tuple(rows)


def fixture_generation(root: str | Path) -> Path:
    """Publish a fresh temporary Phase 10 generation for a test."""
    rows = fixture_rows()
    assignments = tuple(
        SplitAssignment(
            row.example_id,
            "train" if index < 3 else "validation" if index < 5 else "test",
            f"decision:{row.decision['decision_id']}",
        )
        for index, row in enumerate(rows)
    )
    return write_generation(
        root,
        rows,
        (),
        SplitResult(assignments, "COMPLETE"),
        dataset_id="generation-phase11-fixture",
        source_fingerprints=FIXTURE_FINGERPRINTS,
        metadata={"fixture": "phase11-training"},
    )


def tiny_tokenizer() -> Any:
    """Build a tiny fast tokenizer with a native explicit chat template."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "BUY": 7,
        "SELL": 8,
        "HOLD": 9,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>' }} {{ message['content'] }} {{ eos_token }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
    )
    tokenizer.name_or_path = "phase11-tiny-tokenizer"
    tokenizer.model_max_length = 256
    return tokenizer


def tiny_decoder_model(tokenizer: Any | None = None) -> Any:
    """Build a deterministic GPT-2-style decoder without loading a checkpoint."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    tokenizer = tokenizer or tiny_tokenizer()
    torch.manual_seed(1729)
    config = GPT2Config(
        vocab_size=max(int(tokenizer.vocab_size), 16),
        n_positions=256,
        n_ctx=256,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return GPT2LMHeadModel(config)


__all__ = ["FIXTURE_FINGERPRINTS", "fixture_generation", "fixture_rows", "tiny_decoder_model", "tiny_tokenizer"]
