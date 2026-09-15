from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

from tests.fixtures.phase11_training import (
    fixture_generation,
    tiny_decoder_model,
    tiny_tokenizer,
)
from tradingagents.finetuning.artifacts import publish_run, validate_run_hashes
from tradingagents.finetuning.formatting import SFTFormatter
from tradingagents.finetuning.models import LoraConfig, Phase11Status, TrainingConfig
from tradingagents.finetuning.phase10 import Phase10Generation
from tradingagents.finetuning.preparation import prepare_generation
from tradingagents.finetuning.tokenization import TokenizationPolicy
from tradingagents.finetuning.training import train
from tradingagents.finetuning.validation import validate_run

pytestmark = pytest.mark.smoke


def _training_stack_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft"))


def test_tiny_cpu_lora_smoke_is_offline_and_keeps_test_split_untouched(tmp_path: Path, monkeypatch) -> None:
    if not _training_stack_available():
        pytest.skip("optional training extra missing: install torch, transformers, and peft")

    # This guard proves the explicit tiny-model path cannot accidentally leave
    # the local process for a hosted service during the smoke.
    import socket

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network call")))
    generation_path = fixture_generation(tmp_path / "generation")
    generation = Phase10Generation.open(generation_path)
    tokenizer = tiny_tokenizer()
    policy = TokenizationPolicy(tokenizer=tokenizer, max_length=256)
    prepared = prepare_generation(generation, tmp_path / "prepared", SFTFormatter(), policy)
    test_before = (generation_path / "test.jsonl").read_bytes()

    base = tiny_decoder_model(tokenizer)
    base_path = tmp_path / "base-model"
    tokenizer.save_pretrained(base_path)
    base.save_pretrained(base_path, safe_serialization=True)
    config = TrainingConfig(
        base_model=str(base_path),
        epochs=1,
        max_steps=2,
        batch_size=1,
        gradient_checkpointing=False,
        lora=LoraConfig(r=2, alpha=4, target_modules=("c_attn",)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = train(
            prepared,
            model=base,
            tokenizer_policy=policy,
            config=config,
            output_dir=tmp_path / "adapter",
        )
    assert result.status is Phase11Status.COMPLETE
    assert result.optimizer_steps == 2
    assert result.metrics is not None
    assert result.metrics.examples == 3
    assert result.metrics.steps == 2
    assert result.metrics.trainable_parameters > 0
    assert result.metrics.total_parameters > result.metrics.trainable_parameters
    assert result.validation_loss is not None
    assert (generation_path / "test.jsonl").read_bytes() == test_before

    prepared_manifest = json.loads((prepared.output_root / "prepared" / "manifest.json").read_text(encoding="utf-8"))
    assert prepared_manifest["train_count"] == 3
    assert prepared_manifest["validation_count"] == 2
    rows = []
    for split in ("train", "validation"):
        rows.extend(json.loads(line) for line in (prepared.output_root / "prepared" / f"{split}.sft.jsonl").read_text(encoding="utf-8").splitlines() if line)
    assert {row["target"]["action"] for row in rows} == {"BUY", "SELL", "HOLD"}
    serialized = json.dumps(rows, sort_keys=True).lower()
    assert not any(term in serialized for term in ("prompt", "completion", "reasoning", "chain_of_thought", "secret", "api_key"))

    attachment = result.attachment
    assert attachment is not None
    assert all(
        not parameter.requires_grad or "lora_" in name.lower()
        for name, parameter in attachment.model.named_parameters()
    )
    assert attachment.trainable_parameters == result.metrics.trainable_parameters

    from peft import PeftModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reloaded = PeftModel.from_pretrained(tiny_decoder_model(tokenizer), tmp_path / "adapter", is_trainable=False)
    assert any("lora_" in name.lower() for name, _ in reloaded.named_parameters())
    resolved_config = config.to_dict()
    assert result.provenance is not None
    resolved_config["model_provenance"] = result.provenance.to_dict()
    run = publish_run(
        tmp_path / "runs",
        run_id="run-tiny-smoke",
        config=resolved_config,
        dataset=prepared_manifest,
        metrics=result.metrics.to_dict(),
        prepared={
            "manifest.json": prepared.output_root / "prepared" / "manifest.json",
            "train.sft.jsonl": prepared.output_root / "prepared" / "train.sft.jsonl",
            "validation.sft.jsonl": prepared.output_root / "prepared" / "validation.sft.jsonl",
        },
        adapter=tmp_path / "adapter",
    )
    assert (run / "metrics.json").is_file()
    assert (run / "run_manifest.json").is_file()
    assert validate_run_hashes(run)
    report = validate_run(run)
    assert report.valid, report.errors
