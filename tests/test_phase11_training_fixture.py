from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.fixtures.phase11_training import tiny_decoder_model, tiny_tokenizer


def test_tiny_decoder_and_fast_tokenizer_save_reload_without_download(tmp_path: Path, monkeypatch) -> None:
    if not all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "tokenizers")):
        pytest.skip("fixture ML dependencies missing: install torch, transformers, and tokenizers")

    import torch
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model download")))
    tokenizer = tiny_tokenizer()
    model = tiny_decoder_model(tokenizer)
    source = tmp_path / "tiny-source"
    tokenizer.save_pretrained(source)
    model.save_pretrained(source, safe_serialization=True)

    reloaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(source, local_files_only=True)
    reloaded_model = GPT2LMHeadModel.from_pretrained(source, local_files_only=True)
    assert reloaded_tokenizer.chat_template
    assert reloaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert reloaded_model.config.vocab_size == model.config.vocab_size
    assert sum(parameter.numel() for parameter in reloaded_model.parameters()) == sum(parameter.numel() for parameter in model.parameters())
