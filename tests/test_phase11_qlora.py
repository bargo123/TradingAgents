from __future__ import annotations

import sys

import pytest

from tradingagents.finetuning.models import LoraConfig, Phase11Status, QloraConfig, TrainingConfig
from tradingagents.finetuning.tokenization import TokenizationPolicy


def test_qlora_config_is_strict_nf4_double_quant() -> None:
    from tradingagents.finetuning.qlora import QloraCapabilityError, validate_qlora_config

    assert validate_qlora_config().quant_type == "nf4"
    with pytest.raises(QloraCapabilityError):
        validate_qlora_config(QloraConfig(double_quant=False))


def test_cpu_is_gpu_required_and_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from tradingagents.finetuning.qlora import QloraCapabilityError, check_qlora_capability

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(QloraCapabilityError, match="GPU_REQUIRED"):
        check_qlora_capability()
    assert "bitsandbytes" not in sys.modules


def test_quantization_kwargs_are_explicit() -> None:
    from tradingagents.finetuning.qlora import quantization_kwargs

    assert quantization_kwargs() == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }


def test_training_qlora_mode_fails_closed_instead_of_running_lora(monkeypatch) -> None:
    from tests.test_phase11_lora import example
    from tradingagents.finetuning import training
    from tradingagents.finetuning.qlora import QloraCapabilityError

    class StubCuda:
        @staticmethod
        def is_available():
            return False

    class StubTorch:
        cuda = StubCuda()

        @staticmethod
        def manual_seed(_seed):
            return None

    monkeypatch.setattr(training, "_load_optional_stack", lambda: (StubTorch, object(), None))
    monkeypatch.setattr(
        training,
        "check_qlora_capability",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            QloraCapabilityError("GPU_REQUIRED: CUDA is unavailable")
        ),
    )
    result = training.train(
        {"train": [example()], "validation": []},
        model=object(),
        tokenizer_policy=TokenizationPolicy(tokenizer=object(), max_length=128),
        config=TrainingConfig(
            mode="qlora",
            epochs=1,
            max_steps=1,
            gradient_checkpointing=False,
            lora=LoraConfig(r=2, alpha=4, target_modules=("c_attn",)),
        ),
    )
    assert result.status is Phase11Status.GPU_REQUIRED
