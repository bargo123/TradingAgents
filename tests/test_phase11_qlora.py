from __future__ import annotations

import sys

import pytest

from tradingagents.finetuning.models import QloraConfig


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
