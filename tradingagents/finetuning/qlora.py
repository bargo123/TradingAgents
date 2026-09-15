"""Fail-closed QLoRA capability checks (optional dependencies stay lazy)."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from .models import QloraConfig


class QloraCapabilityError(RuntimeError):
    """QLoRA cannot be used with the requested immutable settings."""


@dataclass(frozen=True, slots=True)
class QloraCapability:
    available: bool
    config: QloraConfig
    cuda: bool
    bitsandbytes: bool
    reason: str = ""


def validate_qlora_config(config: QloraConfig | None = None) -> QloraConfig:
    cfg = config or QloraConfig()
    if not cfg.load_in_4bit or cfg.quant_type != "nf4" or not cfg.double_quant:
        raise QloraCapabilityError("QLORA_UNAVAILABLE: QLoRA requires 4-bit NF4 double quantization")
    if cfg.compute_dtype not in {"bfloat16", "float16"}:
        raise QloraCapabilityError("QLORA_UNAVAILABLE: unsupported compute dtype")
    return cfg


def check_qlora_capability(config: QloraConfig | None = None) -> QloraCapability:
    """Check CUDA and bitsandbytes without importing either on module import."""
    cfg = validate_qlora_config(config)
    try:
        import torch  # noqa: PLC0415
        cuda = bool(torch.cuda.is_available())
    except (ImportError, AttributeError, RuntimeError):
        cuda = False
    if not cuda:
        raise QloraCapabilityError("GPU_REQUIRED: QLoRA requires CUDA")
    available = importlib.util.find_spec("bitsandbytes") is not None
    if not available:
        raise QloraCapabilityError("QLORA_UNAVAILABLE: bitsandbytes is not installed")
    # Import only after CUDA and module presence checks; this catches broken wheels.
    try:
        import bitsandbytes  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - platform-specific wheel failures
        raise QloraCapabilityError("QLORA_UNAVAILABLE: incompatible bitsandbytes") from exc
    return QloraCapability(True, cfg, True, True)


def quantization_kwargs(config: QloraConfig | None = None) -> dict[str, Any]:
    """Return BitsAndBytesConfig kwargs only after a successful capability check."""
    cfg = validate_qlora_config(config)
    return {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": cfg.compute_dtype}

