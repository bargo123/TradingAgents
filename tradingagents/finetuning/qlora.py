"""Fail-closed QLoRA capability checks (optional dependencies stay lazy)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from .models import Phase11Status, QloraConfig


class QloraCapabilityError(RuntimeError):
    """QLoRA cannot be used with the requested immutable settings."""

    def __init__(self, message: str, status: Phase11Status | None = None) -> None:
        super().__init__(message)
        self.status = status or (
            Phase11Status.GPU_REQUIRED
            if message.startswith("GPU_REQUIRED")
            else Phase11Status.QLORA_UNAVAILABLE
        )


@dataclass(frozen=True, slots=True)
class QloraCapability:
    available: bool
    config: QloraConfig
    cuda: bool
    bitsandbytes: bool
    reason: str = ""


def validate_qlora_config(config: QloraConfig | None = None) -> QloraConfig:
    if config is not None and not isinstance(config, QloraConfig):
        raise QloraCapabilityError("QLORA_UNAVAILABLE: invalid QLoRA configuration")
    cfg = config or QloraConfig()
    if not cfg.load_in_4bit or cfg.quant_type != "nf4" or not cfg.double_quant:
        raise QloraCapabilityError(
            "QLORA_UNAVAILABLE: QLoRA requires 4-bit NF4 double quantization"
        )
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
        raise QloraCapabilityError("GPU_REQUIRED: QLoRA requires CUDA", Phase11Status.GPU_REQUIRED)
    try:
        available = importlib.util.find_spec("bitsandbytes") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    if not available:
        raise QloraCapabilityError("QLORA_UNAVAILABLE: bitsandbytes is not installed")
    # Import only after CUDA and module presence checks; this catches broken wheels.
    try:
        import bitsandbytes  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - platform-specific wheel failures
        raise QloraCapabilityError("QLORA_UNAVAILABLE: incompatible bitsandbytes") from exc
    return QloraCapability(True, cfg, True, True)


def quantization_kwargs(
    config: QloraConfig | None = None, *, torch_module: Any | None = None
) -> dict[str, Any]:
    """Return explicit BitsAndBytesConfig kwargs after config validation.

    The public no-framework form keeps the serializable dtype name.  The
    training loader supplies ``torch_module`` so the framework receives the
    actual ``torch.bfloat16``/``torch.float16`` object.
    """
    cfg = validate_qlora_config(config)
    dtype: Any = cfg.compute_dtype
    if torch_module is not None:
        dtype = getattr(torch_module, cfg.compute_dtype)
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": dtype,
    }


def load_qlora_model(
    base_model: str,
    revision: str | None,
    config: QloraConfig,
    torch_module: Any,
) -> Any:
    """Load a 4-bit model only after capability checks; never fall back."""
    check_qlora_capability(config)
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # noqa: PLC0415

        quant = BitsAndBytesConfig(**quantization_kwargs(config, torch_module=torch_module))
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            revision=revision,
            local_files_only=True,
            quantization_config=quant,
            device_map="auto",
        )
    except QloraCapabilityError:
        raise
    except (ImportError, ModuleNotFoundError) as exc:
        raise QloraCapabilityError(
            "QLORA_UNAVAILABLE: quantization dependencies are unavailable",
            Phase11Status.QLORA_UNAVAILABLE,
        ) from exc
    except Exception as exc:
        raise QloraCapabilityError(
            f"QLORA_UNAVAILABLE: quantized model load failed ({type(exc).__name__})",
            Phase11Status.QLORA_UNAVAILABLE,
        ) from exc


def prepare_kbit_model(model: Any) -> Any:
    """Prepare an already 4-bit model for PEFT without changing precision."""
    try:
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415
    except (ImportError, ModuleNotFoundError) as exc:
        raise QloraCapabilityError(
            "QLORA_UNAVAILABLE: PEFT k-bit preparation is unavailable",
            Phase11Status.QLORA_UNAVAILABLE,
        ) from exc
    return prepare_model_for_kbit_training(model)


__all__ = [
    "QloraCapability",
    "QloraCapabilityError",
    "check_qlora_capability",
    "load_qlora_model",
    "prepare_kbit_model",
    "quantization_kwargs",
    "validate_qlora_config",
]
