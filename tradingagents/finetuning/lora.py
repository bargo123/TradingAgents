"""Lazy PEFT LoRA attachment for the Phase 11 offline trainer.

The module deliberately imports neither torch nor PEFT at import time.  This
keeps inspection/status commands usable on the base installation, while the
actual attachment remains a standard PEFT ``get_peft_model`` operation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .models import LoraConfig, Phase11Status


class TargetModuleMissingError(ValueError):
    """Raised when a requested LoRA target is not present in the model."""

    status = Phase11Status.TARGET_MODULE_MISSING


class TrainingDependencyMissingError(RuntimeError):
    """Raised by direct attachment when PEFT is not installed."""

    status = Phase11Status.TRAINING_DEPENDENCY_MISSING


@dataclass(frozen=True, slots=True)
class TargetModuleReport:
    """Resolved target suffixes and concrete module paths."""

    requested: tuple[str, ...]
    found: tuple[str, ...]
    module_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoraAttachment:
    """An attached model and bounded parameter accounting."""

    model: Any
    target_modules: tuple[str, ...]
    total_parameters: int
    trainable_parameters: int
    adapter_parameters: int

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_parameters / self.total_parameters if self.total_parameters else 0.0

    @property
    def trainable_parameter_count(self) -> int:
        return self.trainable_parameters

    @property
    def total_parameter_count(self) -> int:
        return self.total_parameters


def _requested(value: LoraConfig | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TargetModuleMissingError("invalid target module configuration")
    modules = value.target_modules if isinstance(value, LoraConfig) else tuple(value)
    if (
        not modules
        or any(not isinstance(item, str) or not item for item in modules)
        or len(set(modules)) != len(modules)
    ):
        raise TargetModuleMissingError("invalid target module configuration")
    result = tuple(modules)
    return result


def _named_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    method = getattr(model, "named_modules", None)
    if not callable(method):
        raise TargetModuleMissingError("model does not expose named_modules()")
    try:
        raw_modules = tuple(method())
        modules = []
        for item in raw_modules:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("module entry must contain name and module")
            modules.append((str(item[0]), item[1]))
    except Exception as exc:  # malformed model should remain a bounded error
        raise TargetModuleMissingError("unable to inspect model modules") from exc
    if not modules:
        raise TargetModuleMissingError("model has no modules")
    return tuple(modules)


def inspect_target_modules(model: Any, requested: LoraConfig | Iterable[str]) -> TargetModuleReport:
    """Inspect model module names before constructing a PEFT config.

    PEFT accepts suffixes, so ``attention.q_proj`` satisfies a request for
    ``q_proj``.  Every requested suffix must occur at least once.
    """

    wanted = _requested(requested)
    modules = _named_modules(model)
    paths: list[str] = []
    found: list[str] = []
    for suffix in wanted:
        matches = [name for name, _module in modules if name and (name == suffix or name.endswith("." + suffix))]
        if matches:
            found.append(suffix)
            paths.extend(matches)
    missing = tuple(item for item in wanted if item not in found)
    if missing:
        raise TargetModuleMissingError("missing target modules: " + ", ".join(missing))
    return TargetModuleReport(wanted, tuple(found), tuple(sorted(set(paths))))


def _parameter_counts(model: Any) -> tuple[int, int, int]:
    method = getattr(model, "named_parameters", None)
    if not callable(method):
        raise TargetModuleMissingError("model does not expose named_parameters()")
    total = trainable = adapters = 0
    for name, parameter in method():
        try:
            count = int(parameter.numel())
        except (AttributeError, TypeError, ValueError) as exc:
            raise TargetModuleMissingError("malformed model parameter") from exc
        if count < 0:
            raise TargetModuleMissingError("malformed model parameter count")
        total += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable += count
            if "lora_" in str(name).lower():
                adapters += count
    return total, trainable, adapters


def attach_lora(
    model: Any,
    config: LoraConfig | None = None,
    *,
    base_model_name_or_path: str | None = None,
    revision: str | None = None,
) -> LoraAttachment:
    """Attach standard PEFT LoRA after fail-closed module inspection."""

    if model is None:
        raise TargetModuleMissingError("model is required")
    if config is None:
        resolved = LoraConfig()
    elif isinstance(config, LoraConfig):
        resolved = config
    elif isinstance(config, Iterable):
        resolved = LoraConfig(target_modules=tuple(config))
    else:
        raise TargetModuleMissingError("invalid LoRA configuration")
    # This check intentionally precedes the PEFT import and PEFT config
    # construction.  A missing target must not be hidden by dependency errors.
    inspect_target_modules(model, resolved)
    try:
        from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
    except (ImportError, ModuleNotFoundError) as exc:
        raise TrainingDependencyMissingError("TRAINING_DEPENDENCY_MISSING: peft") from exc

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TargetModuleMissingError("model does not expose named_parameters()")
    for _name, parameter in named_parameters():
        if hasattr(parameter, "requires_grad"):
            parameter.requires_grad = False
    try:
        peft_config = PeftLoraConfig(
            r=resolved.r,
            lora_alpha=resolved.alpha,
            lora_dropout=resolved.dropout,
            target_modules=list(resolved.target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        attached = get_peft_model(model, peft_config)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise TargetModuleMissingError("unable to attach LoRA to model") from exc

    # Programmatically constructed models do not have a model-card identity,
    # so PEFT otherwise serializes an empty base-model binding.  Carry the
    # operator-supplied immutable identity into the adapter config when one is
    # available; validation can then bind the adapter to the exact snapshot.
    if base_model_name_or_path or revision:
        configs = getattr(attached, "peft_config", {})
        if isinstance(configs, dict):
            for value in configs.values():
                if base_model_name_or_path:
                    value.base_model_name_or_path = str(base_model_name_or_path)
                if revision and hasattr(value, "revision"):
                    value.revision = str(revision)

    total, trainable, adapters = _parameter_counts(attached)
    if not trainable or not adapters or trainable != adapters:
        raise TargetModuleMissingError("LoRA adapter parameters are not the only trainable parameters")
    return LoraAttachment(attached, tuple(resolved.target_modules), total, trainable, adapters)


__all__ = [
    "LoraAttachment",
    "TargetModuleMissingError",
    "TargetModuleReport",
    "TrainingDependencyMissingError",
    "attach_lora",
    "inspect_target_modules",
]
