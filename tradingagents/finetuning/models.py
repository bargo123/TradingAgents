"""Frozen, JSON-safe contracts shared by the Phase 11 pipeline.

This module intentionally has no training-framework imports.  The contracts
are useful to inspection/status commands even when optional ML dependencies
are not installed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import ContractError, InvalidStatusError, UnknownVersionError

SFT_FORMAT_VERSION = "phase11-sft-format.v1"
TRAINING_POLICY_VERSION = "phase11-training-policy.v1"
RUN_MANIFEST_VERSION = "phase11-run-manifest.v1"
ADAPTER_PACKAGE_VERSION = "phase11-adapter-package.v1"

_VERSIONS = {
    SFT_FORMAT_VERSION,
    TRAINING_POLICY_VERSION,
    RUN_MANIFEST_VERSION,
    ADAPTER_PACKAGE_VERSION,
}
_SENSITIVE = re.compile(
    r"(?:secret|password|credential|api[_ -]?key|access[_ -]?token|private[_ -]?key|prompt|completion|reasoning|chain[_ -]?of[_ -]?thought|\bcot\b|scratch)",
    re.I,
)


class Phase11Status(str, Enum):
    EMPTY_ELIGIBLE_SET = "EMPTY_ELIGIBLE_SET"
    PHASE10_INVALID = "PHASE10_INVALID"
    TRAINING_DEPENDENCY_MISSING = "TRAINING_DEPENDENCY_MISSING"
    BASE_MODEL_NOT_FOUND = "BASE_MODEL_NOT_FOUND"
    BASE_MODEL_REVISION_UNPINNED = "BASE_MODEL_REVISION_UNPINNED"
    TOKENIZER_INVALID = "TOKENIZER_INVALID"
    CHAT_TEMPLATE_UNAVAILABLE = "CHAT_TEMPLATE_UNAVAILABLE"
    TARGET_MODULE_MISSING = "TARGET_MODULE_MISSING"
    GPU_REQUIRED = "GPU_REQUIRED"
    QLORA_UNAVAILABLE = "QLORA_UNAVAILABLE"
    SEQUENCE_TOO_LONG = "SEQUENCE_TOO_LONG"
    TRAINING_FAILED = "TRAINING_FAILED"
    ADAPTER_INVALID = "ADAPTER_INVALID"
    COMPLETE = "COMPLETE"


def _safe(value: Any, key: str = "") -> None:
    if key and _SENSITIVE.search(key):
        raise ContractError(f"forbidden sensitive field: {key}")
    if dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            _safe(getattr(value, f.name), f.name)
    elif isinstance(value, Mapping):
        for k, v in value.items():
            _safe(v, str(k))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            _safe(v)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ContractError("non-finite numeric value")
    elif isinstance(value, str) and _SENSITIVE.search(value):
        raise ContractError("forbidden sensitive value")
    elif value is not None and not isinstance(value, (str, int, float, bool, Path, Enum)) and not dataclasses.is_dataclass(value):
        raise ContractError(f"value is not JSON-safe: {type(value).__name__}")


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            f.name: _plain(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if getattr(value, f.name) is not None
        }
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_plain(v) for v in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False)) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    _safe(value)
    return json.dumps(
        _plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _bounded_text(value: Any, name: str, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ContractError(f"{name} must be a bounded non-empty string")
    return value


class Contract:
    """Convenience serialization surface shared by all persisted contracts."""

    def to_dict(self) -> dict[str, Any]:
        return contract_to_dict(self)

    def to_json(self) -> str:
        return canonical_json(self)


@dataclass(frozen=True, slots=True)
class LoraConfig(Contract):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self):
        if not isinstance(self.r, int) or not 1 <= self.r <= 1024:
            raise ContractError("r out of bounds")
        if not isinstance(self.alpha, int) or not 1 <= self.alpha <= 4096:
            raise ContractError("alpha out of bounds")
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ContractError("dropout out of bounds")
        if not isinstance(self.target_modules, (tuple, list)) or not self.target_modules or any(
            not isinstance(x, str) or not x for x in self.target_modules
        ):
            raise ContractError("invalid target modules")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))


@dataclass(frozen=True, slots=True)
class QloraConfig(Contract):
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"

    def __post_init__(self):
        if not isinstance(self.load_in_4bit, bool) or not isinstance(self.double_quant, bool):
            raise ContractError("quantization flags must be bool")
        if self.quant_type != "nf4":
            raise ContractError("only NF4 is supported")
        if self.compute_dtype not in {"bfloat16", "float16"}:
            raise ContractError("unsupported compute dtype")


@dataclass(frozen=True, slots=True)
class TrainingConfig(Contract):
    mode: str = "lora"
    base_model: str = ""
    base_model_revision: str | None = None
    seed: int = 42
    max_sequence_length: int = 4096
    learning_rate: float = 2e-4
    epochs: int = 2
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "linear"
    gradient_clipping: float = 1.0
    gradient_checkpointing: bool = True
    validation_interval: str = "epoch"
    logging_steps: int = 10
    checkpoint_interval: str = "epoch"
    early_stopping: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)
    qlora: QloraConfig = field(default_factory=QloraConfig)
    version: str = TRAINING_POLICY_VERSION

    def __post_init__(self):
        if self.mode not in {"lora", "qlora"}:
            raise ContractError("mode must be lora or qlora")
        if self.version != TRAINING_POLICY_VERSION:
            raise UnknownVersionError(self.version)
        if self.base_model:
            _bounded_text(self.base_model, "base_model")
        if self.max_sequence_length < 1 or self.max_sequence_length > 1_000_000:
            raise ContractError("max_sequence_length out of bounds")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ContractError("learning_rate out of bounds")
        if (
            self.epochs < 1
            or self.batch_size < 1
            or self.gradient_accumulation_steps < 1
            or self.warmup_steps < 0
        ):
            raise ContractError("training bounds invalid")
        if self.max_steps is not None and self.max_steps < 1:
            raise ContractError("max_steps out of bounds")
        if self.weight_decay < 0 or self.gradient_clipping <= 0:
            raise ContractError("optimizer bounds invalid")
        if self.logging_steps < 1 or self.validation_interval not in {"epoch", "steps"}:
            raise ContractError("logging/validation interval invalid")
        if self.checkpoint_interval not in {"epoch", "steps", "never"}:
            raise ContractError("checkpoint interval invalid")
        if not isinstance(self.lora, LoraConfig) or not isinstance(self.qlora, QloraConfig):
            raise ContractError("invalid adapter config")


@dataclass(frozen=True, slots=True)
class DatasetBinding(Contract):
    generation_id: str
    generation_path: str | Path = ""
    manifest_hash: str = ""
    train_count: int = 0
    validation_count: int = 0
    test_count: int = 0
    schema_version: str = "phase10.dataset.v1"

    def __post_init__(self):
        _bounded_text(self.generation_id, "generation_id")
        if not isinstance(self.generation_path, (str, Path)):
            raise ContractError("generation_path must be a path")
        if self.test_count:
            raise ContractError("test rows cannot be bound for training")
        for n in (self.train_count, self.validation_count, self.test_count):
            if n < 0:
                raise ContractError("counts cannot be negative")


@dataclass(frozen=True, slots=True)
class SFTExample(Contract):
    example_id: str
    split: str
    messages: tuple[Mapping[str, Any], ...]
    target: Mapping[str, Any]
    version: str = SFT_FORMAT_VERSION

    def __post_init__(self):
        _bounded_text(self.example_id, "example_id")
        if self.split not in {"train", "validation"}:
            raise ContractError("invalid split")
        if len(self.messages) < 2:
            raise ContractError("at least two messages required")
        if self.version != SFT_FORMAT_VERSION:
            raise UnknownVersionError(self.version)
        if not isinstance(self.messages, (tuple, list)) or any(not isinstance(x, Mapping) for x in self.messages):
            raise ContractError("messages must be mappings")
        if not isinstance(self.target, Mapping):
            raise ContractError("target must be a mapping")
        _safe(self.messages)
        _safe(self.target)
        object.__setattr__(self, "messages", tuple(_freeze(x) for x in self.messages))
        object.__setattr__(self, "target", _freeze(self.target))


@dataclass(frozen=True, slots=True)
class PreparedManifest(Contract):
    generation_id: str
    train_count: int = 0
    validation_count: int = 0
    train_hash: str = ""
    validation_hash: str = ""
    version: str = SFT_FORMAT_VERSION

    def __post_init__(self):
        _bounded_text(self.generation_id, "generation_id")
        if self.version != SFT_FORMAT_VERSION:
            raise UnknownVersionError(self.version)
        if self.train_count < 0 or self.validation_count < 0:
            raise ContractError("counts cannot be negative")


@dataclass(frozen=True, slots=True)
class RunManifest(Contract):
    run_id: str
    status: Phase11Status = Phase11Status.COMPLETE
    config: Mapping[str, Any] = field(default_factory=dict)
    dataset: Mapping[str, Any] = field(default_factory=dict)
    version: str = RUN_MANIFEST_VERSION

    def __post_init__(self):
        _bounded_text(self.run_id, "run_id")
        if not isinstance(self.status, Phase11Status):
            try:
                object.__setattr__(self, "status", Phase11Status(self.status))
            except ValueError as exc:
                raise InvalidStatusError(str(self.status)) from exc
        if self.version != RUN_MANIFEST_VERSION:
            raise UnknownVersionError(self.version)
        if not isinstance(self.config, Mapping) or not isinstance(self.dataset, Mapping):
            raise ContractError("config and dataset must be mappings")
        _safe(self.config)
        _safe(self.dataset)
        object.__setattr__(self, "config", _freeze(self.config))
        object.__setattr__(self, "dataset", _freeze(self.dataset))


@dataclass(frozen=True, slots=True)
class Metrics(Contract):
    loss: float | None = None
    steps: int = 0
    epochs: float = 0.0
    examples: int = 0
    tokens: int = 0
    runtime_seconds: float = 0.0
    trainable_parameters: int = 0
    total_parameters: int = 0

    def __post_init__(self):
        if self.loss is not None and (not math.isfinite(self.loss) or self.loss < 0):
            raise ContractError("loss out of bounds")
        if any(
            not isinstance(x, int) or x < 0
            for x in (
                self.steps,
                self.examples,
                self.tokens,
                self.trainable_parameters,
                self.total_parameters,
            )
        ):
            raise ContractError("metric counts out of bounds")
        if self.runtime_seconds < 0 or not math.isfinite(self.runtime_seconds):
            raise ContractError("runtime out of bounds")


@dataclass(frozen=True, slots=True)
class TrainingReport(Contract):
    status: Phase11Status
    metrics: Metrics | None = None
    message: str = ""
    run_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.status, Phase11Status):
            try:
                object.__setattr__(self, "status", Phase11Status(self.status))
            except ValueError as exc:
                raise InvalidStatusError(str(self.status)) from exc
        if len(self.message) > 4096:
            raise ContractError("message too long")
        if self.metrics is not None and not isinstance(self.metrics, Metrics):
            raise ContractError("invalid metrics")
        if self.run_id is not None:
            _bounded_text(self.run_id, "run_id")
        _safe(self.message)


@dataclass(frozen=True, slots=True)
class ValidationReport(Contract):
    valid: bool
    status: Phase11Status = Phase11Status.COMPLETE
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.valid, bool):
            raise ContractError("valid must be bool")
        if not isinstance(self.status, Phase11Status):
            try:
                object.__setattr__(self, "status", Phase11Status(self.status))
            except ValueError as exc:
                raise InvalidStatusError(str(self.status)) from exc
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if any(not isinstance(x, str) or len(x) > 4096 for x in self.errors + self.warnings):
            raise ContractError("report messages out of bounds")
        _safe(self.errors)
        _safe(self.warnings)


def _freeze(value: Any) -> Any:
    """Recursively freeze mappings/containers while retaining JSON types."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    return value


def contract_to_dict(value: Any) -> dict[str, Any]:
    _safe(value)
    return _plain(value)
