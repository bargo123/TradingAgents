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
PHASE10_DATASET_SCHEMA_VERSION = "phase10.dataset.v1"
_UNRESOLVED_REVISIONS = {"latest", "main", "master", "head", "default"}
_SENSITIVE = re.compile(
    r"(?:secret|password|credential|api[_ -]?key|access[_ -]?token|private[_ -]?key|prompt|completion|reasoning|chain[_ -]?of[_ -]?thought|\bcot\b|scratch)",
    re.I,
)
_EVIDENCE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_TARGET_STATUSES = {"USED", "NONE_RELEVANT", "UNAVAILABLE", "DISABLED", "INCOMPLETE", "INVALID"}
_REJECTION_REASONS = {
    "CONFLICTS_WITH_CURRENT_STATE", "LOW_RELEVANCE", "INSUFFICIENT_SAMPLE", "DIAGNOSTIC_ONLY", "REDUNDANT",
}


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


def _bounded_int(value: Any, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name} out of bounds")


def _bounded_finite(value: Any, name: str, *, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ContractError(f"{name} out of bounds")


class Contract:
    """Convenience serialization surface shared by all persisted contracts."""

    def to_dict(self) -> dict[str, Any]:
        return contract_to_dict(self)

    def to_json(self) -> str:
        return canonical_json(self)


def _validate_target(value: Mapping[str, Any]) -> None:
    allowed = {"action", "evidence_refs", "evidence_use_status", "evidence_refs_rejected"}
    required = {"action", "evidence_refs"}
    if not required.issubset(value) or set(value) - allowed:
        raise ContractError("target contains unsupported or missing fields")
    if value.get("action") not in {"BUY", "SELL", "HOLD"}:
        raise ContractError("target action must be BUY, SELL, or HOLD")
    refs = value.get("evidence_refs")
    if not isinstance(refs, (list, tuple)) or any(not isinstance(ref, str) or not _EVIDENCE_REF.fullmatch(ref) for ref in refs):
        raise ContractError("target evidence references are invalid")
    if list(refs) != sorted(set(refs)):
        raise ContractError("target evidence references must be unique and sorted")
    status = value.get("evidence_use_status")
    if status is not None:
        if not isinstance(status, str) or status not in _TARGET_STATUSES:
            raise ContractError("target evidence_use_status is invalid")
        if status == "NONE_RELEVANT" and refs:
            raise ContractError("NONE_RELEVANT cannot contain used evidence")
    rejected = value.get("evidence_refs_rejected")
    if "evidence_refs_rejected" in value and rejected is None:
        raise ContractError("target rejected evidence references are invalid")
    if rejected is not None:
        if not isinstance(rejected, (list, tuple)):
            raise ContractError("target rejected evidence references are invalid")
        seen: set[str] = set()
        for item in rejected:
            if not isinstance(item, Mapping) or set(item) != {"ref", "reason"}:
                raise ContractError("target rejected evidence entries are invalid")
            ref, reason = item["ref"], item["reason"]
            if not isinstance(ref, str) or not _EVIDENCE_REF.fullmatch(ref) or ref in seen:
                raise ContractError("target rejected evidence references are invalid")
            if not isinstance(reason, str) or reason not in _REJECTION_REASONS:
                raise ContractError("target rejected evidence reason is invalid")
            seen.add(ref)
        if [item["ref"] for item in rejected] != sorted(seen):
            raise ContractError("target rejected evidence references must be sorted")
        if set(refs) & seen:
            raise ContractError("target evidence references cannot overlap")


def _validate_assistant_target(messages: tuple[Mapping[str, Any], ...], target: Mapping[str, Any]) -> None:
    """Ensure the visible assistant message is exactly the structured target."""
    content = messages[-1].get("content")
    if not isinstance(content, str):
        raise ContractError("assistant target content is required")
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise ContractError("assistant target is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ContractError("assistant target must be a JSON object")
    _validate_target(parsed)
    if canonical_json(parsed) != canonical_json(target):
        raise ContractError("assistant target does not match structured target")


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
        _bounded_int(self.r, "r", minimum=1)
        _bounded_int(self.alpha, "alpha", minimum=1)
        if self.r > 1024:
            raise ContractError("r out of bounds")
        if self.alpha > 4096:
            raise ContractError("alpha out of bounds")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ContractError("dropout out of bounds")
        if not isinstance(self.target_modules, (tuple, list)) or not self.target_modules or any(
            not isinstance(x, str) or not x for x in self.target_modules
        ):
            raise ContractError("invalid target modules")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ContractError("target modules must be unique")
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
        if not isinstance(self.quant_type, str) or self.quant_type != "nf4":
            raise ContractError("only NF4 is supported")
        if not isinstance(self.compute_dtype, str) or self.compute_dtype not in {"bfloat16", "float16"}:
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
        if not isinstance(self.mode, str) or self.mode not in {"lora", "qlora"}:
            raise ContractError("mode must be lora or qlora")
        if self.version != TRAINING_POLICY_VERSION:
            raise UnknownVersionError(self.version)
        if not isinstance(self.base_model, str):
            raise ContractError("base_model must be a string")
        if self.base_model:
            _bounded_text(self.base_model, "base_model")
        if self.base_model_revision is not None:
            _bounded_text(self.base_model_revision, "base_model_revision")
            if self.base_model_revision.casefold() in _UNRESOLVED_REVISIONS:
                raise ContractError("base_model_revision must be immutable")
        _bounded_int(self.seed, "seed", minimum=0)
        _bounded_int(self.max_sequence_length, "max_sequence_length", minimum=1)
        if self.max_sequence_length > 1_000_000:
            raise ContractError("max_sequence_length out of bounds")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ContractError("learning_rate out of bounds")
        _bounded_int(self.epochs, "epochs", minimum=1)
        _bounded_int(self.batch_size, "batch_size", minimum=1)
        _bounded_int(self.gradient_accumulation_steps, "gradient_accumulation_steps", minimum=1)
        _bounded_int(self.warmup_steps, "warmup_steps")
        if self.max_steps is not None:
            _bounded_int(self.max_steps, "max_steps", minimum=1)
        if (isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, (int, float)) or not math.isfinite(self.weight_decay) or self.weight_decay < 0 or
            isinstance(self.gradient_clipping, bool) or not isinstance(self.gradient_clipping, (int, float)) or not math.isfinite(self.gradient_clipping) or self.gradient_clipping <= 0):
            raise ContractError("optimizer bounds invalid")
        _bounded_int(self.logging_steps, "logging_steps", minimum=1)
        if not isinstance(self.optimizer, str) or self.optimizer not in {"adamw"}:
            raise ContractError("unsupported optimizer")
        if not isinstance(self.scheduler, str) or self.scheduler not in {"linear"}:
            raise ContractError("unsupported scheduler")
        if not isinstance(self.validation_interval, str) or self.validation_interval not in {"epoch", "steps"}:
            raise ContractError("logging/validation interval invalid")
        if not isinstance(self.checkpoint_interval, str) or self.checkpoint_interval not in {"epoch", "steps", "never"}:
            raise ContractError("checkpoint interval invalid")
        if not isinstance(self.gradient_checkpointing, bool) or not isinstance(self.early_stopping, bool):
            raise ContractError("checkpointing/early-stopping flags must be bool")
        if not isinstance(self.lora, LoraConfig) or not isinstance(self.qlora, QloraConfig):
            raise ContractError("invalid adapter config")

    def to_dict(self) -> dict[str, Any]:
        """Persist optional effective settings explicitly, including nulls."""
        value = contract_to_dict(self)
        value["base_model_revision"] = self.base_model_revision
        value["max_steps"] = self.max_steps
        return value


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
        if self.schema_version != PHASE10_DATASET_SCHEMA_VERSION:
            raise UnknownVersionError(self.schema_version)
        if not isinstance(self.generation_path, (str, Path)):
            raise ContractError("generation_path must be a path")
        if self.test_count:
            raise ContractError("test rows cannot be bound for training")
        for name, n in (("train_count", self.train_count), ("validation_count", self.validation_count), ("test_count", self.test_count)):
            _bounded_int(n, name)


@dataclass(frozen=True, slots=True)
class SFTExample(Contract):
    example_id: str
    split: str
    messages: tuple[Mapping[str, Any], ...]
    target: Mapping[str, Any]
    version: str = SFT_FORMAT_VERSION

    def __post_init__(self):
        _bounded_text(self.example_id, "example_id")
        if not isinstance(self.split, str) or self.split not in {"train", "validation"}:
            raise ContractError("invalid split")
        if self.version != SFT_FORMAT_VERSION:
            raise UnknownVersionError(self.version)
        if not isinstance(self.messages, (tuple, list)):
            raise ContractError("messages must be a sequence")
        if len(self.messages) != 3:
            raise ContractError("exactly three SYSTEM, USER, and ASSISTANT messages are required")
        if any(not isinstance(x, Mapping) for x in self.messages):
            raise ContractError("messages must be mappings")
        if not isinstance(self.target, Mapping):
            raise ContractError("target must be a mapping")
        _validate_target(self.target)
        roles = [message.get("role") for message in self.messages]
        if roles != ["system", "user", "assistant"]:
            raise ContractError("messages must be exactly SYSTEM/USER/ASSISTANT")
        for message in self.messages:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip() or len(content) > 256 * 1024:
                raise ContractError("message content must be bounded non-empty text")
        _validate_assistant_target(tuple(self.messages), self.target)
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
        _bounded_int(self.train_count, "train_count")
        _bounded_int(self.validation_count, "validation_count")


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
            except (TypeError, ValueError) as exc:
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
    validation_loss: float | None = None
    throughput_tokens_per_second: float | None = None
    peak_cuda_memory_bytes: int | None = None
    version: str = RUN_MANIFEST_VERSION

    def __post_init__(self):
        if self.version != RUN_MANIFEST_VERSION:
            raise UnknownVersionError(self.version)
        if self.loss is not None and (isinstance(self.loss, bool) or not isinstance(self.loss, (int, float)) or not math.isfinite(self.loss) or self.loss < 0):
            raise ContractError("loss out of bounds")
        if self.validation_loss is not None and (
            isinstance(self.validation_loss, bool)
            or not isinstance(self.validation_loss, (int, float))
            or not math.isfinite(self.validation_loss)
            or self.validation_loss < 0
        ):
            raise ContractError("validation_loss out of bounds")
        if self.throughput_tokens_per_second is not None:
            _bounded_finite(self.throughput_tokens_per_second, "throughput_tokens_per_second")
        if self.peak_cuda_memory_bytes is not None:
            _bounded_int(self.peak_cuda_memory_bytes, "peak_cuda_memory_bytes")
        for name, value in (("steps", self.steps), ("examples", self.examples), ("tokens", self.tokens), ("trainable_parameters", self.trainable_parameters), ("total_parameters", self.total_parameters)):
            _bounded_int(value, name)
        _bounded_finite(self.epochs, "epochs")
        _bounded_finite(self.runtime_seconds, "runtime_seconds")


@dataclass(frozen=True, slots=True)
class TrainingReport(Contract):
    status: Phase11Status
    metrics: Metrics | None = None
    message: str = ""
    run_id: str | None = None
    version: str = RUN_MANIFEST_VERSION

    def __post_init__(self):
        if not isinstance(self.status, Phase11Status):
            try:
                object.__setattr__(self, "status", Phase11Status(self.status))
            except (TypeError, ValueError) as exc:
                raise InvalidStatusError(str(self.status)) from exc
        if self.version != RUN_MANIFEST_VERSION:
            raise UnknownVersionError(self.version)
        if not isinstance(self.message, str) or len(self.message) > 4096:
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
    version: str = RUN_MANIFEST_VERSION

    def __post_init__(self):
        if self.version != RUN_MANIFEST_VERSION:
            raise UnknownVersionError(self.version)
        if not isinstance(self.valid, bool):
            raise ContractError("valid must be bool")
        if not isinstance(self.status, Phase11Status):
            try:
                object.__setattr__(self, "status", Phase11Status(self.status))
            except ValueError as exc:
                raise InvalidStatusError(str(self.status)) from exc
        if not isinstance(self.errors, (tuple, list)) or not isinstance(self.warnings, (tuple, list)):
            raise ContractError("report messages must be sequences")
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
    if isinstance(value, (set, frozenset)):
        frozen = tuple(_freeze(v) for v in value)
        return tuple(sorted(frozen, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False)))
    return value


def contract_to_dict(value: Any) -> dict[str, Any]:
    _safe(value)
    return _plain(value)
