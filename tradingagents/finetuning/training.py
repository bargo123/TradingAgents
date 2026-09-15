"""Small, deterministic, offline causal-LM trainer for Phase 11 LoRA adapters."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lora import (
    LoraAttachment,
    TrainingDependencyMissingError,
    attach_lora,
)
from .models import Metrics, Phase11Status, SFTExample, TrainingConfig
from .provenance import ModelProvenance, ProvenanceError
from .tokenization import TokenizationPolicy, TokenizedExample, tokenize_example


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Bounded result returned by ``train`` for both success and fail-closed paths."""

    status: Phase11Status
    metrics: Metrics | None = None
    validation_loss: float | None = None
    adapter_path: Path | None = None
    attachment: LoraAttachment | None = None
    provenance: ModelProvenance | None = None
    optimizer_steps: int = 0
    message: str = ""

    @property
    def loss(self) -> float | None:
        return self.metrics.loss if self.metrics is not None else None

    @property
    def trainable_parameters(self) -> int:
        return self.metrics.trainable_parameters if self.metrics is not None else 0

    @property
    def total_parameters(self) -> int:
        return self.metrics.total_parameters if self.metrics is not None else 0


def _as_example(value: Any, split: str) -> SFTExample:
    if isinstance(value, SFTExample):
        if value.split != split:
            return SFTExample(value.example_id, split, value.messages, value.target, value.version)
        return value
    if not isinstance(value, Mapping):
        raise ValueError("prepared row must be a mapping or SFTExample")
    try:
        return SFTExample(
            str(value["example_id"]),
            split,
            tuple(value["messages"]),
            dict(value["target"]),
            value.get("version", "phase11-sft-format.v1"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed prepared SFT row") from exc


def _read_jsonl(path: Path, split: str) -> list[SFTExample]:
    if not path.is_file():
        return []
    rows: list[SFTExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(_as_example(json.loads(line), split))
    return rows


def _dataset(prepared: Any) -> tuple[list[SFTExample], list[SFTExample]]:
    """Read only prepared train/validation data, never a test split."""

    if isinstance(prepared, (str, Path)):
        root = Path(prepared)
        folder = root / "prepared" if (root / "prepared").is_dir() else root
        return _read_jsonl(folder / "train.sft.jsonl", "train"), _read_jsonl(folder / "validation.sft.jsonl", "validation")
    output_root = getattr(prepared, "output_root", None)
    if output_root is not None:
        return _dataset(Path(output_root))
    if isinstance(prepared, Mapping):
        train_rows = [_as_example(row, "train") for row in prepared.get("train", ())]
        validation_rows = [_as_example(row, "validation") for row in prepared.get("validation", ())]
        return train_rows, validation_rows
    if isinstance(prepared, tuple) and len(prepared) == 2:
        return ([_as_example(row, "train") for row in prepared[0]], [_as_example(row, "validation") for row in prepared[1]])
    if isinstance(prepared, Iterable) and not isinstance(prepared, (str, bytes)):
        rows = [_as_example(row, getattr(row, "split", "train")) for row in prepared]
        return [row for row in rows if row.split == "train"], [row for row in rows if row.split == "validation"]
    raise ValueError("prepared training data is required")


def _load_optional_stack() -> tuple[Any | None, Any | None, str | None]:
    """Import the optional framework only after preparation has been checked."""

    try:
        import peft
        import torch
    except (ImportError, ModuleNotFoundError) as exc:
        return None, None, f"TRAINING_DEPENDENCY_MISSING: {exc.name or 'torch/peft'}"
    except Exception as exc:
        # Broken optional installations should have the same bounded status as
        # absent installations; no traceback is exposed in the result.
        return None, None, f"TRAINING_DEPENDENCY_MISSING: {type(exc).__name__}"
    return torch, peft, None


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenize_rows(rows: list[SFTExample], policy: TokenizationPolicy) -> list[TokenizedExample]:
    return [tokenize_example(row, policy) for row in rows]


def _batch(torch: Any, rows: list[TokenizedExample], pad_id: int) -> tuple[Any, Any, Any]:
    width = max(len(row.input_ids) for row in rows)
    input_ids = [list(row.input_ids) + [pad_id] * (width - len(row.input_ids)) for row in rows]
    labels = [list(row.labels) + [-100] * (width - len(row.labels)) for row in rows]
    attention = [[1] * len(row.input_ids) + [0] * (width - len(row.input_ids)) for row in rows]
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def _loss_from_output(torch: Any, output: Any, labels: Any) -> Any:
    loss = output.get("loss") if isinstance(output, Mapping) else getattr(output, "loss", None)
    if loss is not None:
        return loss
    logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
    if logits is None:
        raise ValueError("causal model output has neither loss nor logits")
    if logits.ndim != 3:
        raise ValueError("causal model logits must be rank three")
    return torch.nn.functional.cross_entropy(logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]), labels[..., 1:].contiguous().view(-1), ignore_index=-100)


def _scheduler(torch: Any, optimizer: Any, total_steps: int, warmup_steps: int) -> Any:
    total = max(1, int(total_steps))
    warmup = max(0, min(int(warmup_steps), total))

    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / warmup
        remaining = max(0, total - step - 1)
        denominator = max(1, total - warmup)
        return float(remaining) / denominator

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _failed(status: Phase11Status, message: str) -> TrainingResult:
    return TrainingResult(status=status, message=message[:4096])


def train(
    prepared: Any,
    model: Any = None,
    tokenizer_policy: TokenizationPolicy | None = None,
    config: TrainingConfig | None = None,
    output_dir: str | Path | None = None,
    *,
    provenance: ModelProvenance | None = None,
) -> TrainingResult:
    """Train a tiny LoRA adapter from prepared train/validation rows.

    Preparation is loaded before any optional ML import.  A zero-row train
    split therefore returns ``EMPTY_ELIGIBLE_SET`` without constructing a
    model, tokenizer, optimizer, or accelerator.
    """

    try:
        train_rows, validation_rows = _dataset(prepared)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return _failed(Phase11Status.TRAINING_FAILED, f"malformed preparation: {exc}")
    if not train_rows:
        return _failed(Phase11Status.EMPTY_ELIGIBLE_SET, "no eligible training rows")
    resolved = config or TrainingConfig()
    if not isinstance(resolved, TrainingConfig):
        return _failed(Phase11Status.TRAINING_FAILED, "config must be TrainingConfig")
    torch, _peft, missing = _load_optional_stack()
    if missing:
        return _failed(Phase11Status.TRAINING_DEPENDENCY_MISSING, missing)
    if tokenizer_policy is None:
        return _failed(Phase11Status.TOKENIZER_INVALID, "tokenizer policy is required")

    try:
        _set_seed(torch, resolved.seed)
        if provenance is None and resolved.base_model:
            candidate = Path(resolved.base_model).expanduser()
            if candidate.is_dir():
                try:
                    provenance = ModelProvenance.inspect(
                        candidate,
                        revision=resolved.base_model_revision,
                        tokenizer_path=candidate,
                    )
                except ProvenanceError as exc:
                    return _failed(Phase11Status.BASE_MODEL_NOT_FOUND, str(exc))
        encoded_train = _tokenize_rows(train_rows, tokenizer_policy)
        encoded_validation = _tokenize_rows(validation_rows, tokenizer_policy)
        if model is None:
            if not resolved.base_model:
                return _failed(Phase11Status.BASE_MODEL_NOT_FOUND, "base model is required")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(resolved.base_model, revision=resolved.base_model_revision, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(resolved.base_model, revision=resolved.base_model_revision, local_files_only=True)
            tokenizer_policy = TokenizationPolicy(tokenizer=tokenizer, max_length=resolved.max_sequence_length)
        attachment = attach_lora(model, resolved.lora)
        model = attachment.model
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            return _failed(Phase11Status.TRAINING_FAILED, "no trainable adapter parameters")
        optimizer = torch.optim.AdamW(parameters, lr=resolved.learning_rate, weight_decay=resolved.weight_decay)
        batches_per_epoch = max(1, math.ceil(len(encoded_train) / resolved.batch_size / resolved.gradient_accumulation_steps))
        requested_steps = resolved.max_steps or (resolved.epochs * batches_per_epoch)
        total_steps = max(1, int(requested_steps))
        scheduler = _scheduler(torch, optimizer, total_steps, resolved.warmup_steps)
        model.train()
        pad_id = int(getattr(tokenizer_policy.load(), "pad_token_id", 0) or 0)
        losses: list[float] = []
        optimizer_steps = 0
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _epoch in range(resolved.epochs):
            for start in range(0, len(encoded_train), resolved.batch_size):
                batch_rows = encoded_train[start : start + resolved.batch_size]
                input_ids, attention, labels = _batch(torch, batch_rows, pad_id)
                output = model(input_ids=input_ids, attention_mask=attention, labels=labels)
                loss = _loss_from_output(torch, output, labels)
                if not bool(torch.isfinite(loss).item()):
                    return _failed(Phase11Status.TRAINING_FAILED, "non-finite training loss")
                losses.append(float(loss.detach().cpu().item()))
                (loss / resolved.gradient_accumulation_steps).backward()
                is_accumulation_boundary = ((start // resolved.batch_size + 1) % resolved.gradient_accumulation_steps == 0)
                is_last_batch = start + resolved.batch_size >= len(encoded_train)
                if is_accumulation_boundary or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(parameters, resolved.gradient_clipping)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if optimizer_steps >= total_steps:
                        break
            if optimizer_steps >= total_steps:
                break
        runtime = max(0.0, time.perf_counter() - started)
        val_loss = _validation_loss(torch, model, encoded_validation, pad_id, resolved.batch_size)
        if output_dir is not None:
            adapter_path = Path(output_dir)
            adapter_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_path)
        else:
            adapter_path = None
        tokens = sum(len(row.input_ids) for row in encoded_train)
        metrics = Metrics(
            loss=sum(losses) / len(losses) if losses else 0.0,
            steps=optimizer_steps,
            epochs=optimizer_steps / batches_per_epoch,
            examples=len(train_rows),
            tokens=tokens,
            runtime_seconds=runtime,
            trainable_parameters=attachment.trainable_parameters,
            total_parameters=attachment.total_parameters,
        )
        return TrainingResult(Phase11Status.COMPLETE, metrics, val_loss, adapter_path, attachment, provenance, optimizer_steps)
    except TrainingDependencyMissingError as exc:
        return _failed(Phase11Status.TRAINING_DEPENDENCY_MISSING, str(exc))
    except ProvenanceError as exc:
        return _failed(Phase11Status.BASE_MODEL_NOT_FOUND, str(exc))
    except (OSError, ValueError, TypeError, RuntimeError, AttributeError) as exc:
        return _failed(Phase11Status.TRAINING_FAILED, f"training failed: {type(exc).__name__}")


def _validation_loss(torch: Any, model: Any, rows: list[TokenizedExample], pad_id: int, batch_size: int) -> float | None:
    if not rows:
        return None
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            input_ids, attention, labels = _batch(torch, rows[start : start + batch_size], pad_id)
            output = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            loss = _loss_from_output(torch, output, labels)
            if not bool(torch.isfinite(loss).item()):
                return None
            losses.append(float(loss.detach().cpu().item()))
    return sum(losses) / len(losses) if losses else None


def train_lora(model: Any, train_rows: Any, validation_rows: Any, tokenizer_policy: TokenizationPolicy, config: TrainingConfig | None = None, output_dir: str | Path | None = None) -> TrainingResult:
    """Compatibility wrapper with the model-first calling convention."""

    return train({"train": train_rows, "validation": validation_rows}, model=model, tokenizer_policy=tokenizer_policy, config=config, output_dir=output_dir)


train_prepared = train


__all__ = ["TrainingResult", "train", "train_lora", "train_prepared"]
