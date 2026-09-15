from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tradingagents.finetuning.models import LoraConfig, Phase11Status, SFTExample, TrainingConfig
from tradingagents.finetuning.tokenization import TokenizationPolicy


class TinyTokenizer:
    model_max_length = 256
    pad_token_id = 0
    name_or_path = "phase11-tiny"
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        return self.encode(text) if tokenize else text

    def encode(self, value, **kwargs):
        return [ord(char) % 31 + 1 for char in value]


def example(example_id: str = "ex-1", split: str = "train") -> SFTExample:
    return SFTExample(
        example_id,
        split,
        (
            {"role": "system", "content": "Use state."},
            {"role": "user", "content": '{"decision":{"action":"BUY"}}'},
            {"role": "assistant", "content": '{"action":"BUY","evidence_refs":[]}'},
        ),
        {"action": "BUY", "evidence_refs": []},
    )


def tiny_model():
    if not importlib.util.find_spec("transformers") or not importlib.util.find_spec("peft"):
        pytest.skip("optional training stack is absent")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(7)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=256,
            n_ctx=256,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def test_target_module_inspection_and_missing_module() -> None:
    from tradingagents.finetuning.lora import TargetModuleMissingError, inspect_target_modules

    class Model:
        def named_modules(self):
            return [("", self), ("block.q_proj", object()), ("block.v_proj", object())]

    report = inspect_target_modules(Model(), ("q_proj", "v_proj"))
    assert report.found == ("q_proj", "v_proj")
    with pytest.raises(TargetModuleMissingError, match="missing"):
        inspect_target_modules(Model(), ("q_proj", "k_proj"))

    with pytest.raises(TargetModuleMissingError, match="configuration"):
        inspect_target_modules(Model(), "q_proj")
    with pytest.raises(TargetModuleMissingError, match="configuration"):
        inspect_target_modules(Model(), ("q_proj", 1))


def test_attach_lora_freezes_base_and_exposes_counts() -> None:
    from tradingagents.finetuning.lora import attach_lora

    attached = attach_lora(tiny_model(), LoraConfig(r=2, alpha=4, target_modules=("c_attn",)))
    assert attached.trainable_parameters > 0
    assert attached.total_parameters > attached.trainable_parameters
    assert all(not parameter.requires_grad or "lora_" in name.lower() for name, parameter in attached.model.named_parameters())


def test_attach_lora_binds_operator_base_identity() -> None:
    from tradingagents.finetuning.lora import attach_lora

    attached = attach_lora(
        tiny_model(),
        LoraConfig(r=2, alpha=4, target_modules=("c_attn",)),
        base_model_name_or_path="C:/models/tiny",
    )
    assert attached.model.peft_config["default"].base_model_name_or_path == "C:/models/tiny"


def test_train_runs_real_optimizer_step_and_saves_adapter(tmp_path: Path) -> None:
    from tradingagents.finetuning.training import train

    result = train(
        {"train": [example()], "validation": [example("val-1", "validation")]},
        model=tiny_model(),
        tokenizer_policy=TokenizationPolicy(tokenizer=TinyTokenizer(), max_length=128),
        config=TrainingConfig(epochs=1, max_steps=1, gradient_checkpointing=False, lora=LoraConfig(r=2, alpha=4, target_modules=("c_attn",))),
        output_dir=tmp_path / "adapter",
    )
    assert result.status is Phase11Status.COMPLETE
    assert result.optimizer_steps == 1
    assert result.metrics is not None and result.metrics.loss is not None
    assert result.validation_loss is not None
    assert (tmp_path / "adapter" / "adapter_config.json").is_file()


def test_training_reports_missing_optional_dependency(monkeypatch) -> None:
    from tradingagents.finetuning import training

    monkeypatch.setattr(training, "_load_optional_stack", lambda: (None, None, "peft is unavailable"))
    result = training.train(
        {"train": [example()], "validation": []},
        model=object(),
        tokenizer_policy=TokenizationPolicy(tokenizer=TinyTokenizer()),
        config=TrainingConfig(epochs=1, max_steps=1, gradient_checkpointing=False),
    )
    assert result.status is Phase11Status.TRAINING_DEPENDENCY_MISSING


def test_empty_preparation_short_circuits_before_optional_import(monkeypatch) -> None:
    from tradingagents.finetuning import training

    called = False

    def fail_import():
        nonlocal called
        called = True
        raise AssertionError("optional stack must not load for empty preparation")

    monkeypatch.setattr(training, "_load_optional_stack", fail_import)
    result = training.train({"train": [], "validation": []}, model=object())
    assert result.status is Phase11Status.EMPTY_ELIGIBLE_SET
    assert not called


def test_missing_base_model_fails_before_optional_stack_load(monkeypatch) -> None:
    from tradingagents.finetuning import training

    monkeypatch.setattr(
        training,
        "_load_optional_stack",
        lambda: (_ for _ in ()).throw(AssertionError("optional stack must not load")),
    )
    result = training.train(
        {"train": [example()], "validation": []},
        model=None,
        tokenizer_policy=TokenizationPolicy(tokenizer=TinyTokenizer()),
        config=TrainingConfig(epochs=1, max_steps=1, gradient_checkpointing=False),
    )
    assert result.status is Phase11Status.BASE_MODEL_NOT_FOUND


def test_unpinned_remote_base_model_returns_typed_status(monkeypatch) -> None:
    from tradingagents.finetuning import training

    monkeypatch.setattr(training, "_load_optional_stack", lambda: (object(), object(), None))
    monkeypatch.setattr(training, "_set_seed", lambda *_args: None)
    result = training.train(
        {"train": [example()], "validation": []},
        model=None,
        tokenizer_policy=TokenizationPolicy(tokenizer=TinyTokenizer()),
        config=TrainingConfig(
            base_model="org/model",
            epochs=1,
            max_steps=1,
            gradient_checkpointing=False,
        ),
    )
    assert result.status is Phase11Status.BASE_MODEL_REVISION_UNPINNED


def test_malformed_structured_target_fails_closed_before_model_use() -> None:
    from tradingagents.finetuning.training import train

    bad = {
        "example_id": "bad",
        "split": "train",
        "messages": [
            {"role": "system", "content": "Use state."},
            {"role": "user", "content": "{}"},
            {"role": "assistant", "content": "{}"},
        ],
        "target": {"action": "BUY", "evidence_refs": [], "unexpected": True},
    }
    result = train({"train": [bad], "validation": []}, model=object())
    assert result.status is Phase11Status.TRAINING_FAILED
    assert "malformed preparation" in result.message


def test_malformed_example_id_is_not_coerced_into_a_valid_row() -> None:
    from tradingagents.finetuning.training import train

    row = {
        "example_id": None,
        "split": "train",
        "messages": [
            {"role": "system", "content": "Use state."},
            {"role": "user", "content": "{}"},
            {"role": "assistant", "content": '{"action":"BUY","evidence_refs":[]}'},
        ],
        "target": {"action": "BUY", "evidence_refs": []},
    }
    result = train({"train": [row], "validation": []}, model=object())
    assert result.status is Phase11Status.TRAINING_FAILED


def test_test_split_rows_cannot_be_relabelled_as_training() -> None:
    from tradingagents.finetuning.training import train

    row = {
        "example_id": "test-only",
        "split": "test",
        "messages": [
            {"role": "system", "content": "Use state."},
            {"role": "user", "content": "{}"},
            {"role": "assistant", "content": '{"action":"BUY","evidence_refs":[]}'},
        ],
        "target": {"action": "BUY", "evidence_refs": []},
    }
    result = train({"train": [row], "validation": []}, model=object())
    assert result.status is Phase11Status.TRAINING_FAILED


def test_explicit_test_mapping_is_rejected_instead_of_ignored() -> None:
    from tradingagents.finetuning.training import train

    result = train({"test": [example("test-only", "train")]}, model=object())
    assert result.status is Phase11Status.TRAINING_FAILED
    assert "malformed preparation" in result.message


def test_sequence_capacity_failure_returns_typed_status(monkeypatch) -> None:
    from tradingagents.finetuning import training
    from tradingagents.finetuning.errors import SequenceTooLongError

    monkeypatch.setattr(training, "_load_optional_stack", lambda: (object(), object(), None))
    monkeypatch.setattr(training, "_set_seed", lambda *_args: None)
    monkeypatch.setattr(training, "_dataset", lambda _prepared: ([example()], []))
    monkeypatch.setattr(training, "_tokenize_rows", lambda *_args: (_ for _ in ()).throw(SequenceTooLongError("too long")))
    result = training.train(
        {"train": [example()]},
        model=object(),
        tokenizer_policy=TokenizationPolicy(tokenizer=TinyTokenizer()),
        config=TrainingConfig(epochs=1, max_steps=1, gradient_checkpointing=False),
    )
    assert result.status is Phase11Status.SEQUENCE_TOO_LONG


def test_invalid_tokenizer_returns_typed_status(monkeypatch) -> None:
    from tradingagents.finetuning import training

    monkeypatch.setattr(training, "_load_optional_stack", lambda: (object(), object(), None))
    monkeypatch.setattr(training, "_set_seed", lambda *_args: None)
    result = training.train(
        {"train": [example()]},
        model=object(),
        tokenizer_policy=TokenizationPolicy(tokenizer=object()),
        config=TrainingConfig(epochs=1, max_steps=1, gradient_checkpointing=False),
    )
    assert result.status is Phase11Status.TOKENIZER_INVALID


def test_missing_target_module_returns_typed_status(monkeypatch) -> None:
    from tradingagents.finetuning import training
    from tradingagents.finetuning.lora import TargetModuleMissingError

    monkeypatch.setattr(training, "_load_optional_stack", lambda: (object(), object(), None))
    monkeypatch.setattr(training, "_set_seed", lambda *_args: None)
    monkeypatch.setattr(
        training,
        "attach_lora",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TargetModuleMissingError("missing")),
    )
    result = training.train(
        {"train": [example()]},
        model=object(),
        tokenizer_policy=TokenizationPolicy(tokenizer=TinyTokenizer()),
        config=TrainingConfig(epochs=1, max_steps=1, gradient_checkpointing=False),
    )
    assert result.status is Phase11Status.TARGET_MODULE_MISSING
