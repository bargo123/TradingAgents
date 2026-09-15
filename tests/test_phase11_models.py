import dataclasses
import math

import pytest

from tradingagents.finetuning import (
    ADAPTER_PACKAGE_VERSION,
    LoraConfig,
    Phase11Status,
    TrainingConfig,
    canonical_hash,
    canonical_json,
)
from tradingagents.finetuning.errors import ContractError, InvalidStatusError, UnknownVersionError


def test_defaults_and_frozen_json_serialization():
    cfg = TrainingConfig()
    assert cfg.learning_rate == 2e-4
    assert cfg.max_sequence_length == 4096
    assert cfg.lora.r == 16
    assert dataclasses.is_dataclass(cfg) and dataclasses.is_dataclass(LoraConfig())
    assert '"mode":"lora"' in cfg.to_json()
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.mode = "qlora"


def test_statuses_are_closed_and_versions_are_checked():
    assert Phase11Status.COMPLETE.value == "COMPLETE"
    with pytest.raises(InvalidStatusError):
        from tradingagents.finetuning import TrainingReport

        TrainingReport("not-a-status")
    with pytest.raises(UnknownVersionError):
        TrainingConfig(version="phase11-training-policy.v999")
    with pytest.raises(UnknownVersionError):
        from tradingagents.finetuning import SFTExample

        SFTExample(
            "x",
            "train",
            ({"role": "user", "content": "x"}, {"role": "assistant", "content": "{}"}),
            {},
            "bad",
        )


def test_bounds_and_sensitive_nonfinite_rejection():
    with pytest.raises(ContractError):
        LoraConfig(dropout=1.0)
    with pytest.raises(ContractError):
        TrainingConfig(max_sequence_length=0)
    with pytest.raises(ContractError):
        canonical_json({"loss": math.nan})
    with pytest.raises(ContractError):
        canonical_json({"api_key": "do-not-serialize"})
    with pytest.raises(ContractError):
        canonical_json({"text": "hidden chain of thought"})


def test_package_exports_are_lazy_and_version_constant_present():
    assert ADAPTER_PACKAGE_VERSION == "phase11-adapter-package.v1"
    import tradingagents.finetuning as package

    assert "torch" not in package.__dict__
