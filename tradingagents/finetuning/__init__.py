"""Lazy public API for Phase 11 fine-tuning contracts."""

from importlib import import_module

_NAMES = dict.fromkeys(
    (
        "SFT_FORMAT_VERSION",
        "TRAINING_POLICY_VERSION",
        "RUN_MANIFEST_VERSION",
        "ADAPTER_PACKAGE_VERSION",
        "Phase11Status",
        "TrainingConfig",
        "LoraConfig",
        "QloraConfig",
        "DatasetBinding",
        "SFTExample",
        "PreparedManifest",
        "RunManifest",
        "Metrics",
        "TrainingReport",
        "ValidationReport",
        "canonical_json",
        "canonical_hash",
        "contract_to_dict",
    ),
    "tradingagents.finetuning.models",
)

__all__ = sorted(_NAMES)


def __getattr__(name):
    if name in _NAMES:
        return getattr(import_module(_NAMES[name]), name)
    raise AttributeError(name)
