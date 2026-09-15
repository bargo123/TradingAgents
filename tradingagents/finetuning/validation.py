"""Read-only validation of published Phase 11 adapter runs.

Validation deliberately operates on local files only.  It never changes an
adapter and optional ML frameworks are imported only while performing the
reload smoke check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .artifacts import validate_run_hashes
from .fingerprints import directory_hash
from .models import Phase11Status, ValidationReport
from .phase10 import Phase10Generation


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("must be an object")
    return value


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _check_hashes(root: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    try:
        for key in ("prepared_hashes", "adapter_hashes"):
            if not isinstance(manifest.get(key), dict) or not manifest[key]:
                errors.append(f"{key} missing")
        if not validate_run_hashes(root):
            errors.append("artifact hash mismatch")
        for folder, key in (("prepared", "prepared_hashes"), ("adapter", "adapter_hashes")):
            expected = manifest.get(key)
            if expected is not None and directory_hash(root / folder) != expected:
                errors.append(f"{folder} file set/hash mismatch")
    except Exception as exc:
        errors.append(f"artifact hashes unavailable: {type(exc).__name__}")


def _check_prepared(root: Path, run: dict[str, Any], dataset: dict[str, Any], errors: list[str]) -> None:
    prepared = root / "prepared"
    manifest_path = prepared / "manifest.json"
    if not manifest_path.is_file():
        errors.append("prepared manifest missing")
        return
    try:
        pm = _read_json(manifest_path)
    except Exception:
        errors.append("prepared manifest invalid")
        return
    if pm.get("status") != "PREPARED":
        errors.append("prepared status is not PREPARED")
    if pm.get("generation_id") != dataset.get("generation_id"):
        errors.append("prepared generation binding mismatch")
    for name in ("train.sft.jsonl", "validation.sft.jsonl"):
        info = pm.get("files", {}).get(name)
        path = prepared / name
        if not path.is_file() or not isinstance(info, dict) or _digest(path) != info.get("sha256"):
            errors.append(f"prepared hash mismatch: {name}")
        if path.is_file():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip() and json.loads(line).get("split") == "test":
                        errors.append("test split leaked into prepared data")
                        break
            except Exception:
                errors.append(f"prepared file invalid: {name}")
    if pm.get("source_fingerprints") != dataset.get("source_fingerprints"):
        errors.append("prepared source fingerprints mismatch")
    config = run.get("config", {})
    if isinstance(config, dict):
        if pm.get("formatter_version") and config.get("formatter_version") not in (None, pm.get("formatter_version")):
            errors.append("formatter compatibility mismatch")
        if pm.get("tokenization_policy_version") and config.get("tokenization_policy_version") not in (None, pm.get("tokenization_policy_version")):
            errors.append("tokenizer policy compatibility mismatch")


def _check_generation(dataset: dict[str, Any], errors: list[str]) -> None:
    path = dataset.get("generation_path") or dataset.get("path")
    if not path:
        errors.append("Phase 10 generation path missing")
        return
    try:
        generation = Phase10Generation.open(path)
    except Exception as exc:
        errors.append(f"Phase 10 generation invalid: {type(exc).__name__}")
        return
    if dataset.get("generation_id") != generation.generation_id:
        errors.append("Phase 10 generation ID mismatch")
    expected = dataset.get("manifest_hash")
    if expected and expected != generation.manifest_hash:
        errors.append("Phase 10 manifest binding mismatch")
    if dataset.get("source_fingerprints") != generation.source_fingerprints:
        errors.append("Phase 10 source fingerprints mismatch")
    if dataset.get("train_count") is not None and dataset.get("train_count") != len(generation.train):
        errors.append("Phase 10 train count mismatch")
    if dataset.get("validation_count") is not None and dataset.get("validation_count") != len(generation.validation):
        errors.append("Phase 10 validation count mismatch")


def _check_adapter(root: Path, run: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any] | None:
    adapter = root / "adapter"
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        errors.append("adapter_config.json missing")
        return None
    try:
        config = _read_json(config_path)
    except Exception:
        errors.append("adapter config invalid")
        return None
    if str(config.get("peft_type", "LORA")).upper() != "LORA":
        errors.append("unsupported adapter type")
    if not isinstance(config.get("target_modules"), (list, tuple)) or not config.get("target_modules"):
        errors.append("adapter target modules missing")
    if not isinstance(config.get("r"), int) or config.get("r", 0) < 1:
        errors.append("adapter rank invalid")
    if not any(item.is_file() and item.name != "adapter_config.json" for item in adapter.iterdir()):
        errors.append("adapter weights missing")
    resolved = run.get("resolved_config", {})
    if isinstance(resolved, dict):
        lora = resolved.get("lora", resolved.get("adapter", {}))
        if isinstance(lora, dict) and lora.get("target_modules") and set(lora["target_modules"]) != set(config.get("target_modules", ())):
            errors.append("adapter target modules mismatch")
        configured_base = resolved.get("base_model") or resolved.get("base_model_name_or_path")
        if configured_base and str(config.get("base_model_name_or_path")) != str(configured_base):
            errors.append("adapter base model compatibility mismatch")
    base = config.get("base_model_name_or_path")
    if not base:
        errors.append("adapter base model binding missing")
    return config


def _reload_smoke(root: Path, config: dict[str, Any] | None, warnings: list[str], errors: list[str]) -> None:
    if config is None:
        return
    try:
        import torch  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
    except (ImportError, ModuleNotFoundError):
        warnings.append("PEFT reload skipped: optional dependencies unavailable")
        return
    base = config.get("base_model_name_or_path")
    if not base or not Path(str(base)).is_dir():
        errors.append("adapter base model snapshot unavailable for reload")
        return
    try:
        model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True)
        before = [p.detach().clone() for p in model.parameters() if not p.requires_grad]
        loaded = PeftModel.from_pretrained(model, str(root / "adapter"), is_trainable=False, local_files_only=True)
        if any(p.requires_grad for p in loaded.base_model.parameters()):
            errors.append("frozen-base invariant failed")
        with torch.no_grad():
            sample = torch.ones((1, 2), dtype=torch.long)
            output = loaded(input_ids=sample)
            if getattr(output, "logits", None) is None and not isinstance(output, dict):
                errors.append("adapter forward pass returned no logits")
        after = [p.detach() for p in model.parameters() if not p.requires_grad]
        if len(before) != len(after) or any(not torch.equal(a, b) for a, b in zip(before, after, strict=True)):
            errors.append("frozen base weights changed")
    except Exception:
        errors.append("PEFT adapter reload/forward failed")


def validate_run(path: str | Path) -> ValidationReport:
    """Validate one immutable run and return a bounded fail-closed report."""
    root = Path(path).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = _read_json(root / "run_manifest.json")
    except Exception:
        return ValidationReport(False, Phase11Status.ADAPTER_INVALID, ("run manifest missing or invalid",))
    if manifest.get("status") != "COMPLETE":
        errors.append("run status is not COMPLETE")
    if manifest.get("run_id") != root.name:
        errors.append("run identity mismatch")
    _check_hashes(root, manifest, errors)
    try:
        dataset = _read_json(root / "dataset_manifest.json")
    except Exception:
        dataset = {}
        errors.append("dataset manifest missing or invalid")
    run_config = _read_json(root / "resolved_config.json") if (root / "resolved_config.json").is_file() else {}
    merged = dict(manifest)
    merged["resolved_config"] = run_config
    if not dataset.get("generation_id"):
        errors.append("dataset generation binding missing")
    else:
        _check_generation(dataset, errors)
    _check_prepared(root, merged, dataset, errors)
    adapter_config = _check_adapter(root, merged, errors, warnings)
    _reload_smoke(root, adapter_config, warnings, errors)
    return ValidationReport(not errors, Phase11Status.COMPLETE if not errors else Phase11Status.ADAPTER_INVALID, tuple(errors[:32]), tuple(warnings[:16]))


__all__ = ["validate_run"]
