"""Read-only validation of published Phase 11 adapter runs.

Validation deliberately operates on local files only.  It never changes an
adapter and optional ML frameworks are imported only while performing the
reload smoke check.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import validate_run_hashes
from .fingerprints import directory_hash
from .models import ADAPTER_PACKAGE_VERSION, RUN_MANIFEST_VERSION, Phase11Status, ValidationReport
from .phase10 import Phase10Generation
from .provenance import ModelProvenance, ProvenanceError


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
    files = pm.get("files")
    if not isinstance(files, Mapping):
        errors.append("prepared file metadata missing")
        files = {}
    for name in ("train.sft.jsonl", "validation.sft.jsonl"):
        info = files.get(name)
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
    if not adapter.is_dir():
        errors.append("adapter directory missing")
        return None
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
    targets = config.get("target_modules")
    if (
        not isinstance(targets, (list, tuple))
        or not targets
        or any(not isinstance(item, str) or not item for item in targets)
        or len(set(targets)) != len(targets)
    ):
        errors.append("adapter target modules missing")
    if type(config.get("r")) is not int or config.get("r", 0) < 1:
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
        provenance = resolved.get("model_provenance")
        if configured_base and not isinstance(provenance, Mapping):
            errors.append("model provenance missing")
        if isinstance(provenance, Mapping):
            provenance_base = provenance.get("base_model") or provenance.get("base_model_name_or_path")
            if configured_base and provenance_base and _same_model_identity(configured_base, provenance_base) is False:
                errors.append("model provenance binding mismatch")
            if provenance_base and config.get("base_model_name_or_path") and _same_model_identity(provenance_base, config.get("base_model_name_or_path")) is False:
                errors.append("adapter provenance compatibility mismatch")
    base = config.get("base_model_name_or_path")
    if not base:
        errors.append("adapter base model binding missing")
    return config


def _same_model_identity(left: Any, right: Any) -> bool | None:
    """Compare local paths case-insensitively while preserving remote IDs."""
    if not isinstance(left, (str, Path)) or not isinstance(right, (str, Path)):
        return False
    left_text, right_text = str(left), str(right)
    left_path, right_path = Path(left_text).expanduser(), Path(right_text).expanduser()
    if left_path.is_absolute() or right_path.is_absolute() or left_path.exists() or right_path.exists():
        try:
            return str(left_path.resolve()).casefold() == str(right_path.resolve()).casefold()
        except OSError:
            return left_text.casefold() == right_text.casefold()
    return left_text == right_text


def _check_model_provenance(config: Mapping[str, Any], errors: list[str]) -> None:
    """Recompute the immutable local model identity when a run records one."""
    provenance = config.get("model_provenance")
    if not isinstance(provenance, Mapping):
        return
    expected = provenance.get("fingerprint")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        errors.append("model provenance fingerprint missing or invalid")
        return
    base = provenance.get("base_model") or config.get("base_model")
    if not isinstance(base, (str, Path)) or not str(base).strip():
        errors.append("model provenance base model binding missing")
        return
    base_path = Path(str(base)).expanduser()
    if not base_path.is_dir():
        errors.append("model provenance snapshot unavailable")
        return
    tokenizer_path = provenance.get("tokenizer_path") or base_path
    try:
        current = ModelProvenance.inspect(
            base_path,
            revision=provenance.get("revision"),
            tokenizer_path=tokenizer_path,
        )
    except (OSError, ProvenanceError, TypeError, ValueError):
        errors.append("model provenance snapshot invalid")
        return
    if current.fingerprint != expected:
        errors.append("model provenance fingerprint mismatch")


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
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        loaded = PeftModel.from_pretrained(model, str(root / "adapter"), is_trainable=False, local_files_only=True)
        if any(p.requires_grad for p in loaded.base_model.parameters()):
            errors.append("frozen-base invariant failed")
        with torch.no_grad():
            vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 2) or 2)
            sample = torch.ones((1, 2), dtype=torch.long) % max(1, vocab_size)
            output = loaded(input_ids=sample)
            if getattr(output, "logits", None) is None and not isinstance(output, dict):
                errors.append("adapter forward pass returned no logits")
        after = {
            _normalise_base_name(name): parameter.detach()
            for name, parameter in loaded.base_model.named_parameters()
            if "lora_" not in name.lower()
        }
        for name, expected in before.items():
            actual = after.get(name)
            if actual is None or not torch.equal(expected, actual):
                errors.append("frozen base weights changed")
                break
    except Exception:
        errors.append("PEFT adapter reload/forward failed")


def _normalise_base_name(name: str) -> str:
    """Remove PEFT wrapper prefixes so base parameters compare by identity."""
    value = name
    for prefix in ("base_model.model.", "model."):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value.replace(".base_layer.", ".")


def validate_run(path: str | Path) -> ValidationReport:
    """Validate one immutable run and return a bounded fail-closed report."""
    root = Path(path).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = _read_json(root / "run_manifest.json")
    except Exception:
        return ValidationReport(False, Phase11Status.ADAPTER_INVALID, ("run manifest missing or invalid",))
    if manifest.get("version") != RUN_MANIFEST_VERSION:
        errors.append("run manifest version is unsupported")
    if manifest.get("adapter_package_version") != ADAPTER_PACKAGE_VERSION:
        errors.append("adapter package version is unsupported")
    request_fp = manifest.get("request_fingerprint")
    if not isinstance(request_fp, str) or not re.fullmatch(r"[0-9a-f]{64}", request_fp):
        errors.append("request fingerprint missing or invalid")
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
    config_path = root / "resolved_config.json"
    try:
        run_config = _read_json(config_path) if config_path.is_file() else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        run_config = {}
        errors.append("resolved config missing or invalid")
    if not config_path.is_file():
        errors.append("resolved config missing")
    if isinstance(run_config, Mapping):
        _check_model_provenance(run_config, errors)
    for required in ("metrics.json", "environment.json"):
        if not (root / required).is_file():
            errors.append(f"{required} missing")
    merged = dict(manifest)
    merged["resolved_config"] = run_config
    merged["config"] = run_config
    if not dataset.get("generation_id"):
        errors.append("dataset generation binding missing")
    else:
        _check_generation(dataset, errors)
    _check_prepared(root, merged, dataset, errors)
    adapter_config = _check_adapter(root, merged, errors, warnings)
    _reload_smoke(root, adapter_config, warnings, errors)
    return ValidationReport(not errors, Phase11Status.COMPLETE if not errors else Phase11Status.ADAPTER_INVALID, tuple(errors[:32]), tuple(warnings[:16]))


__all__ = ["validate_run"]
