from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import EmptyEligibleSetError, Phase10InvalidError
from .formatting import SFTFormatter
from .models import SFT_FORMAT_VERSION, canonical_json
from .phase10 import Phase10Generation
from .tokenization import TokenizationPolicy, tokenize_example


@dataclass(frozen=True, slots=True)
class PreparationResult:
    status: str
    manifest: dict[str, Any]
    output_root: Path


def validate_prepared(output_root: str | Path) -> dict[str, Any]:
    """Validate a prepared manifest and its train/validation file hashes."""
    root = Path(output_root).resolve()
    folder = root / "prepared" if (root / "prepared").is_dir() else root
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise Phase10InvalidError("prepared manifest is required")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Phase10InvalidError("prepared manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "PREPARED":
        raise Phase10InvalidError("prepared manifest status is invalid")
    if not isinstance(manifest.get("generation_id"), str) or not manifest["generation_id"].strip():
        raise Phase10InvalidError("prepared generation binding is missing")
    if manifest.get("format_version") != SFT_FORMAT_VERSION:
        raise Phase10InvalidError("prepared format version is unsupported")
    if manifest.get("formatter_version") != SFT_FORMAT_VERSION:
        raise Phase10InvalidError("prepared formatter version is unsupported")
    if not isinstance(manifest.get("tokenization_policy_version"), str) or not manifest["tokenization_policy_version"].strip():
        raise Phase10InvalidError("prepared tokenization policy version is missing")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise Phase10InvalidError("prepared file metadata is missing")
    expected_files = {"manifest.json", "train.sft.jsonl", "validation.sft.jsonl"}
    actual_files = {
        str(item.relative_to(folder)).replace("\\", "/")
        for item in folder.rglob("*")
        if item.is_file()
    }
    if actual_files != expected_files:
        raise Phase10InvalidError("prepared destination contains unexpected files")
    if set(files) != {"train.sft.jsonl", "validation.sft.jsonl"}:
        raise Phase10InvalidError("prepared file metadata contains unexpected files")
    generation_path = manifest.get("generation_path")
    if not isinstance(generation_path, str) or not generation_path.strip():
        raise Phase10InvalidError("prepared Phase 10 generation path is missing")
    try:
        generation = Phase10Generation.open(generation_path)
    except EmptyEligibleSetError as exc:
        raise Phase10InvalidError("prepared generation is empty") from exc
    except Exception as exc:
        raise Phase10InvalidError("prepared Phase 10 generation is invalid") from exc
    if generation.generation_id != manifest.get("generation_id"):
        raise Phase10InvalidError("prepared generation identity mismatch")
    if manifest.get("generation_manifest_hash") != generation.manifest_hash:
        raise Phase10InvalidError("prepared generation manifest binding mismatch")
    if manifest.get("generation_file_hashes") != generation.file_hashes:
        raise Phase10InvalidError("prepared generation file binding mismatch")
    if manifest.get("generation_policy_versions") != generation.policy_versions:
        raise Phase10InvalidError("prepared generation policy binding mismatch")
    if manifest.get("source_fingerprints") != generation.source_fingerprints:
        raise Phase10InvalidError("prepared generation source binding mismatch")
    for name, count_key in (("train.sft.jsonl", "train_count"), ("validation.sft.jsonl", "validation_count")):
        path = folder / name
        info = files.get(name)
        if not path.is_file() or not isinstance(info, dict):
            raise Phase10InvalidError(f"prepared file is missing: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != info.get("sha256"):
            raise Phase10InvalidError(f"prepared file hash mismatch: {name}")
        count = manifest.get(count_key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise Phase10InvalidError(f"prepared count is invalid: {count_key}")
    return manifest


def _jsonl(rows) -> bytes:
    return b"".join((canonical_json(r) + "\n").encode("utf-8") for r in rows)


def _row(ex):
    return {"example_id": ex.example_id, "split": ex.split, "messages": [dict(m) for m in ex.messages], "target": dict(ex.target), "version": ex.version}


def prepare_generation(generation, output_root, formatter: SFTFormatter, tokenizer_policy: TokenizationPolicy) -> PreparationResult:
    try:
        # Re-open an object as well as a path so every invocation re-checks the
        # immutable Phase 10 hashes before constructing a tokenizer.
        path = generation.path if isinstance(generation, Phase10Generation) else generation
        gen = Phase10Generation.open(path)
    except EmptyEligibleSetError:
        return PreparationResult("EMPTY_ELIGIBLE_SET", {"status": "EMPTY_ELIGIBLE_SET"}, Path(output_root))
    except Exception as exc:
        if isinstance(exc, Phase10InvalidError):
            raise
        raise Phase10InvalidError(str(exc)) from exc
    out = Path(output_root).resolve()
    if out == gen.path or gen.path in out.parents:
        raise Phase10InvalidError("prepared output must not overlap Phase 10 source")
    out.mkdir(parents=True, exist_ok=True)

    def prepare(rows, split):
        output = []
        for source in sorted(rows, key=lambda r: r.example_id):
            item = formatter.format(source, split)
            tokenize_example(item, tokenizer_policy)
            output.append(_row(item))
        return output
    train = prepare(gen.train, "train")
    validation = prepare(gen.validation, "validation")
    files = {"train.sft.jsonl": _jsonl(train), "validation.sft.jsonl": _jsonl(validation)}
    manifest = {
        "status": "PREPARED", "generation_id": gen.generation_id,
        "generation_path": str(gen.path), "generation_manifest_hash": gen.manifest_hash,
        "generation_file_hashes": gen.file_hashes, "generation_policy_versions": gen.policy_versions,
        "source_fingerprints": gen.source_fingerprints,
        "formatter_version": formatter.policy_version, "format_version": SFT_FORMAT_VERSION,
        "tokenization_policy_version": tokenizer_policy.policy_version,
        "tokenizer": tokenizer_policy.fingerprint(),
        "train_count": len(train), "validation_count": len(validation),
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)} for name, data in files.items()},
    }
    manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
    files["manifest.json"] = manifest_bytes
    stage = Path(tempfile.mkdtemp(prefix=".prepared-staging-", dir=str(out)))
    failed = True
    try:
        for name, data in files.items():
            (stage / name).write_bytes(data)
        prepared = out / "prepared"
        if prepared.exists():
            # Idempotent reruns are allowed only when the complete byte/hash
            # contract is unchanged.  Never overwrite a published preparation.
            if not prepared.is_dir():
                raise Phase10InvalidError("prepared destination already exists and is not a directory")
            existing_names = {
                str(item.relative_to(prepared)).replace("\\", "/")
                for item in prepared.rglob("*")
                if item.is_file()
            }
            if existing_names != set(files):
                raise Phase10InvalidError("prepared destination contains unexpected files")
            existing = {
                name: (prepared / name).read_bytes() if (prepared / name).is_file() else None
                for name in files
            }
            if any(existing[name] != data for name, data in files.items()):
                raise Phase10InvalidError("prepared destination already exists with different content")
            failed = False
            return PreparationResult("PREPARED", manifest, out)
        try:
            os.rename(stage, prepared)
        except OSError as exc:
            if prepared.exists():
                raise Phase10InvalidError("prepared destination already exists") from exc
            raise
        stage = None
        failed = False
    finally:
        # Keep interrupted staging directories intact for operator diagnosis.
        # Only clean the temporary tree after an idempotent success; a failed
        # publication remains visibly recoverable under the output root.
        if stage is not None and not failed:
            with suppress(OSError):
                for item in stage.iterdir():
                    item.unlink()
                stage.rmdir()
    return PreparationResult("PREPARED", manifest, out)


__all__ = ["PreparationResult", "prepare_generation", "validate_prepared"]
