"""Create-only publication and validation for Phase 11A generations."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

FILES = ("manifest.json", "examples.jsonl", "excluded.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl", "source_index.json")


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return value.value
    return value


def _json(value: Any) -> bytes:
    return (json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _row_id(row: Any) -> str:
    value = getattr(row, "example_id", None)
    if value is None and isinstance(row, Mapping):
        value = row.get("example_id") or row.get("id")
    return str(value or hashlib.sha256(_json(row)).hexdigest())


def _rows(items: Iterable[Any]) -> list[Any]:
    return sorted((_plain(x) for x in items), key=lambda x: str(x.get("example_id", x.get("id", ""))) if isinstance(x, Mapping) else _row_id(x))


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_generation(output_root: str | Path, examples: Iterable[Any], exclusions: Iterable[Any], assignments: Any = None, metadata: Mapping[str, Any] | None = None) -> Path:
    """Publish one immutable generation using a staging directory and atomic rename."""
    examples = _rows(examples)
    exclusions = _rows(exclusions)
    metadata = dict(metadata or {})
    generation_id = str(metadata.pop("generation_id", "generation-" + _hash(_json({"examples": examples, "excluded": exclusions, "assignments": _plain(assignments), "metadata": metadata}))[:24]))
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / generation_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"generation already exists: {generation_id}")
    split_map: dict[str, str] = {}
    if isinstance(assignments, Mapping):
        split_map = {str(k): str(v) for k, v in assignments.items()}
    elif assignments is not None:
        values = getattr(assignments, "assignments", assignments)
        for item in values:
            item = _plain(item)
            if isinstance(item, Mapping):
                split_map[str(item.get("example_id"))] = str(item.get("split"))
    by_id = {_row_id(x): x for x in examples}
    files_data: dict[str, bytes] = {
        "examples.jsonl": b"".join(_json(x) for x in examples),
        "excluded.jsonl": b"".join(_json(x) for x in exclusions),
    }
    for name, split in (("train.jsonl", "train"), ("validation.jsonl", "validation"), ("test.jsonl", "test")):
        files_data[name] = b"".join(_json(by_id[key]) for key, value in sorted(split_map.items()) if value == split and key in by_id)
    source_index = metadata.get("source_index", {})
    files_data["source_index.json"] = _json(source_index)
    manifest = {"schema_version": "phase11a-manifest.v1", "generation_id": generation_id, "status": "PUBLISHED" if examples else "EMPTY", "counts": {"examples": len(examples), "excluded": len(exclusions)}, "source_fingerprints": metadata.get("source_fingerprints", {}), "policy_versions": metadata.get("policy_versions", {}), "metadata": metadata, "files": {name: {"sha256": _hash(data), "bytes": len(data)} for name, data in files_data.items()}}
    files_data["manifest.json"] = _json(manifest)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{generation_id}-", dir=str(root)))
    try:
        for name, data in files_data.items():
            (staging / name).write_bytes(data)
        os.replace(staging, destination)
    except Exception:
        # Preserve staging for diagnosis; never publish partial generations.
        raise
    return destination


def validate_generation(path: str | Path):
    """Validate hashes and required files without modifying the generation."""
    root = Path(path)
    errors: list[str] = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"manifest unreadable: {exc}")
        manifest = {}
    for name in FILES:
        if not (root / name).is_file():
            errors.append(f"missing file: {name}")
    for name, info in manifest.get("files", {}).items():
        target = root / name
        if target.is_file() and isinstance(info, Mapping) and info.get("sha256") != _hash(target.read_bytes()):
            errors.append(f"hash mismatch: {name}")
    try:
        from .models import ValidationReport
        return ValidationReport(valid=not errors, errors=tuple(errors))
    except (ImportError, TypeError):
        return {"valid": not errors, "errors": errors}


__all__ = ["write_generation", "validate_generation"]
