"""Immutable, hash-bound Phase 11 run artifact publication."""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .fingerprints import directory_hash, environment_versions, file_sha256, request_fingerprint
from .models import ADAPTER_PACKAGE_VERSION, RUN_MANIFEST_VERSION, canonical_json


class ArtifactError(ValueError):
    pass


def new_run_id() -> str:
    return "run-" + secrets.token_hex(12)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def _safe_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ArtifactError("artifact source is not a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _safe_name(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not name.strip():
        raise ArtifactError("artifact name must be relative")
    return relative


def publish_run(root: str | Path, *, run_id: str | None = None,
                config: Mapping[str, Any], dataset: Mapping[str, Any],
                metrics: Mapping[str, Any], prepared: Mapping[str, str | Path] | None = None,
                adapter: str | Path | None = None, environment: Mapping[str, Any] | None = None,
                request: Mapping[str, Any] | None = None) -> Path:
    """Stage then atomically publish one run; an existing run is never changed."""
    root = Path(root)
    rid = run_id or new_run_id()
    if not isinstance(rid, str) or not rid or len(_safe_name(rid).parts) != 1:
        raise ArtifactError("run_id must be a relative single path component")
    destination = root / rid
    if destination.exists():
        raise ArtifactError("immutable destination already exists")
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{rid}-staging-", dir=str(root)))
    try:
        for folder in ("prepared", "checkpoints", "adapter", "logs"):
            (stage / folder).mkdir()
        _write_json(stage / "resolved_config.json", config)
        _write_json(stage / "dataset_manifest.json", dataset)
        _write_json(stage / "metrics.json", metrics)
        _write_json(stage / "environment.json", environment or environment_versions())
        if request is not None:
            req_fp = request_fingerprint(request)
        else:
            req_fp = request_fingerprint({"config": config, "dataset": dataset})
        manifest = {"version": RUN_MANIFEST_VERSION, "run_id": rid, "request_fingerprint": req_fp,
                    "status": "COMPLETE"}
        if prepared:
            for name, source in prepared.items():
                _safe_copy(Path(source), stage / "prepared" / _safe_name(str(name)))
            manifest["prepared_hashes"] = directory_hash(stage / "prepared")
        if adapter:
            source = Path(adapter)
            if source.is_dir():
                for item in source.rglob("*"):
                    if item.is_file():
                        _safe_copy(item, stage / "adapter" / _safe_name(str(item.relative_to(source))))
            else:
                _safe_copy(source, stage / "adapter" / _safe_name(source.name))
            manifest["adapter_hashes"] = directory_hash(stage / "adapter")
            manifest["adapter_package_version"] = ADAPTER_PACKAGE_VERSION
        _write_json(stage / "run_manifest.json", manifest)
        # Directory rename is atomic and has no-replace semantics on the
        # supported platforms.  The checks above provide the common fast path;
        # translating a race here preserves create-only publication.
        if destination.exists():
            raise ArtifactError("immutable destination already exists")
        try:
            os.rename(stage, destination)
        except OSError as exc:
            if destination.exists():
                raise ArtifactError("immutable destination already exists") from exc
            raise
        return destination
    except Exception:
        # Keep stage for diagnosis as required by the recovery contract.
        raise


def validate_run_hashes(path: str | Path) -> bool:
    root = Path(path).resolve()
    try:
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            return False
        for folder, key in (("prepared", "prepared_hashes"), ("adapter", "adapter_hashes")):
            entries = manifest.get(key)
            if not isinstance(entries, Mapping):
                return False
            base = (root / folder).resolve()
            expected_entries: dict[str, str] = {}
            for rel, expected in entries.items():
                if not isinstance(rel, str) or not isinstance(expected, str):
                    return False
                safe = _safe_name(rel)
                item = (base / safe).resolve()
                if base != item and base not in item.parents:
                    return False
                if not item.is_file() or file_sha256(item) != expected:
                    return False
                expected_entries[str(safe).replace("\\", "/")] = expected
            if directory_hash(base) != expected_entries:
                return False
        return True
    except (ArtifactError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


__all__ = ["ArtifactError", "new_run_id", "publish_run", "validate_run_hashes"]
