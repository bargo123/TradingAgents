"""Immutable, hash-bound Phase 11 run artifact publication."""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .fingerprints import directory_hash, environment_versions, file_sha256, request_fingerprint
from .models import RUN_MANIFEST_VERSION, canonical_json


class ArtifactError(ValueError):
    pass


def new_run_id() -> str:
    return "run-" + secrets.token_hex(12)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def _safe_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def publish_run(root: str | Path, *, run_id: str | None = None,
                config: Mapping[str, Any], dataset: Mapping[str, Any],
                metrics: Mapping[str, Any], prepared: Mapping[str, str | Path] | None = None,
                adapter: str | Path | None = None, environment: Mapping[str, Any] | None = None,
                request: Mapping[str, Any] | None = None) -> Path:
    """Stage then atomically publish one run; an existing run is never changed."""
    root = Path(root)
    rid = run_id or new_run_id()
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
                _safe_copy(Path(source), stage / "prepared" / name)
            manifest["prepared_hashes"] = directory_hash(stage / "prepared")
        if adapter:
            source = Path(adapter)
            if source.is_dir():
                for item in source.rglob("*"):
                    if item.is_file(): _safe_copy(item, stage / "adapter" / item.relative_to(source))
            else: _safe_copy(source, stage / "adapter" / source.name)
            manifest["adapter_hashes"] = directory_hash(stage / "adapter")
        _write_json(stage / "run_manifest.json", manifest)
        # os.replace is atomic; refusing destination above plus a second existence check
        # prevents accidental replacement under normal concurrent publication.
        if destination.exists():
            raise ArtifactError("immutable destination already exists")
        os.replace(stage, destination)
        return destination
    except Exception:
        # Keep stage for diagnosis as required by the recovery contract.
        raise


def validate_run_hashes(path: str | Path) -> bool:
    root = Path(path)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    for folder, key in (("prepared", "prepared_hashes"), ("adapter", "adapter_hashes")):
        for rel, expected in manifest.get(key, {}).items():
            item = root / folder / rel
            if not item.is_file() or file_sha256(item) != expected:
                return False
    return True


__all__ = ["ArtifactError", "new_run_id", "publish_run", "validate_run_hashes"]
