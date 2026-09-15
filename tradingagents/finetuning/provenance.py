"""Offline, deterministic model and tokenizer provenance inspection."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProvenanceError(ValueError):
    """Raised when a model cannot be identified by an immutable local snapshot."""


_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$", re.IGNORECASE)
_MUTABLE_REVISIONS = {"latest", "main", "master", "default", "head"}
_SAMPLE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ModelProvenance:
    base_model: str
    revision: str
    tokenizer_path: str
    fingerprint: str
    config_fingerprint: str
    tokenizer_fingerprint: str
    weight_fingerprint: str
    architecture: tuple[str, ...]
    dtype: str | None
    parameter_count: int
    files: tuple[dict[str, Any], ...]

    @property
    def model_architecture(self) -> tuple[str, ...]:
        """Compatibility alias used by provenance consumers."""
        return self.architecture

    @property
    def weight_index_fingerprint(self) -> str:
        return self.weight_fingerprint

    @classmethod
    def inspect(
        cls,
        base_model: str | Path,
        revision: str | None = None,
        tokenizer_path: str | Path | None = None,
    ) -> ModelProvenance:
        model_path = Path(base_model).expanduser() if isinstance(base_model, Path) else Path(str(base_model))
        if not model_path.exists() or not model_path.is_dir():
            if revision is None or not _IMMUTABLE_REVISION.fullmatch(str(revision)):
                raise ProvenanceError("remote model requires an immutable revision and local snapshot")
            raise ProvenanceError("remote model inspection requires a local snapshot; network loading is disabled")
        if revision is not None and (
            str(revision).lower() in _MUTABLE_REVISIONS
            or str(revision).startswith("refs/")
            or not (str(revision).startswith("local-") or _IMMUTABLE_REVISION.fullmatch(str(revision)))
        ):
            raise ProvenanceError("revision must be immutable or an explicit local snapshot")
        resolved_revision = str(revision) if revision is not None else "local-filesystem"
        tok_path = Path(tokenizer_path).expanduser() if tokenizer_path is not None else model_path
        if not tok_path.exists() or not tok_path.is_dir():
            raise ProvenanceError("tokenizer path must be a local snapshot")

        config = _read_json(model_path / "config.json")
        architecture = _as_tuple(config.get("architectures") or config.get("architectures", []))
        dtype = _string(config.get("torch_dtype") or config.get("dtype"))
        files = _collect_files(model_path, tok_path)
        config_files = tuple(item for item in files if item["path"] == "config.json")
        tokenizer_files = tuple(item for item in files if item["kind"] == "tokenizer")
        weight_files = tuple(item for item in files if item["kind"] == "weights")
        config_fp = _digest(config_files)
        tokenizer_fp = _digest(tokenizer_files)
        weight_fp = _digest(weight_files)
        payload = {"base_model": str(model_path.resolve()), "revision": resolved_revision, "config": config_fp, "tokenizer": tokenizer_fp, "weights": weight_fp}
        fingerprint = hashlib.sha256(_canonical(payload)).hexdigest()
        return cls(str(model_path.resolve()), resolved_revision, str(tok_path.resolve()), fingerprint, config_fp, tokenizer_fp, weight_fp, architecture, dtype, _parameter_count(config, model_path), files)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(x) for x in value) if isinstance(value, (list, tuple)) else ()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProvenanceError("local model is missing config.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProvenanceError("invalid model config.json") from exc
    return value if isinstance(value, dict) else {}


def _collect_files(model: Path, tokenizer: Path) -> tuple[dict[str, Any], ...]:
    roots = [(model, "model"), (tokenizer, "tokenizer")] if tokenizer != model else [(model, "model")]
    selected: dict[str, dict[str, Any]] = {}
    for root, _root_kind in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            relevant = name == "config.json" or "tokenizer" in name or name in {"special_tokens_map.json", "chat_template.jinja"} or name.endswith(".index.json") or name.endswith((".safetensors", ".bin", ".pt", ".pth"))
            if not relevant:
                continue
            rel = str(path.relative_to(root)).replace("\\", "/") if root == model else "tokenizer/" + str(path.relative_to(root)).replace("\\", "/")
            kind = "config" if name == "config.json" else "tokenizer" if ("tokenizer" in name or name in {"special_tokens_map.json", "chat_template.jinja"}) else "weights"
            selected[rel] = {"path": rel, "kind": kind, "size": path.stat().st_size, "sha256": _sample_hash(path)}
    return tuple(selected[key] for key in sorted(selected))


def _sample_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        head = stream.read(_SAMPLE_BYTES)
        digest.update(head)
        if path.stat().st_size > _SAMPLE_BYTES:
            stream.seek(max(0, path.stat().st_size - _SAMPLE_BYTES))
            digest.update(stream.read(_SAMPLE_BYTES))
    return digest.hexdigest()


def _digest(items: tuple[dict[str, Any], ...]) -> str:
    return hashlib.sha256(_canonical(items)).hexdigest()


def _parameter_count(config: dict[str, Any], model_path: Path) -> int:
    for key in ("num_parameters", "num_params", "n_parameters"):
        if isinstance(config.get(key), int):
            return max(0, config[key])
    for index in sorted(model_path.glob("*.index.json")):
        try:
            metadata = json.loads(index.read_text(encoding="utf-8")).get("metadata", {})
            for key in ("total_parameters", "num_parameters", "num_params"):
                if isinstance(metadata.get(key), int):
                    return max(0, metadata[key])
        except (OSError, ValueError, AttributeError):
            continue
    total = 0
    for weights in sorted(model_path.glob("*.safetensors")):
        total += _safetensors_parameter_count(weights)
    if total:
        return total
    return 0


def _safetensors_parameter_count(path: Path) -> int:
    """Read only the bounded safetensors header; never deserialize tensor data."""
    try:
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > _SAMPLE_BYTES * 4:
                return 0
            header = json.loads(stream.read(header_size))
    except (OSError, ValueError, struct.error, TypeError):
        return 0
    count = 0
    for tensor in header.values() if isinstance(header, dict) else ():
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        if isinstance(shape, list):
            product = 1
            for dimension in shape:
                if not isinstance(dimension, int) or dimension < 0:
                    product = 0
                    break
                product *= dimension
            count += product
    return count
