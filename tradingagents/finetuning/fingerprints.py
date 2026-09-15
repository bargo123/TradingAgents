"""Deterministic hashes and privacy-safe runtime metadata."""
from __future__ import annotations

import hashlib
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

from .models import canonical_json


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_fingerprint(request: Any) -> str:
    return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()


def directory_hash(path: str | Path) -> dict[str, str]:
    root = Path(path)
    return {str(item.relative_to(root)).replace("\\", "/"): file_sha256(item)
            for item in sorted(root.rglob("*")) if item.is_file()}


_SECRET = ("key", "token", "secret", "password", "credential", "authorization")
_SECRET_VALUE = re.compile(
    r"(?:api[_ -]?key|access[_ -]?token|secret|password|credential|authorization|bearer\s+)[=: \t]+\S+",
    re.IGNORECASE,
)


def redact_environment(values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return bounded environment metadata with secret values omitted."""
    source = dict(os.environ if values is None else values)
    secret_count = 0
    safe: dict[str, str] = {}
    for key, value in sorted(source.items()):
        name = str(key)
        text = str(value)
        if any(word in name.casefold() for word in _SECRET) or _SECRET_VALUE.search(text):
            secret_count += 1
            continue
        safe[name] = text[:256]
    # Do not persist variable names: names such as OPENAI_API_KEY are sensitive
    # metadata too, and the contract scanner intentionally rejects them.
    return {"variables": safe, "redacted_variable_count": secret_count,
            "python": sys.version.split()[0], "platform": platform.platform()}


def environment_versions() -> dict[str, Any]:
    result = redact_environment()
    for module in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            # Version inspection never causes an optional package import/download.
            imported = sys.modules.get(module)
            if imported is None:
                result[module] = None
                continue
            result[module] = getattr(imported, "__version__", "installed")
        except Exception:
            result[module] = None
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    try:
        cuda_available = bool(cuda.is_available()) if cuda is not None else False
    except Exception:
        cuda_available = False
    try:
        gpu_count = int(cuda.device_count()) if cuda_available else 0
    except (AttributeError, TypeError, ValueError, RuntimeError):
        gpu_count = 0
    models: list[str] = []
    if cuda_available and cuda is not None:
        for index in range(max(0, gpu_count)):
            try:
                name = cuda.get_device_name(index)
            except Exception:
                name = None
            if isinstance(name, str) and name:
                models.append(name[:256])
    torch_version = getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None
    result.update(
        {
            "cuda_available": cuda_available,
            "cuda_version": str(torch_version) if torch_version is not None else None,
            "gpu_count": gpu_count,
            "gpu_models": models,
        }
    )
    return result
