"""Deterministic hashes and privacy-safe runtime metadata."""
from __future__ import annotations

import hashlib
import json
import os
import platform
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


def redact_environment(values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return bounded environment metadata with secret values omitted."""
    source = dict(os.environ if values is None else values)
    secret_count = sum(any(word in k.casefold() for word in _SECRET) for k in source)
    # Do not persist variable names: names such as OPENAI_API_KEY are sensitive
    # metadata too, and the contract scanner intentionally rejects them.
    safe = {k: str(v)[:256] for k, v in sorted(source.items())
            if not any(word in k.casefold() for word in _SECRET)}
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
    return result
