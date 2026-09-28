"""Small shared validation helpers for user-supplied filesystem paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def require_nonempty_path(value: Any, name: str) -> Path:
    """Return a path while rejecting empty values and the current directory."""

    if isinstance(value, (bytes, bytearray)):
        raise ValueError(f"{name} must be a non-empty path")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    try:
        path = Path(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty path") from exc
    if not str(path).strip() or str(path) == ".":
        raise ValueError(f"{name} must be a non-empty path")
    return path


__all__ = ["require_nonempty_path"]
