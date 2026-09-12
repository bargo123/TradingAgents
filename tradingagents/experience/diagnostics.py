"""Bounded, sanitized diagnostic labels; no report prose is retained."""

from __future__ import annotations

import re
from typing import Any

_SAFE = re.compile(r"[^A-Za-z0-9_.:/=-]+")


def sanitize_diagnostic(value: Any, limit: int = 256) -> str:
    text = _SAFE.sub("_", str(value).replace("\r", " ").replace("\n", " "))
    return text[:limit].strip("_")


def diagnostic_label(code: str, **metadata: Any) -> str:
    parts = [sanitize_diagnostic(code)]
    for key in sorted(metadata):
        parts.append(f"{sanitize_diagnostic(key)}={sanitize_diagnostic(metadata[key])}")
    return " ".join(parts)[:1024]


__all__ = ["sanitize_diagnostic", "diagnostic_label"]
