"""Bounded, machine-readable diagnostics for knowledge ingestion.

The catalog records identifiers and compact failure metadata only.  It never
copies parsed source text, credentials, or arbitrary exception objects into
SQLite diagnostic rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

from .models import IngestionState


_MAX_ERROR_TEXT = 512


def bounded_error_metadata(error: BaseException | str | None) -> tuple[str | None, str | None, str | None]:
    """Return a bounded ``(type, message, fingerprint)`` diagnostic tuple."""

    if error is None:
        return None, None, None
    error_type = type(error).__name__ if isinstance(error, BaseException) else "Message"
    message = str(error).replace("\x00", " ").strip()[:_MAX_ERROR_TEXT]
    fingerprint = hashlib.sha256(f"{error_type}:{message}".encode("utf-8")).hexdigest()
    return error_type, message, fingerprint


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    """One bounded per-resource diagnostic event from the catalog."""

    event_id: int
    state: IngestionState
    stage: str
    resource_id: str | None = None
    document_id: str | None = None
    attempted_hash: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_fingerprint: str | None = None
    counters: Mapping[str, int] | None = None
    created_at: str | None = None


__all__ = ["QuarantineRecord", "bounded_error_metadata"]
