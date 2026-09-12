"""Stable path and byte identity helpers for knowledge resources."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import unicodedata
from pathlib import Path


class SourceChangedError(RuntimeError):
    """Raised when a source file changes while its bytes are being read."""


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_relative_text(value: str | os.PathLike[str]) -> str:
    text = unicodedata.normalize("NFC", os.fspath(value)).replace("\\", "/")
    # A relative path is the identity input.  Drive prefixes and leading
    # slashes would make the digest depend on the machine where it was read.
    if text.startswith("/") or (len(text) >= 2 and text[1] == ":"):
        raise ValueError("relative path required")
    parts: list[str] = []
    for raw_part in text.split("/"):
        part = unicodedata.normalize("NFC", raw_part)
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("relative path must not escape its root")
        parts.append(part.casefold())
    if not parts:
        raise ValueError("relative path must be non-empty")
    return "/".join(parts)


def canonical_relative_path(path: str | os.PathLike[str], source_root: str | os.PathLike[str]) -> str:
    """Return a normalized, case-folded path relative to ``source_root``.

    The containment check uses resolved paths, so a caller cannot derive an
    identity for a path outside the configured source tree.  The returned
    identity uses forward slashes and Unicode NFC normalization.
    """

    root = Path(source_root).expanduser().resolve(strict=False)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        # A relative source-root-shaped path (``root/A.pdf``) is interpreted
        # relative to the current working directory, as Path.relative_to does.
        candidate = candidate.resolve(strict=False)
    else:
        candidate = candidate.resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside source_root: {candidate}") from exc
    return _canonical_relative_text(relative)


def resource_id_for(relative_path: str | os.PathLike[str]) -> str:
    """Return the stable resource identity for a source-tree relative path."""

    canonical = _canonical_relative_text(relative_path)
    return "res_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def document_id_for(source_hash: str) -> str:
    """Return the content identity for a SHA-256 source hash."""

    normalized = str(source_hash).strip().casefold()
    if not _HASH_RE.fullmatch(normalized):
        raise ValueError("source_hash must be a 64-character hexadecimal SHA-256 value")
    return "doc_" + normalized


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    """Hash file bytes and return ``(sha256_hex, byte_count)``.

    The file is opened read-only and statted before and after the read.  A
    changed size/mtime/inode is rejected so a staged artifact can never claim
    to represent a moving source file.
    """

    if isinstance(chunk_size, bool) or int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    before = source.stat()
    lstat = source.stat(follow_symlinks=False)
    if os.path.islink(source) or bool(getattr(lstat, "st_file_attributes", 0) & 0x0400):
        raise ValueError(f"source path must be a regular file, not a link: {source}")
    if not stat.S_ISREG(lstat.st_mode) or not source.is_file():
        raise ValueError(f"source path is not a regular file: {source}")
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    after = source.stat()
    identity_before = (before.st_size, before.st_mtime_ns, getattr(before, "st_ino", None))
    identity_after = (after.st_size, after.st_mtime_ns, getattr(after, "st_ino", None))
    if identity_before != identity_after or total != after.st_size:
        raise SourceChangedError(f"source changed while hashing: {source}")
    return digest.hexdigest(), total


__all__ = [
    "SourceChangedError",
    "canonical_relative_path",
    "document_id_for",
    "resource_id_for",
    "sha256_file",
]
