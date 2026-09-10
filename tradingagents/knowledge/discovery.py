"""Deterministic, read-only source-tree discovery for approved resources."""

from __future__ import annotations

import os
from pathlib import Path
import stat

from .config import KnowledgeConfig
from .identity import SourceChangedError, resource_id_for, sha256_file
from .models import DiscoveredResource, IngestionState


_REPARSE_POINT = 0x0400
_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".epub"})
_METADATA_FILES = frozenset(
    {
        "desktop.ini",
        "thumbs.db",
        "ehthumbs.db",
        ".ds_store",
    }
)
_METADATA_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__macosx",
        "$recycle.bin",
        "system volume information",
        ".spotlight-v100",
        ".trashes",
        "lost+found",
    }
)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_reparse(entry: os.DirEntry[str]) -> bool:
    try:
        info = entry.stat(follow_symlinks=False)
    except OSError:
        return True
    return entry.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)


class SourceScanner:
    """Enumerate all eligible regular files without writing to the source tree."""

    supported_extensions = _SUPPORTED_EXTENSIONS

    def __init__(self, config: KnowledgeConfig):
        if not isinstance(config, KnowledgeConfig):
            raise TypeError("SourceScanner requires a KnowledgeConfig")
        self.config = config

    def discover(self) -> tuple[DiscoveredResource, ...]:
        root = self.config.source_root
        discovered: list[DiscoveredResource] = []
        visited_directories: set[str] = set()

        def walk(directory: Path, display_parts: tuple[str, ...]) -> None:
            try:
                resolved_directory = directory.resolve(strict=True)
            except (OSError, RuntimeError):
                return
            if not _within(resolved_directory, root):
                return
            # Junctions/symlinked directories can create cycles.  Realpath is
            # used only for the visited guard; display paths retain the alias.
            directory_key = os.path.normcase(os.path.normpath(str(resolved_directory)))
            if directory_key in visited_directories:
                return
            visited_directories.add(directory_key)
            try:
                with os.scandir(directory) as iterator:
                    entries = sorted(
                        iterator,
                        key=lambda item: (item.name.casefold(), item.name),
                    )
            except OSError:
                return
            try:
                for entry in entries:
                    name = entry.name
                    lowered = name.casefold()
                    if lowered in _METADATA_FILES:
                        continue
                    display_path = "/".join((*display_parts, name))
                    entry_path = Path(entry.path)
                    reparsed = _is_reparse(entry)
                    try:
                        entry_info = entry.stat(follow_symlinks=not reparsed)
                    except OSError:
                        # An inaccessible entry is not a regular resource we
                        # can safely hash.  Continue scanning its siblings.
                        continue

                    if stat.S_ISDIR(entry_info.st_mode):
                        if lowered in _METADATA_DIRECTORIES:
                            continue
                        target = entry_path.resolve(strict=False)
                        if reparsed and not _within(target, root):
                            continue
                        walk(entry_path, (*display_parts, name))
                        continue

                    # A symlink/reparse target is admitted only when its
                    # resolved destination remains below the source root.
                    target = entry_path.resolve(strict=False) if reparsed else entry_path
                    if reparsed and not _within(target, root):
                        continue
                    if not stat.S_ISREG(entry_info.st_mode) and not entry_path.is_file():
                        continue
                    extension = entry_path.suffix.casefold()
                    if extension not in _SUPPORTED_EXTENSIONS:
                        discovered.append(
                            DiscoveredResource(
                                resource_id=resource_id_for(display_path),
                                relative_path=display_path,
                                display_path=display_path,
                                path=entry_path,
                                state=IngestionState.UNSUPPORTED,
                                size_bytes=int(entry_info.st_size),
                                modified_ns=int(entry_info.st_mtime_ns),
                                format=extension[1:] or None,
                            )
                        )
                        continue

                    try:
                        # Hash the resolved destination for an internal link;
                        # the alias path remains the provenance identity.
                        source_hash, size = sha256_file(target if reparsed else entry_path)
                        state = IngestionState.HASHED
                    except (SourceChangedError, ValueError):
                        source_hash = None
                        size = int(entry_info.st_size)
                        state = IngestionState.SOURCE_CHANGED
                    except OSError:
                        source_hash = None
                        size = int(entry_info.st_size)
                        state = IngestionState.SOURCE_CHANGED
                    discovered.append(
                        DiscoveredResource(
                            resource_id=resource_id_for(display_path),
                            relative_path=display_path,
                            display_path=display_path,
                            path=entry_path,
                            state=state,
                            source_hash=source_hash,
                            size_bytes=size,
                            modified_ns=int(entry_info.st_mtime_ns),
                            format=extension[1:],
                        )
                    )
            finally:
                # The scanner keeps only materialized DirEntry metadata; the
                # scandir iterator itself is already closed above.
                pass

        walk(root, ())
        discovered.sort(key=lambda item: (item.relative_path.casefold(), item.display_path or ""))
        return tuple(discovered)


__all__ = ["SourceScanner"]
