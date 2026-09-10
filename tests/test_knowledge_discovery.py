"""Deterministic source scanner tests."""

from pathlib import Path

import pytest

from tradingagents.knowledge.config import KnowledgeConfig
from tradingagents.knowledge.discovery import SourceScanner
from tradingagents.knowledge.models import IngestionState


def test_discovery_exposes_unsupported_regular_file_without_parser_or_embed(tmp_path):
    source = tmp_path / "new books"
    (source / "nested").mkdir(parents=True)
    (source / "book.pdf").write_bytes(b"pdf-fixture")
    (source / "notes.txt").write_text("not a book", encoding="utf-8")
    (source / "desktop.ini").write_text("[.ShellClassInfo]", encoding="utf-8")
    (source / "nested" / "paper.epub").write_bytes(b"epub-fixture")

    resources = SourceScanner(
        KnowledgeConfig(source_root=source, artifact_root=tmp_path / "artifacts")
    ).discover()

    assert [(item.relative_path, item.state) for item in resources] == [
        ("book.pdf", IngestionState.HASHED),
        ("nested/paper.epub", IngestionState.HASHED),
        ("notes.txt", IngestionState.UNSUPPORTED),
    ]
    assert all(
        item.source_hash is None
        for item in resources
        if item.state is IngestionState.UNSUPPORTED
    )


def test_discovery_retains_display_spelling_and_sorts_casefolded_paths(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Zeta.PDF").write_bytes(b"z")
    (source / "alpha.epub").write_bytes(b"a")
    (source / "ALPHA.txt").write_bytes(b"t")
    (source / "Thumbs.db").write_bytes(b"metadata")
    (source / ".git").mkdir()
    (source / ".git" / "ignored.pdf").write_bytes(b"metadata")

    resources = SourceScanner(
        KnowledgeConfig(source_root=source, artifact_root=tmp_path / "artifacts")
    ).discover()

    assert [item.relative_path for item in resources] == ["alpha.epub", "ALPHA.txt", "Zeta.PDF"]
    assert [item.display_path for item in resources] == ["alpha.epub", "ALPHA.txt", "Zeta.PDF"]


def test_discovery_skips_external_symlink_without_reading_outside_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    link = source / "escape.pdf"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        return

    resources = SourceScanner(
        KnowledgeConfig(source_root=source, artifact_root=tmp_path / "artifacts")
    ).discover()
    assert resources == ()


def test_discovery_rejects_missing_source_before_scanning(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="source_root"):
        KnowledgeConfig(source_root=missing, artifact_root=tmp_path / "artifacts")
