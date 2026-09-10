"""Content and path identity tests for knowledge resources."""

from pathlib import Path

import pytest

from tradingagents.knowledge.identity import (
    canonical_relative_path,
    document_id_for,
    resource_id_for,
    sha256_file,
)


def test_identity_is_stable_and_exact_duplicates_share_document_id():
    assert canonical_relative_path(Path("root/A/B.pdf"), Path("root")) == "a/b.pdf"
    assert resource_id_for("a/b.pdf") == resource_id_for("A\\B.PDF")
    assert document_id_for("ab" * 32) == document_id_for("ab" * 32)
    assert document_id_for("ab" * 32) != document_id_for("cd" * 32)
    assert resource_id_for("a/b.pdf").startswith("res_")
    assert document_id_for("ab" * 32).startswith("doc_")


def test_canonical_relative_path_rejects_paths_outside_source_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="outside"):
        canonical_relative_path(tmp_path / "other.pdf", source)


def test_sha256_file_returns_content_hash_and_byte_count(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"fixture-bytes")
    source_hash, size = sha256_file(path)
    assert source_hash == "c16a40a4584e5bccc84b45172fcdfa922f59ff1edebf3adba7b8266ea04eb39a"
    assert size == len(b"fixture-bytes")
