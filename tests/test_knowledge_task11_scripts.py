"""Task 11 command-boundary regressions for explicit setup and offline smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    path = _SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_provision_requires_explicit_network_opt_in(tmp_path: Path) -> None:
    provision = _load_script("provision_knowledge_models.py")

    result = provision.main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--docling-artifacts-path",
            str(tmp_path / "docling"),
            "--embedding-model-path",
            str(tmp_path / "embedding"),
        ]
    )

    assert result != 0
    assert not (tmp_path / "artifacts").exists()


def test_smoke_fails_before_parsing_when_local_artifacts_are_missing(tmp_path: Path) -> None:
    smoke = _load_script("knowledge_phase7_smoke.py")
    source = tmp_path / "source"
    source.mkdir()
    (source / "book.pdf").write_bytes(b"%PDF-1.4\n")

    result = smoke.main(
        [
            "--source-root",
            str(source),
            "--artifact-root",
            str(tmp_path / "smoke-artifacts"),
            "--docling-artifacts-path",
            str(tmp_path / "missing-docling"),
            "--embedding-model-path",
            str(tmp_path / "missing-embedding"),
            "--offline",
        ]
    )

    assert result != 0
    assert not (tmp_path / "smoke-artifacts").exists()


def test_smoke_reports_only_public_knowledge_hit_provenance_fields() -> None:
    smoke = _load_script("knowledge_phase7_smoke.py")
    from tradingagents.knowledge.models import KnowledgeHit

    fields = smoke._hit_fields(
        KnowledgeHit(
            chunk_id="chunk",
            document_id="document",
            source_hash="sha256:source",
            source_filename="book.pdf",
            source_relative_path="book.pdf",
            page=7,
            epub_spine_item="chapter.xhtml",
            anchor="section",
            parser_version="docling-2",
            chunker_version="structure-v2",
            index_version="index-v1",
        )
    )

    assert fields["page"] == 7
    assert fields["epub_spine_item"] == "chapter.xhtml"
    assert fields["anchor"] == "section"
    assert fields["index_version"] == "index-v1"
