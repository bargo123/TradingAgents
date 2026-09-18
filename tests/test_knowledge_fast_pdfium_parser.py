"""Fast offline PDF parser and resumable parse-cache contracts."""

from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from tradingagents.knowledge.fast_pdfium_parser import CachedPdfiumParser, PdfiumParser
from tradingagents.knowledge.identity import resource_id_for
from tradingagents.knowledge.models import (
    DiscoveredResource,
    DocumentMetadata,
    IngestionState,
    ParsedDocument,
)
from tradingagents.knowledge.scanned import ScannedDetector


def _resource(tmp_path: Path, name: str = "book.pdf") -> DiscoveredResource:
    path = tmp_path / name
    path.write_bytes(b"fixture-pdf")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return DiscoveredResource(
        resource_id=resource_id_for(name),
        relative_path=name,
        display_path=name,
        path=path,
        state=IngestionState.HASHED,
        source_hash=source_hash,
        size_bytes=path.stat().st_size,
        modified_ns=path.stat().st_mtime_ns,
        format="pdf",
    )


@dataclass
class _FakeTextPage:
    text: str

    def get_text_bounded(self) -> str:
        return self.text


@dataclass
class _FakePage:
    text: str

    def get_textpage(self) -> _FakeTextPage:
        return _FakeTextPage(self.text)


class _FakePdfDocument:
    pages = (_FakePage("page one evidence"), _FakePage(""), _FakePage("page three evidence"))

    def __init__(self, _: str) -> None:
        self._pages = self.pages

    def __len__(self) -> int:
        return len(self._pages)

    def __getitem__(self, index: int) -> _FakePage:
        return self._pages[index]


def test_pdfium_parser_preserves_page_provenance_and_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdfium2", types.SimpleNamespace(PdfDocument=_FakePdfDocument))
    resource = _resource(tmp_path)
    parser = PdfiumParser()

    first = parser.parse(resource, tmp_path / "staging")
    second = parser.parse(resource, tmp_path / "staging")

    assert first.to_json() == second.to_json()
    assert first.parser_id == "pypdfium2"
    assert [block.page_start for block in first.blocks] == [1, 3]
    assert first.parser_provenance["page_text_chars"] == (17, 0, 19)
    assert first.parser_provenance["image_pages"] == (False, True, False)
    assert all(block.text for block in first.blocks)


def test_image_only_document_is_needs_ocr_without_generated_text(tmp_path, monkeypatch):
    class ImageOnlyDocument(_FakePdfDocument):
        pages = (_FakePage(""), _FakePage(""))

    monkeypatch.setitem(sys.modules, "pypdfium2", types.SimpleNamespace(PdfDocument=ImageOnlyDocument))
    document = PdfiumParser().parse(_resource(tmp_path), tmp_path / "staging")

    decision = ScannedDetector().classify(document)
    assert decision.state is IngestionState.NEEDS_OCR
    assert document.text_blocks == ()
    assert document.parser_provenance["ocr"] is False


def test_cached_pdfium_parser_persists_and_reuses_validated_document(tmp_path):
    resource = _resource(tmp_path)
    calls = 0

    class StubParser:
        def parse(self, resource, staging_dir):
            nonlocal calls
            calls += 1
            return ParsedDocument(
                document_id="doc_" + resource.source_hash,
                source_hash=resource.source_hash,
                metadata=DocumentMetadata(
                    format="pdf",
                    source_filename=resource.path.name,
                    source_relative_path=resource.relative_path,
                    source_hash=resource.source_hash,
                ),
                parser_id="stub",
                parser_version="stub-v1",
                parser_config_hash="stub-config",
            )

    parser = CachedPdfiumParser(tmp_path / "parse-cache", parser=StubParser())
    first = parser.parse(resource, tmp_path / "staging")
    second = parser.parse(resource, tmp_path / "staging")

    assert calls == 1
    assert first.to_json() == second.to_json()
    assert (tmp_path / "parse-cache" / f"{first.document_id}.json").is_file()
