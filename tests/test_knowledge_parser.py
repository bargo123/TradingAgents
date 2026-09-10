"""Contract tests for the offline, structure-preserving parser adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import zipfile

import pytest

from tradingagents.knowledge.config import KnowledgeConfig
from tradingagents.knowledge.identity import document_id_for, resource_id_for
from tradingagents.knowledge.models import DiscoveredResource, IngestionState


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "knowledge" / "structured_ir.json"


def fixture_resource(name: str, tmp_path: Path) -> DiscoveredResource:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    path = source / name
    path.write_bytes(b"fixture-source")
    source_hash = "ab" * 32
    return DiscoveredResource(
        resource_id=resource_id_for(name),
        relative_path=name,
        display_path=name,
        path=path,
        source_hash=source_hash,
        state=IngestionState.HASHED,
        format=path.suffix.lstrip("."),
    )


def structured_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@dataclass
class FakePipeline:
    payload: dict[str, object]

    def convert(self, _: Path) -> dict[str, object]:
        return self.payload


@dataclass
class FakeOptions:
    do_ocr: bool
    do_formula_enrichment: bool
    artifacts_path: str | None = None


def make_config(tmp_path: Path, **overrides: object) -> KnowledgeConfig:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    artifacts = tmp_path / "docling-artifacts"
    artifacts.mkdir(exist_ok=True)
    return KnowledgeConfig(
        source_root=source,
        artifact_root=tmp_path / "artifacts",
        docling_artifacts_path=artifacts,
        **overrides,
    )


def test_parser_preserves_equation_table_caption_and_locations(tmp_path):
    from tradingagents.knowledge.parser import normalize_parsed_document

    resource = fixture_resource("paper.pdf", tmp_path)
    document = normalize_parsed_document(
        structured_fixture(), resource, parser_id="fake", parser_version="1", parser_config_hash="cfg"
    )

    assert document.title == "Microstructure Fixture"
    assert document.blocks[1].content_type.value == "EQUATION"
    assert document.blocks[1].equation.latex == r"p_t = m_t + \lambda I_t"
    assert document.blocks[2].content_type.value == "TABLE"
    assert document.blocks[2].table.headers == ("Horizon", "Accuracy")
    assert document.blocks[2].table.caption == "Prediction accuracy"
    assert document.blocks[2].page_start == 4
    assert document.blocks[2].section_path == ("Results", "Prediction")


def test_docling_pdf_options_disable_ocr_and_image_only_text_is_not_ocr_generated(tmp_path):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser
    from tradingagents.knowledge.scanned import ScannedDetector

    options_seen: list[FakeOptions] = []

    def options_factory(**kwargs: object) -> FakeOptions:
        options = FakeOptions(**kwargs)
        options_seen.append(options)
        return options

    parser = DoclingDocumentParser(
        make_config(tmp_path),
        converter_factory=lambda _options, _format: FakePipeline(
            {"metadata": {"format": "pdf"}, "blocks": [], "page_text_chars": [0, 0], "image_pages": [True, True]}
        ),
        pdf_options_factory=options_factory,
    )
    document = parser.parse(fixture_resource("scanned.pdf", tmp_path), tmp_path / "staging")

    assert options_seen[0].do_ocr is False
    assert not document.text_blocks
    assert ScannedDetector().classify(document).state is IngestionState.NEEDS_OCR


@pytest.mark.integration
def test_installed_docling_pdf_pipeline_options_are_explicitly_ocr_disabled(tmp_path):
    pytest.importorskip("docling")
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    parser = DoclingDocumentParser(make_config(tmp_path))
    options = parser.build_pdf_pipeline_options()
    assert options.do_ocr is False
    assert parser.config.docling_offline is True


def test_missing_docling_artifacts_fails_offline_without_network(tmp_path, monkeypatch):
    from tradingagents.knowledge.docling_parser import (
        DoclingArtifactsUnavailable,
        DoclingDocumentParser,
    )

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network"))
    source = tmp_path / "source"
    source.mkdir()
    config = KnowledgeConfig(
        source_root=source,
        artifact_root=tmp_path / "artifacts",
        docling_artifacts_path=tmp_path / "missing-docling",
    )

    with pytest.raises(DoclingArtifactsUnavailable):
        DoclingDocumentParser(config).parse(fixture_resource("paper.pdf", tmp_path), tmp_path / "staging")


def test_formula_enrichment_is_local_optional_and_preserves_native_equation(tmp_path):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    parser = DoclingDocumentParser(
        make_config(tmp_path),
        converter_factory=lambda _options, _format: FakePipeline(
            {
                "metadata": {"format": "pdf"},
                "blocks": [
                    {
                        "block_id": "eq-1",
                        "content_type": "EQUATION",
                        "text": "parser-native-equation",
                        "equation": {"parser_native": "parser-native-equation"},
                    }
                ],
                "formula_enrichment_status": "UNAVAILABLE_NATIVE_PRESERVED",
            }
        ),
        pdf_options_factory=FakeOptions,
    )
    document = parser.parse(fixture_resource("paper.pdf", tmp_path), tmp_path / "staging")

    assert document.blocks[0].equation.parser_native == "parser-native-equation"
    assert document.parser_provenance["formula_enrichment_status"] == "UNAVAILABLE_NATIVE_PRESERVED"


def test_requested_formula_enrichment_requires_local_artifact_without_network(tmp_path, monkeypatch):
    from tradingagents.knowledge.docling_parser import (
        DoclingArtifactsUnavailable,
        DoclingDocumentParser,
    )

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network"))
    source = tmp_path / "source"
    source.mkdir()
    docling_artifacts = tmp_path / "docling-artifacts"
    docling_artifacts.mkdir()
    formula = tmp_path / "formula-artifacts"
    formula.mkdir()
    config = KnowledgeConfig(
        source_root=source,
        artifact_root=tmp_path / "artifacts",
        docling_artifacts_path=docling_artifacts,
        formula_enrichment_enabled=True,
        formula_artifacts_path=formula,
    )
    formula.rmdir()

    with pytest.raises(DoclingArtifactsUnavailable):
        DoclingDocumentParser(config).build_pdf_pipeline_options()


def test_native_epub_path_is_preferred_and_fallback_records_spine_anchor(tmp_path):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    parser = DoclingDocumentParser(
        make_config(tmp_path),
        converter_factory=lambda _options, _format: FakePipeline(
            {
                "metadata": {"format": "epub"},
                "blocks": [
                    {
                        "block_id": "native-1",
                        "content_type": "PROSE",
                        "text": "Native EPUB paragraph.",
                        "epub_spine_item": "chapter-1.xhtml",
                        "anchor": "p-1",
                    }
                ],
            }
        ),
        pdf_options_factory=FakeOptions,
    )
    document = parser.parse(fixture_resource("chapter.epub", tmp_path), tmp_path / "staging")

    assert document.parser_provenance["adapter_path"] == "NATIVE_DOCLING_EPUB"
    assert document.blocks[0].epub_spine_item and document.blocks[0].anchor


def test_epub_opf_fallback_is_used_only_when_native_provenance_fails(tmp_path):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    resource = fixture_resource("chapter.epub", tmp_path)
    with zipfile.ZipFile(resource.path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>',
        )
        archive.writestr(
            "OPS/book.opf",
            '<package><metadata><title>Fallback EPUB</title></metadata><manifest>'
            '<item id="chapter" href="chapter.xhtml"/></manifest><spine>'
            '<itemref idref="chapter"/></spine></package>',
        )
        archive.writestr(
            "OPS/chapter.xhtml",
            '<html><body><p id="first">Spine fallback paragraph.</p></body></html>',
        )
    parser = DoclingDocumentParser(
        make_config(tmp_path),
        converter_factory=lambda _options, _format: FakePipeline(
            {
                "metadata": {"format": "epub"},
                "blocks": [
                    {
                        "block_id": "native-without-location",
                        "content_type": "PROSE",
                        "text": "Native output missing locations.",
                    }
                ],
            }
        ),
        pdf_options_factory=FakeOptions,
    )

    document = parser.parse(resource, tmp_path / "staging")

    assert document.parser_provenance["adapter_path"] == "OPF_SPINE_FALLBACK"
    assert document.blocks[0].epub_spine_item == "OPS/chapter.xhtml"
    assert document.blocks[0].anchor == "first"


def test_converter_failure_keeps_the_resource_identity_for_isolated_diagnostics(tmp_path):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser
    from tradingagents.knowledge.parser import ParseFailure

    class BrokenPipeline:
        def convert(self, _: Path) -> dict[str, object]:
            raise RuntimeError("broken converter")

    parser = DoclingDocumentParser(
        make_config(tmp_path),
        converter_factory=lambda _options, _format: BrokenPipeline(),
        pdf_options_factory=FakeOptions,
    )
    resource = fixture_resource("broken.pdf", tmp_path)

    with pytest.raises(ParseFailure) as caught:
        parser.parse(resource, tmp_path / "staging")

    assert caught.value.resource_id == resource.resource_id


def test_parser_exception_is_parse_failed_not_scanned(tmp_path):
    from tradingagents.knowledge.parser import ParseFailure

    class FailingParser:
        def parse(self, resource: DiscoveredResource, staging_dir: Path):
            raise ParseFailure("broken fixture", resource_id=resource.resource_id)

    with pytest.raises(ParseFailure):
        FailingParser().parse(fixture_resource("broken.pdf", tmp_path), tmp_path / "staging")
