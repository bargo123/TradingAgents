"""Contract tests for the offline, structure-preserving parser adapters."""

from __future__ import annotations

import hashlib
import json
import sys
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from tradingagents.knowledge.config import KnowledgeConfig
from tradingagents.knowledge.identity import resource_id_for
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
    formula_artifacts_path: str | None = None
    formula_model_id: str | None = None
    formula_model_version: str | None = None
    formula_artifact_hash: str | None = None


def provision_docling_artifacts(artifacts: Path, *, version: str = "fake-docling-v1") -> Path:
    model = artifacts / "models" / "layout.bin"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"local-docling-artifact")
    (artifacts / "docling-artifacts.json").write_text(
        json.dumps(
            {
                "schema_version": "docling-artifacts-v1",
                "docling_version": version,
                "artifacts": [
                    {
                        "path": "models/layout.bin",
                        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return artifacts


def provision_formula_artifacts(
    artifacts: Path,
    *,
    model_id: str,
    model_version: str,
) -> tuple[Path, str]:
    model = artifacts / "formula.bin"
    artifacts.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"local-formula-artifact")
    artifact_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    (artifacts / "formula-artifacts.json").write_text(
        json.dumps(
            {
                "schema_version": "formula-artifacts-v1",
                "model_id": model_id,
                "model_version": model_version,
                "artifact_hash": artifact_hash,
                "artifacts": [{"path": "formula.bin", "sha256": artifact_hash}],
            }
        ),
        encoding="utf-8",
    )
    return artifacts, artifact_hash


def write_minimal_epub(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>',
        )
        archive.writestr(
            "OPS/book.opf",
            '<package><metadata><title>Fixture EPUB</title></metadata><manifest>'
            '<item id="chapter" href="chapter.xhtml"/></manifest><spine>'
            '<itemref idref="chapter"/></spine></package>',
        )
        archive.writestr(
            "OPS/chapter.xhtml",
            '<html><body><p id="first">Spine fallback paragraph.</p></body></html>',
        )


def write_two_anchor_epub(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>',
        )
        archive.writestr(
            "OPS/book.opf",
            '<package><metadata><title>Two anchors</title></metadata><manifest>'
            '<item id="chapter" href="chapter.xhtml"/></manifest><spine>'
            '<itemref idref="chapter"/></spine></package>',
        )
        archive.writestr(
            "OPS/chapter.xhtml",
            '<html><body><p id="first">Verified first paragraph.</p>'
            '<p id="second">Verified second paragraph.</p></body></html>',
        )


def make_config(tmp_path: Path, **overrides: object) -> KnowledgeConfig:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    artifacts = tmp_path / "docling-artifacts"
    provision_docling_artifacts(artifacts)
    return KnowledgeConfig(
        source_root=source,
        artifact_root=tmp_path / "artifacts",
        docling_artifacts_path=artifacts,
        **overrides,
    )


def test_docling_export_mapping_preserves_ordered_structure_and_page_inventory(tmp_path):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    docling_export = {
        "origin": {"filename": "actual.pdf", "mimetype": "application/pdf"},
        "pages": {
            "1": {"image": {"uri": "page-1.png"}},
            "2": {"image": None},
        },
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "section_header",
                "text": "Results",
                "prov": [{"page_no": 1}],
            },
            {
                "self_ref": "#/texts/1",
                "label": "text",
                "text": "Order flow predicts short-horizon returns.",
                "prov": [{"page_no": 1}],
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "caption": "Prediction accuracy",
                "prov": [{"page_no": 2}],
                "data": {
                    "table_cells": [
                        {"text": "Horizon", "row": 0, "col": 0},
                        {"text": "Accuracy", "row": 0, "col": 1},
                        {"text": "1", "row": 1, "col": 0},
                        {"text": "0.71", "row": 1, "col": 1},
                    ]
                },
            }
        ],
        "formulas": [
            {
                "self_ref": "#/formulas/0",
                "text": "p_t = m_t + lambda I_t",
                "latex": "p_t = m_t + \\lambda I_t",
                "prov": [{"page_no": 2}],
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "caption": "Queue imbalance figure",
                "prov": [{"page_no": 2}],
            }
        ],
        "references": [
            {
                "self_ref": "#/references/0",
                "text": "Kyle (1985)",
                "identifier": "doi:fixture",
                "prov": [{"page_no": 2}],
            }
        ],
        "body": {
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/formulas/0"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/pictures/0"},
                {"$ref": "#/references/0"},
            ]
        },
    }
    parser = DoclingDocumentParser(
        make_config(tmp_path),
        converter_factory=lambda _options, _format: FakePipeline(docling_export),
        pdf_options_factory=FakeOptions,
    )

    document = parser.parse(fixture_resource("actual.pdf", tmp_path), tmp_path / "staging")

    assert [block.content_type.value for block in document.blocks] == [
        "PROSE",
        "PROSE",
        "EQUATION",
        "TABLE",
        "FIGURE_CAPTION",
        "REFERENCE",
    ]
    assert [block.reading_order for block in document.blocks] == [0, 1, 2, 3, 4, 5]
    assert document.blocks[1].section_path == ("Results",)
    assert document.blocks[2].equation.latex == r"p_t = m_t + \lambda I_t"
    assert document.blocks[3].table.headers == ("Horizon", "Accuracy")
    assert document.blocks[3].table.cells == (("1", "0.71"),)
    assert document.blocks[4].figure_metadata.caption == "Queue imbalance figure"
    assert document.blocks[5].reference_metadata.identifier == "doi:fixture"
    assert document.parser_provenance["page_text_chars"] == (45, 85)
    assert document.parser_provenance["image_pages"] == (True, False)


def test_docling_refitems_resolve_floating_captions_notes_and_nested_body_order(tmp_path):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    docling_export = {
        "texts": [
            {"self_ref": "#/texts/0", "text": "Table caption", "prov": [{"page_no": 1}]},
            {"self_ref": "#/texts/1", "text": "Figure caption", "prov": [{"page_no": 1}]},
            {"self_ref": "#/texts/2", "text": "Related citation", "prov": [{"page_no": 1}]},
            {"self_ref": "#/texts/3", "text": "Footnote text", "prov": [{"page_no": 1}]},
            {"self_ref": "#/texts/4", "text": "Nested list text", "prov": [{"page_no": 1}]},
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "captions": [{"$ref": "#/texts/0"}],
                "references": [{"$ref": "#/texts/2"}],
                "footnotes": [{"$ref": "#/texts/3"}],
                "data": {"table_cells": [{"text": "H", "row": 0, "col": 0}]},
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "captions": [{"$ref": "#/texts/1"}],
                "references": [{"$ref": "#/texts/2"}],
                "footnotes": [{"$ref": "#/texts/3"}],
            }
        ],
        "list_items": [
            {"self_ref": "#/list_items/0", "text": "Nested list text", "prov": [{"page_no": 1}]}
        ],
        "groups": [
            {
                "self_ref": "#/groups/0",
                "children": [
                    {"$ref": "#/tables/0"},
                    {"$ref": "#/groups/1"},
                ],
            },
            {
                "self_ref": "#/groups/1",
                "children": [
                    {"$ref": "#/pictures/0"},
                    {"$ref": "#/list_items/0"},
                ],
            },
        ],
        "body": {"children": [{"$ref": "#/groups/0"}]},
    }
    parser = DoclingDocumentParser(
        make_config(tmp_path),
        converter_factory=lambda _options, _format: FakePipeline(docling_export),
        pdf_options_factory=FakeOptions,
    )

    document = parser.parse(fixture_resource("refs.pdf", tmp_path), tmp_path / "staging")

    assert [block.content_type.value for block in document.blocks] == [
        "TABLE",
        "FIGURE_CAPTION",
        "LIST",
    ]
    table, figure, _ = document.blocks
    assert table.table.caption == "Table caption"
    assert table.table.notes == ("Footnote text",)
    assert table.table.extra["related_references"] == ("Related citation",)
    assert figure.figure_metadata.caption == "Figure caption"
    assert figure.figure_metadata.extra["related_references"] == ("Related citation",)
    assert figure.figure_metadata.extra["footnotes"] == ("Footnote text",)


def test_incomplete_docling_artifact_manifest_fails_before_converter_runs(tmp_path):
    from tradingagents.knowledge.docling_parser import (
        DoclingArtifactsUnavailable,
        DoclingDocumentParser,
    )

    config = make_config(tmp_path)
    (config.docling_artifacts_path / "models" / "layout.bin").unlink()
    calls: list[object] = []
    parser = DoclingDocumentParser(
        config,
        converter_factory=lambda _options, _format: calls.append("converter"),
        pdf_options_factory=FakeOptions,
    )

    with pytest.raises(DoclingArtifactsUnavailable):
        parser.parse(fixture_resource("paper.pdf", tmp_path), tmp_path / "staging")

    assert calls == []


def test_default_adapter_binds_offline_local_runtime_for_pdf_and_epub(tmp_path, monkeypatch):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    calls: list[dict[str, object]] = []

    class InstalledPdfOptions:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class InstalledPdfFormatOption:
        def __init__(self, *, pipeline_options: object) -> None:
            self.pipeline_options = pipeline_options

    class InstalledConverter:
        def __init__(self, **kwargs: object) -> None:
            calls.append(dict(kwargs))

        def convert(self, path: Path) -> dict[str, object]:
            return {"metadata": {"format": path.suffix.lstrip(".")}, "blocks": []}

    docling = types.ModuleType("docling")
    docling.__path__ = []
    docling.__version__ = "fake-docling-v1"
    datamodel = types.ModuleType("docling.datamodel")
    datamodel.__path__ = []
    base_models = types.ModuleType("docling.datamodel.base_models")
    base_models.InputFormat = types.SimpleNamespace(PDF="PDF", EPUB="EPUB")
    pipeline_options = types.ModuleType("docling.datamodel.pipeline_options")
    pipeline_options.PdfPipelineOptions = InstalledPdfOptions
    converter_module = types.ModuleType("docling.document_converter")
    converter_module.DocumentConverter = InstalledConverter
    converter_module.PdfFormatOption = InstalledPdfFormatOption
    monkeypatch.setitem(sys.modules, "docling", docling)
    monkeypatch.setitem(sys.modules, "docling.datamodel", datamodel)
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", base_models)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", pipeline_options)
    monkeypatch.setitem(sys.modules, "docling.document_converter", converter_module)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network"))
    monkeypatch.setattr("socket.create_connection", lambda *args, **kwargs: pytest.fail("network"))
    parser = DoclingDocumentParser(make_config(tmp_path))

    parser.parse(fixture_resource("paper.pdf", tmp_path), tmp_path / "staging")
    epub = fixture_resource("chapter.epub", tmp_path)
    write_minimal_epub(epub.path)
    parser.parse(epub, tmp_path / "staging")

    assert len(calls) == 2
    assert all(call["offline"] is True for call in calls)
    assert all(call["allow_download"] is False for call in calls)
    assert all(call["artifacts_path"] == str(parser.config.docling_artifacts_path) for call in calls)
    assert calls[0]["format_options"]["PDF"].pipeline_options.do_ocr is False


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


def test_parser_tolerates_empty_docling_table_headers_without_losing_cells(tmp_path):
    from tradingagents.knowledge.parser import normalize_parsed_document

    resource = fixture_resource("paper.pdf", tmp_path)
    payload = {
        "metadata": {"title": "Docling table fixture"},
        "blocks": [
            {
                "block_id": "table-with-empty-header",
                "content_type": "TABLE",
                "text": "Bid Ask 1.10 1.11",
                "page_start": 2,
                "table": {
                    "caption": "Quotes",
                    "headers": ["", "Ask"],
                    "cells": [["1.10", "1.11"]],
                },
            }
        ],
    }

    document = normalize_parsed_document(
        payload,
        resource,
        parser_id="docling",
        parser_version="2.126.0",
        parser_config_hash="cfg",
    )

    table = document.blocks[0].table
    assert table is not None
    assert table.headers == ("Ask",)
    assert table.cells == (("1.10", "1.11"),)
    assert table.caption == "Quotes"
    assert document.blocks[0].page_start == 2


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
    docling_artifacts = provision_docling_artifacts(tmp_path / "docling-artifacts")
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


def test_requested_formula_enrichment_requires_complete_local_identity(tmp_path):
    from tradingagents.knowledge.docling_parser import (
        DoclingArtifactsUnavailable,
        DoclingDocumentParser,
    )

    formula = tmp_path / "formula-artifacts"
    formula.mkdir()
    config = make_config(
        tmp_path,
        formula_enrichment_enabled=True,
        formula_artifacts_path=formula,
    )

    with pytest.raises(DoclingArtifactsUnavailable):
        DoclingDocumentParser(config, pdf_options_factory=FakeOptions).build_pdf_pipeline_options()


def test_formula_pipeline_uses_local_identity_and_disabled_native_equation_is_preserved(tmp_path):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    formula, artifact_hash = provision_formula_artifacts(
        tmp_path / "formula-artifacts", model_id="local-formula", model_version="v1"
    )
    config = make_config(
        tmp_path,
        formula_enrichment_enabled=True,
        formula_artifacts_path=formula,
        formula_model_id="local-formula",
        formula_model_version="v1",
        formula_artifact_hash=artifact_hash,
    )
    options = DoclingDocumentParser(config, pdf_options_factory=FakeOptions).build_pdf_pipeline_options()

    assert options.formula_artifacts_path == str(formula)
    assert options.formula_model_id == "local-formula"
    assert options.formula_model_version == "v1"
    assert options.formula_artifact_hash == artifact_hash

    parser = DoclingDocumentParser(
        make_config(tmp_path),
        converter_factory=lambda _options, _format: FakePipeline(
            {
                "metadata": {"format": "pdf"},
                "blocks": [
                    {
                        "block_id": "native-eq",
                        "content_type": "EQUATION",
                        "equation": {"parser_native": "x=y"},
                    }
                ],
            }
        ),
        pdf_options_factory=FakeOptions,
    )
    document = parser.parse(fixture_resource("paper.pdf", tmp_path), tmp_path / "staging")

    assert document.blocks[0].equation.parser_native == "x=y"
    assert document.blocks[0].equation.latex is None
    assert document.parser_provenance["formula_enrichment_status"] == "DISABLED_NATIVE_PRESERVED"


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
                        "epub_spine_item": "OPS/chapter.xhtml",
                        "anchor": "first",
                    }
                ],
            }
        ),
        pdf_options_factory=FakeOptions,
    )
    resource = fixture_resource("chapter.epub", tmp_path)
    write_minimal_epub(resource.path)
    document = parser.parse(resource, tmp_path / "staging")

    assert document.parser_provenance["adapter_path"] == "NATIVE_DOCLING_EPUB"
    assert document.blocks[0].epub_spine_item and document.blocks[0].anchor


def test_epub_opf_fallback_is_used_only_when_native_provenance_fails(tmp_path):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    resource = fixture_resource("chapter.epub", tmp_path)
    write_minimal_epub(resource.path)
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


@pytest.mark.parametrize(
    ("spine_item", "anchor"),
    (("missing.xhtml", "first"), ("OPS/chapter.xhtml", "missing-anchor")),
)
def test_invalid_native_epub_location_uses_verified_opf_block(
    tmp_path, spine_item, anchor
):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    resource = fixture_resource("chapter.epub", tmp_path)
    write_minimal_epub(resource.path)
    parser = DoclingDocumentParser(
        make_config(tmp_path),
        converter_factory=lambda _options, _format: FakePipeline(
            {
                "metadata": {"format": "epub"},
                "blocks": [
                    {
                        "block_id": "native-table",
                        "content_type": "TABLE",
                        "text": "Horizon Accuracy",
                        "epub_spine_item": spine_item,
                        "anchor": anchor,
                        "table": {
                            "caption": "Native retained table",
                            "headers": ["Horizon", "Accuracy"],
                            "cells": [["1", "0.71"]],
                        },
                    }
                ],
            }
        ),
        pdf_options_factory=FakeOptions,
    )

    document = parser.parse(resource, tmp_path / "staging")

    assert document.parser_provenance["adapter_path"] == "OPF_SPINE_FALLBACK"
    assert document.blocks[0].content_type.value == "PROSE"
    assert document.blocks[0].text == "Spine fallback paragraph."
    assert document.blocks[0].epub_spine_item == "OPS/chapter.xhtml"
    assert document.blocks[0].anchor == "first"


def test_epub_fallback_never_repeats_anchor_for_unmatched_native_structured_blocks(tmp_path):
    from tradingagents.knowledge.docling_parser import DoclingDocumentParser

    resource = fixture_resource("chapter.epub", tmp_path)
    write_two_anchor_epub(resource.path)
    parser = DoclingDocumentParser(
        make_config(tmp_path),
        converter_factory=lambda _options, _format: FakePipeline(
            {
                "metadata": {"format": "epub"},
                "blocks": [
                    {
                        "block_id": "native-heading",
                        "content_type": "PROSE",
                        "text": "Unmatched heading",
                    },
                    {
                        "block_id": "native-table",
                        "content_type": "TABLE",
                        "text": "Unmatched table",
                        "table": {"headers": ["H"], "cells": [["1"]]},
                    },
                    {
                        "block_id": "native-figure",
                        "content_type": "FIGURE_CAPTION",
                        "text": "Unmatched figure",
                    },
                ],
            }
        ),
        pdf_options_factory=FakeOptions,
    )

    document = parser.parse(resource, tmp_path / "staging")

    assert document.parser_provenance["adapter_path"] == "OPF_SPINE_FALLBACK"
    assert [block.text for block in document.blocks] == [
        "Verified first paragraph.",
        "Verified second paragraph.",
    ]
    assert [block.anchor for block in document.blocks] == ["first", "second"]
    assert all(block.content_type.value == "PROSE" for block in document.blocks)


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
