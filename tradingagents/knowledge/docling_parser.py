"""Docling-first, offline PDF/EPUB parsing with provenance-safe fallback."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
import zipfile

from .config import KnowledgeConfig
from .models import ParsedDocument
from .parser import (
    DoclingArtifactsUnavailable,
    DocumentParser,
    ParseFailure,
    ParserDependencyUnavailable,
    normalize_parsed_document,
)


class _XhtmlTextExtractor(HTMLParser):
    """Small deterministic XHTML reader used only after native provenance fails."""

    _TEXT_TAGS = frozenset({"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict[str, str]] = []
        self._tag: str | None = None
        self._anchor: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._TEXT_TAGS:
            self._finish()
            self._tag = tag
            self._anchor = dict(attrs).get("id")
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._tag is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._tag == tag.lower():
            self._finish()

    def close(self) -> None:
        super().close()
        self._finish()

    def _finish(self) -> None:
        if self._tag is not None:
            text = " ".join("".join(self._parts).split())
            if text:
                self.blocks.append({"text": text, "anchor": self._anchor or "top"})
        self._tag = None
        self._anchor = None
        self._parts = []


class DoclingDocumentParser(DocumentParser):
    """Optional-dependency adapter which never enables OCR or auto-downloads."""

    parser_id = "docling"

    def __init__(
        self,
        config: KnowledgeConfig,
        *,
        converter_factory: Callable[..., Any] | None = None,
        pdf_options_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._converter_factory = converter_factory
        self._pdf_options_factory = pdf_options_factory

    def parse(self, resource, staging_dir: Path) -> ParsedDocument:
        suffix = resource.path.suffix.casefold()
        if suffix not in {".pdf", ".epub"}:
            raise ParseFailure("unsupported parser format", resource_id=resource.resource_id)
        self._require_local_artifacts(resource.resource_id)
        staged = self._stage_source(resource.path, staging_dir, resource.resource_id)
        if suffix == ".pdf":
            options = self.build_pdf_pipeline_options(resource.resource_id)
            self._assert_ocr_disabled(options, resource.resource_id)
            payload = self._coerce_payload(
                self._convert(staged, options, "pdf", resource.resource_id), resource.resource_id
            )
            return self._normalized(payload, resource, adapter_path="DOCLING_PDF")

        native_payload = self._coerce_payload(
            self._convert(staged, None, "epub", resource.resource_id), resource.resource_id
        )
        native = self._normalized(native_payload, resource, adapter_path="NATIVE_DOCLING_EPUB")
        if self._has_epub_locations(native):
            return native
        return self._parse_epub_spine_fallback(staged, resource)

    def build_pdf_pipeline_options(self, resource_id: str | None = None) -> Any:
        """Build options explicitly; OCR is checked again immediately before use."""

        self._require_local_artifacts(resource_id)
        factory = self._pdf_options_factory or self._load_pdf_options_factory(resource_id)
        kwargs = {
            "do_ocr": False,
            "do_formula_enrichment": self.config.formula_enrichment_enabled,
            "artifacts_path": str(self.config.docling_artifacts_path),
        }
        try:
            options = factory(**kwargs)
        except TypeError:
            # Some supported Docling releases accept artifacts on a nested
            # model.  They still must receive explicit OCR/formula settings.
            try:
                options = factory(
                    do_ocr=False,
                    do_formula_enrichment=self.config.formula_enrichment_enabled,
                )
            except TypeError as exc:
                raise ParserDependencyUnavailable(
                    "installed Docling PdfPipelineOptions is incompatible",
                    resource_id=resource_id,
                ) from exc
            try:
                setattr(options, "artifacts_path", str(self.config.docling_artifacts_path))
            except (AttributeError, TypeError, ValueError):
                pass
        self._assert_ocr_disabled(options, resource_id)
        return options

    def _require_local_artifacts(self, resource_id: str | None) -> None:
        if not self.config.docling_offline or self.config.docling_do_ocr:
            raise ParseFailure("invalid offline OCR parser configuration", resource_id=resource_id)
        if not self.config.docling_artifacts_path.is_dir():
            raise DoclingArtifactsUnavailable(
                "local Docling artifacts are unavailable; provisioning is required",
                resource_id=resource_id,
            )
        if self.config.formula_enrichment_enabled and not (
            self.config.formula_artifacts_path and self.config.formula_artifacts_path.is_dir()
        ):
            raise DoclingArtifactsUnavailable(
                "requested local formula artifacts are unavailable",
                resource_id=resource_id,
            )

    @staticmethod
    def _stage_source(source: Path, staging_dir: Path, resource_id: str) -> Path:
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise ParseFailure("source cannot be read", resource_id=resource_id) from exc
        staging_dir.mkdir(parents=True, exist_ok=True)
        target = staging_dir / f"{resource_id}{source.suffix.casefold()}"
        try:
            target.write_bytes(data)
        except OSError as exc:
            raise ParseFailure("parser staging write failed", resource_id=resource_id) from exc
        return target

    def _load_pdf_options_factory(self, resource_id: str | None) -> Callable[..., Any]:
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError as exc:
            raise ParserDependencyUnavailable(
                "Docling is not installed locally", resource_id=resource_id
            ) from exc
        return PdfPipelineOptions

    def _convert(self, staged: Path, options: Any, format_name: str, resource_id: str) -> Any:
        converter = self._make_converter(options, format_name, resource_id)
        try:
            return converter.convert(staged)
        except ParseFailure:
            raise
        except Exception as exc:  # optional parser exceptions are implementation-specific
            raise ParseFailure(
                "Docling conversion failed", resource_id=resource_id, stage="PARSE"
            ) from exc

    def _make_converter(self, options: Any, format_name: str, resource_id: str) -> Any:
        if self._converter_factory is not None:
            try:
                return self._converter_factory(options, format_name)
            except TypeError:
                return self._converter_factory(options)
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise ParserDependencyUnavailable(
                "Docling is not installed locally", resource_id=resource_id
            ) from exc
        if format_name == "pdf":
            return DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return DocumentConverter()

    @staticmethod
    def _coerce_payload(value: Any, resource_id: str) -> Mapping[str, object]:
        if isinstance(value, Mapping):
            return value
        candidate = getattr(value, "document", value)
        for name in ("export_to_dict", "model_dump", "dict"):
            method = getattr(candidate, name, None)
            if callable(method):
                result = method()
                if isinstance(result, Mapping):
                    return result
        raise ParseFailure("Docling output cannot be normalized without flattening", resource_id=resource_id)

    @staticmethod
    def _assert_ocr_disabled(options: Any, resource_id: str | None) -> None:
        if getattr(options, "do_ocr", None) is not False:
            raise ParseFailure("Docling PDF options must disable OCR", resource_id=resource_id)

    def _normalized(self, payload: Mapping[str, object], resource, *, adapter_path: str) -> ParsedDocument:
        formula_status = payload.get("formula_enrichment_status")
        if formula_status is None:
            formula_status = "ENABLED_LOCAL" if self.config.formula_enrichment_enabled else "DISABLED"
        provenance = {
            "adapter_path": adapter_path,
            "docling_offline": True,
            "docling_do_ocr": False,
            "docling_artifacts_path": str(self.config.docling_artifacts_path),
            "formula_enrichment_requested": self.config.formula_enrichment_enabled,
            "formula_enrichment_status": str(formula_status),
        }
        if self.config.formula_enrichment_enabled:
            provenance.update(
                {
                    "formula_artifacts_path": str(self.config.formula_artifacts_path),
                    "formula_model_id": self.config.formula_model_id,
                    "formula_model_version": self.config.formula_model_version,
                    "formula_artifact_hash": self.config.formula_artifact_hash,
                }
            )
        return normalize_parsed_document(
            payload,
            resource,
            parser_id=self.parser_id,
            parser_version=self._docling_version(),
            parser_config_hash=self.config.component_fingerprints["parser"],
            parser_provenance=provenance,
        )

    @staticmethod
    def _docling_version() -> str:
        try:
            from importlib.metadata import version

            return version("docling")
        except Exception:
            return "unavailable-fake-adapter"

    @staticmethod
    def _has_epub_locations(document: ParsedDocument) -> bool:
        return bool(document.blocks) and all(
            block.epub_spine_item and block.anchor for block in document.blocks
        )

    def _parse_epub_spine_fallback(self, staged: Path, resource) -> ParsedDocument:
        try:
            payload = self._opf_spine_payload(staged)
        except (OSError, KeyError, ElementTree.ParseError, zipfile.BadZipFile) as exc:
            raise ParseFailure("EPUB OPF/spine fallback failed", resource_id=resource.resource_id) from exc
        return self._normalized(payload, resource, adapter_path="OPF_SPINE_FALLBACK")

    @staticmethod
    def _opf_spine_payload(epub_path: Path) -> dict[str, object]:
        with zipfile.ZipFile(epub_path) as archive:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next(
                item.attrib["full-path"]
                for item in container.iter()
                if item.tag.endswith("rootfile") and item.attrib.get("full-path")
            )
            opf = ElementTree.fromstring(archive.read(rootfile))
            opf_parent = Path(rootfile).parent
            manifest = {
                item.attrib["id"]: item.attrib["href"]
                for item in opf.iter()
                if item.tag.endswith("item") and item.attrib.get("id") and item.attrib.get("href")
            }
            spine_ids = [
                item.attrib["idref"] for item in opf.iter() if item.tag.endswith("itemref") and item.attrib.get("idref")
            ]
            title = next(
                (item.text for item in opf.iter() if item.tag.endswith("title") and item.text), None
            )
            blocks: list[dict[str, object]] = []
            order = 0
            for item_id in spine_ids:
                href = manifest.get(item_id)
                if href is None:
                    continue
                spine_item = (opf_parent / href).as_posix()
                parser = _XhtmlTextExtractor()
                parser.feed(archive.read(spine_item).decode("utf-8", errors="replace"))
                parser.close()
                for position, item in enumerate(parser.blocks):
                    blocks.append(
                        {
                            "block_id": f"{item_id}-{position}",
                            "content_type": "PROSE",
                            "text": item["text"],
                            "reading_order": order,
                            "epub_spine_item": spine_item,
                            "anchor": item["anchor"],
                        }
                    )
                    order += 1
        if not blocks:
            raise ParseFailure("EPUB spine fallback found no structured text")
        return {"metadata": {"title": title, "format": "epub"}, "blocks": blocks}


__all__ = [
    "DoclingArtifactsUnavailable",
    "DoclingDocumentParser",
    "ParserDependencyUnavailable",
]
