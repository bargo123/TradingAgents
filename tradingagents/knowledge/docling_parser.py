"""Docling-first, offline PDF/EPUB parsing with provenance-safe fallback.

Docling is deliberately imported only on the real-adapter path. The adapter
accepts its exported document model, not a test-only flattened IR, and turns
that model into the immutable contracts shared by later knowledge stages.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from html.parser import HTMLParser
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree
import hashlib
import inspect
import json
import os
import zipfile

from .config import KnowledgeConfig
from .models import ContentType, ParsedDocument
from .parser import (
    DoclingArtifactsUnavailable,
    DocumentParser,
    ParseFailure,
    ParserDependencyUnavailable,
    normalize_parsed_document,
)


_DOCLING_MANIFEST = "docling-artifacts.json"
_DOCLING_MANIFEST_SCHEMA = "docling-artifacts-v1"
_FORMULA_MANIFEST = "formula-artifacts.json"
_FORMULA_MANIFEST_SCHEMA = "formula-artifacts-v1"


class _XhtmlTextExtractor(HTMLParser):
    """Small deterministic XHTML reader used only after native provenance fails."""

    _TEXT_TAGS = frozenset({"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict[str, str]] = []
        self.anchors: set[str] = {"top"}
        self._tag: str | None = None
        self._anchor: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.anchors.add(str(attributes["id"]))
        tag = tag.lower()
        if tag in self._TEXT_TAGS:
            self._finish()
            self._tag = tag
            self._anchor = attributes.get("id")
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
    """Optional Docling adapter which is offline-only and OCR-disabled."""

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
        manifest = self._require_local_artifacts(resource.resource_id)
        staged = self._stage_source(resource.path, staging_dir, resource.resource_id)
        if suffix == ".pdf":
            options = self.build_pdf_pipeline_options(resource.resource_id)
            self._assert_ocr_disabled(options, resource.resource_id)
            payload = self._coerce_payload(
                self._convert(staged, options, "pdf", resource.resource_id, manifest),
                resource.resource_id,
            )
            return self._normalized(payload, resource, adapter_path="DOCLING_PDF")

        native_payload = self._coerce_payload(
            self._convert(staged, None, "epub", resource.resource_id, manifest),
            resource.resource_id,
        )
        native = self._normalized(native_payload, resource, adapter_path="NATIVE_DOCLING_EPUB")
        if self._has_epub_locations(native, staged):
            return native
        return self._parse_epub_spine_fallback(staged, resource, native)

    def build_pdf_pipeline_options(self, resource_id: str | None = None) -> Any:
        """Build explicit, local-only PDF options and recheck OCR immediately."""

        self._require_local_artifacts(resource_id)
        factory = self._pdf_options_factory or self._load_pdf_options_factory(resource_id)
        values: dict[str, Any] = {
            "do_ocr": False,
            "do_formula_enrichment": self.config.formula_enrichment_enabled,
            "artifacts_path": str(self.config.docling_artifacts_path),
            "offline": True,
            "allow_download": False,
            "enable_remote_services": False,
            "cache_dir": str(self.config.docling_artifacts_path),
        }
        if self.config.formula_enrichment_enabled:
            values.update(
                {
                    "formula_artifacts_path": str(self.config.formula_artifacts_path),
                    "formula_model_id": self.config.formula_model_id,
                    "formula_model_version": self.config.formula_model_version,
                    "formula_artifact_hash": self.config.formula_artifact_hash,
                }
            )
        try:
            options = factory(**values)
        except TypeError:
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
            required = {"artifacts_path": values["artifacts_path"]}
            if self.config.formula_enrichment_enabled:
                required.update(
                    {
                        key: values[key]
                        for key in (
                            "formula_artifacts_path",
                            "formula_model_id",
                            "formula_model_version",
                            "formula_artifact_hash",
                        )
                    }
                )
            for name, value in required.items():
                self._set_required_option(options, name, value, resource_id)
            for name in ("offline", "allow_download", "enable_remote_services", "cache_dir"):
                self._set_if_supported(options, name, values[name])
        self._assert_ocr_disabled(options, resource_id)
        return options

    def _require_local_artifacts(self, resource_id: str | None) -> Mapping[str, Any]:
        if not self.config.docling_offline or self.config.docling_do_ocr:
            raise ParseFailure("invalid offline OCR parser configuration", resource_id=resource_id)
        manifest = self._verify_artifact_manifest(
            self.config.docling_artifacts_path,
            _DOCLING_MANIFEST,
            _DOCLING_MANIFEST_SCHEMA,
            resource_id,
        )
        if self.config.formula_enrichment_enabled:
            self._verify_formula_artifacts(resource_id)
        return manifest

    def _verify_formula_artifacts(self, resource_id: str | None) -> None:
        required = {
            "formula_model_id": self.config.formula_model_id,
            "formula_model_version": self.config.formula_model_version,
            "formula_artifact_hash": self.config.formula_artifact_hash,
        }
        if any(not isinstance(value, str) or not value.strip() for value in required.values()):
            raise DoclingArtifactsUnavailable(
                "formula enrichment requires local model id, version, and artifact hash",
                resource_id=resource_id,
            )
        if self.config.formula_artifacts_path is None:
            raise DoclingArtifactsUnavailable(
                "formula enrichment requires local formula artifacts", resource_id=resource_id
            )
        manifest = self._verify_artifact_manifest(
            self.config.formula_artifacts_path,
            _FORMULA_MANIFEST,
            _FORMULA_MANIFEST_SCHEMA,
            resource_id,
        )
        for key, expected in (
            ("model_id", self.config.formula_model_id),
            ("model_version", self.config.formula_model_version),
            ("artifact_hash", self.config.formula_artifact_hash),
        ):
            if manifest.get(key) != expected:
                raise DoclingArtifactsUnavailable(
                    f"local formula artifact manifest does not match configured {key}",
                    resource_id=resource_id,
                )

    @staticmethod
    def _verify_artifact_manifest(
        root: Path,
        name: str,
        schema: str,
        resource_id: str | None,
    ) -> Mapping[str, Any]:
        manifest_path = root / name
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DoclingArtifactsUnavailable(
                f"local {name} is unavailable or invalid", resource_id=resource_id
            ) from exc
        if not isinstance(payload, Mapping) or payload.get("schema_version") != schema:
            raise DoclingArtifactsUnavailable(
                f"local {name} has an unsupported schema", resource_id=resource_id
            )
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)) or not artifacts:
            raise DoclingArtifactsUnavailable(
                f"local {name} has no provisioned artifact set", resource_id=resource_id
            )
        root_resolved = root.resolve(strict=False)
        for item in artifacts:
            if not isinstance(item, Mapping):
                raise DoclingArtifactsUnavailable(
                    f"local {name} has an invalid artifact entry", resource_id=resource_id
                )
            relative = item.get("path")
            expected_hash = item.get("sha256")
            if not isinstance(relative, str) or not relative or not isinstance(expected_hash, str):
                raise DoclingArtifactsUnavailable(
                    f"local {name} has an incomplete artifact entry", resource_id=resource_id
                )
            target = (root / relative).resolve(strict=False)
            try:
                target.relative_to(root_resolved)
            except ValueError as exc:
                raise DoclingArtifactsUnavailable(
                    f"local {name} references an artifact outside its cache", resource_id=resource_id
                ) from exc
            try:
                actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            except OSError as exc:
                raise DoclingArtifactsUnavailable(
                    f"local {name} is missing a required artifact", resource_id=resource_id
                ) from exc
            if actual_hash != expected_hash.casefold():
                raise DoclingArtifactsUnavailable(
                    f"local {name} artifact hash mismatch", resource_id=resource_id
                )
        return payload

    @staticmethod
    def _set_if_supported(options: Any, name: str, value: Any) -> None:
        try:
            setattr(options, name, value)
        except (AttributeError, TypeError, ValueError):
            return

    @staticmethod
    def _set_required_option(options: Any, name: str, value: Any, resource_id: str | None) -> None:
        try:
            setattr(options, name, value)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ParserDependencyUnavailable(
                f"installed Docling does not expose required local option {name}",
                resource_id=resource_id,
            ) from exc
        if getattr(options, name, None) != value:
            raise ParserDependencyUnavailable(
                f"installed Docling did not bind required local option {name}",
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

    def _convert(
        self,
        staged: Path,
        options: Any,
        format_name: str,
        resource_id: str,
        manifest: Mapping[str, Any],
    ) -> Any:
        try:
            with self._offline_runtime():
                converter = self._make_converter(options, format_name, resource_id, manifest)
                return converter.convert(staged)
        except ParseFailure:
            raise
        except Exception as exc:  # optional parser exceptions are implementation-specific
            raise ParseFailure(
                "Docling conversion failed", resource_id=resource_id, stage="PARSE"
            ) from exc

    @contextmanager
    def _offline_runtime(self) -> Iterator[None]:
        """Bind standard resolver/cache guards for converter construction and use."""

        values = {
            "DOCLING_OFFLINE": "1",
            "DOCLING_ARTIFACTS_PATH": str(self.config.docling_artifacts_path),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(self.config.docling_artifacts_path),
        }
        previous = {name: os.environ.get(name) for name in values}
        changed_settings: list[tuple[Any, str, Any]] = []
        try:
            os.environ.update(values)
            try:
                module = import_module("docling.datamodel.settings")
            except ImportError:
                module = None
            if module is not None:
                for target_name in ("settings", "DOCLING_SETTINGS"):
                    target = getattr(module, target_name, None)
                    if target is None:
                        continue
                    for name, value in (
                        ("offline", True),
                        ("allow_download", False),
                        ("artifacts_path", str(self.config.docling_artifacts_path)),
                        ("cache_dir", str(self.config.docling_artifacts_path)),
                    ):
                        if hasattr(target, name):
                            old = getattr(target, name)
                            try:
                                setattr(target, name, value)
                            except (AttributeError, TypeError, ValueError):
                                continue
                            changed_settings.append((target, name, old))
            yield
        finally:
            for target, name, old in reversed(changed_settings):
                try:
                    setattr(target, name, old)
                except (AttributeError, TypeError, ValueError):
                    pass
            for name, old in previous.items():
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old

    def _make_converter(
        self,
        options: Any,
        format_name: str,
        resource_id: str,
        manifest: Mapping[str, Any],
    ) -> Any:
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
        self._assert_docling_manifest_version(manifest, resource_id)
        format_options: dict[Any, Any] = {}
        if format_name == "pdf":
            format_options[getattr(InputFormat, "PDF")] = PdfFormatOption(pipeline_options=options)
        elif format_name == "epub" and hasattr(InputFormat, "EPUB"):
            # EPUB has no OCR pipeline, but receives the same local resolver controls below.
            format_options = {}
        values = {
            "format_options": format_options,
            "artifacts_path": str(self.config.docling_artifacts_path),
            "cache_dir": str(self.config.docling_artifacts_path),
            "offline": True,
            "allow_download": False,
        }
        return self._construct_converter(DocumentConverter, values, resource_id)

    @staticmethod
    def _construct_converter(factory: Callable[..., Any], values: dict[str, Any], resource_id: str) -> Any:
        try:
            return factory(**values)
        except TypeError:
            try:
                signature = inspect.signature(factory)
            except (TypeError, ValueError) as exc:
                raise ParserDependencyUnavailable(
                    "installed Docling converter does not expose local runtime controls",
                    resource_id=resource_id,
                ) from exc
            supports_varargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if supports_varargs:
                raise
            supported = {
                name: value for name, value in values.items() if name in signature.parameters
            }
            try:
                return factory(**supported)
            except TypeError as exc:
                raise ParserDependencyUnavailable(
                    "installed Docling converter is incompatible with local runtime controls",
                    resource_id=resource_id,
                ) from exc

    def _assert_docling_manifest_version(
        self, manifest: Mapping[str, Any], resource_id: str
    ) -> None:
        expected = manifest.get("docling_version")
        if not isinstance(expected, str) or not expected.strip():
            raise DoclingArtifactsUnavailable(
                "local Docling artifact manifest has no version", resource_id=resource_id
            )
        actual = self._docling_version()
        if actual != expected:
            raise DoclingArtifactsUnavailable(
                "local Docling artifact manifest does not match installed Docling",
                resource_id=resource_id,
            )

    @staticmethod
    def _coerce_payload(value: Any, resource_id: str) -> Mapping[str, object]:
        if isinstance(value, Mapping):
            payload = value
        else:
            candidate = getattr(value, "document", value)
            payload = None
            for name in ("export_to_dict", "model_dump", "dict"):
                method = getattr(candidate, name, None)
                if callable(method):
                    result = method()
                    if isinstance(result, Mapping):
                        payload = result
                        break
            if payload is None:
                raise ParseFailure(
                    "Docling output cannot be normalized without flattening",
                    resource_id=resource_id,
                )
        if "blocks" in payload:
            return payload
        if any(key in payload for key in ("body", "texts", "tables", "formulas", "pictures")):
            return DoclingDocumentParser._map_docling_export(payload)
        raise ParseFailure("unrecognized Docling document export", resource_id=resource_id)

    @staticmethod
    def _map_docling_export(export: Mapping[str, object]) -> Mapping[str, object]:
        """Map the stable exported Docling body/item model into normalized IR input."""

        collections = (
            "texts",
            "tables",
            "formulas",
            "equations",
            "pictures",
            "figures",
            "references",
            "list_items",
            "key_value_items",
            "groups",
        )
        indexed: dict[str, tuple[Mapping[str, object], str]] = {}
        all_items: list[tuple[Mapping[str, object], str]] = []
        for collection in collections:
            values = export.get(collection)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for position, value in enumerate(values):
                if not isinstance(value, Mapping):
                    continue
                item = value
                reference = item.get("self_ref") or item.get("ref") or f"#/{collection}/{position}"
                indexed[str(reference)] = (item, collection)
                all_items.append((item, collection))

        ordered: list[tuple[Mapping[str, object], str]] = []
        seen: set[str] = set()

        def resolve_ref_texts(value: object, visited: set[str] | None = None) -> tuple[str, ...]:
            values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else (value,)
            visited = set() if visited is None else visited
            resolved: list[str] = []
            for entry in values:
                if isinstance(entry, str):
                    reference = entry
                elif isinstance(entry, Mapping):
                    reference = entry.get("$ref") or entry.get("ref") or entry.get("self_ref")
                else:
                    reference = None
                if reference is None:
                    continue
                key = str(reference)
                if key in visited or key not in indexed:
                    continue
                visited.add(key)
                item, collection = indexed[key]
                if collection == "groups":
                    resolved.extend(resolve_ref_texts(item.get("children") or (), visited))
                    continue
                text = DoclingDocumentParser._item_text(item)
                if text:
                    resolved.append(text)
            return tuple(resolved)

        def visit(node: object) -> None:
            if not isinstance(node, Mapping):
                return
            reference = node.get("$ref") or node.get("ref")
            if reference is not None and str(reference) in indexed:
                item, collection = indexed[str(reference)]
                if collection == "groups":
                    for child in item.get("children") or ():
                        visit(child)
                    return
                key = str(item.get("self_ref") or reference)
                if key not in seen:
                    seen.add(key)
                    ordered.append((item, collection))
                for child in item.get("children") or ():
                    visit(child)
                return
            if node.get("self_ref") is not None and str(node["self_ref"]) in indexed:
                item, collection = indexed[str(node["self_ref"])]
                if collection == "groups":
                    for child in item.get("children") or ():
                        visit(child)
                    return
                key = str(item.get("self_ref"))
                if key not in seen:
                    seen.add(key)
                    ordered.append((item, collection))
            children = node.get("children")
            if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
                for child in children:
                    visit(child)

        body = export.get("body")
        if isinstance(body, Mapping):
            children = body.get("children")
            if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
                for child in children:
                    visit(child)
        if not ordered:
            for item, collection in all_items:
                if collection == "groups":
                    continue
                key = str(item.get("self_ref") or id(item))
                if key not in seen:
                    seen.add(key)
                    ordered.append((item, collection))

        blocks: list[dict[str, object]] = []
        section_path: list[str] = []
        for order, (item, collection) in enumerate(ordered):
            label = str(item.get("label") or collection).casefold()
            text = DoclingDocumentParser._item_text(item)
            heading = DoclingDocumentParser._is_heading(label)
            if heading and text:
                level = DoclingDocumentParser._heading_level(item, label)
                section_path = section_path[: max(0, level - 1)]
                section_path.append(text)
            content_type = DoclingDocumentParser._content_type(label, collection)
            location = DoclingDocumentParser._item_location(item)
            block: dict[str, object] = {
                "block_id": str(item.get("self_ref") or f"{collection}-{order}"),
                "content_type": content_type.value,
                "text": text,
                "reading_order": order,
                "section_path": tuple(section_path),
            }
            block.update(location)
            if section_path:
                block["chapter"] = section_path[0]
            if heading:
                block["metadata"] = {"is_heading": True, "docling_label": label}
            elif item.get("metadata") and isinstance(item["metadata"], Mapping):
                block["metadata"] = dict(item["metadata"])
            if content_type is ContentType.EQUATION:
                block["equation"] = DoclingDocumentParser._equation_data(item, text)
            elif content_type is ContentType.TABLE:
                block["table"] = DoclingDocumentParser._table_data(item, resolve_ref_texts)
            elif content_type is ContentType.FIGURE_CAPTION:
                captions = resolve_ref_texts(item.get("captions") or item.get("caption_refs") or ())
                related_references = resolve_ref_texts(item.get("references") or ())
                footnotes = resolve_ref_texts(item.get("footnotes") or ())
                block["figure_metadata"] = {
                    "caption": str(item.get("caption") or (captions[0] if captions else text) or "") or None,
                    "figure_id": str(item.get("self_ref") or "") or None,
                    "extra": {
                        "related_references": related_references,
                        "footnotes": footnotes,
                    },
                }
            elif content_type is ContentType.REFERENCE:
                block["reference_metadata"] = {
                    "citation": text or None,
                    "identifier": item.get("identifier") or item.get("doi"),
                }
            blocks.append(block)

        metadata: dict[str, object] = {}
        raw_metadata = export.get("metadata")
        if isinstance(raw_metadata, Mapping):
            metadata.update(raw_metadata)
        origin = export.get("origin")
        if isinstance(origin, Mapping):
            filename = origin.get("filename")
            if isinstance(filename, str) and filename:
                metadata.setdefault("source_filename", filename)
            mimetype = origin.get("mimetype")
            if isinstance(mimetype, str) and mimetype:
                metadata.setdefault("extra", {"docling_origin_mimetype": mimetype})
        name = export.get("name")
        if isinstance(name, str) and name.strip():
            metadata.setdefault("title", name.strip())
        page_text, image_pages, spine_text, spine_images = DoclingDocumentParser._page_inventory(
            export, blocks
        )
        payload: dict[str, object] = {"metadata": metadata, "blocks": blocks}
        if page_text or image_pages:
            payload["page_text_chars"] = page_text
            payload["image_pages"] = image_pages
        if spine_text or spine_images:
            payload["spine_text_chars"] = spine_text
            payload["image_spine_items"] = spine_images
        return payload

    @staticmethod
    def _item_text(item: Mapping[str, object]) -> str:
        for key in ("text", "caption", "name"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        data = item.get("data")
        if isinstance(data, Mapping):
            for key in ("text", "caption"):
                value = data.get(key)
                if isinstance(value, str) and value:
                    return value
        return ""

    @staticmethod
    def _is_heading(label: str) -> bool:
        return label in {"title", "section_header", "section-header", "heading", "header"}

    @staticmethod
    def _heading_level(item: Mapping[str, object], label: str) -> int:
        value = item.get("level") or item.get("heading_level")
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1 if label == "title" else 2

    @staticmethod
    def _content_type(label: str, collection: str) -> ContentType:
        kind = f"{label} {collection}".casefold()
        if "table" in kind:
            return ContentType.TABLE
        if any(value in kind for value in ("formula", "equation", "math")):
            return ContentType.EQUATION
        if any(value in kind for value in ("picture", "figure", "image")):
            return ContentType.FIGURE_CAPTION
        if any(value in kind for value in ("reference", "citation", "bibliography")):
            return ContentType.REFERENCE
        if "list" in kind:
            return ContentType.LIST
        if any(value in kind for value in ("key_value", "definition", "glossary")):
            return ContentType.DEFINITION
        return ContentType.PROSE

    @staticmethod
    def _item_location(item: Mapping[str, object]) -> dict[str, object]:
        sources: list[Mapping[str, object]] = [item]
        for key in ("prov", "provenance", "locations"):
            value = item.get(key)
            if isinstance(value, Mapping):
                sources.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                sources.extend(value for value in value if isinstance(value, Mapping))
        pages: list[int] = []
        spine_item: str | None = None
        anchor: str | None = None
        for source in sources:
            for key in ("page_no", "page", "page_number"):
                value = source.get(key)
                try:
                    if value is not None:
                        pages.append(int(value))
                except (TypeError, ValueError):
                    pass
            if spine_item is None:
                for key in ("epub_spine_item", "spine_item", "href", "source_item"):
                    value = source.get(key)
                    if isinstance(value, str) and value:
                        spine_item = value.split("#", 1)[0]
                        if "#" in value and anchor is None:
                            anchor = value.split("#", 1)[1]
                        break
            if anchor is None:
                for key in ("anchor", "fragment", "id"):
                    value = source.get(key)
                    if isinstance(value, str) and value:
                        anchor = value.lstrip("#")
                        break
        result: dict[str, object] = {}
        if pages:
            result["page_start"] = min(pages)
            result["page_end"] = max(pages)
        if spine_item:
            result["epub_spine_item"] = spine_item
        if anchor:
            result["anchor"] = anchor
        return result

    @staticmethod
    def _equation_data(item: Mapping[str, object], text: str) -> dict[str, object]:
        data = item.get("data") if isinstance(item.get("data"), Mapping) else {}
        assert isinstance(data, Mapping)
        return {
            "latex": item.get("latex") or data.get("latex"),
            "mathml": item.get("mathml") or data.get("mathml"),
            "parser_native": item.get("parser_native") or item.get("text") or data.get("text") or text,
            "plain_text": item.get("plain_text") or data.get("plain_text"),
            "representation_format": item.get("representation_format") or data.get("format"),
        }

    @staticmethod
    def _table_data(
        item: Mapping[str, object], resolve_ref_texts: Callable[[object], tuple[str, ...]]
    ) -> dict[str, object]:
        data = item.get("data") if isinstance(item.get("data"), Mapping) else {}
        assert isinstance(data, Mapping)
        raw_cells = data.get("table_cells") or item.get("cells") or data.get("cells") or ()
        cells_by_row: dict[int, dict[int, str]] = {}
        if isinstance(raw_cells, Sequence) and not isinstance(raw_cells, (str, bytes)):
            for index, value in enumerate(raw_cells):
                if not isinstance(value, Mapping):
                    continue
                row = value.get("row", value.get("start_row_offset_idx", 0))
                column = value.get("col", value.get("start_col_offset_idx", index))
                try:
                    row_number, column_number = int(row), int(column)
                except (TypeError, ValueError):
                    continue
                cell_text = value.get("text") or value.get("content") or ""
                cells_by_row.setdefault(row_number, {})[column_number] = str(cell_text)
        rows = [
            tuple(row[column] for column in range(max(row) + 1)) if row else ()
            for _, row in sorted(cells_by_row.items())
        ]
        headers = tuple(rows[0]) if rows else tuple(str(item) for item in (data.get("headers") or ()))
        body = tuple(rows[1:]) if rows else tuple(tuple(row) for row in (data.get("rows") or ()))
        captions = resolve_ref_texts(item.get("captions") or item.get("caption_refs") or ())
        related_references = resolve_ref_texts(item.get("references") or ())
        footnotes = resolve_ref_texts(item.get("footnotes") or ())
        return {
            "caption": item.get("caption") or data.get("caption") or (captions[0] if captions else None),
            "headers": headers,
            "cells": body,
            "units": tuple(str(item) for item in (data.get("units") or ())),
            "notes": tuple(str(item) for item in (data.get("notes") or ())) + footnotes,
            "row_count": len(body),
            "column_count": len(headers),
            "extra": {"related_references": related_references},
        }

    @staticmethod
    def _page_inventory(
        export: Mapping[str, object], blocks: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[int, ...], tuple[bool, ...], tuple[int, ...], tuple[bool, ...]]:
        pages = export.get("pages")
        page_entries: dict[int, Mapping[str, object]] = {}
        if isinstance(pages, Mapping):
            iterable = pages.items()
        elif isinstance(pages, Sequence) and not isinstance(pages, (str, bytes)):
            iterable = enumerate(pages, start=1)
        else:
            iterable = ()
        for key, value in iterable:
            if not isinstance(value, Mapping):
                continue
            try:
                page_entries[int(key)] = value
            except (TypeError, ValueError):
                continue
        for block in blocks:
            page = block.get("page_start")
            try:
                if page is not None:
                    page_entries.setdefault(int(page), {})
            except (TypeError, ValueError):
                pass
        if page_entries:
            maximum = max(page_entries)
            text_counts = [0] * maximum
            image_flags = [False] * maximum
            for page, value in page_entries.items():
                image_flags[page - 1] = bool(
                    value.get("image") or value.get("has_image") or value.get("page_image")
                )
            for block in blocks:
                page = block.get("page_start")
                try:
                    index = int(page) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= index < maximum:
                    text_counts[index] += DoclingDocumentParser._block_char_count(block)
            return tuple(text_counts), tuple(image_flags), (), ()

        spine_order: list[str] = []
        text_counts: dict[str, int] = {}
        images: dict[str, bool] = {}
        for block in blocks:
            spine = block.get("epub_spine_item")
            if not isinstance(spine, str) or not spine:
                continue
            if spine not in text_counts:
                spine_order.append(spine)
                text_counts[spine] = 0
                images[spine] = False
            text_counts[spine] += DoclingDocumentParser._block_char_count(block)
        return (), (), tuple(text_counts[item] for item in spine_order), tuple(
            images[item] for item in spine_order
        )

    @staticmethod
    def _block_char_count(block: Mapping[str, object]) -> int:
        pieces = [str(block.get("text") or "")]
        equation = block.get("equation")
        if isinstance(equation, Mapping) and not pieces[0].strip():
            pieces.append(
                str(
                    equation.get("parser_native")
                    or equation.get("latex")
                    or equation.get("mathml")
                    or equation.get("plain_text")
                    or ""
                )
            )
        table = block.get("table")
        if isinstance(table, Mapping):
            pieces.extend(str(value) for value in table.get("headers", ()))
            for row in table.get("cells", ()):
                if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                    pieces.extend(str(value) for value in row)
        return len("".join(" ".join(pieces).split()))

    @staticmethod
    def _assert_ocr_disabled(options: Any, resource_id: str | None) -> None:
        if getattr(options, "do_ocr", None) is not False:
            raise ParseFailure("Docling PDF options must disable OCR", resource_id=resource_id)

    def _normalized(self, payload: Mapping[str, object], resource, *, adapter_path: str) -> ParsedDocument:
        native_equation_present = any(
            isinstance(block, Mapping)
            and isinstance(block.get("equation"), Mapping)
            and bool(block["equation"].get("parser_native"))
            for block in (payload.get("blocks") or ())
        )
        formula_status = payload.get("formula_enrichment_status")
        if formula_status is None:
            if self.config.formula_enrichment_enabled:
                formula_status = "ENABLED_LOCAL"
            elif native_equation_present:
                formula_status = "DISABLED_NATIVE_PRESERVED"
            else:
                formula_status = "DISABLED"
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
        document = normalize_parsed_document(
            payload,
            resource,
            parser_id=self.parser_id,
            parser_version=self._docling_version(),
            parser_config_hash=self.config.component_fingerprints["parser"],
            parser_provenance=provenance,
        )
        if not self.config.formula_enrichment_enabled:
            blocks = tuple(
                replace(
                    block,
                    equation=replace(
                        block.equation, formula_enrichment_status=str(formula_status)
                    ),
                )
                if block.equation is not None and block.equation.parser_native
                else block
                for block in document.blocks
            )
            document = replace(document, blocks=blocks)
        return document

    @staticmethod
    def _docling_version() -> str:
        try:
            module = import_module("docling")
            version_value = getattr(module, "__version__", None)
            if isinstance(version_value, str) and version_value:
                return version_value
        except ImportError:
            pass
        try:
            from importlib.metadata import version

            return version("docling")
        except Exception:
            return "unavailable-fake-adapter"

    def _has_epub_locations(self, document: ParsedDocument, epub_path: Path) -> bool:
        if not document.blocks:
            return False
        try:
            _, spine = self._read_epub_spine(epub_path)
        except (OSError, KeyError, ElementTree.ParseError, zipfile.BadZipFile):
            return False
        valid = {item: anchors for _, item, _, anchors in spine}
        for block in document.blocks:
            if not block.epub_spine_item or not block.anchor:
                return False
            item = self._canonical_spine_item(block.epub_spine_item)
            anchor = block.anchor.lstrip("#")
            if item not in valid or anchor not in valid[item]:
                return False
        return True

    @staticmethod
    def _canonical_spine_item(value: str) -> str:
        return PurePosixPath(value.lstrip("/").replace("\\", "/")).as_posix()

    def _parse_epub_spine_fallback(
        self, staged: Path, resource, native: ParsedDocument
    ) -> ParsedDocument:
        try:
            payload = self._opf_spine_payload(staged)
        except (OSError, KeyError, ElementTree.ParseError, zipfile.BadZipFile) as exc:
            raise ParseFailure("EPUB OPF/spine fallback failed", resource_id=resource.resource_id) from exc
        fallback = self._normalized(payload, resource, adapter_path="OPF_SPINE_FALLBACK")
        # A failed native provenance contract provides no auditable one-to-one
        # source mapping. Do not match by ordinal or reuse an unrelated anchor:
        # the OPF document is the only verified source-location representation.
        provenance = dict(fallback.parser_provenance)
        provenance.update(
            {
                "adapter_path": "OPF_SPINE_FALLBACK",
                "native_epub_location_contract": "FAILED_USED_VERIFIED_OPF",
            }
        )
        return replace(fallback, parser_provenance=provenance)

    @classmethod
    def _opf_spine_payload(cls, epub_path: Path) -> dict[str, object]:
        title, spine = cls._read_epub_spine(epub_path)
        blocks: list[dict[str, object]] = []
        order = 0
        for item_id, spine_item, markup, _ in spine:
            parser = _XhtmlTextExtractor()
            parser.feed(markup)
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

    @classmethod
    def _read_epub_spine(
        cls, epub_path: Path
    ) -> tuple[str | None, tuple[tuple[str, str, str, frozenset[str]], ...]]:
        with zipfile.ZipFile(epub_path) as archive:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next(
                item.attrib["full-path"]
                for item in container.iter()
                if item.tag.endswith("rootfile") and item.attrib.get("full-path")
            )
            opf = ElementTree.fromstring(archive.read(rootfile))
            opf_parent = PurePosixPath(rootfile).parent
            manifest = {
                item.attrib["id"]: item.attrib["href"].split("#", 1)[0]
                for item in opf.iter()
                if item.tag.endswith("item") and item.attrib.get("id") and item.attrib.get("href")
            }
            spine_ids = [
                item.attrib["idref"]
                for item in opf.iter()
                if item.tag.endswith("itemref") and item.attrib.get("idref")
            ]
            title = next(
                (item.text for item in opf.iter() if item.tag.endswith("title") and item.text), None
            )
            entries: list[tuple[str, str, str, frozenset[str]]] = []
            for item_id in spine_ids:
                href = manifest.get(item_id)
                if href is None:
                    continue
                spine_item = cls._canonical_spine_item(str(opf_parent / href))
                markup = archive.read(spine_item).decode("utf-8", errors="replace")
                extractor = _XhtmlTextExtractor()
                extractor.feed(markup)
                extractor.close()
                entries.append((item_id, spine_item, markup, frozenset(extractor.anchors)))
        return title, tuple(entries)


__all__ = [
    "DoclingArtifactsUnavailable",
    "DoclingDocumentParser",
    "ParserDependencyUnavailable",
]
