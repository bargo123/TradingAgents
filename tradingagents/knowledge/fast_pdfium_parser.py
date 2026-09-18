"""Fast, offline PDF extraction with page-level provenance.

This adapter is intentionally conservative: it extracts only parser-native page
text and never performs OCR.  Parsed documents are cached by source hash and
parser identity so an interrupted corpus run can resume without reparsing
completed sources.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .identity import document_id_for
from .models import ContentType, DiscoveredResource, DocumentMetadata, ParsedBlock, ParsedDocument
from .parser import DocumentParser, ParseFailure, ParserDependencyUnavailable


class PdfiumParser:
    """Extract text from digital PDFs using the locally installed PDFium wheel."""

    parser_id = "pypdfium2"
    parser_version = "pypdfium2-page-v1"
    parser_config_hash = "sha256:" + hashlib.sha256(
        b"pypdfium2-page-v1|ocr=false|deterministic-page-order=true"
    ).hexdigest()

    def parse(self, resource: DiscoveredResource, staging_dir: Path) -> ParsedDocument:
        if resource.path.suffix.casefold() != ".pdf":
            raise ParseFailure("pypdfium2 parser supports PDF only", resource_id=resource.resource_id)
        if not resource.source_hash:
            raise ParseFailure("source hash is required", resource_id=resource.resource_id)
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise ParserDependencyUnavailable(
                "pypdfium2 is required for the fast local PDF parser",
                resource_id=resource.resource_id,
            ) from exc

        try:
            pdf = pdfium.PdfDocument(str(resource.path))
            blocks: list[ParsedBlock] = []
            page_text_chars: list[int] = []
            image_pages: list[bool] = []
            for page_index in range(len(pdf)):
                text = pdf[page_index].get_textpage().get_text_bounded() or ""
                normalized = str(text).replace("\x00", "").replace("\r\n", "\n").strip()
                page_number = page_index + 1
                page_text_chars.append(len(normalized))
                image_pages.append(not bool(normalized))
                if normalized:
                    blocks.append(
                        ParsedBlock(
                            block_id=f"page-{page_number:06d}",
                            content_type=ContentType.PROSE,
                            text=normalized,
                            reading_order=page_index,
                            page_start=page_number,
                            page_end=page_number,
                            metadata={
                                "page_number": page_number,
                                "image_bearing": not bool(normalized),
                                "extraction": "pypdfium2",
                            },
                        )
                    )
        except Exception as exc:
            raise ParseFailure(
                f"pypdfium2 PDF extraction failed: {type(exc).__name__}: {exc}",
                resource_id=resource.resource_id,
            ) from exc

        metadata = DocumentMetadata(
            title=resource.path.stem,
            title_source="filename",
            format="pdf",
            source_filename=resource.path.name,
            source_relative_path=resource.relative_path,
            source_hash=resource.source_hash,
        )
        return ParsedDocument(
            document_id=document_id_for(resource.source_hash),
            source_hash=resource.source_hash,
            metadata=metadata,
            blocks=tuple(blocks),
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            parser_config_hash=self.parser_config_hash,
            parser_provenance={
                "ocr": False,
                "parser": self.parser_id,
                "page_count": len(page_text_chars),
                "page_text_chars": tuple(page_text_chars),
                "image_pages": tuple(image_pages),
                "deterministic_order": "page-ascending",
            },
        )


class CachedPdfiumParser:
    """Persist one validated parsed-document artifact per source hash."""

    cache_schema = "pdfium-parse-cache-v1"

    def __init__(self, cache_root: str | Path, parser: DocumentParser | None = None) -> None:
        self.cache_root = Path(cache_root)
        self.parser = parser or PdfiumParser()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def parse(self, resource: DiscoveredResource, staging_dir: Path) -> ParsedDocument:
        if not resource.source_hash:
            raise ParseFailure("source hash is required", resource_id=resource.resource_id)
        path = self.cache_root / f"{document_id_for(resource.source_hash)}.json"
        if path.is_file():
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                document = ParsedDocument.from_dict(envelope["document"])
                if (
                    envelope.get("schema") != self.cache_schema
                    or envelope.get("source_hash") != resource.source_hash
                    or envelope.get("parser_id") != document.parser_id
                    or envelope.get("parser_version") != document.parser_version
                    or envelope.get("parser_config_hash") != document.parser_config_hash
                    or document.source_hash != resource.source_hash
                ):
                    raise ValueError("parse cache identity mismatch")
                return document
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ParseFailure(
                    f"invalid parse cache for {resource.relative_path}: {exc}",
                    resource_id=resource.resource_id,
                ) from exc

        document = self.parser.parse(resource, staging_dir)
        payload: dict[str, Any] = {
            "schema": self.cache_schema,
            "source_hash": resource.source_hash,
            "parser_id": document.parser_id,
            "parser_version": document.parser_version,
            "parser_config_hash": document.parser_config_hash,
            "document": document.to_dict(),
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)
        return document


__all__ = ["CachedPdfiumParser", "PdfiumParser"]
