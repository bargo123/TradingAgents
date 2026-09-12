"""Parser boundary and normalized document-IR construction helpers.

The parser package deliberately contains no ingestion orchestration.  Adapters
raise typed, resource-scoped errors; a later coordinator can record each error
as ``PARSE_FAILED`` while continuing with independent resources.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from .identity import document_id_for
from .models import DiscoveredResource, DocumentMetadata, ParsedBlock, ParsedDocument


class ParseFailure(RuntimeError):
    """A bounded, per-resource parser failure suitable for quarantine."""

    def __init__(
        self,
        message: str,
        *,
        resource_id: str | None = None,
        stage: str = "PARSE",
    ) -> None:
        super().__init__(message)
        self.resource_id = resource_id
        self.stage = stage


class ParserDependencyUnavailable(ParseFailure):
    """The requested optional parser package is not installed locally."""


class DoclingArtifactsUnavailable(ParseFailure):
    """Required local Docling or formula artifacts were not provisioned."""


@runtime_checkable
class DocumentParser(Protocol):
    """A source-read-only parser that emits the shared normalized IR."""

    def parse(self, resource: DiscoveredResource, staging_dir: Path) -> ParsedDocument:
        """Parse one discovered resource using only temporary staging output."""


def normalize_parsed_document(
    payload: Mapping[str, object],
    resource: DiscoveredResource,
    *,
    parser_id: str,
    parser_version: str,
    parser_config_hash: str,
    parser_provenance: Mapping[str, object] | None = None,
) -> ParsedDocument:
    """Create the immutable IR without flattening structured parser blocks."""

    if resource.source_hash is None:
        raise ParseFailure("supported resource has no source hash", resource_id=resource.resource_id)
    if not isinstance(payload, Mapping):
        raise ParseFailure("parser output must be a mapping", resource_id=resource.resource_id)

    raw_metadata = payload.get("metadata") or {}
    if not isinstance(raw_metadata, Mapping):
        raise ParseFailure("parser metadata must be a mapping", resource_id=resource.resource_id)
    metadata_payload = dict(raw_metadata)
    metadata_payload.setdefault("format", resource.format or resource.path.suffix.lstrip("."))
    metadata_payload.setdefault("source_filename", resource.path.name)
    metadata_payload.setdefault("source_relative_path", resource.relative_path)
    metadata_payload.setdefault("source_hash", resource.source_hash)

    raw_blocks = payload.get("blocks") or ()
    if isinstance(raw_blocks, (str, bytes)):
        raise ParseFailure("parser blocks must be a sequence", resource_id=resource.resource_id)
    try:
        blocks = tuple(
            item if isinstance(item, ParsedBlock) else ParsedBlock.from_dict(item)
            for item in raw_blocks  # type: ignore[union-attr]
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ParseFailure("parser emitted an invalid structured block", resource_id=resource.resource_id) from exc

    raw_warnings = payload.get("warnings") or ()
    if isinstance(raw_warnings, str):
        raw_warnings = (raw_warnings,)
    provenance = dict(parser_provenance or {})
    incoming_provenance = payload.get("parser_provenance") or {}
    if isinstance(incoming_provenance, Mapping):
        provenance.update(incoming_provenance)
    for inventory_key in ("page_text_chars", "image_pages", "spine_text_chars", "image_spine_items"):
        if inventory_key in payload:
            provenance[inventory_key] = payload[inventory_key]

    try:
        return ParsedDocument(
            document_id=document_id_for(resource.source_hash),
            source_hash=resource.source_hash,
            metadata=DocumentMetadata.from_dict(metadata_payload),
            blocks=blocks,
            parser_id=parser_id,
            parser_version=parser_version,
            parser_config_hash=parser_config_hash,
            warnings=tuple(str(item) for item in raw_warnings),
            parser_provenance=provenance,
        )
    except (TypeError, ValueError) as exc:
        raise ParseFailure("parser emitted invalid document metadata", resource_id=resource.resource_id) from exc


__all__ = [
    "DoclingArtifactsUnavailable",
    "DocumentParser",
    "ParseFailure",
    "ParserDependencyUnavailable",
    "normalize_parsed_document",
]
