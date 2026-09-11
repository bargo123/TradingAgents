"""Fail-closed provenance checks for published knowledge evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import KnowledgeHit


class ProvenanceError(ValueError):
    """A retrieval result is missing a required source identity or location."""


def _required(value: Any, field: str) -> None:
    if value is None or not str(value).strip():
        raise ProvenanceError(f"knowledge hit is missing {field}")


def validate_hit_provenance(hit: KnowledgeHit | Mapping[str, Any]) -> None:
    """Require enough identity to audit a returned chunk back to its source."""

    values = hit if isinstance(hit, Mapping) else hit.to_dict()
    for field in ("document_id", "chunk_id", "source_hash", "content_type"):
        _required(values.get(field), field)
    _required(values.get("source_filename"), "source filename")
    _required(values.get("source_relative_path"), "source relative path")

    location = next(
        (
            values.get(field)
            for field in (
                "page", "page_start", "chapter", "section", "section_path",
                "epub_spine_item", "anchor",
            )
            if values.get(field) not in (None, "", ())
        ),
        None,
    )
    if location is None or location in ("", (), []):
        raise ProvenanceError("knowledge hit is missing source location")
    for field in ("parser_version", "chunker_version", "index_version"):
        _required(values.get(field), field)

    extra = values.get("extra") or {}
    if not isinstance(extra, Mapping):
        raise ProvenanceError("knowledge hit provenance metadata is malformed")
    generation = (
        extra.get("projection_generation")
        or extra.get("generation_id")
        or values.get("projection_generation")
    )
    _required(generation, "generation identity")


__all__ = ["ProvenanceError", "validate_hit_provenance"]
