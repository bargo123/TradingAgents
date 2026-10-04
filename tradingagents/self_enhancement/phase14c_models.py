"""Immutable value objects shared by offline Phase 14C discovery stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Phase14CGenerationPin:
    generation_id: str
    generation_fingerprint: str
    population_hash: str
    status: str
    vector_ready: bool
    lexical_ready: bool
    vector_location: Path
    lexical_location: Path
    inventory_schema_version: str
    document_count: int
    chunk_count: int

    def __post_init__(self) -> None:
        for name in ("generation_id", "generation_fingerprint", "population_hash", "status"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(self, "vector_location", Path(self.vector_location))
        object.__setattr__(self, "lexical_location", Path(self.lexical_location))
        if self.document_count < 0 or self.chunk_count < 0:
            raise ValueError("generation counts must be non-negative")


@dataclass(frozen=True, slots=True)
class CorpusInventory:
    unique_source_count: int
    indexed_document_count: int
    queryable_document_count: int
    alias_count: int
    needs_ocr_count: int
    unavailable_resource_count: int
    active_chunk_count: int
    source_manifest_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "unique_source_count",
            "indexed_document_count",
            "queryable_document_count",
            "alias_count",
            "needs_ocr_count",
            "unavailable_resource_count",
            "active_chunk_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if len(self.source_manifest_fingerprint) != 64:
            raise ValueError("source_manifest_fingerprint must be a SHA-256 hex digest")
