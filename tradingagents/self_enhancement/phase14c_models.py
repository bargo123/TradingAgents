"""Immutable value objects shared by offline Phase 14C discovery stages."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from tradingagents.knowledge.models import ContentType, KnowledgeHit


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


def _stable_code(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value):
        raise ValueError(f"{field_name} must be a stable lowercase code")
    return value


def _score(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class DiscoveryQuery:
    family_id: str
    formulation_id: str
    text: str
    content_types: tuple[ContentType, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _stable_code(self.family_id, "family_id"))
        object.__setattr__(self, "formulation_id", _stable_code(self.formulation_id, "formulation_id"))
        text = str(self.text).strip()
        if not text:
            raise ValueError("query text must be non-empty")
        object.__setattr__(self, "text", text)
        try:
            content_types = tuple(ContentType(value) for value in self.content_types)
        except ValueError as exc:
            raise ValueError("query contains an unsupported content type") from exc
        if len(content_types) != len(set(content_types)):
            raise ValueError("query content_types must not contain duplicates")
        object.__setattr__(self, "content_types", content_types)


@dataclass(frozen=True, slots=True)
class QueryReference:
    family_id: str
    formulation_id: str
    rank: int
    semantic_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _stable_code(self.family_id, "family_id"))
        object.__setattr__(self, "formulation_id", _stable_code(self.formulation_id, "formulation_id"))
        rank = int(self.rank)
        if rank < 1:
            raise ValueError("query reference rank must be positive")
        object.__setattr__(self, "rank", rank)
        for name in ("semantic_score", "lexical_score", "fused_score", "rerank_score"):
            object.__setattr__(self, name, _score(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class DiscoveryHitRecord:
    hit: KnowledgeHit
    references: tuple[QueryReference, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hit, KnowledgeHit):
            raise TypeError("discovery record hit must be KnowledgeHit")
        references = tuple(self.references)
        if not references or any(not isinstance(ref, QueryReference) for ref in references):
            raise ValueError("discovery record requires QueryReference provenance")
        object.__setattr__(self, "references", references)


@dataclass(frozen=True, slots=True)
class DiscoveryQueryFailure:
    family_id: str
    formulation_id: str
    code: str
    exception_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _stable_code(self.family_id, "family_id"))
        object.__setattr__(self, "formulation_id", _stable_code(self.formulation_id, "formulation_id"))
        object.__setattr__(self, "code", _stable_code(self.code.lower(), "code"))
        exception_type = str(self.exception_type).strip()
        if not exception_type or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", exception_type):
            raise ValueError("exception_type must be a safe type name")
        object.__setattr__(self, "exception_type", exception_type)


@dataclass(frozen=True, slots=True)
class DiscoveryRetrievalReport:
    generation_id: str
    population_hash: str
    query_bank_version: str
    query_bank_fingerprint: str
    status: str
    records: tuple[DiscoveryHitRecord, ...] = ()
    raw_hit_count: int = 0
    unique_hit_count: int = 0
    completed_query_count: int = 0
    query_failures: tuple[DiscoveryQueryFailure, ...] = ()

    def __post_init__(self) -> None:
        for name in ("generation_id", "population_hash", "query_bank_version", "query_bank_fingerprint"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.status not in {"COMPLETE", "FAILED"}:
            raise ValueError("retrieval status must be COMPLETE or FAILED")
        records = tuple(self.records)
        failures = tuple(self.query_failures)
        if any(not isinstance(record, DiscoveryHitRecord) for record in records):
            raise TypeError("records must contain DiscoveryHitRecord values")
        if any(not isinstance(failure, DiscoveryQueryFailure) for failure in failures):
            raise TypeError("query_failures must contain DiscoveryQueryFailure values")
        counts = (int(self.raw_hit_count), int(self.unique_hit_count), int(self.completed_query_count))
        if any(value < 0 for value in counts):
            raise ValueError("retrieval counts must be non-negative")
        if self.status == "FAILED" and (records or counts[0] or counts[1]):
            raise ValueError("failed retrieval must not claim partial hit results")
        if self.status == "COMPLETE" and failures:
            raise ValueError("complete retrieval cannot contain query failures")
        if counts[1] != len(records) or counts[0] < counts[1]:
            raise ValueError("retrieval hit counts do not match records")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "query_failures", failures)
        object.__setattr__(self, "raw_hit_count", counts[0])
        object.__setattr__(self, "unique_hit_count", counts[1])
        object.__setattr__(self, "completed_query_count", counts[2])
