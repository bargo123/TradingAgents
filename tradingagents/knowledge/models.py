"""Immutable contracts shared by the local Phase 7 knowledge pipeline.

The knowledge package deliberately keeps these contracts independent from the
TradingAgents decision, forex, and MT5 packages.  They are small value objects
so catalog rows, JSON manifests, and later adapters can exchange the same
stable representation without importing an optional parser or embedding
dependency.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class IngestionState(str, Enum):
    """Resource/document states used by discovery and incremental ingestion."""

    DISCOVERED = "DISCOVERED"
    UNSUPPORTED = "UNSUPPORTED"
    HASHED = "HASHED"
    UNCHANGED = "UNCHANGED"
    DUPLICATE = "DUPLICATE"
    PARSING = "PARSING"
    PARSED = "PARSED"
    CHUNKED = "CHUNKED"
    EMBEDDING = "EMBEDDING"
    EMBED_FAILED = "EMBED_FAILED"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    NEEDS_OCR = "NEEDS_OCR"
    PARSE_FAILED = "PARSE_FAILED"
    INDEX_FAILED = "INDEX_FAILED"
    REMOVED = "REMOVED"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    INTERRUPTED = "INTERRUPTED"
    RETAINED_PREVIOUS = "RETAINED_PREVIOUS"


class AliasRelation(str, Enum):
    """Relationship of a source path to its latest successfully indexed hash."""

    CURRENT = "CURRENT"
    RETAINED_PREVIOUS = "RETAINED_PREVIOUS"
    REMOVED = "REMOVED"


class ContentType(str, Enum):
    """Retrievable semantic block types in the V1 normalized IR."""

    PROSE = "PROSE"
    EQUATION = "EQUATION"
    TABLE = "TABLE"
    FIGURE_CAPTION = "FIGURE_CAPTION"
    DEFINITION = "DEFINITION"
    REFERENCE = "REFERENCE"
    LIST = "LIST"


def _enum_value(value: Enum | Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _json_value(value: Any) -> Any:
    """Convert nested contract values to JSON-safe primitives."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(_json_value(key)): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


class SerializableModel:
    """Small JSON/manifest helper shared by all public value objects."""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _string_tuple(value: Sequence[str] | str | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{name} entries must be non-empty strings")
    return result


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return {str(key): item for key, item in value.items()}


def _content_type(value: ContentType | str) -> ContentType:
    if isinstance(value, ContentType):
        return value
    try:
        return ContentType(str(value).strip().upper())
    except ValueError as exc:
        raise ValueError(f"unsupported content_type: {value!r}") from exc


def _finite_float(value: float | int, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class DiscoveredResource(SerializableModel):
    """One regular source-tree file discovered during a scan."""

    resource_id: str
    relative_path: str
    path: Path
    state: IngestionState = IngestionState.DISCOVERED
    source_hash: str | None = None
    size_bytes: int | None = None
    modified_ns: int | None = None
    display_path: str | None = None
    format: str | None = None

    def __post_init__(self) -> None:
        if not str(self.resource_id).strip():
            raise ValueError("resource_id must be non-empty")
        if not str(self.relative_path).strip():
            raise ValueError("relative_path must be non-empty")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "state", IngestionState(self.state))
        if self.display_path is None:
            object.__setattr__(self, "display_path", str(self.relative_path))
        if self.format is not None:
            object.__setattr__(self, "format", str(self.format).strip().lower().lstrip("."))
        if self.size_bytes is not None:
            size = int(self.size_bytes)
            if size < 0:
                raise ValueError("size_bytes must be non-negative")
            object.__setattr__(self, "size_bytes", size)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DiscoveredResource:
        payload = dict(value)
        payload["path"] = Path(payload["path"])
        payload["state"] = IngestionState(payload.get("state", IngestionState.DISCOVERED))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class DocumentMetadata(SerializableModel):
    """Metadata preserved from a parsed document and its source alias."""

    title: str | None = None
    alternate_title: str | None = None
    authors: tuple[str, ...] = ()
    publication_year: int | None = None
    document_type: str | None = None
    format: str | None = None
    source_filename: str | None = None
    source_relative_path: str | None = None
    source_hash: str | None = None
    title_source: str | None = None
    language: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "authors", _string_tuple(self.authors, "authors"))
        if self.publication_year is not None:
            if isinstance(self.publication_year, bool):
                raise ValueError("publication_year must be an integer")
            year = int(self.publication_year)
            if year < 0 or year > 9999:
                raise ValueError("publication_year must be between 0 and 9999")
            object.__setattr__(self, "publication_year", year)
        if self.format is not None:
            object.__setattr__(self, "format", str(self.format).strip().lower().lstrip("."))
        object.__setattr__(self, "extra", _mapping(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DocumentMetadata:
        payload = dict(value)
        payload["authors"] = tuple(payload.get("authors") or ())
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EquationMetadata(SerializableModel):
    """Structured equation representation and optional variable definitions."""

    latex: str | None = None
    mathml: str | None = None
    parser_native: str | None = None
    plain_text: str | None = None
    variable_definitions: tuple[str, ...] = ()
    representation_format: str | None = None
    parser_confidence: float | None = None
    formula_enrichment_status: str | None = None
    formula_model_id: str | None = None
    formula_model_version: str | None = None
    formula_artifact_hash: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variable_definitions",
            _string_tuple(self.variable_definitions, "variable_definitions"),
        )
        if self.parser_confidence is not None:
            object.__setattr__(
                self,
                "parser_confidence",
                _finite_float(self.parser_confidence, "parser_confidence"),
            )
        object.__setattr__(self, "extra", _mapping(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EquationMetadata:
        payload = dict(value)
        payload["variable_definitions"] = tuple(payload.get("variable_definitions") or ())
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class TableMetadata(SerializableModel):
    """Table structure retained for lexical context and provenance."""

    caption: str | None = None
    headers: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    cells: tuple[tuple[str, ...], ...] = ()
    notes: tuple[str, ...] = ()
    row_count: int | None = None
    column_count: int | None = None
    nearby_explanation: str | None = None
    row_group_is_complete: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", _string_tuple(self.headers, "headers"))
        object.__setattr__(self, "units", _string_tuple(self.units, "units"))
        object.__setattr__(self, "notes", _string_tuple(self.notes, "notes"))
        cells: list[tuple[str, ...]] = []
        for row in self.cells or ():
            cells.append(tuple(str(item) for item in row))
        object.__setattr__(self, "cells", tuple(cells))
        if self.row_count is None:
            object.__setattr__(self, "row_count", len(cells))
        else:
            object.__setattr__(self, "row_count", max(0, int(self.row_count)))
        if self.column_count is None:
            object.__setattr__(self, "column_count", len(self.headers))
        else:
            object.__setattr__(self, "column_count", max(0, int(self.column_count)))
        object.__setattr__(self, "row_group_is_complete", bool(self.row_group_is_complete))
        object.__setattr__(self, "extra", _mapping(self.extra))

    @property
    def rows(self) -> tuple[tuple[str, ...], ...]:
        """Compatibility/readability alias for the canonical ``cells`` field."""

        return self.cells

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TableMetadata:
        payload = dict(value)
        payload["headers"] = tuple(payload.get("headers") or ())
        payload["units"] = tuple(payload.get("units") or ())
        payload["notes"] = tuple(payload.get("notes") or ())
        payload["cells"] = tuple(tuple(row) for row in (payload.get("cells") or ()))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class FigureMetadata(SerializableModel):
    caption: str | None = None
    nearby_explanation: str | None = None
    figure_id: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _mapping(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FigureMetadata:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ReferenceMetadata(SerializableModel):
    citation: str | None = None
    identifier: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _mapping(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferenceMetadata:
        return cls(**dict(value))


# Friendly aliases used by parser adapters and external integrations.
EquationData = EquationMetadata
TableData = TableMetadata


@dataclass(frozen=True, slots=True)
class ParsedBlock(SerializableModel):
    """One ordered, structure-preserving block from the normalized IR."""

    block_id: str
    content_type: ContentType
    text: str = ""
    reading_order: int = 0
    page_start: int | None = None
    page_end: int | None = None
    chapter: str | None = None
    section_path: tuple[str, ...] = ()
    epub_spine_item: str | None = None
    anchor: str | None = None
    equation: EquationMetadata | None = None
    table: TableMetadata | None = None
    figure_metadata: FigureMetadata | Mapping[str, Any] | None = None
    reference_metadata: ReferenceMetadata | Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.block_id).strip():
            raise ValueError("block_id must be non-empty")
        object.__setattr__(self, "content_type", _content_type(self.content_type))
        object.__setattr__(self, "text", str(self.text or ""))
        order = int(self.reading_order)
        if order < 0:
            raise ValueError("reading_order must be non-negative")
        object.__setattr__(self, "reading_order", order)
        object.__setattr__(self, "section_path", _string_tuple(self.section_path, "section_path"))
        if isinstance(self.equation, Mapping):
            object.__setattr__(self, "equation", EquationMetadata.from_dict(self.equation))
        if isinstance(self.table, Mapping):
            object.__setattr__(self, "table", TableMetadata.from_dict(self.table))
        if isinstance(self.figure_metadata, Mapping):
            object.__setattr__(
                self,
                "figure_metadata",
                FigureMetadata(**self.figure_metadata),
            )
        if isinstance(self.reference_metadata, Mapping):
            object.__setattr__(
                self,
                "reference_metadata",
                ReferenceMetadata(**self.reference_metadata),
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def table_metadata(self) -> dict[str, Any] | None:
        if self.table is None:
            return None
        return self.table.to_dict()

    @property
    def equation_metadata(self) -> dict[str, Any] | None:
        if self.equation is None:
            return None
        return self.equation.to_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ParsedBlock:
        payload = dict(value)
        payload["content_type"] = _content_type(payload["content_type"])
        payload["section_path"] = tuple(payload.get("section_path") or ())
        if payload.get("equation") is not None:
            payload["equation"] = EquationMetadata.from_dict(payload["equation"])
        if payload.get("table") is not None:
            payload["table"] = TableMetadata.from_dict(payload["table"])
        if payload.get("figure_metadata") is not None and isinstance(
            payload["figure_metadata"], Mapping
        ):
            payload["figure_metadata"] = FigureMetadata(**payload["figure_metadata"])
        if payload.get("reference_metadata") is not None and isinstance(
            payload["reference_metadata"], Mapping
        ):
            payload["reference_metadata"] = ReferenceMetadata(**payload["reference_metadata"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ParsedDocument(SerializableModel):
    """Normalized document IR returned by a parser adapter."""

    document_id: str
    source_hash: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    blocks: tuple[ParsedBlock, ...] = ()
    parser_id: str = "unknown"
    parser_version: str = "unknown"
    parser_config_hash: str = ""
    warnings: tuple[str, ...] = ()
    parser_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.document_id).strip():
            raise ValueError("document_id must be non-empty")
        if not str(self.source_hash).strip():
            raise ValueError("source_hash must be non-empty")
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", DocumentMetadata.from_dict(self.metadata))
        object.__setattr__(self, "blocks", tuple(self._coerce_block(item) for item in self.blocks))
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, "warnings"))
        object.__setattr__(self, "parser_provenance", _mapping(self.parser_provenance))

    @staticmethod
    def _coerce_block(value: ParsedBlock | Mapping[str, Any]) -> ParsedBlock:
        return value if isinstance(value, ParsedBlock) else ParsedBlock.from_dict(value)

    @property
    def title(self) -> str | None:
        return self.metadata.title

    @property
    def alternate_title(self) -> str | None:
        return self.metadata.alternate_title

    @property
    def authors(self) -> tuple[str, ...]:
        return self.metadata.authors

    @property
    def publication_year(self) -> int | None:
        return self.metadata.publication_year

    @property
    def document_type(self) -> str | None:
        return self.metadata.document_type

    @property
    def format(self) -> str | None:
        return self.metadata.format

    @property
    def text_blocks(self) -> tuple[ParsedBlock, ...]:
        return tuple(item for item in self.blocks if item.text.strip())

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ParsedDocument:
        payload = dict(value)
        payload["metadata"] = DocumentMetadata.from_dict(payload.get("metadata") or {})
        payload["blocks"] = tuple(ParsedBlock.from_dict(item) for item in (payload.get("blocks") or ()))
        payload["warnings"] = tuple(payload.get("warnings") or ())
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EmbeddingSpec(SerializableModel):
    """Complete vector-semantic identity for corpus and query embeddings."""

    model_id: str = "BAAI/bge-small-en-v1.5"
    resolved_model_version: str = "unknown"
    runtime: str = "onnx-cpu"
    artifact_hash: str = ""
    dimensions: int = 384
    normalization_policy: str = "l2"
    tokenizer_fingerprint: str = ""
    model_max_input_tokens: int = 512
    special_token_budget: int = 2
    effective_corpus_content_token_limit: int = 510
    corpus_instruction_policy: str = "none-v1"
    corpus_instruction_version: str = "v1"
    query_instruction_policy: str = "bge-search-prefix-v1"
    query_instruction_version: str = "v1"
    truncation: bool = False

    def __post_init__(self) -> None:
        for name in ("model_id", "resolved_model_version", "runtime"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        dimensions = int(self.dimensions)
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        object.__setattr__(self, "dimensions", dimensions)
        model_limit = int(self.model_max_input_tokens)
        special_budget = int(self.special_token_budget)
        effective_limit = int(self.effective_corpus_content_token_limit)
        if model_limit <= 0:
            raise ValueError("model_max_input_tokens must be positive")
        if special_budget < 0:
            raise ValueError("special_token_budget must be non-negative")
        if effective_limit <= 0 or effective_limit > model_limit - special_budget:
            raise ValueError(
                "effective_corpus_content_token_limit must fit model_max_input_tokens"
            )
        object.__setattr__(self, "model_max_input_tokens", model_limit)
        object.__setattr__(self, "special_token_budget", special_budget)
        object.__setattr__(self, "effective_corpus_content_token_limit", effective_limit)
        object.__setattr__(self, "truncation", bool(self.truncation))
        if self.truncation:
            raise ValueError("truncation must be false for the knowledge index")

    @property
    def embedding_model_id(self) -> str:
        return self.model_id

    @property
    def embedding_model_version(self) -> str:
        return self.resolved_model_version

    @property
    def embedding_artifact_hash(self) -> str:
        return self.artifact_hash

    @property
    def embedding_normalization(self) -> str:
        return self.normalization_policy

    @property
    def embedding_tokenizer_fingerprint(self) -> str:
        return self.tokenizer_fingerprint

    @property
    def embedding_max_input_tokens(self) -> int:
        return self.model_max_input_tokens

    @property
    def embedding_effective_content_token_limit(self) -> int:
        return self.effective_corpus_content_token_limit

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EmbeddingSpec:
        payload = dict(value)
        aliases = {
            "embedding_model_id": "model_id",
            "embedding_model_version": "resolved_model_version",
            "embedding_artifact_hash": "artifact_hash",
            "embedding_dimensions": "dimensions",
            "embedding_normalization": "normalization_policy",
            "embedding_tokenizer_fingerprint": "tokenizer_fingerprint",
            "embedding_max_input_tokens": "model_max_input_tokens",
            "embedding_effective_content_token_limit": "effective_corpus_content_token_limit",
        }
        for old, new in aliases.items():
            if new not in payload and old in payload:
                payload[new] = payload[old]
        return cls(**{field.name: payload[field.name] for field in fields(cls) if field.name in payload})


@dataclass(frozen=True, slots=True)
class ChunkRecord(SerializableModel):
    """A deterministic chunk and its complete retrieval provenance."""

    chunk_id: str = ""
    document_id: str = ""
    source_hash: str = ""
    text: str = ""
    content_type: ContentType = ContentType.PROSE
    source_filename: str | None = None
    source_relative_path: str | None = None
    title: str | None = None
    authors: tuple[str, ...] = ()
    publication_year: int | None = None
    document_type: str | None = None
    format: str | None = None
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    chapter: str | None = None
    section_path: tuple[str, ...] = ()
    epub_spine_item: str | None = None
    anchor: str | None = None
    reading_order: int | None = None
    block_range: tuple[int, int] | None = None
    chunk_ordinal: int = 0
    content_hash: str | None = None
    table_metadata: Mapping[str, Any] | None = None
    equation_metadata: Mapping[str, Any] | None = None
    embedding_spec: EmbeddingSpec | Mapping[str, Any] | None = None
    parser_id: str | None = None
    parser_version: str | None = None
    parser_config_hash: str | None = None
    chunker_version: str | None = None
    embedding_model_id: str | None = None
    embedding_model_version: str | None = None
    embedding_artifact_hash: str | None = None
    embedding_dimensions: int | None = None
    embedding_max_input_tokens: int | None = None
    embedding_effective_content_token_limit: int | None = None
    embedding_tokenizer_fingerprint: str | None = None
    embedding_normalization: str | None = None
    embedding_truncation: bool = False
    embedding_corpus_instruction_policy: str | None = None
    embedding_corpus_instruction_version: str | None = None
    embedding_query_instruction_policy: str | None = None
    embedding_query_instruction_version: str | None = None
    lexical_index_version: str | None = None
    index_version: str | None = None
    content_tokens: int | None = None
    embedding_input_tokens: int | None = None
    active: bool = True
    vector_ready: bool = False
    lexical_ready: bool = False
    projection_generation: str | None = None
    source_currentness: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_type", _content_type(self.content_type))
        object.__setattr__(self, "text", str(self.text or ""))
        object.__setattr__(self, "authors", _string_tuple(self.authors, "authors"))
        object.__setattr__(self, "section_path", _string_tuple(self.section_path, "section_path"))
        if self.page is not None and self.page_start is None:
            object.__setattr__(self, "page_start", int(self.page))
        if self.page is not None and self.page_end is None:
            object.__setattr__(self, "page_end", int(self.page))
        if self.page is None and self.page_start is not None and self.page_end in (None, self.page_start):
            object.__setattr__(self, "page", int(self.page_start))
        if self.block_range is not None:
            block_range = tuple(int(item) for item in self.block_range)
            if len(block_range) != 2 or block_range[0] < 0 or block_range[1] < block_range[0]:
                raise ValueError("block_range must be a non-negative ordered pair")
            object.__setattr__(self, "block_range", block_range)
        ordinal = int(self.chunk_ordinal)
        if ordinal < 0:
            raise ValueError("chunk_ordinal must be non-negative")
        object.__setattr__(self, "chunk_ordinal", ordinal)
        for name in ("content_tokens", "embedding_input_tokens"):
            value = getattr(self, name)
            if value is not None:
                value = int(value)
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
                object.__setattr__(self, name, value)
        for name in ("table_metadata", "equation_metadata"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _mapping(value))
        if isinstance(self.embedding_spec, Mapping):
            object.__setattr__(self, "embedding_spec", EmbeddingSpec.from_dict(self.embedding_spec))
        if isinstance(self.embedding_spec, EmbeddingSpec):
            spec_fields = {
                "embedding_model_id": self.embedding_spec.model_id,
                "embedding_model_version": self.embedding_spec.resolved_model_version,
                "embedding_artifact_hash": self.embedding_spec.artifact_hash,
                "embedding_dimensions": self.embedding_spec.dimensions,
                "embedding_max_input_tokens": self.embedding_spec.model_max_input_tokens,
                "embedding_effective_content_token_limit": self.embedding_spec.effective_corpus_content_token_limit,
                "embedding_tokenizer_fingerprint": self.embedding_spec.tokenizer_fingerprint,
                "embedding_normalization": self.embedding_spec.normalization_policy,
                "embedding_truncation": self.embedding_spec.truncation,
                "embedding_corpus_instruction_policy": self.embedding_spec.corpus_instruction_policy,
                "embedding_corpus_instruction_version": self.embedding_spec.corpus_instruction_version,
                "embedding_query_instruction_policy": self.embedding_spec.query_instruction_policy,
                "embedding_query_instruction_version": self.embedding_spec.query_instruction_version,
            }
            for name, value in spec_fields.items():
                if getattr(self, name) is None:
                    object.__setattr__(self, name, value)
        object.__setattr__(self, "extra", _mapping(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChunkRecord:
        payload = dict(value)
        payload["content_type"] = _content_type(payload.get("content_type", ContentType.PROSE))
        payload["authors"] = tuple(payload.get("authors") or ())
        payload["section_path"] = tuple(payload.get("section_path") or ())
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class IndexGeneration(SerializableModel):
    """Matched vector/lexical projection generation metadata."""

    generation_id: str = ""
    vector_location: Path | str = ""
    lexical_location: Path | str = ""
    embedding_spec: EmbeddingSpec = field(default_factory=EmbeddingSpec)
    lexical_index_version: str = "fts5-v1"
    lexical_tokenizer_settings: Mapping[str, Any] = field(default_factory=dict)
    index_version: str = "index-v1"
    population_hash: str = ""
    population_identity: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    vector_ready: bool = False
    lexical_ready: bool = False
    status: str = "STAGED"
    created_at: datetime | str | None = None
    activated_at: datetime | str | None = None
    component_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.generation_id).strip():
            raise ValueError("generation_id must be non-empty")
        object.__setattr__(self, "vector_location", Path(self.vector_location))
        object.__setattr__(self, "lexical_location", Path(self.lexical_location))
        if isinstance(self.embedding_spec, Mapping):
            object.__setattr__(self, "embedding_spec", EmbeddingSpec.from_dict(self.embedding_spec))
        object.__setattr__(self, "lexical_tokenizer_settings", _mapping(self.lexical_tokenizer_settings))
        object.__setattr__(self, "component_versions", _mapping(self.component_versions))
        for name in ("document_count", "chunk_count"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    @property
    def vector_path(self) -> Path:
        return Path(self.vector_location)

    @property
    def lexical_path(self) -> Path:
        return Path(self.lexical_location)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IndexGeneration:
        payload = dict(value)
        payload["embedding_spec"] = EmbeddingSpec.from_dict(payload.get("embedding_spec") or {})
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class KnowledgeQuery(SerializableModel):
    """Read-only query request accepted by the knowledge service."""

    text: str = ""
    top_k: int = 10
    content_types: tuple[ContentType, ...] = ()
    document_ids: tuple[str, ...] = ()
    include_stale: bool = False

    def __post_init__(self) -> None:
        text = str(self.text).strip()
        if not text:
            raise ValueError("query text must be non-empty")
        object.__setattr__(self, "text", text)
        if isinstance(self.top_k, bool):
            raise ValueError("top_k must be a positive integer")
        top_k = int(self.top_k)
        if top_k <= 0 or top_k > 1000:
            raise ValueError("top_k must be between 1 and 1000")
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(
            self,
            "content_types",
            tuple(_content_type(item) for item in (self.content_types or ())),
        )
        document_ids = _string_tuple(self.document_ids, "document_ids")
        object.__setattr__(self, "document_ids", document_ids)
        object.__setattr__(self, "include_stale", bool(self.include_stale))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeQuery:
        payload = dict(value)
        payload["content_types"] = tuple(payload.get("content_types") or ())
        payload["document_ids"] = tuple(payload.get("document_ids") or ())
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class KnowledgeHit(SerializableModel):
    """Provenance-complete evidence returned by a knowledge query."""

    chunk_id: str = ""
    document_id: str = ""
    content_type: ContentType = ContentType.PROSE
    text: str = ""
    score: float = 0.0
    semantic_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    source_filename: str | None = None
    source_relative_path: str | None = None
    source_hash: str | None = None
    title: str | None = None
    authors: tuple[str, ...] = ()
    publication_year: int | None = None
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    chapter: str | None = None
    section: str | None = None
    section_path: tuple[str, ...] = ()
    epub_spine_item: str | None = None
    anchor: str | None = None
    parser_version: str | None = None
    chunker_version: str | None = None
    embedding_model_id: str | None = None
    embedding_model_version: str | None = None
    index_version: str | None = None
    table_metadata: Mapping[str, Any] | None = None
    equation_metadata: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_type", _content_type(self.content_type))
        object.__setattr__(self, "authors", _string_tuple(self.authors, "authors"))
        object.__setattr__(self, "section_path", _string_tuple(self.section_path, "section_path"))
        object.__setattr__(self, "score", _finite_float(self.score, "score"))
        for name in ("semantic_score", "lexical_score", "fused_score", "rerank_score"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_float(value, name))
        if self.page is not None and self.page_start is None:
            object.__setattr__(self, "page_start", int(self.page))
        if self.page is not None and self.page_end is None:
            object.__setattr__(self, "page_end", int(self.page))
        if self.page is None and self.page_start is not None and self.page_end in (None, self.page_start):
            object.__setattr__(self, "page", int(self.page_start))
        for name in ("table_metadata", "equation_metadata"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _mapping(value))
        object.__setattr__(self, "extra", _mapping(self.extra))

    @property
    def section_name(self) -> str | None:
        return self.section or (self.section_path[-1] if self.section_path else None)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeHit:
        payload = dict(value)
        payload["content_type"] = _content_type(payload.get("content_type", ContentType.PROSE))
        payload["authors"] = tuple(payload.get("authors") or ())
        payload["section_path"] = tuple(payload.get("section_path") or ())
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class IngestionRunSummary(SerializableModel):
    """Bounded, serializable outcome summary for one explicit ingestion run."""

    run_id: str = ""
    state: str | IngestionState = "SUCCEEDED"
    counts: Mapping[IngestionState | str, int] = field(default_factory=dict)
    started_at: datetime | str | None = None
    completed_at: datetime | str | None = None
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    source_count: int | None = None
    document_count: int | None = None
    chunk_count: int | None = None

    def __post_init__(self) -> None:
        if not str(self.run_id).strip():
            raise ValueError("run_id must be non-empty")
        normalized_counts: dict[IngestionState | str, int] = {}
        for key, count in (self.counts or {}).items():
            try:
                normalized_key: IngestionState | str = IngestionState(key)
            except ValueError:
                normalized_key = str(key)
            value = int(count)
            if value < 0:
                raise ValueError("ingestion counts must be non-negative")
            normalized_counts[normalized_key] = value
        object.__setattr__(self, "counts", normalized_counts)
        object.__setattr__(
            self,
            "diagnostics",
            tuple(_mapping(item) for item in (self.diagnostics or ())),
        )
        for name in ("source_count", "document_count", "chunk_count"):
            value = getattr(self, name)
            if value is not None:
                value = int(value)
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
                object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        # Explicit base dispatch is robust for frozen/slots dataclasses on
        # Python versions where zero-argument ``super()`` loses its class cell.
        result = SerializableModel.to_dict(self)
        result["counts"] = {
            str(_enum_value(key)): value for key, value in self.counts.items()
        }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IngestionRunSummary:
        payload = dict(value)
        payload["counts"] = dict(payload.get("counts") or {})
        payload["diagnostics"] = tuple(payload.get("diagnostics") or ())
        return cls(**payload)


__all__ = [
    "AliasRelation",
    "ChunkRecord",
    "ContentType",
    "DiscoveredResource",
    "DocumentMetadata",
    "EmbeddingSpec",
    "EquationData",
    "EquationMetadata",
    "FigureMetadata",
    "IndexGeneration",
    "IngestionRunSummary",
    "IngestionState",
    "KnowledgeHit",
    "KnowledgeQuery",
    "ParsedBlock",
    "ParsedDocument",
    "ReferenceMetadata",
    "TableData",
    "TableMetadata",
]
