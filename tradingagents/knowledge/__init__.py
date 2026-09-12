"""Local, read-only published-knowledge contracts and discovery boundary."""

from .config import KnowledgeConfig
from .discovery import SourceScanner
from .fusion import (
    DenseCandidate,
    FusedCandidate,
    LexicalCandidate,
    RankedCandidate,
    RRFConfig,
    reciprocal_rank_fuse,
)
from .identity import (
    SourceChangedError,
    canonical_relative_path,
    document_id_for,
    resource_id_for,
    sha256_file,
)
from .models import (
    AliasRelation,
    ChunkRecord,
    ContentType,
    DiscoveredResource,
    DocumentMetadata,
    EmbeddingSpec,
    EquationData,
    EquationMetadata,
    FigureMetadata,
    IndexGeneration,
    IngestionRunSummary,
    IngestionState,
    KnowledgeHit,
    KnowledgeQuery,
    ParsedBlock,
    ParsedDocument,
    ReferenceMetadata,
    TableData,
    TableMetadata,
)
from .provenance import ProvenanceError, validate_hit_provenance
from .query import KnowledgeQueryService
from .reranking import Reranker

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
    "KnowledgeConfig",
    "KnowledgeHit",
    "KnowledgeQuery",
    "KnowledgeQueryService",
    "DenseCandidate",
    "FusedCandidate",
    "LexicalCandidate",
    "RRFConfig",
    "RankedCandidate",
    "Reranker",
    "ProvenanceError",
    "ParsedBlock",
    "ParsedDocument",
    "ReferenceMetadata",
    "SourceChangedError",
    "SourceScanner",
    "TableData",
    "TableMetadata",
    "canonical_relative_path",
    "document_id_for",
    "resource_id_for",
    "sha256_file",
    "reciprocal_rank_fuse",
    "validate_hit_provenance",
]
