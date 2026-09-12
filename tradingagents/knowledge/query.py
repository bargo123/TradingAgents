"""Read-only local hybrid knowledge retrieval service."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .embeddings import EmbeddingSpecMismatch
from .fusion import DenseCandidate, LexicalCandidate, reciprocal_rank_fuse
from .index_generation import IncompatibleIndexGeneration
from .models import ChunkRecord, ContentType, EmbeddingSpec, KnowledgeHit, KnowledgeQuery
from .provenance import validate_hit_provenance
from .reranking import Reranker


def _spec(value: Any) -> EmbeddingSpec:
    if isinstance(value, EmbeddingSpec):
        return value
    if isinstance(value, Mapping):
        return EmbeddingSpec.from_dict(value)
    raise EmbeddingSpecMismatch("complete embedding specification is required")


def _metadata(reader: Any) -> Mapping[str, Any]:
    value = reader.metadata() if callable(getattr(reader, "metadata", None)) else getattr(reader, "metadata", {})
    return value if isinstance(value, Mapping) else {}


def _call_search(reader: Any, first: Any, request: KnowledgeQuery, limit: int) -> Sequence[Any]:
    method = getattr(reader, "search", None)
    if not callable(method):
        raise TypeError("knowledge index reader must expose search()")
    kwargs = {"limit": limit, "document_ids": request.document_ids, "content_types": request.content_types}
    try:
        signature = inspect.signature(method)
        accepts_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        if not accepts_kwargs:
            kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
    except (TypeError, ValueError):
        pass
    return tuple(method(first, **kwargs))


def _chunk(item: Any) -> ChunkRecord | None:
    if isinstance(item, ChunkRecord):
        return item
    if isinstance(item, Mapping):
        value = item.get("chunk", item.get("record", item.get("payload")))
        if isinstance(value, ChunkRecord):
            return value
        if isinstance(value, Mapping):
            return ChunkRecord.from_dict(value)
        if item.get("chunk_id"):
            try:
                payload = {
                    key: value
                    for key, value in item.items()
                    if key in ChunkRecord.__dataclass_fields__
                }
                for key, target in (("authors_json", "authors"), ("section_path_json", "section_path")):
                    if target not in payload and key in item:
                        payload[target] = tuple(json.loads(item[key]))
                return ChunkRecord.from_dict(payload)
            except (TypeError, ValueError):
                return None
    return None


def _candidate(item: Any, cls: type[DenseCandidate] | type[LexicalCandidate], rank: int) -> Any:
    if isinstance(item, cls):
        return item
    if isinstance(item, Mapping):
        chunk = _chunk(item)
        score_key = "semantic_score" if cls is DenseCandidate else "lexical_score"
        score = item.get(score_key, item.get("score", 0.0))
        return cls(
            chunk_id=str(item.get("chunk_id") or getattr(chunk, "chunk_id", "")),
            rank=int(item.get("rank", rank)),
            score=float(score),
            chunk=chunk,
            metadata=item,
        )
    chunk = _chunk(item)
    if chunk is None:
        raise TypeError(f"unsupported candidate value: {type(item).__name__}")
    return cls(chunk_id=chunk.chunk_id, rank=rank, score=0.0, chunk=chunk, metadata=chunk.to_dict())


def _allowed(candidate: Any, request: KnowledgeQuery, catalog: Any) -> bool:
    chunk = candidate.chunk
    document_id = str(getattr(chunk, "document_id", "") if chunk is not None else candidate.metadata.get("document_id", ""))
    if request.document_ids and document_id not in request.document_ids:
        return False
    content_type = getattr(chunk, "content_type", None) if chunk is not None else candidate.metadata.get("content_type")
    if request.content_types:
        try:
            if ContentType(content_type) not in request.content_types:
                return False
        except (TypeError, ValueError):
            return False
    if chunk is not None and (
        not chunk.active or not chunk.vector_ready or not chunk.lexical_ready
    ):
        return False
    if isinstance(candidate.metadata, Mapping) and candidate.metadata.get("active") is False:
        return False
    ready = getattr(catalog, "document_is_retrieval_ready", None)
    if callable(ready):
        return bool(ready(document_id))
    active = getattr(catalog, "document_is_active", None)
    return bool(active(document_id)) if callable(active) else True


class KnowledgeQueryService:
    """Search one catalog-authoritative, matched local index generation."""

    def __init__(self, vector_reader: Any, lexical_reader: Any, catalog: Any, embedder: Any, reranker: Reranker | None = None) -> None:
        self.vector_reader = vector_reader
        self.lexical_reader = lexical_reader
        self.catalog = catalog
        self.embedder = embedder
        self.reranker = reranker or Reranker()

    def _generation(self) -> Any:
        generation = self.catalog.active_generation()
        if generation is None:
            raise IncompatibleIndexGeneration("no active matched knowledge generation")
        if not generation.vector_ready or not generation.lexical_ready:
            raise IncompatibleIndexGeneration("active generation is missing a complete projection pair")
        if getattr(generation, "status", "VALIDATED") != "VALIDATED":
            raise IncompatibleIndexGeneration("active generation is not validated")
        vector_meta = _metadata(self.vector_reader)
        lexical_meta = _metadata(self.lexical_reader)
        expected_id = generation.generation_id
        ids = (vector_meta.get("generation_id"), lexical_meta.get("generation_id"), getattr(self.vector_reader, "generation_id", expected_id), getattr(self.lexical_reader, "generation_id", expected_id))
        if any(value != expected_id for value in ids if value is not None):
            raise IncompatibleIndexGeneration("vector and lexical generations do not match the active generation")
        for metadata in (vector_meta, lexical_meta):
            if metadata.get("population_hash") not in (None, generation.population_hash):
                raise IncompatibleIndexGeneration("projection population does not match the active generation")
            stored_spec = metadata.get("embedding_spec")
            if stored_spec is not None and _spec(stored_spec).to_dict() != _spec(generation.embedding_spec).to_dict():
                raise EmbeddingSpecMismatch("projection embedding specification differs from active generation")
        for reader in (self.vector_reader, self.lexical_reader):
            reader_spec = getattr(reader, "embedding_spec", None)
            if reader_spec is not None and _spec(reader_spec).to_dict() != _spec(generation.embedding_spec).to_dict():
                raise EmbeddingSpecMismatch("reader embedding specification differs from active generation")
        return generation

    @staticmethod
    def _assert_specs(expected: EmbeddingSpec, actual: Any) -> None:
        actual = _spec(actual)
        if expected.to_dict() != actual.to_dict():
            raise EmbeddingSpecMismatch("query embedding specification differs from active index")
        if not actual.model_id.strip() or not actual.resolved_model_version.strip():
            raise EmbeddingSpecMismatch("embedding model identity must be non-empty")

    def _query_vector(self, request: KnowledgeQuery, spec: EmbeddingSpec) -> Sequence[float]:
        self._assert_specs(
            spec,
            getattr(self.embedder, "spec", getattr(self.embedder, "embedding_spec", None)),
        )
        embed = self.embedder.embed
        try:
            params = inspect.signature(embed).parameters
            kwargs = {"purpose": "query"} if "purpose" in params else {}
            if "truncation" in params:
                kwargs["truncation"] = False
        except (TypeError, ValueError):
            kwargs = {"purpose": "query"}
        result = embed((request.text,), **kwargs)
        if not result:
            raise EmbeddingSpecMismatch("query embedder returned no vector")
        first = result[0]
        return result if isinstance(first, (int, float)) else first

    @staticmethod
    def _hit(candidate: Any) -> KnowledgeHit:
        chunk = candidate.chunk
        if chunk is None:
            chunk = ChunkRecord.from_dict(candidate.metadata)
        extra = dict(getattr(chunk, "extra", {}) or {})
        extra.setdefault(
            "projection_generation",
            candidate.metadata.get("projection_generation", getattr(chunk, "projection_generation", "")),
        )
        extra.setdefault("projection_population_hash", candidate.metadata.get("projection_population_hash", ""))
        extra.setdefault("fusion_version", "rrf-v1")
        extra.setdefault("reranker_version", "feature-v1")
        payload = chunk.to_dict()
        payload.update(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content_type=chunk.content_type,
            text=chunk.text,
            score=candidate.rerank_score if candidate.rerank_score is not None else candidate.fused_score,
            semantic_score=candidate.semantic_score,
            lexical_score=candidate.lexical_score,
            fused_score=candidate.fused_score,
            rerank_score=candidate.rerank_score,
            extra=extra,
        )
        # KnowledgeHit intentionally exposes only published evidence fields;
        # discard chunk-only indexing fields while retaining all provenance.
        fields = set(KnowledgeHit.__dataclass_fields__)
        return KnowledgeHit(**{key: value for key, value in payload.items() if key in fields})

    def search(self, request: KnowledgeQuery) -> tuple[KnowledgeHit, ...]:
        if not isinstance(request, KnowledgeQuery):
            raise TypeError("search expects a KnowledgeQuery")
        generation = self._generation()
        spec = _spec(generation.embedding_spec)
        vector = self._query_vector(request, spec)
        limit = min(200, max(50, request.top_k * 5))
        if callable(getattr(self.vector_reader, "search", None)):
            dense_rows = _call_search(self.vector_reader, vector, request, limit)
        else:
            dense_rows = self._search_vector_rows(vector, request, limit)
        dense = tuple(_candidate(item, DenseCandidate, index) for index, item in enumerate(dense_rows, 1))
        lexical = tuple(_candidate(item, LexicalCandidate, index) for index, item in enumerate(
            _call_search(self.lexical_reader, request.text, request, limit), 1
        ))
        dense = tuple(item for item in dense if _allowed(item, request, self.catalog))
        lexical = tuple(item for item in lexical if _allowed(item, request, self.catalog))
        fused = reciprocal_rank_fuse(dense, lexical)
        ranked = self.reranker.rerank(request, fused)
        hits: list[KnowledgeHit] = []
        for candidate in ranked[: request.top_k]:
            hit = self._hit(candidate)
            validate_hit_provenance(hit)
            hits.append(hit)
        return tuple(hits)

    def _search_vector_rows(self, vector: Sequence[float], request: KnowledgeQuery, limit: int) -> tuple[Mapping[str, Any], ...]:
        """Small fallback for the metadata/rows-only VectorIndexReader adapter."""
        rows = self.vector_reader.rows()
        scored: list[dict[str, Any]] = []
        norm = math.sqrt(sum(float(value) ** 2 for value in vector)) or 1.0
        for row in rows:
            values = row.get("vector") if isinstance(row, Mapping) else None
            if not values:
                continue
            row_norm = math.sqrt(sum(float(value) ** 2 for value in values)) or 1.0
            score = sum(float(left) * float(right) for left, right in zip(vector, values, strict=False)) / (norm * row_norm)
            scored.append({**row, "semantic_score": score})
        scored.sort(key=lambda row: (-float(row["semantic_score"]), str(row.get("chunk_id", ""))))
        for index, row in enumerate(scored, 1):
            row["rank"] = index
        return tuple(scored[:limit])


__all__ = ["KnowledgeQueryService"]
