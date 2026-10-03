from __future__ import annotations

from dataclasses import replace
from math import inf, nan
from pathlib import Path

import numpy as np
import pytest

from tradingagents.knowledge.embeddings import EmbeddingSpecMismatch
from tradingagents.knowledge.fusion import (
    DenseCandidate,
    LexicalCandidate,
    RRFConfig,
    reciprocal_rank_fuse,
)
from tradingagents.knowledge.index_generation import IncompatibleIndexGeneration
from tradingagents.knowledge.models import (
    ChunkRecord,
    ContentType,
    EmbeddingSpec,
    IndexGeneration,
    KnowledgeHit,
    KnowledgeQuery,
)
from tradingagents.knowledge.provenance import ProvenanceError, validate_hit_provenance
from tradingagents.knowledge.query import KnowledgeQueryService
from tradingagents.knowledge.reranking import Reranker


@pytest.mark.parametrize("score", [nan, inf, -inf])
def test_fusion_candidates_reject_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        DenseCandidate("chunk-a", 1, score)
    with pytest.raises(ValueError, match="finite"):
        LexicalCandidate("chunk-a", 1, score)


@pytest.mark.parametrize("candidate_type", [DenseCandidate, LexicalCandidate])
@pytest.mark.parametrize("chunk_id", [None, "", "   "])
def test_fusion_candidates_reject_empty_chunk_ids(candidate_type, chunk_id: object) -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        candidate_type(chunk_id, 1, 1.0)


@pytest.mark.parametrize("candidate_type", [DenseCandidate, LexicalCandidate])
@pytest.mark.parametrize("rank", [True, 0, -1, 1.5, "1"])
def test_fusion_candidates_reject_non_positive_integer_ranks(candidate_type, rank: object) -> None:
    with pytest.raises(ValueError, match="rank"):
        candidate_type("chunk-a", rank, 1.0)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "60"])
def test_rrf_config_rejects_non_positive_integer_k(value: object) -> None:
    with pytest.raises(ValueError, match="RRF k"):
        RRFConfig(k=value)


def make_embedding_spec(**overrides: object) -> EmbeddingSpec:
    values: dict[str, object] = {
        "model_id": "fixture/bge",
        "resolved_model_version": "fixture-v1",
        "runtime": "onnx-cpu",
        "artifact_hash": "sha256:fixture-model",
        "dimensions": 3,
        "normalization_policy": "l2",
        "tokenizer_fingerprint": "sha256:fixture-tokenizer",
        "model_max_input_tokens": 512,
        "special_token_budget": 2,
        "effective_corpus_content_token_limit": 510,
        "corpus_instruction_policy": "none-v1",
        "corpus_instruction_version": "v1",
        "query_instruction_policy": "bge-search-prefix-v1",
        "query_instruction_version": "v1",
        "truncation": False,
    }
    values.update(overrides)
    return EmbeddingSpec(**values)


def make_chunk(
    chunk_id: str = "chunk-a",
    document_id: str = "doc-1",
    *,
    text: str = "VPIN measures inventory risk.",
    content_type: ContentType = ContentType.EQUATION,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        source_hash=f"sha256:{document_id}",
        text=text,
        content_type=content_type,
        source_filename=f"{document_id}.pdf",
        source_relative_path=f"books/{document_id}.pdf",
        title=document_id.title(),
        authors=("Author",),
        page_start=3,
        page_end=3,
        chapter="1",
        section_path=("Risk",),
        parser_version="parser-v1",
        parser_id="fixture-parser",
        chunker_version="structure-v2",
        index_version="index-v1",
        embedding_spec=make_embedding_spec(),
        projection_generation="gen-1",
        active=True,
        vector_ready=True,
        lexical_ready=True,
    )


def make_hit(**overrides: object) -> KnowledgeHit:
    values: dict[str, object] = {
        "chunk_id": "chunk-a",
        "document_id": "doc-1",
        "content_type": ContentType.PROSE,
        "text": "VPIN measures inventory risk.",
        "score": 1.0,
        "source_filename": "doc-1.pdf",
        "source_relative_path": "books/doc-1.pdf",
        "source_hash": "sha256:doc-1",
        "page": 3,
        "parser_version": "parser-v1",
        "chunker_version": "structure-v2",
        "index_version": "index-v1",
        "extra": {"projection_generation": "gen-1"},
    }
    values.update(overrides)
    return KnowledgeHit(**values)


class FakeVectorReader:
    def __init__(self, rows=(), *, generation_id="gen-1", embedding_spec=None, fail_if_called=False):
        self.rows = tuple(rows)
        self.generation_id = generation_id
        self.embedding_spec = embedding_spec or make_embedding_spec()
        self.fail_if_called = fail_if_called
        self.calls = []

    def metadata(self):
        return {
            "generation_id": self.generation_id,
            "population_hash": "pop-1",
            "embedding_spec": self.embedding_spec.to_dict(),
            "lexical_index_version": "fts5-v1",
            "index_version": "index-v1",
        }

    def search(self, vector, *, limit=50, document_ids=(), content_types=()):
        if self.fail_if_called:
            raise AssertionError("dense reader must not be called")
        self.calls.append((vector, limit, document_ids, content_types))
        return self.rows


class MissingMetadataVectorReader(FakeVectorReader):
    def metadata(self):
        return {}


class RowsOnlyMalformedVectorReader:
    def __init__(self, rows):
        self._rows = tuple(rows)

    def metadata(self):
        return {
            "generation_id": "gen-1",
            "population_hash": "pop-1",
            "embedding_spec": make_embedding_spec().to_dict(),
            "lexical_index_version": "fts5-v1",
            "index_version": "index-v1",
        }

    def rows(self):
        return self._rows


class FakeLexicalReader:
    def __init__(self, rows=(), *, generation_id="gen-1", embedding_spec=None):
        self.rows = tuple(rows)
        self.generation_id = generation_id
        self.embedding_spec = embedding_spec or make_embedding_spec()
        self.calls = []

    def metadata(self):
        return {
            "generation_id": self.generation_id,
            "population_hash": "pop-1",
            "embedding_spec": self.embedding_spec.to_dict(),
            "lexical_index_version": "fts5-v1",
            "index_version": "index-v1",
        }

    def search(self, query, *, limit=50, document_ids=(), content_types=()):
        self.calls.append((query, limit, document_ids, content_types))
        return self.rows


class FakeCatalog:
    def __init__(self, chunks=(), generation_id="gen-1"):
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.generation = IndexGeneration(
            generation_id=generation_id,
            vector_location=Path("vector") / generation_id,
            lexical_location=Path("keyword") / generation_id / "bm25.sqlite3",
            embedding_spec=make_embedding_spec(),
            population_hash="pop-1",
            population_identity='{"chunk_population_hash":"chunk-pop","document_population_hash":"doc-pop","population_hash":"pop-1"}',
            vector_ready=True,
            lexical_ready=True,
            status="VALIDATED",
            component_versions={"vector": "v1", "lexical": "v1"},
        )

    def active_generation(self):
        return self.generation

    def document_is_retrieval_ready(self, document_id):
        return any(chunk.document_id == document_id and chunk.active for chunk in self.chunks.values())


class FakeEmbedder:
    def __init__(self, spec=None):
        self.spec = spec or make_embedding_spec()
        self.embed_calls = 0

    def embed(self, texts, *, purpose="query"):
        self.embed_calls += 1
        return ((0.1, 0.2, 0.3),)


class OversizedEmbedder(FakeEmbedder):
    def embed(self, texts, *, purpose="query"):
        self.embed_calls += 1
        return ((10**1000, 0.2, 0.3),)


def query_harness(
    *, dense=(), lexical=(), vector_generation="gen-1", lexical_generation="gen-1",
    index_embedding_spec=None, query_embedding_spec=None, dense_reader=None,
):
    chunks = tuple(
        item if isinstance(item, ChunkRecord) else item.get("chunk", make_chunk(item["chunk_id"]))
        for item in (*dense, *lexical)
    )
    catalog = FakeCatalog(chunks, vector_generation)
    vector = dense_reader or FakeVectorReader(dense, generation_id=vector_generation, embedding_spec=index_embedding_spec)
    lexical_reader = FakeLexicalReader(lexical, generation_id=lexical_generation, embedding_spec=index_embedding_spec)
    embedder = FakeEmbedder(query_embedding_spec or index_embedding_spec)
    return KnowledgeQueryService(vector, lexical_reader, catalog, embedder)


def test_exact_term_uses_lexical_candidate_and_hybrid_order_is_deterministic():
    service = query_harness(
        dense=({"chunk_id": "chunk-b", "semantic_score": 0.99, "rank": 1, "chunk": make_chunk("chunk-b", text="semantic inventory risk")},),
        lexical=({"chunk_id": "chunk-a", "lexical_score": -9.0, "rank": 1, "chunk": make_chunk("chunk-a")},),
    )

    result = service.search(KnowledgeQuery(text="VPIN", top_k=2))

    assert [item.chunk_id for item in result] == ["chunk-a", "chunk-b"]
    assert result[0].lexical_score is not None
    assert result[0].source_hash
    assert result[0].page is not None


def test_content_type_and_document_filters_are_applied_before_return():
    service = query_harness(
        dense=({"chunk_id": "chunk-a", "semantic_score": 0.9, "rank": 1, "chunk": make_chunk()},),
        lexical=({"chunk_id": "chunk-a", "lexical_score": -2.0, "rank": 1, "chunk": make_chunk()},),
    )
    result = service.search(
        KnowledgeQuery(text="inventory risk", content_types=(ContentType.EQUATION,), document_ids=("doc-1",))
    )
    assert result
    assert all(item.content_type is ContentType.EQUATION for item in result)
    assert all(item.document_id == "doc-1" for item in result)


def test_anonymous_hit_and_generation_mismatch_fail_closed():
    with pytest.raises(ProvenanceError):
        validate_hit_provenance(make_hit(source_hash=None))
    with pytest.raises(ProvenanceError):
        validate_hit_provenance(make_hit(source_filename=None))
    with pytest.raises(ProvenanceError):
        validate_hit_provenance(make_hit(source_relative_path=None))
    with pytest.raises(IncompatibleIndexGeneration):
        query_harness(vector_generation="gen-a", lexical_generation="gen-b").search(KnowledgeQuery(text="OFI"))


def test_query_embedder_rejects_oversized_numeric_vector():
    service = query_harness()
    service.embedder = OversizedEmbedder(service.embedder.spec)

    with pytest.raises(EmbeddingSpecMismatch, match="malformed vector"):
        service.search(KnowledgeQuery(text="OFI"))


def test_rows_only_vector_reader_rejects_oversized_numeric_vector():
    reader = RowsOnlyMalformedVectorReader(
        ({"chunk_id": "chunk-a", "vector": (10**1000, 0.2, 0.3), "chunk": make_chunk()},)
    )
    service = query_harness(dense_reader=reader)

    with pytest.raises(IncompatibleIndexGeneration, match="malformed embedding"):
        service.search(KnowledgeQuery(text="OFI"))


def test_query_rejects_oversized_candidate_score():
    service = query_harness(
        dense=(
            {
                "chunk_id": "chunk-a",
                "semantic_score": 10**1000,
                "rank": 1,
                "chunk": make_chunk(),
            },
        )
    )

    with pytest.raises(ValueError, match="finite"):
        service.search(KnowledgeQuery(text="OFI"))


def test_missing_projection_metadata_fails_closed_before_retrieval():
    reader = MissingMetadataVectorReader(
        ({"chunk_id": "chunk-a", "semantic_score": 0.9, "rank": 1, "chunk": make_chunk()},)
    )
    service = query_harness(dense_reader=reader)

    with pytest.raises(IncompatibleIndexGeneration):
        service.search(KnowledgeQuery(text="OFI"))


def test_candidate_from_stale_projection_generation_fails_closed():
    stale = replace(make_chunk(), projection_generation="gen-old")
    service = query_harness(
        dense=({"chunk_id": stale.chunk_id, "semantic_score": 0.9, "rank": 1, "chunk": stale},)
    )
    with pytest.raises(IncompatibleIndexGeneration):
        service.search(KnowledgeQuery(text="OFI"))


def test_candidate_from_stale_projection_population_fails_closed():
    service = query_harness(
        dense=(
            {
                "chunk_id": "chunk-a",
                "semantic_score": 0.9,
                "rank": 1,
                "projection_population_hash": "pop-old",
                "chunk": make_chunk(),
            },
        )
    )
    with pytest.raises(IncompatibleIndexGeneration):
        service.search(KnowledgeQuery(text="OFI"))


def test_candidate_chunk_identity_mismatch_fails_closed():
    service = query_harness(
        dense=(
            {
                "chunk_id": "claimed-chunk",
                "semantic_score": 0.9,
                "rank": 1,
                "chunk": make_chunk("actual-chunk"),
            },
        )
    )

    with pytest.raises(ProvenanceError, match="chunk identity"):
        service.search(KnowledgeQuery(text="OFI"))


def test_malformed_query_vector_fails_before_dense_retrieval():
    service = query_harness(
        dense_reader=FakeVectorReader(fail_if_called=True),
    )
    service.embedder.embed = lambda _texts, **_kwargs: ((0.1, float("nan")),)

    with pytest.raises(EmbeddingSpecMismatch):
        service.search(KnowledgeQuery(text="OFI"))


def test_query_vector_accepts_numpy_batch_shape() -> None:
    service = query_harness(
        dense=(
            {"chunk_id": "chunk-a", "semantic_score": 0.9, "rank": 1, "chunk": make_chunk()},
        )
    )
    service.embedder.embed = lambda _texts, **_kwargs: np.array([[0.1, 0.2, 0.3]])

    assert service.search(KnowledgeQuery(text="OFI"))


def test_malformed_rows_only_vector_fails_closed():
    reader = RowsOnlyMalformedVectorReader(
        ({"chunk_id": "chunk-a", "vector": (0.1, 0.2)},),
    )
    service = query_harness(dense_reader=reader)

    with pytest.raises(IncompatibleIndexGeneration):
        service.search(KnowledgeQuery(text="OFI"))


def test_identical_query_and_index_embedding_specs_allow_dense_search():
    spec = make_embedding_spec()
    service = query_harness(
        dense=({"chunk_id": "chunk-a", "semantic_score": 0.9, "rank": 1, "chunk": make_chunk()},),
        index_embedding_spec=spec,
        query_embedding_spec=spec,
    )
    assert service.search(KnowledgeQuery(text="OFI"))


def test_hit_carries_validated_population_identity_when_candidate_omits_it():
    service = query_harness(
        dense=({"chunk_id": "chunk-a", "semantic_score": 0.9, "rank": 1, "chunk": make_chunk()},),
    )

    hit = service.search(KnowledgeQuery(text="OFI"))[0]

    assert hit.extra["projection_generation"] == "gen-1"
    assert hit.extra["projection_population_hash"] == "pop-1"


@pytest.mark.parametrize(
    "field",
    [
        "model_id", "resolved_model_version", "artifact_hash", "normalization_policy",
        "tokenizer_fingerprint", "model_max_input_tokens", "query_instruction_policy",
        "query_instruction_version",
    ],
)
def test_embedding_spec_mismatch_fails_before_dense_retrieval(field):
    index_spec = make_embedding_spec()
    value = 513 if field == "model_max_input_tokens" else f"different-{field}"
    query_spec = make_embedding_spec(**{field: value})
    dense = FakeVectorReader(fail_if_called=True, embedding_spec=index_spec)
    service = query_harness(index_embedding_spec=index_spec, query_embedding_spec=query_spec, dense_reader=dense)
    with pytest.raises(EmbeddingSpecMismatch):
        service.search(KnowledgeQuery(text="inventory risk"))


def test_same_dimensions_do_not_make_different_specs_compatible():
    index_spec = make_embedding_spec(model_id="model-a", dimensions=384)
    query_spec = make_embedding_spec(model_id="model-b", dimensions=384)
    with pytest.raises(EmbeddingSpecMismatch):
        query_harness(index_embedding_spec=index_spec, query_embedding_spec=query_spec).search(KnowledgeQuery(text="microprice"))


def test_search_constructs_read_only_query_embedder_once():
    service = query_harness(
        dense=({"chunk_id": "chunk-a", "semantic_score": 0.9, "rank": 1, "chunk": make_chunk()},),
    )
    result = service.search(KnowledgeQuery(text="queue imbalance"))
    assert result
    assert service.embedder.embed_calls == 1


def test_rrf_retains_single_list_candidates_and_expected_scores():
    fused = reciprocal_rank_fuse(
        (DenseCandidate("dense", rank=1, semantic_score=0.9),),
        (LexicalCandidate("lexical", rank=2, lexical_score=-3.0),),
        RRFConfig(k=60),
    )
    assert [item.chunk_id for item in fused] == ["dense", "lexical"]
    assert fused[0].fused_score == pytest.approx(1 / 61)
    assert fused[1].fused_score == pytest.approx(1 / 62)


def test_reranker_prefers_exact_phrase_and_is_deterministic():
    query = KnowledgeQuery(text="VPIN")
    first = make_chunk("chunk-a", text="VPIN measures inventory risk.")
    second = make_chunk("chunk-b", text="Inventory risk is discussed.")
    candidates = reciprocal_rank_fuse(
        (DenseCandidate("chunk-b", rank=1, semantic_score=0.9, chunk=second),),
        (LexicalCandidate("chunk-a", rank=1, lexical_score=-4.0, chunk=first),),
    )
    result = Reranker().rerank(query, candidates)
    assert [item.chunk_id for item in result] == ["chunk-a", "chunk-b"]
    assert result == Reranker().rerank(query, candidates)
