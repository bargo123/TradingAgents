"""Deterministic retrieval-quality evidence using only local fixture fakes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.config import KnowledgeConfig
from tradingagents.knowledge.discovery import SourceScanner
from tradingagents.knowledge.embeddings import EmbeddingProvider
from tradingagents.knowledge.index_generation import IndexGenerationManager
from tradingagents.knowledge.ingestion import IngestionMode, KnowledgeIngestor
from tradingagents.knowledge.lexical_index import LexicalIndexWriter
from tradingagents.knowledge.models import (
    ContentType,
    DocumentMetadata,
    EmbeddingSpec,
    IngestionState,
    KnowledgeHit,
    KnowledgeQuery,
    ParsedBlock,
    ParsedDocument,
)
from tradingagents.knowledge.quality import (
    load_cases,
    run_benchmark,
    snapshot_tree,
)
from tradingagents.knowledge.vector_index import VectorIndexWriter

FIXTURE = Path(__file__).parent / "fixtures" / "knowledge" / "benchmark_cases.json"


def _hit(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    content_type: ContentType = ContentType.PROSE,
    extra: dict[str, object] | None = None,
) -> KnowledgeHit:
    return KnowledgeHit(
        chunk_id=chunk_id,
        document_id=document_id,
        content_type=content_type,
        text=text,
        score=1.0,
        source_filename=f"{document_id}.pdf",
        source_relative_path=f"books/{document_id}.pdf",
        source_hash=f"sha256:{document_id}",
        page=1,
        parser_version="fixture-parser-v1",
        chunker_version="fixture-chunker-v1",
        index_version="fixture-index-v1",
        extra=extra if extra is not None else {"projection_generation": "fixture-generation"},
    )


@dataclass(frozen=True)
class FixtureQueryService:
    """Read-only local fake whose returned evidence mirrors the public contract."""

    rows: dict[str, tuple[KnowledgeHit, ...]]

    def search(self, request: KnowledgeQuery) -> tuple[KnowledgeHit, ...]:
        rows = self.rows[request.text]
        return tuple(
            hit
            for hit in rows
            if not request.content_types or hit.content_type in request.content_types
        )[: request.top_k]


def fixture_query_service() -> FixtureQueryService:
    rows = {
        "order flow imbalance short horizon prediction": (
            _hit("ofi-1", "doc-ofi", "Order flow imbalance predicts short horizon movement."),
        ),
        "market making inventory risk": (
            _hit("mm-1", "doc-mm", "Avellaneda-Stoikov inventory risk equation.", content_type=ContentType.EQUATION),
        ),
        "queue imbalance price movement": (
            _hit("queue-1", "doc-queue", "Queue imbalance forecasts price movement."),
        ),
        "FX order flow exchange rate microstructure": (
            _hit("fx-1", "doc-fx", "FX order flow explains exchange rate microstructure."),
        ),
        "adverse selection in high-frequency market making": (
            _hit("mm-2", "doc-mm", "Adverse selection in high-frequency market making."),
        ),
        "Hawkes process high frequency order arrivals": (
            _hit("hawkes-1", "doc-hawkes", "Hawkes process models high frequency order arrivals."),
        ),
        "VPIN": (_hit("vpin-1", "doc-vpin", "VPIN measures volume-synchronized toxicity."),),
        "DeepLOB": (_hit("deeplob-1", "doc-deeplob", "DeepLOB learns limit-order-book features."),),
        "Avellaneda-Stoikov": (
            _hit("as-1", "doc-mm", "Avellaneda-Stoikov market making model."),
        ),
        "Kyle lambda": (_hit("kyle-1", "doc-kyle", "Kyle lambda measures price impact."),),
    }
    return FixtureQueryService(rows)


class _Tokenizer:
    model_max_input_tokens = 16

    def encode(self, text, *, add_special_tokens, purpose, truncation):
        assert truncation is False
        return list(range(len(str(text).split()) + (2 if add_special_tokens else 0)))


class _Embedder(EmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(
            spec=EmbeddingSpec(
                model_id="fixture/quality",
                resolved_model_version="v1",
                runtime="fixture",
                artifact_hash="fixture-quality-hash",
                dimensions=2,
                tokenizer_fingerprint="fixture-quality-tokenizer",
                model_max_input_tokens=16,
                special_token_budget=2,
                effective_corpus_content_token_limit=14,
            ),
            tokenizer=_Tokenizer(),
        )

    def _embed_formatted(self, texts):
        return tuple((1.0, float(index)) for index, _ in enumerate(texts))


class _Parser:
    parser_id = "fixture-quality-parser"
    parser_version = "v1"
    parser_config_hash = "fixture-quality-config"

    def parse(self, resource, staging_dir):
        return ParsedDocument(
            document_id=f"doc-{resource.source_hash[:12]}",
            source_hash=resource.source_hash,
            metadata=DocumentMetadata(
                source_filename=resource.path.name,
                source_relative_path=resource.relative_path,
                format=resource.format,
            ),
            blocks=(
                ParsedBlock(
                    block_id="body",
                    content_type=ContentType.PROSE,
                    text="order flow imbalance",
                    reading_order=0,
                ),
            ),
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            parser_config_hash=self.parser_config_hash,
        )


class _VectorBackend:
    identity = "fixture-quality-vector"

    def write(self, location, rows):
        import json

        location.mkdir(parents=True, exist_ok=True)
        (location / "rows.json").write_text(json.dumps(rows), encoding="utf-8")

    def read(self, location):
        import json

        return tuple(json.loads((location / "rows.json").read_text(encoding="utf-8")))


def _ingestion_harness(source: Path, artifact: Path) -> KnowledgeIngestor:
    config = KnowledgeConfig(source_root=source, artifact_root=artifact)
    catalog = KnowledgeCatalog(artifact / "catalog.sqlite3")
    embedder = _Embedder()
    manager = IndexGenerationManager(
        config=config,
        catalog=catalog,
        embedding_spec=embedder.spec,
        vector_writer=VectorIndexWriter(backend=_VectorBackend()),
        lexical_writer=LexicalIndexWriter(),
    )
    return KnowledgeIngestor(
        config,
        scanner=SourceScanner(config),
        catalog=catalog,
        parser=_Parser(),
        embedder=embedder,
        generation_manager=manager,
    )


def test_hft_benchmark_reports_retrieval_metrics_and_provenance():
    """Would fail if ranks, exact terminology, or provenance validation regress."""
    report = run_benchmark(load_cases(FIXTURE), fixture_query_service())

    assert report.recall_at == {1: 1.0, 5: 1.0, 10: 1.0}
    assert report.mrr == 1.0
    assert report.exact_keyword_hit_rate == 1.0
    assert report.content_type_filter_precision == 1.0
    assert report.provenance_correctness == 1.0
    assert report.duplicate_result_rate == 0.0
    assert report.deterministic_ordering is True


def test_benchmark_zero_denominators_have_stable_zero_metrics():
    """Would fail if an empty labeled subset raises or reports a misleading rate."""
    report = run_benchmark(
        ({"query": "empty", "documents": [], "terms": []},),
        FixtureQueryService({"empty": ()}),
    )

    assert report.recall_at == {1: 0.0, 5: 0.0, 10: 0.0}
    assert report.mrr == 0.0
    assert report.exact_keyword_hit_rate == 0.0
    assert report.content_type_filter_precision == 0.0
    assert report.provenance_correctness == 0.0
    assert report.duplicate_result_rate == 0.0


def test_benchmark_rejects_duplicate_or_nondeterministic_ranked_results():
    """Would fail if duplicate chunks or changing ranking order are treated as valid."""

    class UnstableService:
        calls = 0

        def search(self, request: KnowledgeQuery) -> tuple[KnowledgeHit, ...]:
            self.calls += 1
            first = _hit("same", "doc-a", "exact term")
            second = _hit("other", "doc-a", "exact term")
            return (first, first) if self.calls % 2 else (second, first)

    report = run_benchmark(
        ({"query": "exact term", "documents": ["doc-a"], "terms": ["exact term"]},),
        UnstableService(),
    )

    assert report.duplicate_result_rate == 0.5
    assert report.deterministic_ordering is False


def test_indexing_does_not_change_source_bytes_or_metadata(tmp_path):
    """Would fail if any indexing operation writes to or retimestamps a source file."""
    source = tmp_path / "source"
    source.mkdir()
    book = source / "book.pdf"
    book.write_bytes(b"immutable source bytes")
    before = snapshot_tree(source)

    artifact = tmp_path / "artifacts"
    summary = _ingestion_harness(source, artifact).run(IngestionMode.INCREMENTAL)

    assert summary.counts[IngestionState.INDEXED] == 1
    assert snapshot_tree(source) == before


def test_benchmark_results_expose_no_action_or_trade_fields():
    """Would fail if a read-only knowledge result starts carrying decision payloads."""
    report = run_benchmark(load_cases(FIXTURE), fixture_query_service())

    assert report.forbidden_result_fields == ()


def test_benchmark_recursively_rejects_nested_experience_and_action_fields():
    """Would fail if decision or experience payloads hide inside serialized metadata."""
    hit = _hit(
        "nested-1",
        "doc-nested",
        "order flow evidence",
        extra={"metadata": {"experience": {"trade_action": "BUY"}}},
    )
    report = run_benchmark(
        ({"query": "nested", "documents": ["doc-nested"]},),
        FixtureQueryService({"nested": (hit,)}),
    )

    assert report.forbidden_result_fields == (
        "extra.metadata.experience",
        "extra.metadata.experience.trade_action",
    )
