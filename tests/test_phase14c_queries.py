from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tradingagents.knowledge.models import ContentType, KnowledgeHit, KnowledgeQuery
from tradingagents.self_enhancement import phase14c_queries
from tradingagents.self_enhancement.phase14c_models import DiscoveryQuery
from tradingagents.self_enhancement.phase14c_queries import (
    DISCOVERY_QUERY_BANK_VERSION,
    build_discovery_query_bank,
    query_bank_fingerprint,
    retrieve_discovery_evidence,
)

GENERATION_ID = "gen_607de64268a04a6ab09ffa1e160fc280"
POPULATION_HASH = "sha256:fd3ee0c846f2969747aca70576f58e55ffe1c07e8190570d2f9f7e2cb20b9e17"
SOURCE_HASH = "a" * 64


def _hit(*, extra=None, source_hash=SOURCE_HASH, text="A supported market mechanism.") -> KnowledgeHit:
    return KnowledgeHit(
        chunk_id="chunk-1",
        document_id="doc-1",
        content_type=ContentType.PROSE,
        text=text,
        score=0.9,
        semantic_score=0.8,
        lexical_score=0.7,
        fused_score=0.6,
        rerank_score=0.9,
        source_filename="book.pdf",
        source_relative_path="book.pdf",
        source_hash=source_hash,
        title="Fixture book",
        page=12,
        section="3.1",
        parser_version="parser-v1",
        chunker_version="chunker-v1",
        index_version="index-v1",
        extra=extra
        or {
            "projection_generation": GENERATION_ID,
            "projection_population_hash": POPULATION_HASH,
        },
    )


class _Catalog:
    def __init__(self, *, active: bool = True, document_hash: str = SOURCE_HASH, chunk_text: str = "A supported market mechanism."):
        self.generation = SimpleNamespace(
            generation_id=GENERATION_ID,
            population_hash=POPULATION_HASH,
            status="VALIDATED",
            vector_ready=True,
            lexical_ready=True,
        )
        self.document = SimpleNamespace(
            document_id="doc-1",
            source_hash=document_hash,
            active=active,
            vector_ready=True,
            lexical_ready=True,
            projection_generation=GENERATION_ID,
            projection_population_hash=POPULATION_HASH,
        )
        self.chunk_text = chunk_text

    def active_generation(self):
        return self.generation

    def get_document(self, document_id):
        return self.document if document_id == "doc-1" else None

    def chunks_for_document(self, document_id):
        return (
            SimpleNamespace(
                chunk_id="chunk-1",
                document_id="doc-1",
                source_hash=SOURCE_HASH,
                text=self.chunk_text,
                content_type=ContentType.PROSE,
                source_filename="book.pdf",
                source_relative_path="book.pdf",
                title="Fixture book",
                authors=(),
                publication_year=None,
                page=12,
                page_start=12,
                page_end=12,
                chapter=None,
                section_path=(),
                epub_spine_item=None,
                anchor=None,
                parser_version="parser-v1",
                chunker_version="chunker-v1",
                index_version="index-v1",
                table_metadata=None,
                equation_metadata=None,
                active=True,
            ),
        )

    def document_is_retrieval_ready(self, document_id):
        return self.document.active


class _QueryService:
    def __init__(self, responses, *, catalog=None):
        self.responses = list(responses)
        self.catalog = catalog or _Catalog()
        self.requests = []

    def search(self, request):
        assert isinstance(request, KnowledgeQuery)
        self.requests.append(request)
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def test_query_bank_is_versioned_deterministic_and_covers_discovery_families() -> None:
    first = build_discovery_query_bank()
    second = build_discovery_query_bank()
    ids = {(query.family_id, query.formulation_id) for query in first}
    families = {query.family_id for query in first}

    assert first == second
    assert DISCOVERY_QUERY_BANK_VERSION
    assert query_bank_fingerprint(first) == query_bank_fingerprint(second)
    assert len(ids) == len(first)
    assert {
        "scalping",
        "momentum",
        "momentum_continuation",
        "breakout",
        "failed_breakout",
        "range_rejection",
        "mean_reversion",
        "pullback",
        "trend_continuation",
        "trend_reversal",
        "support_resistance",
        "volatility_expansion",
        "volatility_contraction",
        "channel_breakout",
        "price_action",
        "candle_reversal",
        "session_open_breakout",
        "london_session",
        "new_york_session",
        "overlap_session",
        "time_based_entry_exit",
        "trailing_stop",
        "breakeven_protection",
        "failed_move_exit",
        "momentum_reversal_exit",
        "no_progress_exit",
        "volatility_exit",
        "risk_reward",
        "position_sizing",
        "expected_move",
        "tick_scalping",
        "microstructure",
        "intraday_fx",
    } <= families
    assert sum(query.family_id == "failed_breakout" for query in first) >= 2


def test_query_bank_fingerprint_changes_when_a_formulation_changes() -> None:
    bank = build_discovery_query_bank()
    changed = (replace(bank[0], text=bank[0].text + " revised"), *bank[1:])

    assert query_bank_fingerprint(bank) != query_bank_fingerprint(changed)


def test_query_bank_fingerprint_includes_bank_version(monkeypatch) -> None:
    bank = build_discovery_query_bank()
    original = query_bank_fingerprint(bank)
    monkeypatch.setattr(phase14c_queries, "DISCOVERY_QUERY_BANK_VERSION", "phase14c-discovery-query-bank.v2")

    assert query_bank_fingerprint(bank) != original


def test_retrieval_fans_out_one_bounded_query_per_formulation_and_merges_sibling_refs() -> None:
    bank = (
        DiscoveryQuery("failed_breakout", "reversal", "failed breakout reversal"),
        DiscoveryQuery("failed_breakout", "return_inside", "return inside range after false break"),
    )
    hit = _hit()
    service = _QueryService(((hit,), (hit,)))

    report = retrieve_discovery_evidence(
        service,
        bank,
        pinned_generation_id=GENERATION_ID,
        pinned_population_hash=POPULATION_HASH,
    )

    assert len(service.requests) == 2
    assert all(request.top_k == 10 for request in service.requests)
    assert [request.text for request in service.requests] == [query.text for query in bank]
    assert report.status == "COMPLETE"
    assert report.raw_hit_count == 2
    assert report.unique_hit_count == 1
    assert len(report.records) == 1
    assert [(ref.family_id, ref.formulation_id, ref.rank) for ref in report.records[0].references] == [
        ("failed_breakout", "reversal", 1),
        ("failed_breakout", "return_inside", 1),
    ]


@pytest.mark.parametrize(
    ("hit", "catalog"),
    [
        (_hit(extra={"projection_generation": "wrong", "projection_population_hash": POPULATION_HASH}), _Catalog()),
        (_hit(extra={"projection_generation": GENERATION_ID, "projection_population_hash": "wrong"}), _Catalog()),
        (_hit(source_hash="b" * 64), _Catalog()),
        (_hit(text="forged or mismatched text"), _Catalog()),
        (_hit(), _Catalog(active=False)),
        (_hit(), _Catalog(document_hash="b" * 64)),
    ],
)
def test_invalid_hit_provenance_aborts_without_repair_or_partial_success(hit, catalog) -> None:
    service = _QueryService(((hit,),), catalog=catalog)

    report = retrieve_discovery_evidence(
        service,
        (DiscoveryQuery("momentum", "base", "short horizon momentum"),),
        pinned_generation_id=GENERATION_ID,
        pinned_population_hash=POPULATION_HASH,
    )

    assert report.status == "FAILED"
    assert report.records == ()
    assert report.raw_hit_count == 0
    assert report.unique_hit_count == 0
    assert report.query_failures


def test_query_service_error_has_explicit_failure_and_no_partial_success_claim() -> None:
    bank = (
        DiscoveryQuery("momentum", "base", "short horizon momentum"),
        DiscoveryQuery("breakout", "base", "price breakout"),
    )
    service = _QueryService(((_hit(),), RuntimeError("private provider detail")))

    report = retrieve_discovery_evidence(
        service,
        bank,
        pinned_generation_id=GENERATION_ID,
        pinned_population_hash=POPULATION_HASH,
    )

    assert report.status == "FAILED"
    assert report.records == ()
    assert report.raw_hit_count == 0
    assert report.query_failures[-1].formulation_id == "base"
    assert report.query_failures[-1].code == "query_failed"
    assert "private provider detail" not in repr(report)
