"""Deterministic Phase 14C discovery queries over the local hybrid index."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tradingagents.knowledge.models import KnowledgeHit, KnowledgeQuery
from tradingagents.knowledge.provenance import validate_hit_provenance

from .phase14c_models import (
    DiscoveryHitRecord,
    DiscoveryQuery,
    DiscoveryQueryFailure,
    DiscoveryRetrievalReport,
    QueryReference,
)

DISCOVERY_QUERY_BANK_VERSION = "phase14c-discovery-query-bank.v1"
_MAX_TOP_K_PER_FORMULATION = 50
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class _BankEntry:
    family_id: str
    formulation_id: str
    text: str


_BANK_ENTRIES = (
    _BankEntry("scalping", "short_horizon", "short-horizon foreign-exchange scalping entry and exit rules"),
    _BankEntry("momentum", "short_horizon", "short-horizon price momentum rules with exact confirmation conditions"),
    _BankEntry("momentum", "persistence", "momentum persistence and continuation conditions on intraday prices"),
    _BankEntry("momentum_continuation", "base", "momentum continuation entry, confirmation, invalidation, and exit"),
    _BankEntry("momentum_continuation", "pullback", "continuation after a pullback with explicit trigger and failure rule"),
    _BankEntry("breakout", "base", "price breakout strategy with source-stated trigger and confirmation"),
    _BankEntry("breakout", "range_expansion", "range expansion breakout entry and failed-breakout invalidation"),
    _BankEntry("failed_breakout", "reversal", "failed breakout reversal with exact entry and invalidation conditions"),
    _BankEntry("failed_breakout", "return_inside", "price returns inside the prior range after a false break"),
    _BankEntry("failed_breakout", "false_break_entry_exit", "false-breakout entry, confirmation, stop, and exit rules"),
    _BankEntry("range_rejection", "range_edge", "range edge rejection entry and confirmation conditions"),
    _BankEntry("range_rejection", "rejection_candle", "rejection from support or resistance within a defined range"),
    _BankEntry("mean_reversion", "base", "mean-reversion entry, confirmation, invalidation, and target rules"),
    _BankEntry("pullback", "base", "pullback entry in a prevailing trend with explicit confirmation"),
    _BankEntry("trend_continuation", "base", "trend continuation rules for entry, confirmation, and exit"),
    _BankEntry("trend_reversal", "base", "trend reversal trigger with confirmation and invalidation"),
    _BankEntry("support_resistance", "levels", "support and resistance level reactions with measurable rules"),
    _BankEntry("volatility_expansion", "base", "volatility expansion setup with breakout trigger and risk rule"),
    _BankEntry("volatility_contraction", "base", "volatility contraction and subsequent expansion conditions"),
    _BankEntry("channel_breakout", "base", "price channel breakout trigger, confirmation, and failure exit"),
    _BankEntry("price_action", "base", "rule-based price action setup with exact entry and invalidation"),
    _BankEntry("candle_reversal", "base", "candlestick reversal pattern with source-stated confirmation"),
    _BankEntry("session_open_breakout", "base", "session opening range breakout rules and time window"),
    _BankEntry("london_session", "base", "London session foreign-exchange entry or exit conditions"),
    _BankEntry("new_york_session", "base", "New York session foreign-exchange entry or exit conditions"),
    _BankEntry("overlap_session", "base", "London New York overlap trading rules and time conditions"),
    _BankEntry("time_based_entry_exit", "base", "time-based entry, holding horizon, and exit rules"),
    _BankEntry("trailing_stop", "base", "trailing stop placement and deterministic update conditions"),
    _BankEntry("breakeven_protection", "base", "break-even stop and profit-protection rules"),
    _BankEntry("failed_move_exit", "base", "exit after a failed move or failure to follow through"),
    _BankEntry("momentum_reversal_exit", "base", "exit when momentum reverses against an open position"),
    _BankEntry("no_progress_exit", "base", "time or progress-based exit when expected movement does not occur"),
    _BankEntry("volatility_exit", "base", "volatility-based stop, exit, or risk adjustment rules"),
    _BankEntry("risk_reward", "base", "risk reward ratio and stop target construction rules"),
    _BankEntry("position_sizing", "base", "position sizing formula with explicit risk and unit definitions"),
    _BankEntry("expected_move", "base", "expected price move calculation and source-stated assumptions"),
    _BankEntry("intraday_fx", "base", "short-term intraday foreign-exchange strategy with bounded holding period"),
    _BankEntry("tick_scalping", "base", "tick-level scalping entry, spread, and exit conditions"),
    _BankEntry("microstructure", "order_flow", "foreign exchange order flow imbalance and short-horizon price response"),
    _BankEntry("microstructure", "queue_imbalance", "queue or order-book imbalance and short-horizon price movement"),
    _BankEntry("microstructure", "market_impact", "market impact adverse selection and high-frequency execution mechanics"),
)


def build_discovery_query_bank() -> tuple[DiscoveryQuery, ...]:
    """Return the closed, order-stable V1 family/formulation bank."""

    return tuple(DiscoveryQuery(item.family_id, item.formulation_id, item.text) for item in _BANK_ENTRIES)


def query_bank_fingerprint(bank: Sequence[DiscoveryQuery]) -> str:
    queries = tuple(bank)
    if not queries:
        raise ValueError("discovery query bank must not be empty")
    if any(not isinstance(query, DiscoveryQuery) for query in queries):
        raise TypeError("query bank must contain DiscoveryQuery values")
    keys = [(query.family_id, query.formulation_id) for query in queries]
    if len(keys) != len(set(keys)):
        raise ValueError("query bank contains duplicate family/formulation IDs")
    payload = {
        "version": DISCOVERY_QUERY_BANK_VERSION,
        "queries": [
            {
                "family_id": query.family_id,
                "formulation_id": query.formulation_id,
                "text": query.text,
                "content_types": [item.value for item in query.content_types],
            }
            for query in queries
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_active_generation(catalog: Any, generation_id: str, population_hash: str) -> None:
    active = getattr(catalog, "active_generation", None)
    if not callable(active):
        raise ValueError("catalog has no active-generation reader")
    generation = active()
    if (
        generation is None
        or getattr(generation, "generation_id", None) != generation_id
        or getattr(generation, "population_hash", None) != population_hash
        or getattr(generation, "status", None) != "VALIDATED"
        or getattr(generation, "vector_ready", False) is not True
        or getattr(generation, "lexical_ready", False) is not True
    ):
        raise ValueError("active Phase 7 generation differs from the pinned retrieval identity")


def _same_value(actual: Any, expected: Any) -> bool:
    if hasattr(actual, "value"):
        actual = actual.value
    if hasattr(expected, "value"):
        expected = expected.value
    if isinstance(actual, list):
        actual = tuple(actual)
    if isinstance(expected, list):
        expected = tuple(expected)
    return actual == expected


def _validate_hit(hit: Any, catalog: Any, *, generation_id: str, population_hash: str) -> KnowledgeHit:
    if not isinstance(hit, KnowledgeHit):
        raise ValueError("query service returned a non-KnowledgeHit value")
    validate_hit_provenance(hit)
    if not _SHA256_RE.fullmatch(str(hit.source_hash or "")):
        raise ValueError("hit source hash is not a lowercase SHA-256 digest")
    if not isinstance(hit.extra, Mapping):
        raise ValueError("hit projection metadata is malformed")
    if (
        hit.extra.get("projection_generation") != generation_id
        or hit.extra.get("projection_population_hash") != population_hash
    ):
        raise ValueError("hit projection identity differs from the pinned generation")

    get_document = getattr(catalog, "get_document", None)
    chunks_for_document = getattr(catalog, "chunks_for_document", None)
    if not callable(get_document) or not callable(chunks_for_document):
        raise ValueError("catalog lacks exact document/chunk lookup")
    document = get_document(hit.document_id)
    if (
        document is None
        or getattr(document, "active", None) is not True
        or getattr(document, "source_hash", None) != hit.source_hash
        or getattr(document, "vector_ready", None) is not True
        or getattr(document, "lexical_ready", None) is not True
        or getattr(document, "projection_generation", None) != generation_id
        or getattr(document, "projection_population_hash", None) != population_hash
    ):
        raise ValueError("hit document is stale or not ready in the pinned generation")
    retrieval_ready = getattr(catalog, "document_is_retrieval_ready", None)
    if callable(retrieval_ready) and retrieval_ready(hit.document_id) is not True:
        raise ValueError("hit document is not retrieval-ready")

    matching = tuple(
        chunk for chunk in chunks_for_document(hit.document_id) if getattr(chunk, "chunk_id", None) == hit.chunk_id
    )
    if len(matching) != 1:
        raise ValueError("hit does not identify exactly one catalog chunk")
    chunk = matching[0]
    if (
        getattr(chunk, "document_id", None) != hit.document_id
        or getattr(chunk, "source_hash", None) != hit.source_hash
        or getattr(chunk, "text", None) != hit.text
        or getattr(chunk, "active", None) is not True
    ):
        raise ValueError("hit text or identity differs from the exact catalog chunk")

    for field_name in (
        "content_type",
        "source_filename",
        "source_relative_path",
        "title",
        "authors",
        "publication_year",
        "page",
        "page_start",
        "page_end",
        "chapter",
        "section_path",
        "epub_spine_item",
        "anchor",
        "parser_version",
        "chunker_version",
        "index_version",
        "table_metadata",
        "equation_metadata",
    ):
        if not _same_value(getattr(hit, field_name), getattr(chunk, field_name, None)):
            raise ValueError(f"hit source field {field_name} differs from the catalog chunk")
    return hit


def _failed_report(
    *,
    generation_id: str,
    population_hash: str,
    bank_fingerprint: str,
    failure: DiscoveryQueryFailure,
    completed_query_count: int,
) -> DiscoveryRetrievalReport:
    return DiscoveryRetrievalReport(
        generation_id=generation_id,
        population_hash=population_hash,
        query_bank_version=DISCOVERY_QUERY_BANK_VERSION,
        query_bank_fingerprint=bank_fingerprint,
        status="FAILED",
        records=(),
        raw_hit_count=0,
        unique_hit_count=0,
        completed_query_count=completed_query_count,
        query_failures=(failure,),
    )


def _query_failure(
    query: DiscoveryQuery,
    *,
    code: str,
    exc: BaseException,
    generation_id: str,
    population_hash: str,
    bank_fingerprint: str,
    completed_query_count: int,
) -> DiscoveryRetrievalReport:
    return _failed_report(
        generation_id=generation_id,
        population_hash=population_hash,
        bank_fingerprint=bank_fingerprint,
        failure=DiscoveryQueryFailure(
            query.family_id,
            query.formulation_id,
            code,
            type(exc).__name__,
        ),
        completed_query_count=completed_query_count,
    )


def retrieve_discovery_evidence(
    query_service: Any,
    bank: Sequence[DiscoveryQuery],
    *,
    pinned_generation_id: str,
    pinned_population_hash: str,
    top_k_per_formulation: int = 10,
) -> DiscoveryRetrievalReport:
    """Search sequentially and fail closed if any query or hit loses provenance."""

    queries = tuple(bank)
    fingerprint = query_bank_fingerprint(queries)
    if not isinstance(top_k_per_formulation, int) or isinstance(top_k_per_formulation, bool):
        raise TypeError("top_k_per_formulation must be an integer")
    if not 1 <= top_k_per_formulation <= _MAX_TOP_K_PER_FORMULATION:
        raise ValueError(f"top_k_per_formulation must be in [1, {_MAX_TOP_K_PER_FORMULATION}]")
    if not pinned_generation_id.strip() or not pinned_population_hash.strip():
        raise ValueError("explicit Phase 7 generation and population pins are required")
    catalog = getattr(query_service, "catalog", None)
    if catalog is None:
        raise TypeError("query_service must expose its read-only catalog")
    try:
        _validate_active_generation(catalog, pinned_generation_id, pinned_population_hash)
    except Exception as exc:
        first = queries[0]
        return _failed_report(
            generation_id=pinned_generation_id,
            population_hash=pinned_population_hash,
            bank_fingerprint=fingerprint,
            failure=DiscoveryQueryFailure(
                first.family_id,
                first.formulation_id,
                "PINNED_GENERATION_MISMATCH",
                type(exc).__name__,
            ),
            completed_query_count=0,
        )

    records: dict[tuple[str, str, str], tuple[KnowledgeHit, list[QueryReference]]] = {}
    raw_hit_count = 0
    completed = 0
    search = getattr(query_service, "search", None)
    if not callable(search):
        raise TypeError("query_service must expose search(KnowledgeQuery)")
    for query in queries:
        try:
            _validate_active_generation(catalog, pinned_generation_id, pinned_population_hash)
        except Exception as exc:
            return _query_failure(
                query,
                code="PINNED_GENERATION_MISMATCH",
                exc=exc,
                generation_id=pinned_generation_id,
                population_hash=pinned_population_hash,
                bank_fingerprint=fingerprint,
                completed_query_count=completed,
            )
        try:
            request = KnowledgeQuery(
                text=query.text,
                top_k=top_k_per_formulation,
                content_types=query.content_types,
            )
            hits = search(request)
            if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
                raise ValueError("query service returned a malformed hit batch")
        except Exception as exc:
            return _query_failure(
                query,
                code="QUERY_FAILED",
                exc=exc,
                generation_id=pinned_generation_id,
                population_hash=pinned_population_hash,
                bank_fingerprint=fingerprint,
                completed_query_count=completed,
            )
        try:
            validated_hits = tuple(
                _validate_hit(
                    hit,
                    catalog,
                    generation_id=pinned_generation_id,
                    population_hash=pinned_population_hash,
                )
                for hit in hits
            )
        except Exception as exc:
            return _query_failure(
                query,
                code="PROVENANCE_INVALID",
                exc=exc,
                generation_id=pinned_generation_id,
                population_hash=pinned_population_hash,
                bank_fingerprint=fingerprint,
                completed_query_count=completed,
            )
        completed += 1
        raw_hit_count += len(validated_hits)
        for rank, hit in enumerate(validated_hits, start=1):
            try:
                key = (hit.document_id, hit.chunk_id, hit.source_hash)
                reference = QueryReference(
                    family_id=query.family_id,
                    formulation_id=query.formulation_id,
                    rank=rank,
                    semantic_score=hit.semantic_score,
                    lexical_score=hit.lexical_score,
                    fused_score=hit.fused_score,
                    rerank_score=hit.rerank_score,
                )
                existing = records.get(key)
                if existing is None:
                    records[key] = (hit, [reference])
                else:
                    if existing[0].text != hit.text:
                        raise ValueError("duplicate hit identity returned conflicting text")
                    existing[1].append(reference)
            except Exception as exc:
                return _query_failure(
                    query,
                    code="PROVENANCE_INVALID",
                    exc=exc,
                    generation_id=pinned_generation_id,
                    population_hash=pinned_population_hash,
                    bank_fingerprint=fingerprint,
                    completed_query_count=completed,
                )

    result_records = tuple(
        DiscoveryHitRecord(hit, tuple(references)) for hit, references in records.values()
    )
    return DiscoveryRetrievalReport(
        generation_id=pinned_generation_id,
        population_hash=pinned_population_hash,
        query_bank_version=DISCOVERY_QUERY_BANK_VERSION,
        query_bank_fingerprint=fingerprint,
        status="COMPLETE",
        records=result_records,
        raw_hit_count=raw_hit_count,
        unique_hit_count=len(result_records),
        completed_query_count=completed,
    )


__all__ = [
    "DISCOVERY_QUERY_BANK_VERSION",
    "build_discovery_query_bank",
    "query_bank_fingerprint",
    "retrieve_discovery_evidence",
]
