from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tradingagents.knowledge.models import KnowledgeHit
from tradingagents.self_enhancement.book_drafter import (
    DRAFT_PROMPT_VERSION,
    DRAFT_SCHEMA_VERSION,
    DraftResult,
    StrategyDraft,
    StrategyDraftBatch,
    StrategyDraftClaim,
)
from tradingagents.self_enhancement.book_pipeline import (
    BookStrategyPipeline,
    MappingStatus,
    PinnedGenerationMismatch,
    classify_suitability,
    deduplicate_specs,
    map_existing_strategy,
)
from tradingagents.self_enhancement.strategy_specs import (
    EvidenceSpan,
    RuleDirection,
    RuleOperator,
    RuleOrigin,
    RuleStage,
    RuleValidationResult,
    RuleValidationStatus,
    StrategyRuleClaim,
    StrategySpec,
    StrategySuitability,
)

GENERATION = "gen_fixture_1"
FINGERPRINT = "sha256:" + "b" * 64
SOURCE_HASH = "a" * 64
SOURCE_TEXT = "LONG when return_1 > 1.2 points after confirmation"
QUOTE = SOURCE_TEXT


def _hit(
    *,
    document_id: str = "doc-book-1",
    chunk_id: str = "chunk-1",
    source_hash: str = SOURCE_HASH,
    text: str = SOURCE_TEXT,
    generation: str = GENERATION,
    fingerprint: str = FINGERPRINT,
) -> KnowledgeHit:
    return KnowledgeHit(
        chunk_id=chunk_id,
        document_id=document_id,
        source_hash=source_hash,
        text=text,
        source_filename=f"{document_id}.pdf",
        title="Short Horizon Trading",
        page=4,
        section="Entry",
        extra={
            "projection_generation": generation,
            "projection_population_hash": fingerprint,
        },
    )


def _claim(hit: KnowledgeHit, **overrides) -> StrategyDraftClaim:
    values = {
        "stage": RuleStage.ENTRY,
        "operator": RuleOperator.GREATER_THAN,
        "direction": RuleDirection.LONG,
        "value": 1.2,
        "unit": "points",
        "condition": "after confirmation",
        "horizon_seconds": None,
        "origin": RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
        "evidence_ref": "E1",
        "start_offset": 0,
        "end_offset": len(QUOTE),
        "quote": QUOTE,
    }
    values.update(overrides)
    return StrategyDraftClaim(**values)


def _draft(hit: KnowledgeHit, claim: StrategyDraftClaim | None = None, **overrides) -> StrategyDraft:
    values = {
        "name": "Short horizon continuation",
        "family": "MOMENTUM_CONTINUATION",
        "required_data": ["return_1"],
        "implementation_confidence": 0.65,
        "rule_claims": [claim or _claim(hit)],
    }
    values.update(overrides)
    return StrategyDraft(**values)


def _draft_result(*specs: StrategyDraft) -> DraftResult:
    batch = StrategyDraftBatch(specs=list(specs))
    return DraftResult(
        specs=tuple(batch.specs),
        provider="ollama-local",
        model="qwen3.5:2b",
        prompt_version=DRAFT_PROMPT_VERSION,
        schema_version=DRAFT_SCHEMA_VERSION,
        timeout_seconds=120,
        max_output_tokens=2048,
        context_tokens=8192,
        elapsed_seconds=1.0,
        input_tokens=100,
        output_tokens=50,
        finish_reason="stop",
        draft_digest="c" * 64,
    )


class _Catalog:
    def __init__(self, hits: tuple[KnowledgeHit, ...]):
        self.generation = SimpleNamespace(
            generation_id=GENERATION,
            population_hash=FINGERPRINT,
            vector_ready=True,
            lexical_ready=True,
            status="VALIDATED",
        )
        self.documents = {
            hit.document_id: SimpleNamespace(
                document_id=hit.document_id,
                source_hash=hit.source_hash,
                active=True,
                projection_generation=GENERATION,
                projection_population_hash=FINGERPRINT,
            )
            for hit in hits
        }
        self.chunks = {
            hit.chunk_id: SimpleNamespace(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                source_hash=hit.source_hash,
                text=hit.text,
                active=True,
                vector_ready=True,
                lexical_ready=True,
                projection_generation=GENERATION,
                extra={"projection_population_hash": FINGERPRINT},
            )
            for hit in hits
        }

    def active_generation(self):
        return self.generation

    def get_document(self, document_id: str):
        return self.documents.get(document_id)

    def chunks_for_document(self, document_id: str):
        return tuple(chunk for chunk in self.chunks.values() if chunk.document_id == document_id)


class _QueryService:
    def __init__(self, hits: tuple[KnowledgeHit, ...]):
        self.catalog = _Catalog(hits)
        self.hits = hits
        self.requests = []

    def search(self, request):
        self.requests.append(request)
        return self.hits


class _Drafter:
    def __init__(self, result: DraftResult):
        self.result = result
        self.calls = []

    def draft(self, hits, max_specs):
        self.calls.append((tuple(hits), max_specs))
        return self.result


def _pipeline(hit: KnowledgeHit, draft: StrategyDraft, **overrides) -> tuple[BookStrategyPipeline, _Drafter]:
    service = _QueryService((hit,))
    drafter = _Drafter(_draft_result(draft))
    pipeline = BookStrategyPipeline(
        service,
        drafter,
        pinned_generation=GENERATION,
        pinned_fingerprint=FINGERPRINT,
        **overrides,
    )
    return pipeline, drafter


def test_exact_source_span_and_finite_rule_grammar_are_supported() -> None:
    hit = _hit()
    pipeline, drafter = _pipeline(hit, _draft(hit))

    report = pipeline.extract(["short horizon entry rule"], max_specs=2)

    assert report.error_code is None
    assert report.retrieved_count == 1
    assert len(drafter.calls) == 1
    assert len(report.specs) == 1
    spec = report.specs[0]
    assert spec.generation_id == GENERATION
    assert spec.knowledge_fingerprint == FINGERPRINT
    assert spec.source_supported_rules[0].evidence == EvidenceSpan(
        GENERATION,
        hit.document_id,
        hit.chunk_id,
        SOURCE_HASH,
        0,
        len(QUOTE),
        QUOTE,
    )
    assert spec.validation_results[0].status is RuleValidationStatus.SUPPORTED


def test_shifted_source_span_is_rejected_even_when_quote_text_is_valid() -> None:
    hit = _hit()
    shifted = _draft(hit, _claim(hit, start_offset=1, end_offset=1 + len(QUOTE)))
    pipeline, _ = _pipeline(hit, shifted)

    report = pipeline.extract(["entry"], max_specs=1)

    assert report.specs == ()
    assert report.rejected_rules[0].reason_code == "SPAN_MISMATCH"


@pytest.mark.parametrize(
    "changes",
    [
        {"operator": RuleOperator.LESS_THAN},
        {"direction": RuleDirection.SHORT},
        {"value": 1.3},
        {"unit": "pips"},
        {"condition": "before confirmation"},
        {"evidence_ref": "E2"},
    ],
)
def test_mismatched_or_unavailable_rule_claim_is_rejected(changes: dict) -> None:
    hit = _hit()
    pipeline, _ = _pipeline(hit, _draft(hit, _claim(hit, **changes)))

    report = pipeline.extract(["entry"], max_specs=1)

    assert report.specs == ()
    assert report.rejected_rules
    assert report.rejected_rules[0].status is RuleValidationStatus.UNSUPPORTED


def test_exact_quote_for_unrelated_feature_does_not_support_required_data() -> None:
    hit = _hit()
    draft = _draft(hit, required_data=["volatility"])
    pipeline, _ = _pipeline(hit, draft)

    report = pipeline.extract(["entry"], max_specs=1)

    assert report.specs == ()
    assert report.rejected_rules[0].reason_code == "DATA_FEATURE_MISMATCH"


@pytest.mark.parametrize(
    "hit_override",
    [
        {"generation": "gen_stale"},
        {"fingerprint": "sha256:" + "e" * 64},
    ],
)
def test_catalog_identity_or_pinned_projection_mismatch_fails_before_model_call(hit_override: dict) -> None:
    hit = _hit(**hit_override)
    pipeline, drafter = _pipeline(hit, _draft(hit))

    report = pipeline.extract(["entry"], max_specs=1)

    assert report.specs == ()
    assert report.error_code == "KNOWLEDGE_PROVENANCE_INVALID"
    assert drafter.calls == []


@pytest.mark.parametrize("mismatch", ["document_hash", "chunk_identity"])
def test_catalog_document_and_chunk_identity_are_checked(mismatch: str) -> None:
    hit = _hit()
    pipeline, drafter = _pipeline(hit, _draft(hit))
    catalog = pipeline.query_service.catalog
    if mismatch == "document_hash":
        catalog.documents[hit.document_id].source_hash = "d" * 64
    else:
        catalog.chunks.pop(hit.chunk_id)

    report = pipeline.extract(["entry"], max_specs=1)

    assert report.specs == ()
    assert report.error_code == "KNOWLEDGE_PROVENANCE_INVALID"
    assert drafter.calls == []


def test_active_generation_mismatch_fails_closed() -> None:
    hit = _hit()
    pipeline, drafter = _pipeline(hit, _draft(hit))
    pipeline.query_service.catalog.generation.population_hash = "sha256:" + "e" * 64

    with pytest.raises(PinnedGenerationMismatch):
        pipeline.extract(["entry"], max_specs=1)

    assert drafter.calls == []


def _spec_for_suitability(*, extra_data: tuple[str, ...] = ()) -> StrategySpec:
    span = EvidenceSpan(GENERATION, "doc-book-1", "chunk-1", SOURCE_HASH, 0, len(QUOTE), QUOTE)
    claim = StrategyRuleClaim(
        RuleStage.ENTRY,
        RuleOperator.GREATER_THAN,
        RuleDirection.LONG,
        1.2,
        "points",
        "after confirmation",
        None,
        RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
        span,
    )
    return StrategySpec(
        "spec-1",
        "Entry-only",
        "MOMENTUM_CONTINUATION",
        ("return_1", *extra_data),
        (claim,),
        GENERATION,
        FINGERPRINT,
        "ollama-local",
        "qwen3.5:2b",
        DRAFT_PROMPT_VERSION,
        DRAFT_SCHEMA_VERSION,
        datetime(2026, 10, 3, tzinfo=UTC),
        0.65,
        StrategySuitability.HFT_SUITABLE,
        (RuleValidationResult(claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE"),),
    )


def test_unavailable_l2_data_is_not_hft_suitable() -> None:
    result = classify_suitability(_spec_for_suitability(extra_data=("l2_order_book",)), {"return_1"})

    assert result.suitability is StrategySuitability.UNAVAILABLE_DATA
    assert result.missing_features == ("l2_order_book",)


def test_missing_exit_and_horizon_remain_insufficient_specification() -> None:
    result = classify_suitability(_spec_for_suitability(), {"return_1"})

    assert result.suitability is StrategySuitability.INSUFFICIENT_SPECIFICATION
    assert RuleStage.EXIT in result.unspecified_stages
    assert RuleStage.HORIZON in result.unspecified_stages


def test_dedup_collapses_duplicate_chunks_but_preserves_independent_documents() -> None:
    first = _spec_for_suitability()
    second_span = EvidenceSpan(GENERATION, "doc-book-2", "chunk-2", "d" * 64, 0, len(QUOTE), QUOTE)
    second_claim = replace(first.rule_claims[0], evidence=second_span)
    second = replace(
        first,
        spec_id="spec-2",
        rule_claims=(second_claim,),
        validation_results=(
            RuleValidationResult(second_claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE"),
        ),
    )
    duplicate_chunk_span = EvidenceSpan(GENERATION, "doc-book-1", "chunk-99", SOURCE_HASH, 0, len(QUOTE), QUOTE)
    duplicate_chunk_claim = replace(first.rule_claims[0], evidence=duplicate_chunk_span)
    same_book_chunk = replace(
        first,
        spec_id="spec-3",
        rule_claims=(duplicate_chunk_claim,),
        validation_results=(
            RuleValidationResult(duplicate_chunk_claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE"),
        ),
    )

    deduplicated = deduplicate_specs((first, same_book_chunk, second))

    assert len(deduplicated) == 1
    source_documents = {claim.evidence.document_id for claim in deduplicated[0].source_supported_rules}
    assert source_documents == {"doc-book-1", "doc-book-2"}
    assert len(deduplicated[0].source_supported_rules) == 2


@pytest.mark.parametrize("strategy_id", ["range_rejection", "momentum_continuation"])
def test_existing_strategy_mapping_never_fabricates_citations(strategy_id: str) -> None:
    report = map_existing_strategy(strategy_id, (_hit(text="Generic discussion of market structure."),))

    assert report.strategy_id == strategy_id
    assert report.overall_status is MappingStatus.NO_DIRECT_MATCH
    assert all(component.status is MappingStatus.NO_DIRECT_MATCH for component in report.components)
    assert all(component.evidence == () for component in report.components)
