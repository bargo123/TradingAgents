from datetime import UTC, datetime

from tradingagents.self_enhancement.book_factory import BookStrategyFactory
from tradingagents.self_enhancement.candidates import CandidateGenerator
from tradingagents.self_enhancement.models import CandidateState, ExitPolicyConfig, StrategyVersion
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


def _parent():
    return StrategyVersion(
        "range_rejection",
        "incumbent-v1",
        "cfg-v1",
        ExitPolicyConfig().to_dict(),
        "abc123",
    )


def test_candidate_generator_is_deterministic_and_bounded():
    first = CandidateGenerator().generate(_parent())
    second = CandidateGenerator().generate(_parent())
    assert [item.candidate_id for item in first] == [item.candidate_id for item in second]
    assert len(first) == 6
    assert all(item.execution_mode.value == "SHADOW" for item in first)
    assert all(item.state is CandidateState.EXPERIMENTAL for item in first)
    assert all(item.real_money is False for item in first)


def test_book_factory_preserves_exact_phase7_provenance_and_never_approves():
    evidence = {
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "source_filename": "paper.pdf",
        "source_hash": "hash-1",
        "page": 4,
        "section": "3.1",
        "content_type": "PROSE",
        "text": "Protect favorable excursion after failed continuation.",
    }
    candidate = BookStrategyFactory().from_hits((evidence,), parent=_parent())[0]
    assert candidate.state is CandidateState.EXTRACTED
    assert candidate.source_evidence == (evidence,)
    assert candidate.execution_mode.value == "SHADOW"


def test_book_factory_rejects_anonymous_evidence():
    try:
        BookStrategyFactory().from_hits(({"text": "unproven"},), parent=_parent())
    except ValueError as exc:
        assert "provenance" in str(exc)
    else:
        raise AssertionError("anonymous book evidence must fail closed")


def _validated_book_spec():
    quote = "LONG when return_1 > 1.2 points after confirmation"
    span = EvidenceSpan("gen-1", "doc-1", "chunk-1", "a" * 64, 0, len(quote), quote)
    claims = tuple(
        StrategyRuleClaim(
            stage,
            RuleOperator.GREATER_THAN,
            RuleDirection.LONG,
            1.2,
            "points",
            "after confirmation",
            5 if stage in {RuleStage.EXPECTED_MOVE, RuleStage.HORIZON} else None,
            RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
            span,
        )
        for stage in RuleStage
    )
    validations = tuple(
        RuleValidationResult(
            claim.fingerprint,
            RuleValidationStatus.SUPPORTED,
            "EXACT_SOURCE_RULE",
        )
        for claim in claims
    )
    return StrategySpec(
        spec_id="book-spec-1",
        name="Validated fixture",
        family="MOMENTUM_CONTINUATION",
        required_data=("return_1",),
        rule_claims=claims,
        generation_id="gen-1",
        knowledge_fingerprint="b" * 64,
        model_provider="ollama-local",
        model_id="qwen3.5:2b",
        prompt_version="phase14b-draft-v1",
        schema_version="strategy-spec-v1",
        created_at=datetime(2026, 10, 3, tzinfo=UTC),
        implementation_confidence=0.8,
        suitability=StrategySuitability.HFT_SUITABLE,
        validation_results=validations,
    )


def test_book_factory_routes_only_validated_spec_as_extracted_shadow_candidate():
    spec = _validated_book_spec()

    assert spec.is_executable is True
    candidate = BookStrategyFactory().from_validated_spec(spec, parent=_parent())

    assert candidate.state is CandidateState.EXTRACTED
    assert candidate.execution_mode.value == "SHADOW"
    assert candidate.real_money is False
    assert candidate.strategy_id.startswith("book-spec-")
    assert candidate.source_evidence


def test_book_factory_rejects_incomplete_strategy_spec():
    from dataclasses import replace

    from tradingagents.self_enhancement.strategy_specs import StrategySuitability

    incomplete = replace(
        _validated_book_spec(),
        rule_claims=(_validated_book_spec().rule_claims[0],),
        validation_results=(_validated_book_spec().validation_results[0],),
        suitability=StrategySuitability.INSUFFICIENT_SPECIFICATION,
    )

    try:
        BookStrategyFactory().from_validated_spec(incomplete, parent=_parent())
    except ValueError as exc:
        assert "validated" in str(exc)
    else:
        raise AssertionError("incomplete specs must not become candidates")
