from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from tradingagents.self_enhancement.book_pipeline import classify_suitability
from tradingagents.self_enhancement.book_rule_grammar import parse_supported_rule_quote
from tradingagents.self_enhancement.book_strategies import BookStrategyRegistry
from tradingagents.self_enhancement.strategy_specs import (
    EvidenceSpan,
    RuleOperator,
    RuleOrigin,
    RuleStage,
    RuleValidationResult,
    RuleValidationStatus,
    StrategyRuleClaim,
    StrategySpec,
    StrategySuitability,
)

GENERATION = "gen-phase14c-suitability-fixture"
GENERATION_FINGERPRINT = "e" * 64
COMPLETE_QUOTES = (
    (RuleStage.ENTRY, "LONG when momentum > 2 points after confirmation"),
    (
        RuleStage.CONFIRMATION,
        "CONFIRMATION: LONG when direction_persistence >= 0.6 fraction after three ticks",
    ),
    (RuleStage.INVALIDATION, "INVALIDATION: LONG when momentum <= 0 points after reversal"),
    (RuleStage.EXPECTED_MOVE, "EXPECTED_MOVE: LONG target 5 points within 30 seconds after entry"),
    (RuleStage.EXIT, "EXIT: LONG when momentum <= 0 points after reversal"),
    (
        RuleStage.PROFIT_PROTECTION,
        "PROFIT_PROTECTION: LONG when momentum <= 0 points after target retracement",
    ),
    (RuleStage.STOP_BEHAVIOR, "STOP_BEHAVIOR: LONG when momentum <= -3 points after adverse move"),
    (RuleStage.HORIZON, "HORIZON: hold no longer than 30 seconds after entry"),
)


def _spec(
    *,
    family: str = "MOMENTUM_CONTINUATION",
    quotes=COMPLETE_QUOTES,
    required_data: tuple[str, ...] = ("momentum", "direction_persistence"),
) -> StrategySpec:
    text = "\n".join(quote for _, quote in quotes)
    source_hash = hashlib.sha256(b"phase14c-suitability-source").hexdigest()
    claims = []
    cursor = 0
    for stage, quote in quotes:
        parsed = parse_supported_rule_quote(quote, stage)
        assert parsed is not None
        start = text.index(quote, cursor)
        cursor = start + len(quote)
        span = EvidenceSpan(
            GENERATION,
            "doc-suitability-fixture",
            "chunk-suitability-fixture",
            source_hash,
            start,
            cursor,
            quote,
        )
        value = parsed["value"]
        claims.append(
            StrategyRuleClaim(
                stage=stage,
                operator=parsed["operator"],
                direction=parsed["direction"],
                value=int(value) if value == value.to_integral_value() else float(value),
                unit=parsed["unit"],
                condition=parsed["condition"],
                horizon_seconds=parsed["horizon_seconds"],
                origin=RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
                evidence=span,
            )
        )
    validations = tuple(
        RuleValidationResult(claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE")
        for claim in claims
    )
    return StrategySpec(
        spec_id="spec-suitability-fixture",
        name="Complete source-supported momentum fixture",
        family=family,
        required_data=required_data,
        rule_claims=tuple(claims),
        generation_id=GENERATION,
        knowledge_fingerprint=GENERATION_FINGERPRINT,
        model_provider="ollama-local",
        model_id="qwen3.5:2b",
        prompt_version="fixture-v1",
        schema_version="fixture-v1",
        created_at=datetime.now(UTC),
        implementation_confidence=0.0,
        suitability=StrategySuitability.HFT_SUITABLE,
        validation_results=validations,
    )


def test_unavailable_queue_position_data_prevents_hft_suitability() -> None:
    spec = _spec(required_data=("momentum", "direction_persistence", "queue_position"))

    result = classify_suitability(spec, ("momentum", "direction_persistence"))

    assert result.suitability is StrategySuitability.UNAVAILABLE_DATA
    assert result.missing_features == ("queue_position",)
    assert not replace(spec, suitability=result.suitability).is_executable


def test_partial_specification_is_rejected_by_finite_registry() -> None:
    spec = _spec(quotes=COMPLETE_QUOTES[:-1])

    assert not spec.is_executable
    with pytest.raises(ValueError, match="complete validated HFT StrategySpec"):
        BookStrategyRegistry().create(spec)


def test_horizon_over_sixty_seconds_cannot_reach_finite_registry() -> None:
    quotes = (*COMPLETE_QUOTES[:-1], (RuleStage.HORIZON, "HORIZON: hold no longer than 61 seconds after entry"))
    spec = _spec(quotes=quotes)

    suitability = classify_suitability(spec, ("momentum", "direction_persistence"))
    candidate = replace(spec, suitability=suitability.suitability)
    assert suitability.suitability is StrategySuitability.SHORT_TERM_SUITABLE
    assert not candidate.is_executable
    with pytest.raises(ValueError, match="complete validated HFT StrategySpec"):
        BookStrategyRegistry().create(candidate)


def test_conflicting_same_stage_rules_are_rejected_by_finite_registry() -> None:
    conflict = (RuleStage.EXIT, "EXIT: LONG when momentum <= -1 points after reversal")
    spec = _spec(quotes=(*COMPLETE_QUOTES, conflict))

    with pytest.raises(ValueError, match="conflicting independently sourced rules for stage: EXIT"):
        BookStrategyRegistry().create(spec)


def test_unreviewed_family_is_not_accepted_by_finite_registry() -> None:
    spec = _spec(family="MEAN_REVERSION")

    with pytest.raises(ValueError, match="no reviewed HFT permission mapping"):
        BookStrategyRegistry().create(spec)


def test_source_claim_that_disagrees_with_supported_operator_is_rejected() -> None:
    spec = _spec()
    entry = spec.rule_claims[0]
    changed_entry = replace(entry, operator=RuleOperator.LESS_THAN)
    claims = (changed_entry, *spec.rule_claims[1:])
    validations = tuple(
        RuleValidationResult(claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE")
        for claim in claims
    )
    candidate = replace(spec, rule_claims=claims, validation_results=validations)

    with pytest.raises(ValueError, match="source quote and structured rule disagree: ENTRY"):
        BookStrategyRegistry().create(candidate)


def test_only_reviewed_existing_strategy_family_is_constructible() -> None:
    candidate = BookStrategyRegistry().create(_spec())

    assert candidate.strategy_id == "momentum_continuation"
