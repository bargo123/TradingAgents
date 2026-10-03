from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradingagents.self_enhancement.strategy_specs import (
    EXECUTABLE_REQUIRED_STAGES,
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


def _span(**overrides) -> EvidenceSpan:
    values = {
        "generation_id": "gen_fixture_1",
        "document_id": "doc-book-1",
        "chunk_id": "chunk-14",
        "source_hash": "a" * 64,
        "start_offset": 11,
        "end_offset": 11 + len("price > 1.2 points after confirmation"),
        "quote": "price > 1.2 points after confirmation",
    }
    values.update(overrides)
    return EvidenceSpan(**values)


def _claim(**overrides) -> StrategyRuleClaim:
    values = {
        "stage": RuleStage.ENTRY,
        "operator": RuleOperator.GREATER_THAN,
        "direction": "LONG",
        "value": 1.2,
        "unit": "points",
        "condition": "after confirmation",
        "horizon_seconds": None,
        "origin": RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
        "evidence": _span(),
    }
    values.update(overrides)
    return StrategyRuleClaim(**values)


def _spec(**overrides) -> StrategySpec:
    values = {
        "spec_id": "book-spec-001",
        "name": "Short horizon continuation",
        "family": "MOMENTUM_CONTINUATION",
        "required_data": ("bid", "ask", "mid_return_1"),
        "rule_claims": (_claim(),),
        "generation_id": "gen_fixture_1",
        "knowledge_fingerprint": "b" * 64,
        "model_provider": "ollama-local",
        "model_id": "qwen3.5:2b",
        "prompt_version": "phase14b-draft-v1",
        "schema_version": "strategy-spec-v1",
        "created_at": datetime(2026, 10, 3, 12, tzinfo=UTC),
        "implementation_confidence": 0.65,
        "suitability": StrategySuitability.INSUFFICIENT_SPECIFICATION,
    }
    values.update(overrides)
    return StrategySpec(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": "entry"},
        {"operator": "greater_than"},
        {"direction": "long"},
        {"origin": "BOOK"},
        {"value": True},
        {"horizon_seconds": True},
    ],
)
def test_rule_contract_rejects_unknown_enum_values_and_coercion(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        _claim(**kwargs)


def test_evidence_span_requires_exact_bounded_source_identity_and_offsets() -> None:
    with pytest.raises(ValueError, match="source_hash"):
        _span(source_hash="not-a-digest")
    with pytest.raises(ValueError, match="offset"):
        _span(start_offset=42, end_offset=11)
    with pytest.raises(ValueError, match="quote"):
        _span(quote=" ")


def test_strategy_spec_serialization_and_content_hash_are_deterministic() -> None:
    left = _spec()
    right = _spec(required_data=("mid_return_1", "ask", "bid"))

    assert left.content_hash == right.content_hash
    assert left.canonical_json() == right.canonical_json()
    assert left.to_dict()["content_hash"] == left.content_hash


def test_knowledge_fingerprint_accepts_phase7_sha256_identity_format() -> None:
    fingerprint = "sha256:" + "b" * 64

    spec = _spec(knowledge_fingerprint=fingerprint)

    assert spec.knowledge_fingerprint == fingerprint


def test_strategy_spec_rejects_unknown_fields_and_invalid_nested_types() -> None:
    serialized = _spec().to_dict()
    serialized.pop("content_hash")
    with pytest.raises(TypeError, match="unexpected"):
        StrategySpec(**{**serialized, "unexpected": "ignored"})
    with pytest.raises(ValueError, match="rule_claims"):
        _spec(rule_claims=({"stage": "ENTRY"},))


def test_source_supported_concepts_and_research_hypotheses_stay_separate() -> None:
    source_rule = _claim()
    hypothesis = _claim(
        origin=RuleOrigin.RESEARCH_HYPOTHESIS_PARAMETER,
        evidence=None,
        value=0.75,
        unit="fraction",
    )
    spec = _spec(rule_claims=(source_rule, hypothesis))

    assert spec.source_supported_rules == (source_rule,)
    assert spec.research_hypothesis_rules == (hypothesis,)
    assert spec.is_executable is False


def test_missing_source_rule_stages_remain_explicitly_unspecified() -> None:
    spec = _spec(rule_claims=(_claim(),))

    assert RuleStage.ENTRY not in spec.unspecified_stages
    assert set(spec.unspecified_stages) == set(RuleStage) - {RuleStage.ENTRY}
    assert spec.is_executable is False


def test_hypothesis_does_not_satisfy_a_source_supported_stage() -> None:
    hypothesis = _claim(
        origin=RuleOrigin.RESEARCH_HYPOTHESIS_PARAMETER,
        evidence=None,
    )
    spec = _spec(rule_claims=(hypothesis,))

    assert RuleStage.ENTRY in spec.unspecified_stages


def test_rule_validation_result_uses_closed_status_contract() -> None:
    result = RuleValidationResult(
        rule_fingerprint="c" * 64,
        status=RuleValidationStatus.UNSUPPORTED,
        reason_code="GRAMMAR_MISMATCH",
    )

    assert result.status is RuleValidationStatus.UNSUPPORTED
    with pytest.raises(ValueError, match="status"):
        RuleValidationResult("c" * 64, "MAYBE", "GRAMMAR_MISMATCH")


def test_incomplete_source_claim_cannot_become_implementation_ready() -> None:
    incomplete = _claim(value=None, condition=None)
    spec = _spec(
        rule_claims=(incomplete,),
        suitability=StrategySuitability.HFT_SUITABLE,
    )

    assert spec.is_executable is False
    assert RuleStage.ENTRY in spec.unspecified_stages


def _complete_core_spec(*, optional_claim: StrategyRuleClaim | None = None) -> StrategySpec:
    claims = []
    definitions = {
        RuleStage.ENTRY: (RuleOperator.GREATER_THAN, "LONG", 2.0, "points", "after confirmation", None),
        RuleStage.CONFIRMATION: (
            RuleOperator.GREATER_OR_EQUAL,
            "LONG",
            0.6,
            "fraction",
            "after three ticks",
            None,
        ),
        RuleStage.INVALIDATION: (RuleOperator.LESS_OR_EQUAL, "LONG", 0.0, "points", "after reversal", None),
        RuleStage.EXPECTED_MOVE: (RuleOperator.GREATER_OR_EQUAL, "LONG", 5.0, "points", "after entry", 30),
        RuleStage.EXIT: (RuleOperator.LESS_OR_EQUAL, "LONG", 0.0, "points", "after reversal", None),
        RuleStage.PROFIT_PROTECTION: (
            RuleOperator.LESS_OR_EQUAL,
            "LONG",
            0.0,
            "points",
            "after target retracement",
            None,
        ),
        RuleStage.STOP_BEHAVIOR: (RuleOperator.LESS_OR_EQUAL, "LONG", -3.0, "points", "after adverse move", None),
        RuleStage.HORIZON: (RuleOperator.LESS_OR_EQUAL, "BOTH", 30, "seconds", "after entry", 30),
    }
    for stage in EXECUTABLE_REQUIRED_STAGES:
        operator, direction, value, unit, condition, horizon_seconds = definitions[stage]
        quote = f"{stage.value}: {direction} {operator.value} {value} {unit} {condition}"
        claims.append(
            _claim(
                stage=stage,
                operator=operator,
                direction=direction,
                value=value,
                unit=unit,
                condition=condition,
                horizon_seconds=horizon_seconds,
                evidence=_span(start_offset=0, end_offset=len(quote), quote=quote),
            )
        )
    if optional_claim is not None:
        claims.append(optional_claim)
    validations = tuple(
        RuleValidationResult(claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE")
        for claim in claims
    )
    return _spec(
        rule_claims=tuple(claims),
        validation_results=validations,
        required_data=("momentum", "direction_persistence"),
        suitability=StrategySuitability.HFT_SUITABLE,
    )


def test_core_complete_spec_can_be_executable_with_optional_stages_unspecified() -> None:
    spec = _complete_core_spec()

    assert set(spec.unspecified_stages) == set(RuleStage) - EXECUTABLE_REQUIRED_STAGES
    assert spec.is_executable is True


def test_optional_unsupported_claim_still_blocks_execution() -> None:
    optional = _claim(stage=RuleStage.SESSION_FILTER, value="ASIA")
    core = _complete_core_spec()
    claims = (*core.rule_claims, optional)
    validations = tuple(
        RuleValidationResult(
            claim.fingerprint,
            RuleValidationStatus.UNSUPPORTED if claim is optional else RuleValidationStatus.SUPPORTED,
            "UNSUPPORTED_OPTIONAL_RULE" if claim is optional else "EXACT_SOURCE_RULE",
        )
        for claim in claims
    )

    spec = _spec(
        rule_claims=claims,
        validation_results=validations,
        required_data=("momentum", "direction_persistence"),
        suitability=StrategySuitability.HFT_SUITABLE,
    )

    assert spec.is_executable is False
