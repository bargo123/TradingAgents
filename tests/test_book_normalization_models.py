import importlib
from dataclasses import replace
from decimal import Decimal

import pytest

from tradingagents.self_enhancement.strategy_specs import (
    EvidenceSpan,
    RuleDirection,
    RuleOperator,
    RuleStage,
)


def models():
    return importlib.import_module("tradingagents.self_enhancement.book_normalization_models")


def span(start=0, text="a", **changes):
    return replace(EvidenceSpan("gen", "doc", "chunk", "a" * 64, start, start + len(text), text), **changes)


@pytest.mark.parametrize("count", [1, 8])
def test_bundle_accepts_bounds(count):
    assert len(models().EvidenceBundle(tuple(span(i * 2) for i in range(count))).spans) == count


@pytest.mark.parametrize("count", [0, 9])
def test_bundle_rejects_outside_bounds(count):
    with pytest.raises(ValueError):
        models().EvidenceBundle(tuple(span(i * 2) for i in range(count)))


@pytest.mark.parametrize("bad", ["document_id", "chunk_id", "generation_id", "source_hash"])
def test_bundle_rejects_mixed_identity(bad):
    with pytest.raises(ValueError):
        models().EvidenceBundle((span(), span(2, **{bad: "b" * 64})))


@pytest.mark.parametrize("spans", [(span(), span()), (span(2), span()), (span(0, "abc"), span(2))])
def test_bundle_rejects_duplicate_overlap_and_reorder(spans):
    with pytest.raises(ValueError):
        models().EvidenceBundle(spans)


def test_bundle_character_limit():
    assert models().EvidenceBundle((span(text="a" * 600),)).spans[0].end_offset == 600
    with pytest.raises(ValueError):
        models().EvidenceBundle((span(text="a" * 601),))


def rule():
    m = models()
    return m.NormalizedRuleV2(
        "rule", RuleStage.EXPECTED_MOVE, "OTHER_SUPPORTED", "price_distance",
        ("profit",), RuleDirection.UNSPECIFIED, RuleOperator.EQUALS,
        Decimal("0.10"), "pip", None, None, m.EvidenceBundle((span(text="0.10"),)),
        (m.FieldProof("value", 0, 0, 4),), "numeric-distance-v1",
        m.SCHEMA_VERSION, m.GRAMMAR_VERSION, m.FEATURE_CONTRACT_VERSION, "a" * 64,
    )


def test_rule_round_trip_keeps_decimal_and_evidence():
    original = rule()
    assert original.to_dict()["value"] == "0.10"
    assert models().NormalizedRuleV2.from_dict(original.to_dict()) == original
    assert models().normalization_identity(original) != models().normalization_identity(replace(original, value=Decimal("0.11")))


@pytest.mark.parametrize("changes", [{"schema_version": "bad"}, {"grammar_version": "bad"}, {"feature_contract_version": "bad"}, {"value": Decimal("NaN")}, {"value": Decimal("Infinity")}, {"horizon_seconds": True}])
def test_rule_rejects_invalid_fields(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(rule(), **changes)


@pytest.mark.parametrize("proof", [("value", True, 0, 1), ("value", 1, 0, 1), ("value", 0, 0, 5)])
def test_field_proofs_stay_inside_addressed_span(proof):
    with pytest.raises((ValueError, TypeError)):
        replace(rule(), field_proofs=(models().FieldProof(*proof),))


def test_unknown_serialized_keys_and_numeric_values_reject():
    for changes in ({"extra": 1}, {"value": 0.1}):
        with pytest.raises((ValueError, TypeError)):
            models().NormalizedRuleV2.from_dict({**rule().to_dict(), **changes})


def test_envelope_and_results_round_trip():
    m = models()
    env = m.StrategyEnvelopeV2("strategy", "OTHER_SUPPORTED", (rule(),), m.SCHEMA_VERSION)
    result = m.CandidateValidationV2(env, m.NormalizationStatus.SUPPORTED_NONEXECUTABLE, ("MISSING_STAGE",), (RuleStage.ENTRY,))
    assert m.CandidateValidationV2.from_dict(result.to_dict()) == result
    normalized = m.NormalizationResult(m.NormalizationStatus.SUPPORTED_NONEXECUTABLE, rule(), ("MISSING_HORIZON",))
    assert m.NormalizationResult.from_dict(normalized.to_dict()) == normalized
