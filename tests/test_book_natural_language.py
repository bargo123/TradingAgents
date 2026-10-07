from dataclasses import replace
from decimal import Decimal
import importlib

import pytest

from tests.test_book_natural_language_sources import source_cases
from tradingagents.self_enhancement.book_normalization_models import EvidenceBundle, NormalizationStatus
from tradingagents.self_enhancement.strategy_specs import EvidenceSpan, RuleStage, RuleOperator


def resolver():
    return importlib.import_module("tradingagents.self_enhancement.book_natural_language")


def source_lookup(e):
    return " " * e.start_offset + e.quote


def resolve(case):
    return resolver().normalize_rule(EvidenceBundle((EvidenceSpan(**case["evidence"]),)), stage=RuleStage(case["stage"]), family=case["family"], source_lookup=source_lookup)


@pytest.mark.parametrize("case", source_cases(), ids=lambda c: c["case_id"])
def test_genuine_case_meaning_and_blockers(case):
    result = resolve(case)
    assert result.status.value == case["expected_status"]
    assert set(result.reason_codes) >= set(case["expected_blockers"])
    if result.rule:
        assert result.rule.operation == case["expected_operation"]
        assert result.rule.unit in {"pip", "pips"}
        assert result.rule.operands[0] != "momentum"


def test_target_has_exact_value_without_invented_direction_or_horizon():
    r = resolve(source_cases()[0]).rule
    assert r.value == Decimal("10")
    assert r.operator is RuleOperator.EQUALS
    assert r.direction.value == "UNSPECIFIED"
    assert r.horizon_seconds is None
    proof = next(p for p in r.field_proofs if p.field == "value")
    assert proof.start_offset == 164 and proof.end_offset == 166


@pytest.mark.parametrize("change", [
    {"value": Decimal("11")}, {"operator": RuleOperator.LESS_THAN},
    {"stage": RuleStage.ENTRY}, {"semantic_fingerprint": "b" * 64},
    {"field_proofs": ()},
])
def test_serialized_semantics_never_override_source(change):
    original = resolve(source_cases()[0]).rule
    try:
        altered = replace(original, **change)
    except ValueError:
        assert change == {"field_proofs": ()}
        return
    assert resolver().verify_rule(altered, source_lookup=source_lookup).status is NormalizationStatus.REJECTED


@pytest.mark.parametrize("quote", [
    "Do not place an initial protective stop no more than 20 pips below the entry",
    "For example, place an initial protective stop no more than 20 pips below the entry",
    "Place an initial protective stop no more than 20 pips below the entry, unless volatility rises",
    "Place an initial protective stop no more than 20 pips below the entr\ufffdy",
    "The target on each trade is not a non-adjustable one and set to 10 pip of profit.",
    "Holding limit: 10 seconds after entry.",
    "Buy when momentum is greater than 10 points.",
])
def test_unreviewed_or_changed_constructions_reject(quote):
    e = EvidenceSpan("gen", "doc", "chunk", "a" * 64, 0, len(quote), quote)
    assert resolver().normalize_rule(EvidenceBundle((e,)), stage=RuleStage.STOP_BEHAVIOR, family="OTHER_SUPPORTED", source_lookup=source_lookup).status is NormalizationStatus.REJECTED


def test_stale_source_and_wrong_stage_reject():
    case = source_cases()[0]
    e = EvidenceSpan(**case["evidence"])
    kwargs = dict(family=case["family"], stage=RuleStage.EXPECTED_MOVE)
    assert resolver().normalize_rule(EvidenceBundle((e,)), source_lookup=lambda _: "wrong", **kwargs).reason_codes == ("SPAN_MISMATCH",)
    assert resolver().normalize_rule(EvidenceBundle((e,)), source_lookup=source_lookup, **{**kwargs, "stage": RuleStage.ENTRY}).reason_codes == ("STAGE_MISMATCH",)


def test_unaddressed_context_cannot_define_a_missing_direction():
    case = source_cases()[1]
    e = EvidenceSpan(**case["evidence"])
    r = resolver().normalize_rule(EvidenceBundle((e,)), stage=RuleStage.STOP_BEHAVIOR, family=case["family"], source_lookup=lambda _: "Buy long. " + " " * (e.start_offset - 10) + e.quote)
    assert "MISSING_DIRECTION" in r.reason_codes
