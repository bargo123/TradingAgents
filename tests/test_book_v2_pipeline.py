import importlib
from dataclasses import replace

from tests.test_book_natural_language import resolve, source_lookup
from tests.test_book_natural_language_sources import source_cases
from tests.test_book_v2_extraction import _Transport, selection, sentence
from tradingagents.self_enhancement.book_drafter import OllamaStrategyDrafter
from tradingagents.self_enhancement.book_normalization_models import (
    SCHEMA_VERSION,
    NormalizationStatus,
    StrategyEnvelopeV2,
)
from tradingagents.self_enhancement.strategy_specs import RuleStage


def pipeline():
    return importlib.import_module("tradingagents.self_enhancement.book_v2_pipeline")


def envelope():
    return StrategyEnvelopeV2("candidate", "OTHER_SUPPORTED", (resolve(source_cases()[0]).rule,), SCHEMA_VERSION)


def test_incomplete_target_is_not_an_executable_strategy():
    result = pipeline().validate_envelope(envelope(), source_lookup=source_lookup)
    assert result.status is NormalizationStatus.SUPPORTED_NONEXECUTABLE
    assert set(result.missing_stages) == {RuleStage.ENTRY, RuleStage.CONFIRMATION, RuleStage.EXIT, RuleStage.HORIZON, RuleStage.INVALIDATION, RuleStage.PROFIT_PROTECTION, RuleStage.STOP_BEHAVIOR}
    assert "UNSUPPORTED_PRIMITIVE" in result.reason_codes


def test_changed_evidence_or_stage_rejected_across_assembly():
    e = envelope()
    tampered = replace(e, rules=(replace(e.rules[0], stage=RuleStage.ENTRY),))
    assert pipeline().validate_envelope(tampered, source_lookup=source_lookup).status is NormalizationStatus.REJECTED


def test_assembly_does_not_stitch_unrelated_books_or_trust_claimed_status():
    results = tuple(resolve(c) for c in source_cases()[:2])
    asserted = tuple(replace(r, status=NormalizationStatus.EXECUTABLE_ELIGIBLE) for r in results)
    candidates = pipeline().assemble_v2(asserted, source_lookup=source_lookup)
    assert len(candidates) == 2
    assert all(c.status is NormalizationStatus.SUPPORTED_NONEXECUTABLE for c in candidates)


def test_drafter_dispatch_uses_model_and_returns_v2_candidates():
    transport = _Transport(selection)
    drafter = OllamaStrategyDrafter("http://127.0.0.1:11434", "qwen3.5:2b", transport=transport)
    candidates = drafter.draft_v2((sentence().hit,), source_lookup=source_lookup)
    assert candidates[0].envelope.rules[0].value == 10
    assert candidates[0].status is NormalizationStatus.SUPPORTED_NONEXECUTABLE
