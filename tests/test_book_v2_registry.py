import importlib
from dataclasses import replace

import pytest

from tests.test_book_natural_language import source_lookup
from tests.test_book_v2_pipeline import envelope
from tradingagents.self_enhancement.book_normalization_models import (
    CandidateValidationV2,
    NormalizationStatus,
)
from tradingagents.self_enhancement.phase14c_mapping import capture_current_strategy_contracts


def registry():
    return importlib.import_module("tradingagents.self_enhancement.book_v2_registry")


def claimed_candidate():
    return CandidateValidationV2(envelope(), NormalizationStatus.EXECUTABLE_ELIGIBLE, (), ())


def test_claimed_eligible_status_cannot_construct_strategy():
    with pytest.raises(ValueError, match="not replay eligible"):
        registry().create_research_strategy(claimed_candidate(), source_lookup=source_lookup)


def test_pip_distance_is_not_attributed_to_momentum_or_range():
    mappings = registry().map_v2(claimed_candidate(), snapshot=capture_current_strategy_contracts(), source_lookup=source_lookup)
    assert len(mappings) == 2
    assert all(component.status.value == "NO_DIRECT_MATCH" for mapping in mappings for component in mapping.components)


def test_mapping_revalidates_tampered_proof():
    candidate = claimed_candidate()
    candidate = replace(candidate, envelope=replace(candidate.envelope, rules=(replace(candidate.envelope.rules[0], condition="after entry"),)))
    with pytest.raises(ValueError, match="proof"):
        registry().map_v2(candidate, snapshot=capture_current_strategy_contracts(), source_lookup=source_lookup)
