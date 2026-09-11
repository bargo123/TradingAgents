from datetime import datetime, timezone
from dataclasses import FrozenInstanceError

import pytest

from tradingagents.experience.models import (
    EvidenceBundle,
    EvidenceRequest,
    ExperienceHit,
    ExperienceQuery,
    ExperienceRecord,
    ExperienceSearchResult,
    OutcomeStatistics,
    OutcomeStatsRequest,
    TrustTier,
    EvaluationStatus,
)


def test_outcome_stats_request_carries_as_of() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    request = OutcomeStatsRequest(("exp-1",), "ANALYSIS_SNAPSHOT", 300, as_of=cutoff)
    assert request.as_of == cutoff
    assert request.evaluation_basis == "ANALYSIS_SNAPSHOT"


def test_experience_query_defaults_to_tier_a_and_b() -> None:
    assert ExperienceQuery({}).trust_tiers == (
        TrustTier.TIER_A_HIGH_TRUST,
        TrustTier.TIER_B_LIMITED,
    )


def test_evidence_bundle_has_no_recommendation_field() -> None:
    assert not hasattr(EvidenceBundle, "recommendation")


def test_contracts_are_frozen_and_json_serializable() -> None:
    record = ExperienceRecord("exp-1", "db-1", "dec-1", "EURUSD", datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert '"experience_id":"exp-1"' in record.to_json()
    with pytest.raises(FrozenInstanceError):
        record.experience_id = "other"
    result = ExperienceSearchResult((ExperienceHit("exp-1"),), "fp", "gen-1", 1, {})
    assert result.to_dict()["hits"][0]["experience_id"] == "exp-1"


def test_utc_timestamps_are_required() -> None:
    with pytest.raises(ValueError, match="UTC"):
        OutcomeStatsRequest(("x",), "ANALYSIS_SNAPSHOT", 300, as_of=datetime(2026, 1, 1))


def test_training_eligibility_is_only_provenance() -> None:
    record = ExperienceRecord("exp-1", "db-1", "dec-1", "EURUSD", datetime(2026, 1, 1, tzinfo=timezone.utc), provenance={"training_eligible": True})
    assert record.provenance["training_eligible"] is True
    assert "training_eligible" not in ExperienceRecord.__dataclass_fields__


def test_request_and_statistics_contracts_accept_defaults() -> None:
    assert EvidenceRequest().as_of is None
    assert OutcomeStatistics().eligible_sample_denominator == 0


def test_evaluation_status_includes_ineligible() -> None:
    assert EvaluationStatus.INELIGIBLE.value == "INELIGIBLE"


def test_nested_mappings_are_immutable_and_set_serialization_is_deterministic() -> None:
    record = ExperienceRecord("x", "db", "d", "EURUSD", datetime(2026, 1, 1, tzinfo=timezone.utc), market_state={"values": {"b", "a"}})
    with pytest.raises(TypeError):
        record.market_state["new"] = 1
    assert record.to_json() == ExperienceRecord("x", "db", "d", "EURUSD", datetime(2026, 1, 1, tzinfo=timezone.utc), market_state={"values": {"a", "b"}}).to_json()


def test_training_eligible_is_rejected_outside_provenance() -> None:
    with pytest.raises(ValueError, match="training_eligible"):
        ExperienceRecord("x", "db", "d", "EURUSD", datetime(2026, 1, 1, tzinfo=timezone.utc), decision_evidence={"training_eligible": True})


def test_hit_timestamps_require_utc() -> None:
    with pytest.raises(ValueError, match="UTC"):
        ExperienceHit("x", timestamps={"completed": datetime(2026, 1, 1)})


def test_query_market_state_is_deeply_immutable() -> None:
    query = ExperienceQuery({"nested": {"price": 1}})
    with pytest.raises(TypeError):
        query.market_state["nested"]["price"] = 2
    with pytest.raises(ValueError, match="training_eligible"):
        ExperienceQuery({"training_eligible": True})
