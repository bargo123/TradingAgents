from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from tradingagents.experience.models import (
    EvaluationStatus,
    EvidenceBundle,
    EvidenceRequest,
    ExperienceHit,
    ExperienceQuery,
    ExperienceRecord,
    ExperienceSearchResult,
    OutcomeStatistics,
    OutcomeStatsRequest,
    TrustTier,
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
    record = ExperienceRecord(
        "exp-1", "db-1", "dec-1", "EURUSD", datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    assert '"experience_id":"exp-1"' in record.to_json()
    with pytest.raises(FrozenInstanceError):
        record.experience_id = "other"
    result = ExperienceSearchResult((ExperienceHit("exp-1"),), "fp", "gen-1", 1, {})
    assert result.to_dict()["hits"][0]["experience_id"] == "exp-1"


@pytest.mark.parametrize(
    "field",
    ["experience_id", "source_database_id", "source_decision_id", "symbol"],
)
@pytest.mark.parametrize("value", [None, 1, "  "])
def test_experience_record_rejects_invalid_required_identity(field: str, value: object) -> None:
    payload = {
        "experience_id": "exp-1",
        "source_database_id": "db-1",
        "source_decision_id": "dec-1",
        "symbol": "EURUSD",
        "analysis_snapshot_timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        ExperienceRecord(**payload)


def test_experience_record_requires_analysis_snapshot_timestamp() -> None:
    with pytest.raises(ValueError, match="analysis_snapshot_timestamp"):
        ExperienceRecord("exp-1", "db-1", "dec-1", "EURUSD", None)


@pytest.mark.parametrize("value", [None, 1, "  "])
def test_experience_hit_rejects_invalid_identity(value: object) -> None:
    with pytest.raises(ValueError, match="experience_id"):
        ExperienceHit(value)


def test_utc_timestamps_are_required() -> None:
    with pytest.raises(ValueError, match="UTC"):
        OutcomeStatsRequest(("x",), "ANALYSIS_SNAPSHOT", 300, as_of=datetime(2026, 1, 1))


def test_training_eligibility_is_only_provenance() -> None:
    record = ExperienceRecord(
        "exp-1",
        "db-1",
        "dec-1",
        "EURUSD",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        provenance={"training_eligible": True},
    )
    assert record.provenance["training_eligible"] is True
    assert "training_eligible" not in ExperienceRecord.__dataclass_fields__


def test_request_and_statistics_contracts_accept_defaults() -> None:
    assert EvidenceRequest().as_of is None
    assert OutcomeStatistics().eligible_sample_denominator == 0


def test_evaluation_status_includes_ineligible() -> None:
    assert EvaluationStatus.INELIGIBLE.value == "INELIGIBLE"


def test_nested_mappings_are_immutable_and_set_serialization_is_deterministic() -> None:
    record = ExperienceRecord(
        "x",
        "db",
        "d",
        "EURUSD",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        market_state={"values": {"b", "a"}},
    )
    with pytest.raises(TypeError):
        record.market_state["new"] = 1
    assert (
        record.to_json()
        == ExperienceRecord(
            "x",
            "db",
            "d",
            "EURUSD",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            market_state={"values": {"a", "b"}},
        ).to_json()
    )


@pytest.mark.parametrize(
    "field",
    [
        "source_aliases",
        "market_state",
        "decision_evidence",
        "outcome_evidence_by_basis_horizon",
        "provenance",
        "source_evaluation_fingerprints",
    ],
)
def test_experience_record_rejects_non_mapping_payloads(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        ExperienceRecord(
            "x",
            "db",
            "d",
            "EURUSD",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            **{field: ["not", "a", "mapping"]},
        )


def test_experience_record_rejects_malformed_alias_and_fingerprint_entries() -> None:
    base = {
        "experience_id": "x",
        "source_database_id": "db",
        "source_decision_id": "d",
        "symbol": "EURUSD",
        "analysis_snapshot_timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    with pytest.raises(ValueError, match="source_aliases"):
        ExperienceRecord(**base, source_aliases={None: "CURRENT"})
    with pytest.raises(ValueError, match="source_evaluation_fingerprints"):
        ExperienceRecord(**base, source_evaluation_fingerprints={"ANALYSIS_SNAPSHOT:300": 1})


def test_experience_hit_rejects_non_mapping_payloads() -> None:
    with pytest.raises(ValueError, match="market_state"):
        ExperienceHit("x", market_state=["not", "a", "mapping"])


def test_experience_hit_requires_boolean_tombstone_flag() -> None:
    with pytest.raises(ValueError, match="currently_tombstoned"):
        ExperienceHit("x", currently_tombstoned="false")


def test_training_eligible_is_rejected_outside_provenance() -> None:
    with pytest.raises(ValueError, match="training_eligible"):
        ExperienceRecord(
            "x",
            "db",
            "d",
            "EURUSD",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            decision_evidence={"training_eligible": True},
        )


def test_hit_timestamps_require_utc() -> None:
    with pytest.raises(ValueError, match="UTC"):
        ExperienceHit("x", timestamps={"completed": datetime(2026, 1, 1)})


def test_query_market_state_is_deeply_immutable() -> None:
    query = ExperienceQuery({"nested": {"price": 1}})
    with pytest.raises(TypeError):
        query.market_state["nested"]["price"] = 2
    with pytest.raises(ValueError, match="training_eligible"):
        ExperienceQuery({"training_eligible": True})


@pytest.mark.parametrize("value", [True, 1.5, "10"])
def test_experience_query_rejects_non_integer_top_k(value: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        ExperienceQuery({}, top_k=value)


@pytest.mark.parametrize("value", [True, 300.0, "300"])
def test_outcome_stats_request_rejects_non_integer_horizon(value: object) -> None:
    with pytest.raises(ValueError, match="horizon_seconds"):
        OutcomeStatsRequest(("exp-1",), horizon_seconds=value)


def test_outcome_stats_request_rejects_string_experience_ids() -> None:
    with pytest.raises(ValueError, match="experience_ids"):
        OutcomeStatsRequest("exp-1", horizon_seconds=300)


@pytest.mark.parametrize("field", ["knowledge_top_k", "experience_top_k"])
@pytest.mark.parametrize("value", [True, 0, -1, 1001, 1.5, "10"])
def test_evidence_request_rejects_invalid_top_k(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        EvidenceRequest(**{field: value})
