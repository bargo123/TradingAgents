from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from tradingagents.experience.models import (
    EvaluationStatus,
    EvidenceBundle,
    EvidenceRequest,
    EvidenceSourceError,
    EvidenceWarning,
    ExperienceHit,
    ExperienceQuery,
    ExperienceRecord,
    ExperienceSearchResult,
    OrchestrationProvenance,
    OutcomeDirectionStatistics,
    OutcomeHoldStatistics,
    OutcomeStatistics,
    OutcomeStatsRequest,
    TrustTier,
)
from tradingagents.forex.evidence_context import CanonicalKnowledgeQuery


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


@pytest.mark.parametrize("field", ["symbol", "analysis_profile", "analysis_timeframe", "action_filter"])
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_experience_query_rejects_non_text_filters(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        ExperienceQuery({}, **{field: value})


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


@pytest.mark.parametrize("value", [False, 0, "bad", {"experience_id": "exp-1"}])
def test_experience_search_result_rejects_non_hit_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="hits"):
        ExperienceSearchResult((value,))


@pytest.mark.parametrize("field", ["query_normalization_fingerprint", "active_generation_id"])
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_experience_search_result_rejects_non_text_metadata(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        ExperienceSearchResult(**{field: value})


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_experience_search_result_rejects_invalid_candidate_count(value: object) -> None:
    with pytest.raises(ValueError, match="candidate_count"):
        ExperienceSearchResult(candidate_count=value)


def test_experience_search_result_rejects_invalid_excluded_count_values() -> None:
    with pytest.raises(ValueError, match="excluded_counts"):
        ExperienceSearchResult(excluded_counts={"invalid": "1"})


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


@pytest.mark.parametrize(
    "field",
    [
        "source_run_id",
        "requested_symbol",
        "analysis_profile",
        "analysis_timeframe",
        "source_decision_fingerprint",
    ],
)
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_experience_record_rejects_non_text_optional_identity(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        ExperienceRecord(
            "exp-1",
            "db-1",
            "dec-1",
            "EURUSD",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            **{field: value},
        )


@pytest.mark.parametrize("value", [None, 1, "  "])
def test_experience_hit_rejects_invalid_identity(value: object) -> None:
    with pytest.raises(ValueError, match="experience_id"):
        ExperienceHit(value)


@pytest.mark.parametrize(
    "field",
    ["source_decision_id", "action", "feature_schema_version", "similarity_profile_version"],
)
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_experience_hit_rejects_non_text_optional_identity(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        ExperienceHit("exp-1", **{field: value})


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


@pytest.mark.parametrize(
    "field",
    [
        "experience_schema_version",
        "feature_schema_version",
        "feature_extractor_version",
        "similarity_profile_version",
        "trust_policy_version",
        "statistics_policy_version",
    ],
)
@pytest.mark.parametrize("value", [None, False, 0, [], {}, "   ", "x" * 257])
def test_experience_record_rejects_invalid_version_metadata(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        ExperienceRecord(
            "exp-1",
            "db-1",
            "dec-1",
            "EURUSD",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            **{field: value},
        )


def test_request_and_statistics_contracts_accept_defaults() -> None:
    assert EvidenceRequest().as_of is None
    assert OutcomeStatistics().eligible_sample_denominator == 0


def test_outcome_statistics_preserves_explicit_zero_eligible_count() -> None:
    result = OutcomeStatistics(eligible_sample_denominator=7, eligible_count=0)

    assert result.eligible_count == 0


def test_outcome_statistics_preserves_explicit_requested_basis() -> None:
    result = OutcomeStatistics(evaluation_basis="ANALYSIS_SNAPSHOT", requested_basis="")

    assert result.requested_basis == ""


@pytest.mark.parametrize("field", ["buy", "sell", "hold"])
@pytest.mark.parametrize("value", [False, 0, [], {}, "malformed"])
def test_outcome_statistics_rejects_malformed_nested_statistics(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        OutcomeStatistics(**{field: value})


@pytest.mark.parametrize("value", [[], False, 0])
def test_search_result_rejects_malformed_excluded_counts(value: object) -> None:
    with pytest.raises(ValueError, match="excluded_counts"):
        ExperienceSearchResult(excluded_counts=value)


@pytest.mark.parametrize("field", ["mfe_quantiles", "mae_quantiles"])
@pytest.mark.parametrize("value", [[], False, 0])
def test_direction_statistics_reject_malformed_quantile_mappings(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        OutcomeDirectionStatistics(**{field: value})


@pytest.mark.parametrize("value", [False, "bad", {}, [True]])
def test_direction_statistics_reject_malformed_numeric_sequences(value: object) -> None:
    with pytest.raises(ValueError, match="net_points"):
        OutcomeDirectionStatistics(net_points=value)


def test_direction_statistics_reject_invalid_quantile_values() -> None:
    with pytest.raises(ValueError, match="mfe_quantiles"):
        OutcomeDirectionStatistics(mfe_quantiles={"p50": "1"})


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_direction_statistics_reject_invalid_count(value: object) -> None:
    with pytest.raises(ValueError, match="count"):
        OutcomeDirectionStatistics(count=value)


@pytest.mark.parametrize("value", [[], False, 0])
def test_hold_statistics_reject_malformed_counterfactual_mapping(value: object) -> None:
    with pytest.raises(ValueError, match="best_counterfactual_counts"):
        OutcomeHoldStatistics(best_counterfactual_counts=value)


@pytest.mark.parametrize("value", [False, "bad", {}, [True]])
def test_hold_statistics_reject_malformed_numeric_sequences(value: object) -> None:
    with pytest.raises(ValueError, match="opportunity_cost_points"):
        OutcomeHoldStatistics(opportunity_cost_points=value)


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_hold_statistics_reject_invalid_count(value: object) -> None:
    with pytest.raises(ValueError, match="count"):
        OutcomeHoldStatistics(count=value)


@pytest.mark.parametrize(
    "field",
    ["excluded_counts", "source_evaluation_fingerprints", "exclusions_by_status", "exclusions_by_tier"],
)
@pytest.mark.parametrize("value", [[], False, 0])
def test_outcome_statistics_reject_malformed_mappings(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        OutcomeStatistics(**{field: value})


@pytest.mark.parametrize("field", ["eligible_sample_denominator", "eligible_count"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_outcome_statistics_reject_invalid_counts(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        OutcomeStatistics(**{field: value})


@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_outcome_statistics_rejects_non_text_basis(value: object) -> None:
    with pytest.raises(ValueError, match="evaluation_basis"):
        OutcomeStatistics(evaluation_basis=value)


def test_outcome_statistics_rejects_non_text_fingerprint_values() -> None:
    with pytest.raises(ValueError, match="source_evaluation_fingerprints"):
        OutcomeStatistics(source_evaluation_fingerprints={"x": 1})


@pytest.mark.parametrize("field", ["source_status", "provenance"])
@pytest.mark.parametrize("value", [[], False, 0])
def test_evidence_bundle_rejects_malformed_mappings(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        EvidenceBundle(**{field: value})


@pytest.mark.parametrize("field", ["text", "fingerprint", "policy_version"])
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_canonical_knowledge_query_rejects_non_text_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        CanonicalKnowledgeQuery(**{field: value})


@pytest.mark.parametrize("factory", [EvidenceSourceError, EvidenceWarning])
def test_evidence_diagnostics_reject_non_text_fields(factory) -> None:
    kwargs = (
        {"source": False, "error_type": "x", "message": "m"}
        if factory is EvidenceSourceError
        else {"code": False, "message": "m"}
    )
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize("value", [False, 0, [], {}, "UNKNOWN"])
def test_evidence_bundle_rejects_invalid_status(value: object) -> None:
    with pytest.raises(ValueError, match="status"):
        EvidenceBundle(status=value)


@pytest.mark.parametrize("field", ["knowledge", "experience", "warnings", "errors"])
def test_evidence_bundle_rejects_scalar_collections(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        EvidenceBundle(**{field: "not-a-sequence"})


@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_orchestration_provenance_rejects_non_text_fingerprint(value: object) -> None:
    with pytest.raises(ValueError, match="query_normalization_fingerprint"):
        OrchestrationProvenance(query_normalization_fingerprint=value)


@pytest.mark.parametrize("field", ["knowledge_requested", "experience_requested"])
@pytest.mark.parametrize("value", [0, 1, "true", [], {}])
def test_orchestration_provenance_rejects_non_boolean_flags(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        OrchestrationProvenance(**{field: value})


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


@pytest.mark.parametrize("value", [None, False, 0, [], {}])
def test_outcome_stats_request_rejects_non_text_basis(value: object) -> None:
    with pytest.raises(ValueError, match="evaluation_basis"):
        OutcomeStatsRequest(("exp-1",), evaluation_basis=value, horizon_seconds=300)


@pytest.mark.parametrize("field", ["knowledge_top_k", "experience_top_k"])
@pytest.mark.parametrize("value", [True, 0, -1, 1001, 1.5, "10"])
def test_evidence_request_rejects_invalid_top_k(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        EvidenceRequest(**{field: value})


@pytest.mark.parametrize("value", [True, -1, 1.5, "300"])
def test_evidence_request_rejects_invalid_horizon(value: object) -> None:
    with pytest.raises(ValueError, match="horizon_seconds"):
        EvidenceRequest(evaluation_basis="ANALYSIS_SNAPSHOT", horizon_seconds=value)


@pytest.mark.parametrize("field", ["content_types", "document_ids"])
def test_evidence_request_rejects_scalar_filter_sequences(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        EvidenceRequest(**{field: "not-a-sequence"})


@pytest.mark.parametrize(
    "field",
    [
        "research_question",
        "symbol",
        "analysis_profile",
        "analysis_timeframe",
        "evaluation_basis",
        "action_filter",
    ],
)
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_evidence_request_rejects_non_text_optional_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        EvidenceRequest(**{field: value})
