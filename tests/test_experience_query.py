from datetime import datetime, timezone

import pytest

from tradingagents.experience.models import ExperienceQuery, TrustTier
from tradingagents.experience.normalization import (
    NormalizationCohortV1,
    SimilarityProfileV1,
    build_profile,
    query_normalization_fingerprint,
)
from tradingagents.experience.query import ExperienceQueryService

UTC = timezone.utc
COHORT = NormalizationCohortV1("EURUSD", "P", "M15", "experience-features.v1", "extractor")
NAMES = tuple(f"f{i}" for i in range(8))


def profile():
    return SimilarityProfileV1(
        dict.fromkeys(NAMES, 0.0),
        dict.fromkeys(NAMES, 1.0),
        dict.fromkeys(NAMES, 1.0),
        dict.fromkeys(NAMES, 1.0),
        dict.fromkeys(NAMES, 1.0),
        1e-9,
        dict.fromkeys(NAMES, 1.0),
        (-8, 8),
        NAMES,
        "exclude-missing.v1",
        (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED),
        COHORT,
    )


def row(
    eid,
    *,
    tier=TrustTier.TIER_A_HIGH_TRUST,
    completed=datetime(2026, 1, 1, tzinfo=UTC),
    tombstoned=False,
):
    return {
        "experience_id": eid,
        "symbol": "EURUSD",
        "analysis_profile": "P",
        "analysis_timeframe": "M15",
        "analysis_snapshot_timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "decision_completed_timestamp": completed,
        "trust": tier,
        "trust_tier": tier,
        "tombstoned": tombstoned,
        "currently_tombstoned": tombstoned,
        "source_aliases": {"db:d": "REMOVED" if tombstoned else "CURRENT"},
        "values": tuple([0.0] * 8),
        "mask": tuple([True] * 8),
        "feature_names": NAMES,
        "cohort": COHORT,
        "feature_schema_version": COHORT.feature_schema_version,
        "feature_extractor_version": COHORT.feature_extractor_version,
        "accepted": True,
        "provenance_valid": True,
        "feature_fingerprint": "fp",
    }


def query_state(**extra):
    state = {"values": [0.0] * 8, "mask": [True] * 8, "feature_names": NAMES, "cohort": COHORT}
    state.update(extra)
    return state


def test_default_query_excludes_tier_c_and_current_tombstones():
    service = ExperienceQueryService(
        [row("a"), row("c", tier=TrustTier.TIER_C_DIAGNOSTIC_ONLY), row("gone", tombstoned=True)],
        profile=profile(),
        generation_id="g1",
    )
    result = service.search(ExperienceQuery(query_state()))
    assert [h.experience_id for h in result.hits] == ["a"]
    assert result.candidate_count == 3
    assert result.excluded_counts["trust_tier"] == 1
    assert result.excluded_counts["tombstone"] == 1
    assert result.active_generation_id == "g1"


def test_historical_query_allows_tombstones_but_requires_completion_before_cutoff():
    service = ExperienceQueryService(
        [row("gone", tombstoned=True), row("future", completed=datetime(2026, 1, 3, tzinfo=UTC))],
        profile=profile(),
    )
    result = service.search(ExperienceQuery(query_state(), as_of=datetime(2026, 1, 2, tzinfo=UTC)))
    assert [h.experience_id for h in result.hits] == ["gone"]
    assert result.hits[0].currently_tombstoned is True
    assert result.excluded_counts["as_of"] == 1


def test_tier_c_numeric_similarity_is_rejected():
    with pytest.raises(ValueError, match="Tier C"):
        ExperienceQueryService(
            [row("c", tier=TrustTier.TIER_C_DIAGNOSTIC_ONLY)], profile=profile()
        ).search(ExperienceQuery(query_state(), trust_tiers=(TrustTier.TIER_C_DIAGNOSTIC_ONLY,)))


def test_query_excludes_explicit_schema_or_extractor_mismatch():
    bad_schema = row("bad-schema") | {"feature_schema_version": "wrong-schema"}
    bad_extractor = row("bad-extractor") | {"feature_extractor_version": "wrong-extractor"}
    service = ExperienceQueryService([row("good"), bad_schema, bad_extractor], profile=profile())
    result = service.search(ExperienceQuery(query_state()))
    assert [h.experience_id for h in result.hits] == ["good"]
    assert result.excluded_counts["feature_schema"] == 1
    assert result.excluded_counts["feature_extractor"] == 1


def test_query_excludes_invalid_timestamp_under_as_of():
    invalid = row("invalid") | {"analysis_snapshot_timestamp": datetime(2026, 1, 1)}
    service = ExperienceQueryService([row("good"), invalid], profile=profile())
    result = service.search(ExperienceQuery(query_state(), as_of=datetime(2026, 1, 2, tzinfo=UTC)))
    assert [h.experience_id for h in result.hits] == ["good"]
    assert result.excluded_counts["as_of"] == 1


def test_query_excludes_malformed_timestamp_string_under_as_of():
    invalid = row("invalid") | {"analysis_snapshot_timestamp": "not-a-timestamp"}
    service = ExperienceQueryService([row("good"), invalid], profile=profile())
    result = service.search(
        ExperienceQuery(
            {"values": [0.0] * 8, "mask": [True] * 8, "feature_names": NAMES, "cohort": COHORT},
            as_of=datetime(2026, 1, 2, tzinfo=UTC),
        )
    )
    assert [h.experience_id for h in result.hits] == ["good"]
    assert result.excluded_counts["as_of"] == 1


def test_non_default_tiers_use_query_local_profile():
    a = row("a", tier=TrustTier.TIER_A_HIGH_TRUST) | {"values": tuple([0.0] * 8)}
    b = row("b", tier=TrustTier.TIER_B_LIMITED) | {"values": tuple([100.0] * 8)}
    service = ExperienceQueryService([a, b], profile=profile())
    query = ExperienceQuery(query_state(), trust_tiers=(TrustTier.TIER_A_HIGH_TRUST,))
    result = service.search(query)
    local = build_profile([a], COHORT, (TrustTier.TIER_A_HIGH_TRUST,), None)
    assert result.query_normalization_fingerprint == query_normalization_fingerprint(
        local, COHORT, query.trust_tiers, None
    )


def test_action_filter_is_applied_after_identity_gates_without_changing_scores():
    buy = row("buy") | {"action": "BUY"}
    sell = row("sell") | {"action": "SELL", "values": tuple([1.0] * 8)}
    service = ExperienceQueryService([buy, sell], profile=profile())
    unfiltered = service.search(ExperienceQuery(query_state()))
    filtered = service.search(ExperienceQuery(query_state(), action_filter="BUY"))
    assert [h.experience_id for h in filtered.hits] == ["buy"]
    assert filtered.hits[0].distance == next(
        h.distance for h in unfiltered.hits if h.experience_id == "buy"
    )
    assert filtered.excluded_counts["action"] == 1


def test_query_market_state_cohort_must_match_selected_profile():
    service = ExperienceQueryService([row("good")], profile=profile())
    result = service.search(
        ExperienceQuery(
            query_state(
                cohort=NormalizationCohortV1(
                    "XAUUSD", "P", "M15", "experience-features.v1", "extractor"
                )
            )
        )
    )
    assert result.hits == ()
    assert result.excluded_counts["query_cohort"] == 1


def test_query_malformed_non_none_cohort_fails_closed():
    service = ExperienceQueryService([row("good")], profile=profile())
    result = service.search(ExperienceQuery(query_state(cohort="malformed-cohort")))
    assert result.hits == ()
    assert result.excluded_counts["query_cohort_invalid"] == 1


def test_query_market_state_feature_names_must_match_selected_profile():
    service = ExperienceQueryService([row("good")], profile=profile())
    result = service.search(ExperienceQuery(query_state(feature_names=tuple(reversed(NAMES)))))
    assert result.hits == ()
    assert result.excluded_counts["query_feature_names"] == 1
