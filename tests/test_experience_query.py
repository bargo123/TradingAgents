from datetime import datetime, timezone

import pytest

from tradingagents.experience.models import ExperienceQuery, TrustTier
from tradingagents.experience.normalization import NormalizationCohortV1, SimilarityProfileV1
from tradingagents.experience.query import ExperienceQueryService


UTC = timezone.utc
COHORT = NormalizationCohortV1("EURUSD", "P", "M15", "experience-features.v1", "extractor")
NAMES = tuple(f"f{i}" for i in range(8))


def profile():
    return SimilarityProfileV1({n: 0.0 for n in NAMES}, {n: 1.0 for n in NAMES}, {n: 1.0 for n in NAMES}, {n: 1.0 for n in NAMES}, {n: 1.0 for n in NAMES}, 1e-9, {n: 1.0 for n in NAMES}, (-8, 8), NAMES, "exclude-missing.v1", (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED), COHORT)


def row(eid, *, tier=TrustTier.TIER_A_HIGH_TRUST, completed=datetime(2026, 1, 1, tzinfo=UTC), tombstoned=False):
    return {"experience_id": eid, "symbol": "EURUSD", "analysis_profile": "P", "analysis_timeframe": "M15",
            "analysis_snapshot_timestamp": datetime(2026, 1, 1, tzinfo=UTC), "decision_completed_timestamp": completed,
            "trust": tier, "trust_tier": tier, "tombstoned": tombstoned, "currently_tombstoned": tombstoned,
            "source_aliases": {"db:d": "REMOVED" if tombstoned else "CURRENT"},
            "values": tuple([0.0] * 8), "mask": tuple([True] * 8), "feature_names": NAMES,
            "cohort": COHORT, "feature_schema_version": COHORT.feature_schema_version,
            "feature_extractor_version": COHORT.feature_extractor_version, "accepted": True, "provenance_valid": True,
            "feature_fingerprint": "fp"}


def test_default_query_excludes_tier_c_and_current_tombstones():
    service = ExperienceQueryService([row("a"), row("c", tier=TrustTier.TIER_C_DIAGNOSTIC_ONLY), row("gone", tombstoned=True)], profile=profile(), generation_id="g1")
    result = service.search(ExperienceQuery({"values": [0.0] * 8, "mask": [True] * 8, "feature_names": NAMES, "cohort": COHORT}))
    assert [h.experience_id for h in result.hits] == ["a"]
    assert result.candidate_count == 3
    assert result.excluded_counts["trust_tier"] == 1
    assert result.excluded_counts["tombstone"] == 1
    assert result.active_generation_id == "g1"


def test_historical_query_allows_tombstones_but_requires_completion_before_cutoff():
    service = ExperienceQueryService([row("gone", tombstoned=True), row("future", completed=datetime(2026, 1, 3, tzinfo=UTC))], profile=profile())
    result = service.search(ExperienceQuery({"values": [0.0] * 8, "mask": [True] * 8, "feature_names": NAMES, "cohort": COHORT}, as_of=datetime(2026, 1, 2, tzinfo=UTC)))
    assert [h.experience_id for h in result.hits] == ["gone"]
    assert result.hits[0].currently_tombstoned is True
    assert result.excluded_counts["as_of"] == 1


def test_tier_c_numeric_similarity_is_rejected():
    with pytest.raises(ValueError, match="Tier C"):
        ExperienceQueryService([row("c", tier=TrustTier.TIER_C_DIAGNOSTIC_ONLY)], profile=profile()).search(
            ExperienceQuery({"values": [0.0] * 8, "mask": [True] * 8, "feature_names": NAMES, "cohort": COHORT}, trust_tiers=(TrustTier.TIER_C_DIAGNOSTIC_ONLY,)))


def test_query_excludes_explicit_schema_or_extractor_mismatch():
    bad_schema = row("bad-schema") | {"feature_schema_version": "wrong-schema"}
    bad_extractor = row("bad-extractor") | {"feature_extractor_version": "wrong-extractor"}
    service = ExperienceQueryService([row("good"), bad_schema, bad_extractor], profile=profile())
    result = service.search(ExperienceQuery({"values": [0.0] * 8, "mask": [True] * 8, "feature_names": NAMES, "cohort": COHORT}))
    assert [h.experience_id for h in result.hits] == ["good"]
    assert result.excluded_counts["feature_schema"] == 1
    assert result.excluded_counts["feature_extractor"] == 1


def test_query_excludes_invalid_timestamp_under_as_of():
    invalid = row("invalid") | {"analysis_snapshot_timestamp": datetime(2026, 1, 1)}
    service = ExperienceQueryService([row("good"), invalid], profile=profile())
    result = service.search(ExperienceQuery({"values": [0.0] * 8, "mask": [True] * 8, "feature_names": NAMES, "cohort": COHORT}, as_of=datetime(2026, 1, 2, tzinfo=UTC)))
    assert [h.experience_id for h in result.hits] == ["good"]
    assert result.excluded_counts["as_of"] == 1
