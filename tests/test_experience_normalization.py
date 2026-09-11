from datetime import datetime, timezone

import pytest

from tradingagents.experience.features import FEATURE_NAMES_V1
from tradingagents.experience.models import TrustTier
from tradingagents.experience.normalization import (
    NormalizationCohortV1,
    build_profile,
    query_normalization_fingerprint,
)


UTC = timezone.utc
COHORT = NormalizationCohortV1("EURUSD", "INTRADAY", "M15", "experience-features.v1", "phase8-feature-extractor.v1")


def row(value: float, when: datetime, *, tier=TrustTier.TIER_A_HIGH_TRUST, cohort=COHORT, current=True, completion=None):
    values = tuple(value + i * 0.01 for i in range(len(FEATURE_NAMES_V1)))
    return {
        "values": values,
        "mask": tuple(True for _ in values),
        "feature_names": FEATURE_NAMES_V1,
        "cohort": cohort,
        "trust_tier": tier,
        "analysis_snapshot_timestamp": when,
        "decision_completed_timestamp": completion or when,
        "source_aliases": {"db:d1": "CURRENT"} if current else {"db:d1": "REMOVED"},
        "feature_fingerprint": "valid",
        "accepted": True,
    }


def test_zero_iqr_uses_mad_then_versioned_fallback():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [row(1.0, when) for _ in range(5)]
    rows[-1] = row(10.0, datetime(2026, 1, 1, 1, tzinfo=UTC))
    profile = build_profile(rows, cohort=COHORT, trust_tiers=(TrustTier.TIER_A_HIGH_TRUST,), as_of=None)
    assert profile.scales["spread_points"] == profile.mad_scales["spread_points"]
    assert profile.fallback_policy_version == "mad-then-feature-fallback.v1"


def test_unrelated_rows_do_not_change_exact_cohort_profile():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    base_rows = [row(1.0, when), row(2.0, datetime(2026, 1, 1, 1, tzinfo=UTC))]
    unrelated_symbol = row(999.0, when, cohort=NormalizationCohortV1("XAUUSD", "INTRADAY", "M15", COHORT.feature_schema_version, COHORT.feature_extractor_version))
    unrelated_timeframe = row(999.0, when, cohort=NormalizationCohortV1("EURUSD", "SWING", "H1", COHORT.feature_schema_version, COHORT.feature_extractor_version))
    base = build_profile(base_rows, COHORT, (TrustTier.TIER_A_HIGH_TRUST,), None)
    expanded = build_profile(base_rows + [unrelated_symbol, unrelated_timeframe], COHORT, (TrustTier.TIER_A_HIGH_TRUST,), None)
    assert base.population_fingerprint == expanded.population_fingerprint
    assert base.medians == expanded.medians


def test_tier_a_only_changes_population_fingerprint():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [row(1.0, when, tier=TrustTier.TIER_A_HIGH_TRUST), row(2.0, datetime(2026, 1, 1, 1, tzinfo=UTC), tier=TrustTier.TIER_B_LIMITED)]
    ab = build_profile(rows, COHORT, (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED), None)
    a = build_profile(rows, COHORT, (TrustTier.TIER_A_HIGH_TRUST,), None)
    assert a.population_fingerprint != ab.population_fingerprint
    assert a.trust_tiers == (TrustTier.TIER_A_HIGH_TRUST,)


def test_historical_profile_excludes_future_rows_and_removed_current_aliases():
    cutoff = datetime(2026, 1, 2, tzinfo=UTC)
    before = [row(1.0, datetime(2026, 1, 1, tzinfo=UTC), completion=datetime(2026, 1, 1, 1, tzinfo=UTC))]
    future = row(9999.0, datetime(2026, 1, 3, tzinfo=UTC), completion=datetime(2026, 1, 3, 1, tzinfo=UTC))
    first = build_profile(before, COHORT, (TrustTier.TIER_A_HIGH_TRUST,), cutoff)
    expanded = build_profile(before + [future], COHORT, (TrustTier.TIER_A_HIGH_TRUST,), cutoff)
    assert first.to_fingerprint() == expanded.to_fingerprint()


def test_tier_c_and_invalid_historical_rows_are_not_numeric():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    diagnostic = row(1.0, when, tier=TrustTier.TIER_C_DIAGNOSTIC_ONLY)
    with pytest.raises(ValueError):
        build_profile([diagnostic], COHORT, (TrustTier.TIER_C_DIAGNOSTIC_ONLY,), None)
    profile = build_profile([diagnostic], COHORT, (TrustTier.TIER_A_HIGH_TRUST,), None)
    assert profile.population_count == 0


def test_query_fingerprint_contains_policy_and_cutoff():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    profile = build_profile([row(1.0, when)], COHORT, (TrustTier.TIER_A_HIGH_TRUST,), None)
    fingerprint = query_normalization_fingerprint(profile, COHORT, (TrustTier.TIER_A_HIGH_TRUST,), None)
    assert fingerprint != profile.population_fingerprint
