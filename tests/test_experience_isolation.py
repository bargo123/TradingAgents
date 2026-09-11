"""Compatibility, cutoff, alias, recovery, and normalization isolation gates."""
from datetime import datetime, timezone

import pytest

from tradingagents.experience.models import ExperienceQuery, OutcomeStatsRequest, TrustTier
from tradingagents.experience.normalization import NormalizationCohortV1, build_profile
from tradingagents.experience.outcomes import OutcomeStatsCalculator
from tradingagents.experience.query import ExperienceQueryService

UTC = timezone.utc
COHORT = NormalizationCohortV1("EURUSD", "P", "M15", "experience-features.v1", "extractor")
NAMES = tuple(f"f{i}" for i in range(8))


def row(eid, *, cohort=COHORT, tier=TrustTier.TIER_A_HIGH_TRUST, current=True, when=None, completed=None):
    when = when or datetime(2026, 1, 1, tzinfo=UTC)
    return {"experience_id": eid, "symbol": cohort.resolved_symbol, "analysis_profile": cohort.analysis_profile,
            "analysis_timeframe": cohort.analysis_timeframe, "analysis_snapshot_timestamp": when,
            "decision_completed_timestamp": completed or when, "trust": tier, "trust_tier": tier,
            "source_aliases": {"db:d": "CURRENT" if current else "REMOVED"}, "values": (0.0,) * 8,
            "mask": (True,) * 8, "feature_names": NAMES, "cohort": cohort,
            "feature_schema_version": cohort.feature_schema_version, "feature_extractor_version": cohort.feature_extractor_version,
            "feature_fingerprint": eid, "accepted": True, "provenance_valid": True}


def query(**kwargs):
    return ExperienceQuery({"values": (0.0,) * 8, "mask": (True,) * 8, "feature_names": NAMES, "cohort": COHORT}, **kwargs)


def _service(rows):
    return ExperienceQueryService(rows, profile=build_profile([rows[0]], COHORT))


def test_symbol_filter_is_independent():
    unrelated = NormalizationCohortV1("XAUUSD", "P", "M15", COHORT.feature_schema_version, COHORT.feature_extractor_version)
    result = _service([row("good"), row("gold", cohort=unrelated)]).search(query(symbol="EURUSD"))
    assert [hit.experience_id for hit in result.hits] == ["good"]
    assert result.excluded_counts["symbol"] == 1


def test_profile_filter_is_independent():
    other = NormalizationCohortV1("EURUSD", "SWING", "M15", COHORT.feature_schema_version, COHORT.feature_extractor_version)
    result = _service([row("good"), row("other", cohort=other)]).search(query(analysis_profile="P"))
    assert [hit.experience_id for hit in result.hits] == ["good"]
    assert result.excluded_counts["profile"] == 1


def test_timeframe_filter_is_independent():
    other = NormalizationCohortV1("EURUSD", "P", "H1", COHORT.feature_schema_version, COHORT.feature_extractor_version)
    result = _service([row("good"), row("other", cohort=other)]).search(query(analysis_timeframe="M15"))
    assert [hit.experience_id for hit in result.hits] == ["good"]
    assert result.excluded_counts["timeframe"] == 1


def test_cohort_mismatch_is_excluded_even_without_explicit_filters():
    other = NormalizationCohortV1("EURUSD", "SWING", "M15", COHORT.feature_schema_version, COHORT.feature_extractor_version)
    result = _service([row("good"), row("other", cohort=other)]).search(query())
    assert [hit.experience_id for hit in result.hits] == ["good"]
    assert result.excluded_counts["cohort"] == 1


def test_later_alias_removal_preserves_historical_as_of_result():
    gone = row("gone", current=False)
    future = row("future", when=datetime(2026, 1, 3, tzinfo=UTC))
    service = ExperienceQueryService([gone, future], profile=build_profile([gone], COHORT))
    result = service.search(query(as_of=datetime(2026, 1, 2, tzinfo=UTC)))
    assert [hit.experience_id for hit in result.hits] == ["gone"]
    assert result.hits[0].currently_tombstoned is True


def test_tier_a_only_normalization_excludes_tier_b_population():
    rows = [row("a", tier=TrustTier.TIER_A_HIGH_TRUST), row("b", tier=TrustTier.TIER_B_LIMITED)]
    profile = build_profile(rows, COHORT, (TrustTier.TIER_A_HIGH_TRUST,))
    assert profile.population_count == 1
    assert profile.trust_tiers == (TrustTier.TIER_A_HIGH_TRUST,)


def test_tier_c_normalization_is_rejected():
    with pytest.raises(ValueError, match="Tier C"):
        build_profile([row("c", tier=TrustTier.TIER_C_DIAGNOSTIC_ONLY)], COHORT,
                      (TrustTier.TIER_C_DIAGNOSTIC_ONLY,))


def _complete(status, observed, evaluated, fingerprint):
    return {"evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300,
            "evaluation_status": status, "source_context_eligible": 1,
            "observation_timestamp": observed, "evaluated_at": evaluated,
            "selected_action": "BUY", "buy_net_points": 2.0, "sell_net_points": -1.0,
            "hold_opportunity_cost_points": 0.0, "fingerprint": fingerprint}


def test_recovered_evaluation_without_prior_snapshot_is_not_fabricated():
    record = {"experience_id": "late", "trust": TrustTier.TIER_A_HIGH_TRUST,
              "outcome_snapshots": (_complete("COMPLETE", "2026-01-03T12:00:00Z", "2026-01-03T13:00:00Z", "late"),)}
    result = OutcomeStatsCalculator([record]).calculate(
        OutcomeStatsRequest(("late",), horizon_seconds=300,
                            as_of=datetime(2026, 1, 2, tzinfo=UTC)))
    assert result.excluded_counts["EVALUATION_NOT_YET_AVAILABLE"] == 1


def test_recovered_evaluation_with_prior_unavailable_snapshot_preserves_history():
    record = {"experience_id": "recovered", "trust": TrustTier.TIER_A_HIGH_TRUST,
              "outcome_snapshots": (
                  _complete("DATA_UNAVAILABLE", "2026-01-02T12:00:00Z", "2026-01-02T13:00:00Z", "prior"),
                  _complete("COMPLETE", "2026-01-03T12:00:00Z", "2026-01-03T13:00:00Z", "later"),
              )}
    result = OutcomeStatsCalculator([record]).calculate(
        OutcomeStatsRequest(("recovered",), horizon_seconds=300,
                            as_of=datetime(2026, 1, 2, 14, tzinfo=UTC)))
    assert result.excluded_counts["DATA_UNAVAILABLE"] == 1
