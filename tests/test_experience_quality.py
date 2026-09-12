"""Deterministic retrieval quality, provenance, ordering, and resource gates."""

import tracemalloc
from datetime import datetime, timezone
from time import perf_counter

from tradingagents.experience.models import ExperienceQuery, TrustTier
from tradingagents.experience.normalization import NormalizationCohortV1, SimilarityProfileV1
from tradingagents.experience.query import ExperienceQueryService

UTC = timezone.utc
COHORT = NormalizationCohortV1("EURUSD", "P", "M15", "experience-features.v1", "extractor")
NAMES = tuple(f"f{i}" for i in range(8))


def make_row(eid, value, when):
    return {
        "experience_id": eid,
        "symbol": "EURUSD",
        "analysis_profile": "P",
        "analysis_timeframe": "M15",
        "analysis_snapshot_timestamp": when,
        "decision_completed_timestamp": when,
        "trust": TrustTier.TIER_A_HIGH_TRUST,
        "trust_tier": TrustTier.TIER_A_HIGH_TRUST,
        "source_aliases": {"db:d": "CURRENT"},
        "values": (value,) * 8,
        "mask": (True,) * 8,
        "feature_names": NAMES,
        "cohort": COHORT,
        "feature_schema_version": COHORT.feature_schema_version,
        "feature_extractor_version": COHORT.feature_extractor_version,
        "feature_fingerprint": eid,
        "accepted": True,
        "provenance_valid": True,
        "provenance": {"source_decision_id": eid},
    }


def service():
    rows = [
        make_row("gold", 0.0, datetime(2026, 1, 1, tzinfo=UTC)),
        make_row("silver", 1.0, datetime(2026, 1, 2, tzinfo=UTC)),
        make_row("bronze", 2.0, datetime(2026, 1, 3, tzinfo=UTC)),
    ]
    profile = SimilarityProfileV1(
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
        (TrustTier.TIER_A_HIGH_TRUST,),
        COHORT,
    )
    return ExperienceQueryService(rows, profile=profile)


def q(**kwargs):
    return ExperienceQuery(
        {"values": (0.0,) * 8, "mask": (True,) * 8, "feature_names": NAMES, "cohort": COHORT},
        **kwargs,
    )


def test_recall_at_k_and_mrr_are_perfect_for_known_fixture():
    result = service().search(q(top_k=2))
    ranked = [hit.experience_id for hit in result.hits]
    relevant = {"gold", "silver"}
    recall = len(set(ranked) & relevant) / len(relevant)
    reciprocal_rank = 1 / (ranked.index("gold") + 1)
    assert recall == 1.0
    assert reciprocal_rank == 1.0


def test_hits_have_complete_provenance_and_stable_order():
    svc = service()
    first = svc.search(q(top_k=3))
    second = svc.search(q(top_k=3))
    assert [(h.experience_id, h.distance) for h in first.hits] == [
        (h.experience_id, h.distance) for h in second.hits
    ]
    assert first.query_normalization_fingerprint
    assert all(hit.provenance["source_decision_id"] == hit.experience_id for hit in first.hits)
    assert all(hit.feature_schema_version == "experience-features.v1" for hit in first.hits)


def test_query_has_bounded_latency_and_memory_for_tiny_fixture():
    svc = service()
    tracemalloc.start()
    start = perf_counter()
    for _ in range(100):
        svc.search(q(top_k=3))
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert elapsed < 2.0
    assert peak < 2_000_000
