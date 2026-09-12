from datetime import datetime, timezone

import numpy as np
import pytest

from tradingagents.experience.normalization import NormalizationCohortV1, SimilarityProfileV1
from tradingagents.experience.similarity import ExactSimilarityIndex

UTC = timezone.utc
COHORT = NormalizationCohortV1("EURUSD", "P", "M15", "experience-features.v1", "extractor")
NAMES = tuple(f"f{i}" for i in range(8))


def profile():
    return SimilarityProfileV1(
        medians=dict.fromkeys(NAMES, 0.0),
        iqr=dict.fromkeys(NAMES, 1.0),
        mad_scales=dict.fromkeys(NAMES, 1.0),
        scales=dict.fromkeys(NAMES, 1.0),
        fallback_scales=dict.fromkeys(NAMES, 1.0),
        epsilon=1e-9,
        weights=dict.fromkeys(NAMES, 1.0),
        clipping=(-8.0, 8.0),
        feature_order=NAMES,
        mask_policy="exclude-missing.v1",
        trust_tiers=(),
        cohort=COHORT,
    )


def test_search_uses_masked_float32_weighted_rms_and_deterministic_ties():
    idx = ExactSimilarityIndex(
        {
            "b": {"values": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64), "mask": [1] * 8},
            "a": {"values": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64), "mask": [1] * 8},
        },
        metadata={
            "a": {"analysis_snapshot_timestamp": datetime(2026, 1, 1, tzinfo=UTC)},
            "b": {"analysis_snapshot_timestamp": datetime(2026, 1, 1, tzinfo=UTC)},
        },
    )
    hits = idx.search(np.zeros(8), np.ones(8, dtype=bool), ("b", "a"), 10, profile())
    assert [h.experience_id for h in hits] == ["a", "b"]
    assert hits[0].distance == pytest.approx(1 / np.sqrt(8), rel=1e-6)
    assert hits[0].similarity_score == 1 / (1 + hits[0].distance)


def test_search_rejects_insufficient_overlap():
    idx = ExactSimilarityIndex({"x": {"values": [1] * 8, "mask": [1] + [0] * 7}})
    assert idx.search([0] * 8, [1] * 8, ("x",), 5, profile()) == ()


def test_sort_fallback_handles_missing_timestamp_with_aware_timestamp():
    idx = ExactSimilarityIndex(
        {
            "missing": {"values": [0] * 8, "mask": [1] * 8},
            "dated": {"values": [0] * 8, "mask": [1] * 8},
        },
        metadata={"dated": {"analysis_snapshot_timestamp": datetime(2026, 1, 1, tzinfo=UTC)}},
    )
    hits = idx.search([0] * 8, [1] * 8, ("missing", "dated"), 5, profile())
    assert [h.experience_id for h in hits] == ["missing", "dated"]
