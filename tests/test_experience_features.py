from datetime import datetime, timezone

import pytest

from tradingagents.experience.features import FEATURE_NAMES_V1, extract_market_state
from tradingagents.experience.errors import FeatureExtractionIncompleteError


@pytest.fixture
def decision_row():
    features = {tf: {"return_over_bars": 0.01, "range_pct": 0.02,
                     "close_position": 0.5, "average_true_range": 0.003,
                     "direction": "UP"} for tf in ("M1", "M5", "M15", "H1")}
    features["M5"].pop("average_true_range")
    return {
        "resolved_symbol": "EURUSD", "analysis_profile": "INTRADAY",
        "analysis_timeframe": "M15",
        "analysis_snapshot_timestamp": datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        "snapshot_json": {"quote": {"spread_points": 10}, "features": features,
                          "point": 0.00001, "digits": 5},
    }


def test_feature_order_and_missing_mask_are_deterministic(decision_row):
    result = extract_market_state(decision_row)
    assert result.feature_names == FEATURE_NAMES_V1
    assert len(result.values) == len(result.mask)
    assert result.mask[result.feature_names.index("M5.average_true_range")] is False


def test_outcome_fields_cannot_change_market_vector(decision_row):
    changed = {**decision_row, "buy_net_points": 99999, "selected_action": "SELL"}
    assert extract_market_state(decision_row).fingerprint == extract_market_state(changed).fingerprint


def test_malformed_required_metadata_has_typed_diagnostic(decision_row):
    decision_row["snapshot_json"]["digits"] = "five"
    with pytest.raises(FeatureExtractionIncompleteError):
        extract_market_state(decision_row)
