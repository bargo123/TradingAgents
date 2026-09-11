from datetime import datetime, timezone

from tradingagents.experience.features import extract_market_state
from tradingagents.experience.models import TrustTier
from tradingagents.experience.trust import classify_trust


def _row():
    return {"resolved_symbol": "EURUSD", "analysis_profile": "INTRADAY", "analysis_timeframe": "M15",
            "analysis_snapshot_timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "decision_completed_timestamp": datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            "normalization_status": "NORMALIZED", "decision_context_status": "COMPLETE",
            "action": "BUY", "executed": 0,
            "snapshot_json": {"quote": {"spread_points": 10, "bid": 1.1, "ask": 1.1001},
                              "point": 0.00001, "digits": 5,
                              "features": {tf: {"return_over_bars": .01, "range_pct": .02,
                                  "close_position": .5, "average_true_range": .003,
                                  "direction": "UP"} for tf in ("M1", "M5", "M15", "H1")}}}


def test_incomplete_normalization_is_tier_c():
    row = _row(); row["normalization_status"] = "FAILED"
    result = classify_trust(row, extract_market_state(row))
    assert result.tier is TrustTier.TIER_C_DIAGNOSTIC_ONLY
    assert "NORMALIZATION_FAILED" in result.reasons


def test_complete_normalized_decision_is_tier_a():
    row = _row()
    result = classify_trust(row, extract_market_state(row))
    assert result.tier is TrustTier.TIER_A_HIGH_TRUST
