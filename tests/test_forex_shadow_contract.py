from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(name, None)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


structured = load_module(
    "test_forex_shadow_contract_structured",
    "tradingagents/agents/utils/structured.py",
)

StructuredOutputRequiredError = structured.StructuredOutputRequiredError
invoke_structured_only = structured.invoke_structured_only

from tradingagents.agents.schemas import ForexPortfolioDecision  # noqa: E402
from tradingagents.forex.shadow import (  # noqa: E402  # isolated helper import must run first
    PortfolioDecision,
    PortfolioRating,
    ShadowDecisionStore,
    ShadowTradeDecision,
    normalize_portfolio_manager_result,
)


def make_decision(**overrides):
    base = {
        "decision_id": "decision-001",
        "created_at": datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc),
        "snapshot_timestamp": datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc),
        "analysis_date": date(2026, 9, 8),
        "requested_symbol": "EURUSD",
        "resolved_symbol": "EURUSDm",
        "action": "HOLD",
        "raw_portfolio_manager_result": {"rating": "Hold"},
        "normalization_status": "NORMALIZED",
        "normalization_error": None,
        "confidence": 0.75,
        "reference_bid": 1.1,
        "reference_ask": 1.1002,
        "reference_mid": 1.1001,
        "spread": 0.0002,
        "spread_points": 2.0,
        "analysis_timeframe": "M15",
        "trader_summary": "x",
        "portfolio_manager_summary": "y",
        "bull_summary": None,
        "bear_summary": None,
        "llm_provider": "local",
        "quick_model": "qwen",
        "deep_model": None,
        "snapshot_json": {"symbol": "EURUSD"},
        "future_evaluation_status": "PENDING",
        "outcome_raw": None,
        "outcome_alpha": None,
        "outcome_resolved_at": None,
        "reflection": None,
        "source_run_id": "run-001",
        "executed": False,
    }
    base.update(overrides)
    return ShadowTradeDecision(**base)


def test_invoke_structured_only_requires_a_structured_binding() -> None:
    with pytest.raises(StructuredOutputRequiredError):
        invoke_structured_only(None, prompt="ignored", agent_name="Portfolio Manager")


def test_invoke_structured_only_rejects_none_and_non_base_model() -> None:
    class NoneStructured:
        def invoke(self, prompt):
            return None

    class PlainStructured:
        def invoke(self, prompt):
            return {"rating": "Hold"}

    with pytest.raises(StructuredOutputRequiredError):
        invoke_structured_only(NoneStructured(), prompt="ignored", agent_name="Portfolio Manager")
    with pytest.raises(StructuredOutputRequiredError):
        invoke_structured_only(PlainStructured(), prompt="ignored", agent_name="Portfolio Manager")


def test_normalization_uses_exact_portfolio_rating_vocabulary_only() -> None:
    assert normalize_portfolio_manager_result({"rating": "Hold"}).action == "HOLD"
    assert normalize_portfolio_manager_result({"rating": "Overweight"}).action == "BUY"
    assert normalize_portfolio_manager_result({"rating": "Underweight"}).action == "SELL"


def test_normalization_uses_structured_rating_only() -> None:
    result = normalize_portfolio_manager_result(
        PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="x",
            investment_thesis="y",
        )
    )
    assert result.action == "BUY"
    assert result.normalization_status == "NORMALIZED"
    assert result.raw_result["rating"] == "Overweight"


def test_normalization_rejects_prose_and_preserves_failure() -> None:
    result = normalize_portfolio_manager_result("BUY now; HOLD if uncertain")
    assert result.action is None
    assert result.normalization_status == "FAILED"
    assert "structured" in result.normalization_error.lower()


def test_normalization_rejects_whitespace_and_case_variants() -> None:
    result = normalize_portfolio_manager_result({"rating": " Hold "})
    assert result.action is None
    assert result.normalization_status == "FAILED"
    assert "unknown" in result.normalization_error.lower()


def test_normalization_rejects_conflicting_mapping_and_unknown_types() -> None:
    result = normalize_portfolio_manager_result(
        {"rating": "Hold", "recommendation": "Sell"}
    )
    assert result.action is None
    assert result.normalization_status == "FAILED"
    assert "conflicting" in result.normalization_error.lower()

    result = normalize_portfolio_manager_result(SimpleNamespace(rating="Hold"))
    assert result.action is None
    assert result.normalization_status == "FAILED"


def test_forex_normalization_rejects_stock_style_horizons() -> None:
    result = normalize_portfolio_manager_result(
        {
            "rating": "Hold",
            "analysis_profile": "INTRADAY",
            "time_horizon": "3-6 months",
            "valid_for_seconds": 3600,
        },
        forex_profile="INTRADAY",
    )

    assert result.action is None
    assert result.normalization_status == "FAILED"
    assert "horizon" in result.normalization_error.lower()


def test_forex_normalization_rejects_profile_or_validity_mismatches() -> None:
    mismatched = normalize_portfolio_manager_result(
        {"rating": "Hold", "analysis_profile": "SWING"},
        forex_profile="INTRADAY",
    )
    assert mismatched.action is None
    assert mismatched.normalization_status == "FAILED"
    assert "profile" in mismatched.normalization_error.lower()

    invalid_validity = normalize_portfolio_manager_result(
        {"rating": "Hold", "valid_for_seconds": 0},
        forex_profile="INTRADAY",
    )
    assert invalid_validity.action is None
    assert invalid_validity.normalization_status == "FAILED"
    assert "valid" in invalid_validity.normalization_error.lower()


def test_forex_portfolio_schema_rejects_months_and_carries_validity() -> None:
    with pytest.raises(ValueError, match="intraday|month"):
        ForexPortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="x",
            investment_thesis="y",
            time_horizon="3-6 months",
        )

    decision = ForexPortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="x",
        investment_thesis="y",
        time_horizon="minutes to hours",
    )
    assert decision.analysis_profile == "INTRADAY"
    assert decision.valid_for_seconds == 3600


def test_shadow_decision_rejects_executed_true_and_invalid_status() -> None:
    with pytest.raises(ValueError):
        make_decision(executed=1)
    with pytest.raises(ValueError):
        make_decision(action="HOLD", normalization_status="FAILED")
    with pytest.raises(ValueError):
        make_decision(future_evaluation_status="BROKEN")


def test_shadow_decision_requires_utc_timestamps() -> None:
    with pytest.raises(ValueError):
        make_decision(created_at=datetime(2026, 9, 8, 2, 0, tzinfo=timezone(timedelta(hours=2))))
    with pytest.raises(ValueError):
        make_decision(outcome_resolved_at=datetime(2026, 9, 8, 2, 0, tzinfo=timezone(timedelta(hours=2))))


def test_shadow_decision_requires_snapshot_timestamp_round_trip() -> None:
    snapshot_timestamp = datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc)
    decision = make_decision(snapshot_timestamp=snapshot_timestamp)
    assert decision.snapshot_timestamp == snapshot_timestamp


def test_shadow_decision_round_trips_intraday_validity() -> None:
    snapshot_timestamp = datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc)
    decision = make_decision(
        snapshot_timestamp=snapshot_timestamp,
        analysis_profile="INTRADAY",
        valid_for_seconds=3600,
        valid_until=snapshot_timestamp + timedelta(seconds=3600),
    )
    assert decision.valid_for_seconds == 3600
    assert decision.valid_until == snapshot_timestamp + timedelta(seconds=3600)


def test_shadow_decision_store_rejects_executed_true_directly() -> None:
    with pytest.raises(ValueError):
        make_decision(executed=1)


def test_shadow_decision_round_trips_with_canonical_portfolio_decision_class() -> None:
    canonical_module = sys.modules["tradingagents.agents.schemas"]
    canonical_decision = canonical_module.PortfolioDecision(
        rating=canonical_module.PortfolioRating.HOLD,
        executive_summary="x",
        investment_thesis="y",
    )
    result = normalize_portfolio_manager_result(canonical_decision)
    assert result.action == "HOLD"
    assert result.raw_result["rating"] == "Hold"
    assert PortfolioDecision is canonical_module.PortfolioDecision


def test_store_round_trip_is_idempotent_and_preserves_failed_action(
    tmp_path: Path,
) -> None:
    store = ShadowDecisionStore(tmp_path / "shadow.db")
    failed = make_decision(
        action=None,
        normalization_status="FAILED",
        normalization_error="structured output missing",
        raw_portfolio_manager_result={"error": "missing"},
    )
    store.initialize()
    store.record(failed)
    store.record(failed)
    restored = store.get(failed.decision_id)
    assert restored.action is None
    assert restored.normalization_status == "FAILED"
    assert store.list_pending() == [restored]
    assert json.loads(restored.raw_portfolio_manager_result_json)["error"] == "missing"


def test_store_round_trip_preserves_profile_and_validity(tmp_path: Path) -> None:
    store = ShadowDecisionStore(tmp_path / "shadow.db")
    timestamp = datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc)
    decision = make_decision(
        snapshot_timestamp=timestamp,
        analysis_profile="INTRADAY",
        valid_for_seconds=3600,
        valid_until=timestamp + timedelta(seconds=3600),
    )

    store.record(decision)
    restored = store.get(decision.decision_id)

    assert restored.analysis_profile == "INTRADAY"
    assert restored.valid_for_seconds == 3600
    assert restored.valid_until == timestamp + timedelta(seconds=3600)


def test_store_rejects_invalid_future_evaluation_status_via_sql(
    tmp_path: Path,
) -> None:
    store = ShadowDecisionStore(tmp_path / "shadow.db")
    store.initialize()
    with pytest.raises(sqlite3.IntegrityError), sqlite3.connect(store.path) as conn:
        conn.execute(
            """
                INSERT INTO shadow_decisions (
                    decision_id,
                    created_at,
                    analysis_date,
                    requested_symbol,
                    resolved_symbol,
                    snapshot_timestamp,
                    action,
                    normalization_status,
                    normalization_error,
                    raw_portfolio_manager_result,
                    confidence,
                    reference_bid,
                    reference_ask,
                    reference_mid,
                    spread,
                    spread_points,
                    analysis_timeframe,
                    trader_summary,
                    portfolio_manager_summary,
                    bull_summary,
                    bear_summary,
                    llm_provider,
                    quick_model,
                    deep_model,
                    snapshot_json,
                    executed,
                    future_evaluation_status,
                    outcome_raw,
                    outcome_alpha,
                    outcome_resolved_at,
                    reflection,
                    source_run_id
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
            (
                "bad-status",
                "2026-09-08T00:00:00Z",
                "2026-09-08",
                "EURUSD",
                "EURUSDm",
                "2026-09-08T00:00:00Z",
                "HOLD",
                "NORMALIZED",
                None,
                json.dumps({"rating": "Hold"}),
                0.1,
                1.1,
                1.1002,
                1.1001,
                0.0002,
                2.0,
                "M15",
                "x",
                "y",
                None,
                None,
                "local",
                "qwen",
                None,
                json.dumps({"symbol": "EURUSD"}),
                0,
                "BROKEN",
                None,
                None,
                None,
                None,
                None,
            ),
        )
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            """
            INSERT INTO shadow_decisions (
                decision_id,
                created_at,
                analysis_date,
                requested_symbol,
                resolved_symbol,
                snapshot_timestamp,
                action,
                normalization_status,
                normalization_error,
                raw_portfolio_manager_result,
                confidence,
                reference_bid,
                reference_ask,
                reference_mid,
                spread,
                spread_points,
                analysis_timeframe,
                trader_summary,
                portfolio_manager_summary,
                bull_summary,
                bear_summary,
                llm_provider,
                quick_model,
                deep_model,
                snapshot_json,
                executed,
                future_evaluation_status,
                outcome_raw,
                outcome_alpha,
                outcome_resolved_at,
                reflection,
                source_run_id
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                "bad-status-update",
                "2026-09-08T00:00:00Z",
                "2026-09-08",
                "EURUSD",
                "EURUSDm",
                "2026-09-08T00:00:00Z",
                "HOLD",
                "NORMALIZED",
                None,
                json.dumps({"rating": "Hold"}),
                0.1,
                1.1,
                1.1002,
                1.1001,
                0.0002,
                2.0,
                "M15",
                "x",
                "y",
                None,
                None,
                "local",
                "qwen",
                None,
                json.dumps({"symbol": "EURUSD"}),
                0,
                "PENDING",
                None,
                None,
                None,
                None,
                None,
            ),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE shadow_decisions SET future_evaluation_status = ? WHERE decision_id = ?",
                ("BROKEN", "bad-status-update"),
            )


def test_shadow_decision_rejects_non_finite_float_fields() -> None:
    with pytest.raises(ValueError):
        make_decision(reference_bid=math.inf)
    with pytest.raises(ValueError):
        make_decision(snapshot_json={"spread": math.nan})
    with pytest.raises(ValueError):
        make_decision(raw_portfolio_manager_result={"rating": "Hold", "spread": math.inf})
