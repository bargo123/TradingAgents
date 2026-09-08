from __future__ import annotations

import json
import importlib.util
import sys
from datetime import date, datetime, timezone
from pathlib import Path

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

from tradingagents.forex.shadow import (
    PortfolioDecision,
    PortfolioRating,
    ShadowDecisionStore,
    ShadowNormalization,
    ShadowTradeDecision,
    normalize_portfolio_manager_result,
)


def make_decision(**overrides):
    base = dict(
        decision_id="decision-001",
        created_at=datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc),
        analysis_date=date(2026, 9, 8),
        requested_symbol="EURUSD",
        resolved_symbol="EURUSDm",
        action="HOLD",
        raw_portfolio_manager_result={"rating": "Hold"},
        normalization_status="NORMALIZED",
        normalization_error=None,
        confidence=0.75,
        reference_bid=1.1,
        reference_ask=1.1002,
        reference_mid=1.1001,
        spread=0.0002,
        spread_points=2.0,
        analysis_timeframe="M15",
        trader_summary="x",
        portfolio_manager_summary="y",
        bull_summary=None,
        bear_summary=None,
        llm_provider="local",
        quick_model="qwen",
        deep_model=None,
        snapshot_json={"symbol": "EURUSD"},
        future_evaluation_status="PENDING",
        outcome_raw=None,
        outcome_alpha=None,
        outcome_resolved_at=None,
        reflection=None,
        source_run_id="run-001",
        executed=False,
    )
    base.update(overrides)
    return ShadowTradeDecision(**base)


def test_invoke_structured_only_requires_a_structured_binding() -> None:
    with pytest.raises(StructuredOutputRequiredError):
        invoke_structured_only(None, prompt="ignored", agent_name="Portfolio Manager")


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


def test_shadow_decision_rejects_executed_true_and_invalid_status() -> None:
    with pytest.raises(ValueError):
        make_decision(executed=True)
    with pytest.raises(ValueError):
        make_decision(action="HOLD", normalization_status="FAILED")


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
