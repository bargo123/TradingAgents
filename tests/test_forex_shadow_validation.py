from __future__ import annotations

import json
from datetime import date

from tests.test_forex_shadow_runner import _make_runner


def test_phase41_persists_complete_shadow_evidence_and_llm_metrics(tmp_path):
    runner, provider, _, store = _make_runner(
        tmp_path,
        {
            "final_trade_decision": "Hold",
            "portfolio_manager_raw_result": {
                "rating": "Hold",
                "executive_summary": "No directional edge after spread.",
                "investment_thesis": "Evidence is balanced.",
            },
            "investment_debate_state": {
                "bull_history": "Bull evidence",
                "bear_history": "Bear evidence",
            },
            "trader_investment_plan": "Hypothetical hold; no order.",
            "risk_debate_state": {
                "history": "Aggressive, conservative, and neutral risk discussion",
            },
        },
    )
    runner.config.update(
        {
            "llm_provider": "openai",
            "quick_think_llm": "gpt-5.6-luna",
            "deep_think_llm": "gpt-5.6",
        }
    )
    callback = type(
        "Callback",
        (),
        {"get_stats": lambda self: {"llm_calls": 8, "tool_calls": 4}},
    )()

    result = runner.run(
        symbol="EURUSD",
        count=1,
        analysis_date=date(2026, 9, 8),
        callbacks=[callback],
    )

    persisted = store.get(result.decision.decision_id)
    raw = json.loads(persisted.raw_portfolio_manager_result_json)
    assert provider.market_snapshot_calls == 1
    assert result.metrics["llm_calls"] == 8
    assert result.metrics["tool_calls"] == 4
    assert result.elapsed_seconds >= 0
    assert persisted.requested_symbol == "EURUSD"
    assert persisted.resolved_symbol == "EURUSDm"
    assert persisted.snapshot_timestamp.tzinfo is not None
    assert persisted.reference_bid == 1.1
    assert persisted.reference_ask == 1.1002
    assert raw["rating"] == "Hold"
    assert persisted.action == "HOLD"
    assert persisted.normalization_status == "NORMALIZED"
    assert persisted.executed is False

