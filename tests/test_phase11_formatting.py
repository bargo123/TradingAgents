from __future__ import annotations

import json

import pytest

from tradingagents.finetuning.errors import ContractError
from tradingagents.finetuning.formatting import SFTFormatter, parse_target


def row(action: str = "BUY") -> dict:
    return {
        "example_id": "ex-1",
        "decision":{"action": action, "resolved_symbol": "EURUSD", "analysis_timeframe": "M5", "decision_context_status": "COMPLETE"},
        "market":{"snapshot_timestamp": "2026-01-01T00:00:00Z", "quote": {"bid": 1.1, "ask": 1.1001}, "features": {"M5": {"direction": "UP"}}},
        "research":{"evidence_refs_used": ["K1", "E2"], "outcome": "future", "ignored": "ignore"},
        "outcome":{"future_bid": 9, "pnl": 4},
        "trust":{"tier": "A", "status": "WITHHELD"},
        "provenance":{"phase56": {"source_id": "s", "file_sha256": "h", "schema_fingerprint": "f", "snapshot_fingerprint": "t", "canonical_path": "/x", "contract_version": "v"}},
    }


def test_formatter_is_deterministic_and_excludes_future_or_cot() -> None:
    first = SFTFormatter().format(row())
    second = SFTFormatter().format(row())
    assert first.to_json() == second.to_json()
    assert first.messages[-1]["role"] == "assistant"
    assert json.loads(first.messages[-1]["content"]) == {"action": "BUY", "evidence_refs": ["E2", "K1"]}
    rendered = first.to_json()
    assert all(value not in rendered for value in ("future_bid", "pnl", "prompt", "outcome_status", "ignore"))


@pytest.mark.parametrize("action", ["BUY", "SELL", "HOLD"])
def test_strict_target_actions(action: str) -> None:
    assert parse_target(SFTFormatter().format(row(action)).target)["action"] == action


def test_target_rejects_extra_missing_and_bad_evidence() -> None:
    with pytest.raises(ContractError):
        parse_target({"action": "BUY"})
    with pytest.raises(ContractError):
        parse_target({"action": "BUY", "evidence_refs": [], "extra": 1})
    with pytest.raises(ContractError):
        parse_target({"action": "JUMP", "evidence_refs": []})
    with pytest.raises(ContractError):
        parse_target({"action": "BUY", "evidence_refs": ["not safe!"]})


def test_sensitive_source_value_is_rejected() -> None:
    bad = row()
    bad["decision"] = {"action": "BUY", "api_key": "x"}
    with pytest.raises(ContractError):
        SFTFormatter().format(bad)
