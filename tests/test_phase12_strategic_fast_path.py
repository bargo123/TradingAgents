from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from cli.forex_supervisor import build_parser
from tradingagents.agents.schemas import TraderAction, TraderProposal
from tradingagents.forex.runner import _pm_rejection_reason
from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig
from tradingagents.graph.parallel_analysts import run_parallel_nodes
from tradingagents.graph.setup import GraphSetup


def test_parallel_nodes_join_independent_updates_without_losing_sibling_state() -> None:
    state = {
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "count": 0,
        },
    }

    def bull(value):
        time.sleep(0.02)
        return {"investment_debate_state": {"bull_history": "BULL", "current_response": "BULL"}}

    def bear(value):
        time.sleep(0.02)
        return {"investment_debate_state": {"bear_history": "BEAR", "current_response": "BEAR"}}

    merged = run_parallel_nodes(
        (("Bull Researcher", bull), ("Bear Researcher", bear)),
        state,
        nested_key="investment_debate_state",
    )

    debate = merged["investment_debate_state"]
    assert debate["bull_history"] == "BULL"
    assert debate["bear_history"] == "BEAR"
    assert debate["count"] == 2


def test_parallel_nodes_preserve_sibling_fields_from_full_agent_updates() -> None:
    state = {
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "risk_debate_state": {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "latest_speaker": "",
            "judge_decision": "",
            "count": 0,
        },
    }

    def bull(_state):
        return {
            "investment_debate_state": {
                "history": "Bull Analyst: BULL",
                "bull_history": "Bull Analyst: BULL",
                "bear_history": "",
                "current_response": "Bull Analyst: BULL",
                "judge_decision": "",
                "count": 1,
            }
        }

    def bear(_state):
        return {
            "investment_debate_state": {
                "history": "Bear Analyst: BEAR",
                "bull_history": "",
                "bear_history": "Bear Analyst: BEAR",
                "current_response": "Bear Analyst: BEAR",
                "judge_decision": "",
                "count": 1,
            }
        }

    def aggressive(_state):
        return {
            "risk_debate_state": {
                "history": "Aggressive Analyst: AGG",
                "aggressive_history": "Aggressive Analyst: AGG",
                "conservative_history": "",
                "neutral_history": "",
                "current_aggressive_response": "Aggressive Analyst: AGG",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "latest_speaker": "Aggressive",
                "judge_decision": "",
                "count": 1,
            }
        }

    def conservative(_state):
        return {
            "risk_debate_state": {
                "history": "Conservative Analyst: CON",
                "aggressive_history": "",
                "conservative_history": "Conservative Analyst: CON",
                "neutral_history": "",
                "current_aggressive_response": "",
                "current_conservative_response": "Conservative Analyst: CON",
                "current_neutral_response": "",
                "latest_speaker": "Conservative",
                "judge_decision": "",
                "count": 1,
            }
        }

    merged_research = run_parallel_nodes(
        (("Bull Researcher", bull), ("Bear Researcher", bear)),
        state,
        nested_key="investment_debate_state",
    )
    research = merged_research["investment_debate_state"]
    assert research["bull_history"] == "Bull Analyst: BULL"
    assert research["bear_history"] == "Bear Analyst: BEAR"

    merged_risk = run_parallel_nodes(
        (("Aggressive Analyst", aggressive), ("Conservative Analyst", conservative)),
        state,
        nested_key="risk_debate_state",
    )
    risk = merged_risk["risk_debate_state"]
    assert risk["aggressive_history"] == "Aggressive Analyst: AGG"
    assert risk["conservative_history"] == "Conservative Analyst: CON"


def test_graph_setup_exposes_explicit_phase12_mode() -> None:
    setup = GraphSetup(
        SimpleNamespace(),
        SimpleNamespace(),
        {"market": SimpleNamespace(), "news": SimpleNamespace()},
        SimpleNamespace(),
        market_data_mode="forex_mt5",
        mt5_tools=SimpleNamespace(as_tools=lambda: []),
        phase12_strategic=True,
    )
    assert setup.phase12_strategic is True


def test_runtime_config_carries_explicit_phase12_mode() -> None:
    runtime = ForexShadowRuntimeConfig(phase12_strategic=True)
    assert runtime.to_tradingagents_config()["forex_phase12_strategic"] is True
    assert runtime.safe_dict()["phase12_strategic"] is True


def test_trader_proposal_is_the_canonical_action_source() -> None:
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="supported")
    assert proposal.action.value.upper() == "BUY"


def test_legacy_runtime_mode_remains_off_by_default() -> None:
    assert ForexShadowRuntimeConfig().phase12_strategic is False


def test_phase12_deep_model_override_is_explicit() -> None:
    args = build_parser().parse_args(
        ["run", "--phase12-strategic", "--phase12-deep-model", "qwen3.5:2b"]
    )
    assert args.phase12_strategic is True
    assert args.phase12_deep_model == "qwen3.5:2b"


@pytest.mark.parametrize(
    ("trader", "status", "action", "expected"),
    [
        ("BUY", "NORMALIZED", "HOLD", "RISK_REJECTED"),
        ("SELL", "FAILED", None, "SCHEMA_FAILURE"),
        ("HOLD", "NORMALIZED", "HOLD", None),
    ],
)
def test_pm_boundary_classification_uses_only_canonical_state(
    trader: str, status: str, action: str | None, expected: str | None
) -> None:
    assert (
        _pm_rejection_reason(
            {"trader_action": trader},
            normalization_status=status,
            normalized_action=action,
        )
        == expected
    )
