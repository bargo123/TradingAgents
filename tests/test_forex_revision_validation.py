from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone

from tradingagents.forex.evaluation import ShadowEvaluationStore, ShadowOutcomeEvaluation
from tradingagents.forex.revision_validation import (
    _freshness_report,
    aggregate_agent_metrics,
    validate_revision,
)
from tradingagents.forex.shadow import ShadowDecisionStore, ShadowTradeDecision
from tradingagents.forex.watch_store import LeaseOwner, RunEvidence, WatcherStore
from tradingagents.forex.watcher import ScheduledOpportunity

UTC = timezone.utc
NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def test_freshness_report_does_not_infer_without_a_budget():
    report = _freshness_report(
        [
            {"stale_by_completion": 1, "analysis_latency_seconds": 901, "freshness_budget_seconds": 900},
            {"stale_by_completion": None, "analysis_latency_seconds": 1, "freshness_budget_seconds": None},
        ]
    )

    assert report["status_counts"] == {"STALE": 1, "UNAVAILABLE": 1}
    assert report["stale_by_completion_count"] == 1


def test_freshness_report_rejects_malformed_stale_flag():
    report = _freshness_report(
        [
            {
                "stale_by_completion": "false",
                "analysis_latency_seconds": 1,
                "freshness_budget_seconds": 900,
            }
        ]
    )

    assert report["status_counts"] == {"UNAVAILABLE": 1}
    assert report["stale_by_completion_count"] == 0


def test_agent_metrics_aggregate_safe_numeric_telemetry_only():
    report = aggregate_agent_metrics(
        [
            {
                "agents": {
                    "Portfolio Manager": {
                        "model": "qwen3.5:4b",
                        "calls": 1,
                        "elapsed_seconds": 10.0,
                        "tokens_in": 100,
                        "tokens_out": 20,
                        "reasoning_tokens": 0,
                    }
                }
            },
            {
                "agents": {
                    "Portfolio Manager": {
                        "model": "qwen3.5:4b",
                        "calls": 1,
                        "elapsed_seconds": 14.0,
                        "tokens_in": 120,
                        "tokens_out": 24,
                        "reasoning_tokens": 2,
                    },
                    "News Analyst": {
                        "model": "qwen3.5:2b",
                        "calls": 3,
                        "elapsed_seconds": 8.0,
                        "tokens_in": 300,
                        "tokens_out": 40,
                    },
                }
            },
        ]
    )

    assert report["Portfolio Manager"]["run_count"] == 2
    assert report["Portfolio Manager"]["calls"] == 2
    assert report["Portfolio Manager"]["models"] == ["qwen3.5:4b"]
    assert report["Portfolio Manager"]["elapsed_seconds"] == {
        "count": 2,
        "mean": 12.0,
        "median": 12.0,
        "p95": 14.0,
        "max": 14.0,
    }
    assert report["Portfolio Manager"]["tokens_in"] == 220
    assert report["Portfolio Manager"]["tokens_out"] == 44
    assert report["Portfolio Manager"]["reasoning_tokens"] == 2


def test_agent_metrics_ignore_malformed_rows_and_private_fields():
    report = aggregate_agent_metrics(
        [
            {"agents": "not-a-mapping", "prompt": "must not be returned"},
            {
                "agents": {
                    "Trader": {
                        "model": "qwen3.5:4b",
                        "calls": True,
                        "elapsed_seconds": "bad",
                        "tokens_in": float("nan"),
                        "tokens_out": 7,
                        "completion": "must not be returned",
                    }
                }
            },
        ]
    )

    assert report["Trader"] == {
        "run_count": 1,
        "calls": 0,
        "models": ["qwen3.5:4b"],
        "elapsed_seconds": {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        },
        "tokens_in": 0,
        "tokens_out": 7,
        "reasoning_tokens": 0,
    }
    assert "prompt" not in str(report)
    assert "completion" not in str(report)


def test_revision_validation_filters_by_git_commit_without_writing(tmp_path):
    db_path = tmp_path / "watch.db"
    provenance = {
        "git_commit": "abc123",
        "working_tree_dirty": False,
        "prompt_config_version": "forex-shadow.runtime.v1",
        "application_version": "0.4.0",
        "collector_contract_version": "forex-watch.v1",
        "config_fingerprint": "cfg",
        "safe_config_json": "{}",
    }
    store = WatcherStore(db_path, provenance=provenance)
    owner = LeaseOwner("owner", 123, "host", NOW)
    lease = store.acquire_lease(owner, NOW)
    opportunity = ScheduledOpportunity(
        requested_symbol="EURUSD",
        analysis_profile="INTRADAY",
        schedule_timeframe="M15",
        anchor_timestamp=NOW,
        bar_close_timestamp=NOW + timedelta(minutes=15),
        eligible_after=NOW + timedelta(minutes=15, seconds=30),
        config_fingerprint="cfg",
    )
    store.observe_opportunity(opportunity, NOW)
    run = store.claim_opportunity(lease.owner_token, opportunity.opportunity_key, NOW)
    ShadowDecisionStore(db_path).record(
        ShadowTradeDecision(
            decision_id="decision-1",
            created_at=NOW,
            snapshot_timestamp=NOW,
            analysis_date=date(2026, 9, 25),
            requested_symbol="EURUSD",
            resolved_symbol="EURUSD",
            action="HOLD",
            raw_portfolio_manager_result={"rating": "Hold"},
            normalization_status="NORMALIZED",
            normalization_error=None,
            confidence=None,
            reference_bid=1.1,
            reference_ask=1.1001,
            reference_mid=1.10005,
            spread=0.0001,
            spread_points=1,
            analysis_timeframe="M15",
            trader_summary="FINAL TRANSACTION PROPOSAL: **HOLD**",
            portfolio_manager_summary="Hold",
            source_run_id=run.source_run_id,
            decision_context_status="COMPLETE",
        )
    )
    store.finalize_run(
        lease.owner_token,
        run.run_id,
        NOW,
        status="SUCCEEDED",
        evidence=RunEvidence(
            run_id=run.run_id,
            source_run_id=run.source_run_id,
            decision_id="decision-1",
            requested_symbol="EURUSD",
            resolved_symbol="EURUSD",
            decision_context_status="COMPLETE",
            normalization_status="NORMALIZED",
            normalized_action="HOLD",
            analysis_snapshot_timestamp=NOW,
            decision_completed_timestamp=NOW,
            analysis_latency_seconds=0,
            decision_reference_timestamp=NOW,
            decision_reference_delay_seconds=0,
            stale_by_completion=False,
            runtime_seconds=1,
            llm_calls=1,
            tool_calls=0,
            tokens_in=1,
            tokens_out=1,
            reasoning_tokens=0,
            llm_provider="ollama",
            quick_model="qwen3.5:2b",
            deep_model="qwen3.5:4b",
            executed=False,
            decision_reference_status="AVAILABLE",
            freshness_budget_seconds=900,
            metrics_json=json.dumps(
                {
                    "agents": {
                        "Portfolio Manager": {
                            "model": "qwen3.5:4b",
                            "calls": 1,
                            "elapsed_seconds": 3.0,
                            "tokens_in": 20,
                            "tokens_out": 4,
                        }
                    },
                    "state_boundaries": [
                        {
                            "node": "Portfolio Manager",
                            "phase": "after",
                            "duration_seconds": 5.0,
                        }
                    ],
                }
            ),
        ),
    )
    ShadowEvaluationStore(db_path).upsert(
        [
            ShadowOutcomeEvaluation(
                decision_id="decision-1",
                resolved_symbol="EURUSD",
                evaluation_basis="ANALYSIS_SNAPSHOT",
                horizon_seconds=300,
                evaluation_version="v1",
                market_data_source="MT5",
                source_context_eligible=True,
                training_eligible=None,
                training_eligibility_reason="DEFERRED_TO_CORPUS_BUILDER",
                evaluation_status="COMPLETE",
                target_timestamp=NOW,
                observation_timestamp=NOW,
                entry_timestamp=NOW,
                created_at=NOW,
                evaluated_at=NOW,
            ),
            ShadowOutcomeEvaluation(
                decision_id="decision-1",
                resolved_symbol="EURUSD",
                evaluation_basis="ANALYSIS_SNAPSHOT",
                horizon_seconds=900,
                evaluation_version="v1",
                market_data_source="MT5",
                source_context_eligible=True,
                training_eligible=None,
                training_eligibility_reason="DEFERRED_TO_CORPUS_BUILDER",
                evaluation_status="PENDING",
                target_timestamp=NOW,
                created_at=NOW,
            ),
        ]
    )
    before = db_path.read_bytes()

    report = validate_revision(db_path, "abc123")

    assert report["run_count"] == 1
    assert report["research_recommendations"] == {"UNAVAILABLE": 1}
    assert report["trader_actions"] == {"HOLD": 1}
    assert report["portfolio_manager_actions"] == {"HOLD": 1}
    assert report["runtime_seconds"]["p95"] == 1.0
    assert report["runtime_seconds"]["average"] == 1.0
    assert report["runtime_seconds"]["maximum"] == 1.0
    assert report["freshness"] == {
        "status_counts": {"WITHIN_BUDGET": 1},
        "stale_by_completion_count": 0,
        "budget_seconds": {"count": 1, "mean": 900.0, "median": 900.0, "p95": 900.0, "max": 900.0},
    }
    assert report["evaluation_coverage"] == {
        "evaluations_total": 2,
        "decisions_with_evaluations": 1,
        "decision_count": 1,
        "status_counts": {"COMPLETE": 1, "PENDING": 1},
        "by_basis_horizon_status": {
            "ANALYSIS_SNAPSHOT/300/COMPLETE": 1,
            "ANALYSIS_SNAPSHOT/900/PENDING": 1,
        },
    }
    assert report["agent_metrics"]["Portfolio Manager"]["elapsed_seconds"]["p95"] == 3.0
    assert report["latency_bottlenecks"][0]["agent"] == "Portfolio Manager"
    assert report["stage_elapsed_seconds"] == {"Portfolio Manager": 5.0}
    assert db_path.read_bytes() == before

    lower_path = tmp_path / "watch-lower.db"
    lower_path.write_bytes(before)
    with closing(sqlite3.connect(lower_path)) as db, db:
        db.execute(
            "UPDATE shadow_decisions SET research_manager_recommendation='buy'"
        )

    lower_report = validate_revision(lower_path, "abc123")

    assert lower_report["research_recommendations"] == {"UNAVAILABLE": 1}
    assert lower_report["portfolio_manager_actions"] == {"HOLD": 1}
