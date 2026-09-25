from __future__ import annotations

from datetime import date, datetime, timezone

from tradingagents.forex.revision_validation import validate_revision
from tradingagents.forex.shadow import ShadowDecisionStore, ShadowTradeDecision
from tradingagents.forex.watch_store import LeaseOwner, RunEvidence, WatcherStore
from tradingagents.forex.watcher import ScheduledOpportunity

UTC = timezone.utc
NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


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
        bar_close_timestamp=NOW,
        eligible_after=NOW,
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
        ),
    )
    before = db_path.read_bytes()

    report = validate_revision(db_path, "abc123")

    assert report["run_count"] == 1
    assert report["research_recommendations"] == {"UNAVAILABLE": 1}
    assert report["trader_actions"] == {"HOLD": 1}
    assert report["portfolio_manager_actions"] == {"HOLD": 1}
    assert db_path.read_bytes() == before
