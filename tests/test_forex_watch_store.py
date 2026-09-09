from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from tradingagents.forex.shadow import ShadowDecisionStore, ShadowTradeDecision
from tradingagents.forex.watch_store import (
    LeaseLostError,
    LeaseOwner,
    LeaseStatus,
    WatcherStore,
)
from tradingagents.forex.watcher import ScheduledOpportunity

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


class FakeInspector:
    def __init__(self, alive: bool | None, calls: list[str]):
        self.alive = alive
        self.calls = calls

    def proves_alive(self, owner):
        self.calls.append("inspect")
        return self.alive is True


def owner(token="old"):
    return LeaseOwner(
        owner_token=token,
        pid=123,
        host="host-a",
        process_started_at=NOW - timedelta(seconds=10),
    )


def test_nonexpired_lease_cannot_be_stolen_even_when_pid_is_dead(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    assert store.acquire_lease(owner(), NOW).status is LeaseStatus.ACQUIRED
    calls: list[str] = []

    result = store.acquire_lease(
        owner("new"), NOW + timedelta(seconds=1), FakeInspector(False, calls)
    )

    assert result.status is LeaseStatus.WATCHER_ALREADY_RUNNING
    assert calls == []


def test_expired_exact_old_process_alive_requires_operator_review(tmp_path):
    store = WatcherStore(tmp_path / "watch.db", lease_ttl_seconds=10)
    store.acquire_lease(owner(), NOW)
    inspector = FakeInspector(True, [])

    result = store.acquire_lease(owner("new"), NOW + timedelta(seconds=11), inspector)

    assert result.status is LeaseStatus.WATCHER_OPERATOR_REVIEW_REQUIRED
    assert result.owner_token == "old"
    assert inspector.calls == ["inspect"]


@pytest.mark.parametrize("alive", [False, None])
def test_expired_dead_or_unverifiable_owner_can_be_reconciled(tmp_path, alive):
    store = WatcherStore(tmp_path / "watch.db", lease_ttl_seconds=10)
    store.acquire_lease(owner(), NOW)

    result = store.acquire_lease(
        owner("new"), NOW + timedelta(seconds=11), FakeInspector(alive, [])
    )

    assert result.status is LeaseStatus.ACQUIRED
    assert result.owner_token == "new"
    assert result.recovered_expired is True


def test_evaluation_due_flag_is_fenced_and_persisted(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    acquired = store.acquire_lease(owner(), NOW)

    store.mark_evaluation_due(acquired.owner_token, NOW)
    assert store.summary(NOW)["evaluation_due_pending"] is True
    store.clear_evaluation_due(acquired.owner_token, NOW, "OK")
    assert store.summary(NOW)["evaluation_due_pending"] is False

    with pytest.raises(LeaseLostError):
        store.mark_evaluation_due("stale-token", NOW)


def test_schema_is_idempotent_and_preserves_phase5_tables(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    store.initialize()
    store.initialize()

    tables = set(store.table_names())
    assert {"forex_watcher_state", "forex_watch_opportunities", "forex_watch_runs"} <= tables


def test_summary_exposes_evaluation_quality_counts_without_training_labels(tmp_path):
    summary = WatcherStore(tmp_path / "watch.db").summary(NOW)

    assert summary["evaluations_by_basis_and_horizon_and_status"] == {}
    assert summary["fully_terminal_outcome_decision_count"] == 0
    assert "training_eligible" not in summary


def _opportunity(key: str) -> ScheduledOpportunity:
    return ScheduledOpportunity(
        requested_symbol="EURUSD",
        analysis_profile="INTRADAY",
        schedule_timeframe="M15",
        anchor_timestamp=NOW,
        bar_close_timestamp=NOW + timedelta(minutes=15),
        eligible_after=NOW + timedelta(minutes=15, seconds=30),
        config_fingerprint=key,
    )


def _insert_running_run(store: WatcherStore, owner_token: str, source_run_id: str):
    item = _opportunity(source_run_id)
    store.observe_opportunity(item, NOW)
    return store.claim_opportunity(
        owner_token,
        item.opportunity_key,
        NOW,
        source_run_id=source_run_id,
    )


def _record_decision(decision_store: ShadowDecisionStore, source_run_id: str, executed=False):
    decision = ShadowTradeDecision(
        decision_id=f"decision-{source_run_id}-{id(decision_store)}",
        created_at=NOW,
        snapshot_timestamp=NOW,
        analysis_date=date(2026, 9, 9),
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
        spread_points=1.0,
        analysis_timeframe="M15",
        trader_summary="",
        portfolio_manager_summary="",
        source_run_id=source_run_id,
        executed=executed,
        decision_context_status="COMPLETE",
    )
    decision_store.record(decision)
    return decision


def test_reconciliation_links_one_decision_and_never_reruns_old_key(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    decision_store = ShadowDecisionStore(tmp_path / "shadow.db")
    acquired = store.acquire_lease(owner(), NOW)
    run = _insert_running_run(store, acquired.owner_token, source_run_id="run-1")
    decision = _record_decision(decision_store, source_run_id="run-1")

    actions = store.reconcile_stale_runs(acquired.owner_token, NOW, decision_store)

    assert actions == ("DECISION_SAVED",)
    assert store.get_run(run.run_id).decision_id == decision.decision_id
    assert store.get_opportunity(run.opportunity_key).status == "DECISION_SAVED"


def test_reconciliation_abandons_zero_match_and_flags_multiple_matches(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    decision_store = ShadowDecisionStore(tmp_path / "shadow.db")
    acquired = store.acquire_lease(owner(), NOW)
    missing = _insert_running_run(store, acquired.owner_token, source_run_id="missing")
    _insert_running_run(store, acquired.owner_token, source_run_id="ambiguous")
    _record_decision(decision_store, source_run_id="ambiguous")
    # Use a distinct ID while retaining the same source ID to reproduce an
    # ambiguous crash join.
    first = decision_store.find_by_source_run_id("ambiguous")[0]
    decision_store.record(replace(first, decision_id="decision-ambiguous-duplicate"))

    actions = store.reconcile_stale_runs(acquired.owner_token, NOW, decision_store)

    assert actions[0] == "ABANDONED"
    assert "RECONCILIATION_AMBIGUOUS" in actions
    assert store.get_run(missing.run_id).run_status == "ABANDONED"
    assert store.circuit_reason() == "RECONCILIATION_AMBIGUOUS"
