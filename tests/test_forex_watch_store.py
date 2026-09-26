from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig, collect_runtime_provenance
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


def test_read_only_active_lease_does_not_initialize_or_write(tmp_path):
    path = tmp_path / "missing.db"
    store = WatcherStore(path)

    assert store.read_only_active_lease(NOW) is None
    assert not path.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lease_ttl_seconds": True},
        {"lease_ttl_seconds": 10.5},
        {"lease_ttl_seconds": "10"},
        {"lease_ttl_seconds": 0},
        {"busy_timeout_seconds": True},
        {"busy_timeout_seconds": 5.5},
        {"busy_timeout_seconds": "5"},
        {"busy_timeout_seconds": 0},
    ],
)
def test_watcher_store_rejects_non_contract_timeout_values(tmp_path, kwargs):
    with pytest.raises(ValueError):
        WatcherStore(tmp_path / "watch.db", **kwargs)


def test_read_only_active_lease_reads_existing_owner_without_mutation(tmp_path):
    path = tmp_path / "watch.db"
    store = WatcherStore(path)
    acquired = store.acquire_lease(owner(), NOW)
    before = path.read_bytes()

    observed = store.read_only_active_lease(NOW)

    assert acquired.status is LeaseStatus.ACQUIRED
    assert observed is not None
    assert observed.owner_token == "old"
    assert path.read_bytes() == before


def test_read_only_summary_does_not_create_or_initialize_database(tmp_path):
    path = tmp_path / "missing.db"
    store = WatcherStore(path)

    assert store.read_only_summary(NOW) == {}
    assert not path.exists()


def test_read_only_summary_reads_without_initialization_or_mutation(tmp_path):
    path = tmp_path / "watch.db"
    store = WatcherStore(path)
    store.initialize()
    before = path.read_bytes()
    store.initialize = lambda: (_ for _ in ()).throw(AssertionError("read-only summary initialized"))

    summary = store.read_only_summary(NOW)

    assert summary["lifecycle_status"] == "STOPPED"
    assert path.read_bytes() == before


def test_evaluation_due_flag_is_fenced_and_persisted(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    acquired = store.acquire_lease(owner(), NOW)

    store.mark_evaluation_due(acquired.owner_token, NOW)
    assert store.summary(NOW)["evaluation_due_pending"] is True
    store.clear_evaluation_due(acquired.owner_token, NOW, "OK")
    assert store.summary(NOW)["evaluation_due_pending"] is False

    with pytest.raises(LeaseLostError):
        store.mark_evaluation_due("stale-token", NOW)


def test_successful_evaluation_clears_previous_evaluation_error(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    acquired = store.acquire_lease(owner(), NOW)

    store.set_error(
        acquired.owner_token,
        "EVALUATION_FAILED",
        "historical read failed",
        NOW,
    )
    store.clear_evaluation_due(acquired.owner_token, NOW, "OK")

    summary = store.summary(NOW)
    assert summary["last_evaluation_status"] == "OK"
    assert summary["last_error_code"] is None
    assert summary["last_error"] is None


def test_successful_evaluation_preserves_unrelated_error(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    acquired = store.acquire_lease(owner(), NOW)

    store.set_error(acquired.owner_token, "ANALYSIS_FAILED", "analysis failed", NOW)
    store.clear_evaluation_due(acquired.owner_token, NOW, "OK")

    summary = store.summary(NOW)
    assert summary["last_error_code"] == "ANALYSIS_FAILED"
    assert summary["last_error"] == "analysis failed"


def test_error_status_without_code_has_safe_read_only_fallback(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    store.initialize()
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE forex_watcher_state SET last_evaluation_status='ERROR', last_error_code=NULL, last_error=NULL WHERE singleton_id=1"
        )
        conn.commit()

    summary = store.read_only_summary()

    assert summary["last_evaluation_status"] == "ERROR"
    assert summary["last_error_code"] == "EVALUATION_FAILED"
    assert summary["last_error"] is None


def test_schema_is_idempotent_and_preserves_phase5_tables(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    store.initialize()
    store.initialize()

    tables = set(store.table_names())
    assert {"forex_watcher_state", "forex_watch_opportunities", "forex_watch_runs"} <= tables


def test_new_runs_persist_runtime_provenance_at_claim(tmp_path):
    provenance = collect_runtime_provenance(
        ForexShadowRuntimeConfig(), repo_root=tmp_path, git_runner=lambda *a, **k: type(
            "Result", (), {"returncode": 0, "stdout": "abc123\n"}
        )()
    )
    store = WatcherStore(tmp_path / "watch.db", provenance=provenance)
    acquired = store.acquire_lease(owner(), NOW)
    run = _insert_running_run(store, acquired.owner_token, source_run_id="provenance-run")

    persisted = store.get_run(run.run_id)
    assert persisted.git_commit == "abc123"
    assert persisted.prompt_config_version == "forex-shadow.runtime.v1"
    assert persisted.application_version
    assert persisted.collector_contract_version == "forex-watch.v1"
    assert persisted.config_fingerprint
    assert persisted.safe_config_json.startswith("{")


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


def _record_decision(
    decision_store: ShadowDecisionStore,
    source_run_id: str,
    executed=False,
    **changes,
):
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
    if changes:
        decision = replace(decision, **changes)
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


def test_reconciliation_preserves_recovered_freshness_evidence(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    decision_store = ShadowDecisionStore(tmp_path / "shadow.db")
    acquired = store.acquire_lease(owner(), NOW)
    run = _insert_running_run(store, acquired.owner_token, source_run_id="stale-run")
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE forex_watch_runs SET freshness_budget_seconds=900 WHERE run_id=?",
            (run.run_id,),
        )
        conn.commit()
    _record_decision(
        decision_store,
        source_run_id="stale-run",
        decision_completed_timestamp=NOW + timedelta(seconds=901),
        analysis_latency_seconds=901.0,
    )

    actions = store.reconcile_stale_runs(acquired.owner_token, NOW, decision_store)

    assert actions == ("DECISION_SAVED",)
    recovered = store.get_run(run.run_id)
    assert recovered.stale_by_completion is True
    assert recovered.freshness_budget_seconds == 900
    assert store.summary()["last_analysis_completed_at"] == "2026-09-09T12:00:00Z"


def test_reconciliation_preserves_runtime_alert_as_slow_success(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    decision_store = ShadowDecisionStore(tmp_path / "shadow.db")
    acquired = store.acquire_lease(owner(), NOW)
    run = _insert_running_run(store, acquired.owner_token, source_run_id="slow-run")
    store.mark_runtime_alert(acquired.owner_token, run.run_id, NOW + timedelta(seconds=901))
    _record_decision(decision_store, source_run_id="slow-run")

    actions = store.reconcile_stale_runs(acquired.owner_token, NOW, decision_store)

    assert actions == ("DECISION_SAVED",)
    assert store.get_run(run.run_id).run_status == "SUCCEEDED_SLOW"


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
