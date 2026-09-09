# Phase 6 Automatic Shadow Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add a separate, autonomous forex-watch shadow collector that schedules completed forex bars, runs at most one existing ForexShadowRunner analysis, reuses the Phase 5 read-only evaluator, and persists auditable skips, leases, provenance, and outcomes without creating an execution path.

**Architecture:** A foreground WatcherCoordinator owns a UTC bar-close scheduler, SQLite lease/state store, circuit breakers, and a zero-capacity single analysis slot. ReadOnlyMarketProbe, ForexShadowRunner, and ShadowOutcomeEvaluator are injected boundaries protected by a non-reentrant MT5 operation gate; while the runner is active the coordinator only heartbeats, observes buckets, and records skips. Existing graph, snapshot, decision, and Phase 5 evaluation contracts remain authoritative.

**Tech Stack:** Python >=3.10; standard-library dataclasses, datetime, hashlib, json, sqlite3, threading, and concurrent.futures; existing MT5Provider, ForexShadowRunner, ShadowDecisionStore, ShadowOutcomeEvaluator, argparse, pytest, Ruff, and the current pyproject.toml console-script layout. No new runtime dependency.

**Spec:** docs/superpowers/specs/2026-09-09-phase-6-automatic-shadow-collector-design.md

## Global Constraints

- Add only the separate forex-watch entry point; leave the tradingagents stock CLI and existing forex-shadow command unchanged except where the existing Phase 5 evaluator must reject an actively leased database.
- MT5 remains read-only. The watcher may use only MT5Provider read methods through the probe, runner, and Phase 5 evaluator; no order_send, buy/sell, open/close/modify/cancel method, order client, or execution callback may be introduced.
- The v1 invariant is at most one MT5-using operation at a time. During ANALYZING, do not probe, evaluate, create another provider/session, or submit another analysis.
- The runner owns exactly one cached ForexMarketSnapshot; the watcher never fetches or copies a second analysis snapshot.
- Defaults are EURUSD, INTRADAY, analysts market,news, timeframe M15, 15-second polling, 30-second bar settle, 60-second cooldown, one active analysis, and a zero-capacity queue.
- A new opportunity observed while a runner is active is persisted once as SKIPPED / ANALYSIS_ALREADY_RUNNING; it is never replayed or queued.
- A non-expired SQLite lease always returns WATCHER_ALREADY_RUNNING, even if a PID probe says the owner is gone. Only an expired lease may enter takeover/reconciliation; an exact old same-host process proven alive returns WATCHER_OPERATOR_REVIEW_REQUIRED and blocks startup.
- If evaluation becomes due during ANALYZING, persist evaluation_due_pending=1; after the runner returns and shuts down MT5, evaluate immediately before the next probe or runner.
- Reuse Phase 5 evaluation and its PENDING, COMPLETE, DATA_UNAVAILABLE, and INELIGIBLE semantics. Do not copy outcome math or add training labels.
- Persist only safe provenance/configuration metadata. Never accept or log credentials, prompts, completions, raw debate prose, or private chain-of-thought in watcher operational records.
- Preserve separate analysis/reference timestamps, quote sets, latency, reference delay, and staleness evidence; never call an analysis-snapshot quote an executable entry.
- Use injected UTC/fake clocks and deterministic fakes in normal tests. No required test may sleep for model time, call Ollama/hosted LLMs, or require a live terminal.

## File Map

### New files

- tradingagents/forex/watch_store.py — Phase 6 SQLite schema, lease/fencing, opportunities, runs, recovery, due-evaluation flag, and summaries.
- tradingagents/forex/watcher.py — validated WatcherConfig, clock/schedule contracts, read-only probe, MT5 operation gate, single-slot executor, heartbeat, circuit breakers, evaluator coordination, and WatcherCoordinator.
- cli/forex_watch.py — separate forex-watch run|once|status|evaluate CLI; no stock CLI registration.
- tests/test_forex_watch_store.py — schema, canonical keys, lease ownership/takeover, fencing, claim/finalize, due flag, and crash reconciliation tests.
- tests/test_forex_watcher.py — configuration, scheduling, probe/gate, executor, coordinator lifecycle, skip, cooldown, evaluation ordering, circuit, and restart tests.
- tests/test_forex_watch_cli.py — parser, banners, status JSON, exit codes, and forbidden-option tests.
- tests/test_forex_watch_integration.py — mocked bounded end-to-end safety trace plus opt-in local MT5 read-only probe/evaluation checks.

### Existing files with minimal changes

- tradingagents/forex/runner.py — accept an optional source_run_id keyword and persist it unchanged; retain random UUID behavior when omitted.
- tradingagents/forex/shadow.py — add read-only find_by_source_run_id query only.
- cli/forex_evaluate.py — refuse with WATCHER_ALREADY_RUNNING before opening MT5 when the target database has a non-expired watcher lease.
- tradingagents/forex/__init__.py — re-export watcher contracts if public imports are useful; do not import stock CLI code.
- pyproject.toml — add only forex-watch = "cli.forex_watch:main".
- docs/forex-shadow.md — document collector lifecycle, lease/takeover rules, serialized MT5 access, skips, evaluation, status fields, and the safety banner.

No other agent workflow, LangGraph registration, TradingAgents tool registry, stock CLI behavior, MT5 provider code, execution code, training/evaluation-label code, RAG, ONNX, or RiskGovernor code is in scope.

## Deterministic test fixture convention

The test snippets use test-local helpers rather than production shortcuts. Define
these helpers at the top of tests/test_forex_watcher.py before the first test
that references them, and import only pure values into the other Phase 6 test
modules:

- NOW is the aware UTC instant 2026-09-09T12:00:00Z and _utc(text) parses a Z-suffixed timestamp into an aware UTC datetime.
- WatcherHarness(tmp_path, symbols=("EURUSD",), runner_result=None, probe_error=None, process_alive=None) constructs a temporary WatcherStore, FrozenClock, CompletedBarSchedule, fake probe, fake runner, fake evaluator, SerializedMt5OperationGate, SingleSlotAnalysisExecutor, and WatcherCoordinator. Its start(now=None), poll(now=None), heartbeat(monotonic), complete_runner(completed_at=None), crash_without_finalizing(), and shutdown() methods call only the public contracts in this plan. harness.events contains allow-listed event mappings; harness.sequence contains internal test labels such as runner_shutdown, evaluation, and next_probe for ordering assertions.
- _complete_run_result() returns a fake ForexShadowRunResult whose decision has a non-empty source_run_id, decision_context_status COMPLETE, normalization_status NORMALIZED, a BUY/SELL/HOLD action, and executed=False. _deferred_run_result() returns a Future-controlled result with the same evidence.
- The fake probe records requested symbols, the fake evaluator records calls and returns EvaluationRunEvidence(status="OK", errors=(), metrics={"llm_calls": 0}), and the fake provider factory records created providers plus created_while_active. Fakes expose no mutation method.
- The fake executor submitter returns a concurrent.futures.Future without starting a thread; tests complete it with set_result() so slot behavior is deterministic and never waits on wall-clock time.
- _owner(), _insert_running_run(), _record_decision(), and _valid_final_state() are test builders for the lease/reconciliation cases; each returns the concrete dataclasses from Tasks 2 and 3 and never calls MT5 or an LLM.

---

### Task 1: Add typed watcher configuration, UTC clock, and completed-bar schedule

**Files:**
- Create: tradingagents/forex/watcher.py
- Create: tests/test_forex_watcher.py

**Interfaces:**
- Produces WatcherConfig, Clock, SystemClock, FrozenClock, ScheduledOpportunity, SchedulePolicy, CompletedBarSchedule, canonical_opportunity_key, and safe_effective_config for all later tasks.
- WatcherConfig defaults and validation must exactly match spec section 16; there is no separate per-interval symbol-count field.
- SchedulePolicy.opportunities_between(previous: datetime | None, now: datetime) -> tuple[ScheduledOpportunity, ...] accepts timezone-aware UTC values and never calls MT5 or an LLM.

- [ ] Step 1: Write failing configuration and scheduling tests

~~~python
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradingagents.forex.watcher import (
    CompletedBarSchedule,
    WatcherConfig,
    canonical_opportunity_key,
)


def test_watcher_defaults_are_forex_safe_and_single_slot(tmp_path):
    config = WatcherConfig(db_path=tmp_path / "watch.db")

    assert config.symbols == ("EURUSD",)
    assert config.analysis_profile == "INTRADAY"
    assert config.analysts == ("market", "news")
    assert config.schedule_timeframe == "M15"
    assert config.max_concurrent_analyses == 1
    assert config.max_attempts_per_opportunity == 1
    assert config.max_recovery_buckets == 8
    assert not hasattr(config, "max_symbols_per_interval")


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_concurrent_analyses", 2),
        ("poll_interval_seconds", True),
        ("bar_close_settle_seconds", -1),
        ("analysts", ("market", "fundamentals")),
        ("schedule_timeframe", "H4"),
    ],
)
def test_watcher_config_rejects_unsafe_or_invalid_values(tmp_path, field, value):
    with pytest.raises(ValueError):
        WatcherConfig(db_path=tmp_path / "watch.db", **{field: value})


def test_m15_policy_returns_only_settled_completed_bucket():
    policy = CompletedBarSchedule(
        timeframe="M15", settle_seconds=30, config_fingerprint="cfg"
    )
    now = datetime(2026, 9, 9, 12, 15, 31, tzinfo=timezone.utc)

    opportunities = policy.opportunities_between(None, now)

    assert len(opportunities) == 1
    item = opportunities[0]
    assert item.anchor_timestamp == datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    assert item.bar_close_timestamp == datetime(2026, 9, 9, 12, 15, tzinfo=timezone.utc)
    assert item.eligible_after == datetime(2026, 9, 9, 12, 15, 30, tzinfo=timezone.utc)
    assert item.opportunity_key == canonical_opportunity_key(item)


def test_schedule_does_not_emit_forming_bucket_or_duplicate_anchor():
    policy = CompletedBarSchedule(
        timeframe="M15", settle_seconds=30, config_fingerprint="cfg"
    )
    first = datetime(2026, 9, 9, 12, 15, 31, tzinfo=timezone.utc)
    second = datetime(2026, 9, 9, 12, 15, 45, tzinfo=timezone.utc)

    first_items = policy.opportunities_between(None, first)
    second_items = policy.opportunities_between(first, second)

    assert first_items
    assert second_items == ()


def test_safe_effective_config_excludes_credentials():
    from tradingagents.forex.watcher import safe_effective_config

    safe = safe_effective_config(
        {"llm_provider": "openai", "deep_think_llm": "gpt-5.6", "api_key": "secret"}
    )

    assert safe["llm_provider"] == "openai"
    assert "api_key" not in safe
    assert "secret" not in repr(safe)
~~~

- [ ] Step 2: Run the focused tests to verify they fail

Run:

~~~powershell
pytest tests/test_forex_watcher.py -q
~~~

Expected: FAIL because the Phase 6 watcher module and its contracts do not yet exist.

- [ ] Step 3: Implement the domain contracts and validation

Add these concrete types and validation rules:

~~~python
@dataclass(frozen=True, slots=True)
class WatcherConfig:
    symbols: tuple[str, ...] = ("EURUSD",)
    analysis_profile: str = "INTRADAY"
    analysts: tuple[str, ...] = ("market", "news")
    schedule_timeframe: str = "M15"
    poll_interval_seconds: int = 15
    bar_close_settle_seconds: int = 30
    cooldown_seconds: int = 60
    max_concurrent_analyses: int = 1
    analysis_timeout_seconds: int = 7200
    lease_heartbeat_seconds: int = 15
    lease_ttl_seconds: int = 9000
    evaluation_interval_seconds: int = 60
    evaluation_enabled: bool = True
    max_attempts_per_opportunity: int = 1
    max_recovery_buckets: int = 8
    freshness_budget_seconds: int = 900
    max_consecutive_mt5_failures: int = 3
    max_consecutive_analysis_failures: int = 3
    max_consecutive_incomplete: int = 5
    max_consecutive_normalization_failures: int = 3
    max_consecutive_runtime_exceeded: int = 2
    circuit_breaker_cooldown_seconds: int = 1800
    db_busy_timeout_seconds: int = 5
    db_path: Path = Path("data_cache/shadow_decisions.db")
    terminal_path: str | None = None

    def __post_init__(self) -> None:
        normalized_symbols = _normalize_symbol_tuple(self.symbols)
        normalized_analysts = _normalize_analyst_tuple(self.analysts)
        object.__setattr__(self, "symbols", normalized_symbols)
        object.__setattr__(self, "analysts", normalized_analysts)
        _require_choice(self.schedule_timeframe, {"M5", "M15", "H1"}, "schedule_timeframe")
        _require_choice(self.analysis_profile, {"INTRADAY"}, "analysis_profile")
        _require_positive_int(self.poll_interval_seconds, "poll_interval_seconds")
        _require_nonnegative_int(self.bar_close_settle_seconds, "bar_close_settle_seconds")
        _require_nonnegative_int(self.cooldown_seconds, "cooldown_seconds")
        if self.max_concurrent_analyses != 1:
            raise ValueError("max_concurrent_analyses must be exactly 1 in Phase 6 v1")
        _require_positive_int(self.analysis_timeout_seconds, "analysis_timeout_seconds")
        _require_positive_int(self.lease_heartbeat_seconds, "lease_heartbeat_seconds")
        _require_positive_int(self.lease_ttl_seconds, "lease_ttl_seconds")
        if self.lease_heartbeat_seconds >= self.lease_ttl_seconds:
            raise ValueError("lease_heartbeat_seconds must be less than lease_ttl_seconds")
        if self.lease_ttl_seconds <= self.analysis_timeout_seconds + self.lease_heartbeat_seconds:
            raise ValueError(
                "lease_ttl_seconds must exceed analysis_timeout_seconds plus heartbeat grace"
            )
        _require_nonnegative_int(self.evaluation_interval_seconds, "evaluation_interval_seconds")
        _require_positive_int(self.max_attempts_per_opportunity, "max_attempts_per_opportunity")
        if self.max_attempts_per_opportunity != 1:
            raise ValueError("max_attempts_per_opportunity must be 1 in Phase 6 v1")
        _require_positive_int(self.max_recovery_buckets, "max_recovery_buckets")
        _require_nonnegative_int(self.freshness_budget_seconds, "freshness_budget_seconds")
        for field_name in (
            "max_consecutive_mt5_failures",
            "max_consecutive_analysis_failures",
            "max_consecutive_incomplete",
            "max_consecutive_normalization_failures",
            "max_consecutive_runtime_exceeded",
        ):
            _require_positive_int(getattr(self, field_name), field_name)
        _require_positive_int(self.circuit_breaker_cooldown_seconds, "circuit_breaker_cooldown_seconds")
        _require_positive_int(self.db_busy_timeout_seconds, "db_busy_timeout_seconds")
        if not isinstance(self.evaluation_enabled, bool):
            raise ValueError("evaluation_enabled must be a bool")
        if self.terminal_path is not None and not isinstance(self.terminal_path, str):
            raise ValueError("terminal_path must be a string or None")

    @property
    def safe_fingerprint(self) -> str:
        payload = safe_effective_config(asdict(self))
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class Clock(Protocol):
    def now(self) -> datetime:
        pass

    def monotonic(self) -> float:
        pass


@dataclass
class FrozenClock:
    current: datetime
    current_monotonic: float = 0.0

    def now(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.current_monotonic

    def set(self, value: datetime) -> None:
        self.current = value


class SchedulePolicy(Protocol):
    def opportunities_between(
        self, previous: datetime | None, now: datetime
    ) -> tuple[ScheduledOpportunity, ...]:
        pass
~~~

Use datetime.astimezone(timezone.utc) only after requiring tzinfo; floor UTC instants to M5/M15/H1 bucket starts, derive bar_close_timestamp, and set eligible_after = bar_close + settle_seconds. canonical_opportunity_key() hashes sorted UTF-8 JSON containing normalized requested symbol, profile, timeframe, anchor timestamp, analysis contract version, and config fingerprint. safe_effective_config() merges DEFAULT_CONFIG with watcher values and copies only the allow-listed provider/model, analyst/profile, model-limit, prompt/collector-version, and forex-safe fields; reject keys containing credentials instead of serializing them. The production CLI passes the resulting effective non-secret mapping to ForexShadowRunner, while provider/model credentials continue to come only from DEFAULT_CONFIG and TRADINGAGENTS_* environment overrides. Define _normalize_symbol_tuple, _normalize_analyst_tuple, _require_positive_int, _require_nonnegative_int, and _require_choice with explicit bool rejection so booleans cannot pass as integers.

- [ ] Step 4: Run the focused tests to verify they pass

Run:

~~~powershell
pytest tests/test_forex_watcher.py -q
~~~

Expected: PASS for configuration, UTC bucket, dedup key, and secret-redaction tests.

- [ ] Step 5: Commit the self-contained domain work

~~~powershell
git add tradingagents/forex/watcher.py tests/test_forex_watcher.py
git commit -m "feat: add phase 6 watcher config and schedule contracts"
~~~

---

### Task 2: Create the SQLite watcher store and strict lease protocol

**Files:**
- Create: tradingagents/forex/watch_store.py
- Modify: tradingagents/forex/watcher.py (import the shared ScheduledOpportunity contract)
- Test: tests/test_forex_watch_store.py

**Interfaces:**
- Produces LeaseOwner, LeaseResult, LeaseStatus, ProcessIdentity, ProcessInspector, WatchRun, RunEvidence, SkipReason, LeaseLostError, and WatcherStore.
- WatcherStore.acquire_lease(owner: LeaseOwner, now: datetime, inspector: ProcessInspector | None = None) -> LeaseResult performs the strict expiry-first protocol.
- WatcherStore.mark_evaluation_due(owner_token: str, now: datetime) -> None and clear_evaluation_due(owner_token: str, now: datetime, status: str) -> None update the fenced singleton state.
- WatcherStore.observe_opportunity(), claim_opportunity(), record_skip(), finalize_run(), get_run(), get_opportunity(), list_runs(), list_opportunities(), list_skipped(), decision_for_run(), active_lease(), summary(), and table_names() are the read/write seams used by the coordinator, CLI, and deterministic tests. active_lease() is read-only and parses the singleton lease without renewing or taking it.
- All write methods validate the owner token in the same transaction; stale owners raise LeaseLostError and cannot mutate the current owner’s state.

- [ ] Step 1: Write failing schema and lease tests

~~~python
from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.forex.watch_store import (
    LeaseOwner,
    LeaseStatus,
    WatcherStore,
)

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
~~~

- [ ] Step 2: Run the focused store tests to verify they fail

Run:

~~~powershell
pytest tests/test_forex_watch_store.py -q
~~~

Expected: FAIL because the Phase 6 SQLite store and lease types do not exist.

- [ ] Step 3: Implement idempotent schema and expiry-first lease acquisition

Create the three tables exactly from spec section 11, including the singleton evaluation_due_pending column, owner/run provenance, controlled statuses, unique opportunity identity, unique source_run_id, and indexes. Open a fresh SQLite connection per operation, set PRAGMA foreign_keys=ON, use the configured busy timeout, and use BEGIN IMMEDIATE for claims and lease transitions.

The core lease branch must be equivalent to:

~~~python
def acquire_lease(self, owner, now, inspector=None):
    with self._transaction(immediate=True) as conn:
        current = self._read_lease(conn)
        if current is not None and current.lease_expires_at > now:
            return LeaseResult(
                LeaseStatus.WATCHER_ALREADY_RUNNING,
                owner_token=current.owner_token,
                recovered_expired=False,
            )
        if current is not None and inspector is not None:
            if inspector.proves_alive(current):
                return LeaseResult(
                    LeaseStatus.WATCHER_OPERATOR_REVIEW_REQUIRED,
                    owner_token=current.owner_token,
                    recovered_expired=False,
                )
        self._write_new_lease(conn, owner, now)
        return LeaseResult(
            LeaseStatus.ACQUIRED,
            owner_token=owner.owner_token,
            recovered_expired=current is not None,
        )
~~~

Do not call inspector on the non-expired branch. Store canonical UTC timestamps with Z, compare parsed aware values, bound sanitized errors, and add owner-token fencing to heartbeat, due-flag, lifecycle, claim, finalize, and release methods. ProcessInspector.proves_alive() returns true only for the exact same host, PID, and recorded process-start identity; a missing/unverifiable identity after expiry is not proof of liveness.

- [ ] Step 4: Run the focused store tests to verify they pass

Run:

~~~powershell
pytest tests/test_forex_watch_store.py -q
~~~

Expected: PASS, including the regression proving a valid lease is never stolen by a dead-PID observation.

- [ ] Step 5: Commit the store and lease work

~~~powershell
git add tradingagents/forex/watch_store.py tests/test_forex_watch_store.py
git commit -m "feat: add phase 6 sqlite lease store"
~~~

---

### Task 3: Add crash-reconciliation joins and the runner source ID seam

**Files:**
- Modify: tradingagents/forex/runner.py
- Modify: tradingagents/forex/shadow.py
- Modify: tradingagents/forex/watch_store.py
- Test: tests/test_forex_shadow_runner.py
- Test: tests/test_forex_watch_store.py

**Interfaces:**
- ForexShadowRunner.run with optional keyword source_run_id: str | None -> ForexShadowRunResult keeps all existing arguments/callers and uses a validated caller ID when present.
- ShadowDecisionStore.find_by_source_run_id(source_run_id: str) -> tuple[ShadowTradeDecision, ...] is read-only and returns all matching rows for ambiguity detection.
- WatcherStore.reconcile_stale_runs(owner_token: str, now: datetime, decision_store: ShadowDecisionStore) -> tuple[str, ...] links exactly one matching executed=False decision, abandons zero matches, and opens operator review for multiple matches or an execution invariant violation.

The store join value is a safe summary, not a copy of the decision payload:

~~~python
@dataclass(frozen=True, slots=True)
class RunEvidence:
    run_id: str
    source_run_id: str
    decision_id: str | None
    requested_symbol: str
    resolved_symbol: str | None
    decision_context_status: str | None
    normalization_status: str | None
    normalized_action: str | None
    analysis_snapshot_timestamp: datetime | None
    decision_completed_timestamp: datetime | None
    analysis_latency_seconds: float | None
    decision_reference_timestamp: datetime | None
    decision_reference_delay_seconds: float | None
    stale_by_completion: bool | None
    runtime_seconds: float | None
    llm_calls: int | None
    tool_calls: int | None
    tokens_in: int | None
    tokens_out: int | None
    reasoning_tokens: int | None
    llm_provider: str | None
    quick_model: str | None
    deep_model: str | None
    executed: bool
~~~

- [ ] Step 1: Write failing seam and recovery tests

~~~python
def test_runner_uses_watcher_source_run_id(tmp_path):
    runner, _, _, store = _make_runner(tmp_path, _valid_final_state())

    result = runner.run(
        symbol="EURUSD",
        analysis_date="2026-09-09",
        source_run_id="watch-run-001",
    )

    assert result.decision.source_run_id == "watch-run-001"
    assert store.find_by_source_run_id("watch-run-001")[0].decision_id == result.decision.decision_id


def test_runner_rejects_empty_source_run_id(tmp_path):
    runner, _, _, _ = _make_runner(tmp_path, _valid_final_state())

    with pytest.raises(ValueError):
        runner.run(symbol="EURUSD", source_run_id=" ")


def test_reconciliation_links_one_decision_and_never_reruns_old_key(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    decision_store = ShadowDecisionStore(tmp_path / "shadow.db")
    acquired = store.acquire_lease(_owner(), NOW)
    run = _insert_running_run(store, acquired.owner_token, source_run_id="run-1")
    decision = _record_decision(decision_store, source_run_id="run-1", executed=False)

    actions = store.reconcile_stale_runs(acquired.owner_token, NOW, decision_store)

    assert actions == ("DECISION_SAVED",)
    assert store.get_run(run.run_id).decision_id == decision.decision_id
    assert store.get_opportunity(run.opportunity_key).status == "DECISION_SAVED"


def test_reconciliation_abandons_zero_match_and_flags_multiple_matches(tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    decision_store = ShadowDecisionStore(tmp_path / "shadow.db")
    acquired = store.acquire_lease(_owner(), NOW)
    _insert_running_run(store, acquired.owner_token, source_run_id="missing")
    _insert_running_run(store, acquired.owner_token, source_run_id="ambiguous")
    _record_decision(decision_store, source_run_id="ambiguous", executed=False)
    _record_decision(decision_store, source_run_id="ambiguous", executed=False)

    actions = store.reconcile_stale_runs(acquired.owner_token, NOW, decision_store)

    assert actions == ("ABANDONED", "RECONCILIATION_AMBIGUOUS")
    assert store.list_runs()[0].run_status == "ABANDONED"
    assert store.circuit_reason() == "RECONCILIATION_AMBIGUOUS"
~~~

- [ ] Step 2: Run the focused tests to verify they fail

Run:

~~~powershell
pytest tests/test_forex_shadow_runner.py::test_runner_uses_watcher_source_run_id tests/test_forex_watch_store.py -q
~~~

Expected: FAIL because run() does not accept the keyword and the read-only source-ID query/reconciliation seam is absent.

- [ ] Step 3: Implement the smallest compatible seams

Change only the runner keyword signature and the assignment immediately before ShadowTradeDecision construction:

~~~python
def run(
    self,
    symbol="EURUSD",
    count=100,
    analysis_date=None,
    terminal_path=None,
    analysts=None,
    *,
    db_path=None,
    callbacks=None,
    analysis_profile="INTRADAY",
    source_run_id: str | None = None,
):
    if source_run_id is not None:
        if not isinstance(source_run_id, str) or not source_run_id.strip():
            raise ValueError("source_run_id must be a non-empty string")
        source_run_id = source_run_id.strip()
    decision = ShadowTradeDecision(
        # existing fields remain unchanged
        source_run_id=source_run_id or str(uuid.uuid4()),
    )
~~~

Implement find_by_source_run_id() with SELECT * FROM shadow_decisions WHERE source_run_id = ? ORDER BY created_at, decision_id, returning parsed models and never updating the database. Reconciliation must query before changing the watcher row, accept exactly one decision only when executed is False, mark zero as ABANDONED / STALE_RUN_RECOVERED, and mark more than one or any execution violation as RECONCILIATION_AMBIGUOUS plus a persisted circuit/operator-review error. It must never call the runner again for the old opportunity key.

- [ ] Step 4: Run focused and existing runner tests

Run:

~~~powershell
pytest tests/test_forex_shadow_runner.py tests/test_forex_watch_store.py -q
~~~

Expected: PASS with all pre-existing Phase 4/5 runner behavior unchanged.

- [ ] Step 5: Commit the source/recovery seams

~~~powershell
git add tradingagents/forex/runner.py tradingagents/forex/shadow.py tradingagents/forex/watch_store.py tests/test_forex_shadow_runner.py tests/test_forex_watch_store.py
git commit -m "feat: add watcher source run reconciliation seam"
~~~

---

### Task 4: Implement read-only MT5 probing, the operation gate, single-slot execution, and heartbeat

**Files:**
- Modify: tradingagents/forex/watcher.py
- Test: tests/test_forex_watcher.py

**Interfaces:**
- MarketProbeResult contains requested symbol, resolved broker symbol, broker tick timestamp, bid, ask, and finite quote validation.
- ReadOnlyMarketProbe.probe(requested_symbol: str) -> MarketProbeResult creates one provider, initializes/resolves/reads one tick, and shuts it down in finally; it has no mutation method.
- Mt5OperationGate.acquire(operation: str) -> ContextManager[None] is non-reentrant and raises Mt5OperationBusy rather than permitting overlap.
- SingleSlotAnalysisExecutor.submit(fn: Callable[[], T]) -> Future[T] rejects while active; it has no pending queue.
- LeaseHeartbeat.tick(now: datetime, monotonic_now: float) -> None renews/fences state and records one soft-timeout alert without killing the runner.

- [ ] Step 1: Write failing serialization and safety tests

~~~python
import pytest

from tradingagents.forex.watcher import (
    Mt5OperationBusy,
    ReadOnlyMarketProbe,
    SerializedMt5OperationGate,
    SingleSlotAnalysisExecutor,
)


def test_operation_gate_rejects_nested_or_concurrent_operation():
    gate = SerializedMt5OperationGate()
    with gate.acquire("runner"):
        with pytest.raises(Mt5OperationBusy):
            with gate.acquire("evaluator"):
                pass


def test_single_slot_executor_has_zero_capacity_queue():
    submitter = FakeSubmitter()
    executor = SingleSlotAnalysisExecutor(submitter=submitter)
    first = executor.submit(lambda: "done")
    with pytest.raises(RuntimeError, match="analysis slot is busy"):
        executor.submit(lambda: "must-not-queue")
    submitter.complete(first, "done")
    assert first.result() == "done"
    executor.shutdown(wait=True)


def test_probe_shuts_down_provider_and_never_exposes_mutation_api():
    provider = FakeProbeProvider()
    probe = ReadOnlyMarketProbe(lambda terminal_path=None: provider, terminal_path=None)

    result = probe.probe("EURUSD")

    assert result.resolved_symbol == "EURUSDm"
    assert provider.initialize_calls == 1
    assert provider.shutdown_calls == 1
    for name in ("order_send", "buy", "sell", "close_position", "modify_position"):
        assert not hasattr(probe, name)


def test_heartbeat_only_renews_and_marks_soft_timeout():
    heartbeat = _heartbeat_with_fake_store()
    heartbeat.tick(NOW, 0.0)
    heartbeat.tick(NOW + timedelta(seconds=7201), 7201.0)

    assert heartbeat.store.renewals == 2
    assert heartbeat.store.runtime_alerts == 1
    assert heartbeat.runner_future_still_owned is True
~~~

- [ ] Step 2: Run focused tests to verify they fail

Run:

~~~powershell
pytest tests/test_forex_watcher.py -q
~~~

Expected: FAIL because the probe, gate, executor, and heartbeat are not implemented.

- [ ] Step 3: Implement the serialized read-only components

Use threading.Lock with a non-blocking acquire for SerializedMt5OperationGate; include the operation name in a sanitized exception and release in a context manager. The probe must follow this exact lifecycle outside the coordinator active-run branch:

~~~python
def probe(self, requested_symbol):
    provider = None
    try:
        provider = self.provider_factory(terminal_path=self.terminal_path)
        if not provider.initialize():
            raise RuntimeError("MT5 provider initialization failed")
        resolved = provider.ensure_symbol(requested_symbol)
        tick = provider.get_tick(resolved)
        return MarketProbeResult.from_tick(requested_symbol, resolved, tick)
    finally:
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
~~~

MarketProbeResult.from_tick() must require an aware UTC timestamp, finite positive bid/ask values, ask >= bid, and a non-empty resolved symbol; it must not substitute the application clock for a missing broker timestamp. A failed initialize, symbol resolution, tick validation, or shutdown is surfaced as a bounded probe failure and becomes an explicit opportunity skip.

Implement the single slot with an explicit _future guarded by a lock: if a non-done future exists, raise RuntimeError("analysis slot is busy"); submit directly to one ThreadPoolExecutor(max_workers=1) only after the check, and clear _future on completion. Accept an injectable submitter in the constructor so tests can return a manually completed Future without starting a thread. The executor itself never stores a pending application queue. LeaseHeartbeat may be a production daemon thread driven by an event, but its testable tick() must only call lease renewal, active-run heartbeat, and one runtime-alert update; it must never call probe, evaluator, LLM, or execution code.

- [ ] Step 4: Run focused tests and safety scans

Run:

~~~powershell
pytest tests/test_forex_watcher.py -q
python -m compileall -q tradingagents/forex/watcher.py
~~~

Expected: PASS; compile succeeds; no mutation method is defined in watcher.py.

- [ ] Step 5: Commit the serialization components

~~~powershell
git add tradingagents/forex/watcher.py tests/test_forex_watcher.py
git commit -m "feat: serialize watcher mt5 access"
~~~

---

### Task 5: Implement normal coordinator scheduling, claims, skips, cooldown, and run finalization

**Files:**
- Modify: tradingagents/forex/watcher.py
- Modify: tradingagents/forex/watch_store.py
- Test: tests/test_forex_watcher.py
- Test: tests/test_forex_watch_store.py

**Interfaces:**
- WatcherCoordinator(config, store, clock, schedule, probe, runner, evaluator, executor, gate, heartbeat, process_inspector=None, events=None) wires all boundaries; no constructor creates a provider or LLM.
- WatcherCoordinator.start() -> LeaseResult, run_once(now: datetime | None = None) -> WatchCycleResult, run_forever() -> None, request_shutdown() -> None, and shutdown() -> None implement the lifecycle.
- WatchCycleResult exposes lifecycle status, observed keys, skipped keys/reasons, active run ID, evaluation status, and safe error code; it never contains report/prose content.
- WatcherStore.observe_opportunity(), claim_opportunity(), record_skip(), and finalize_run() are atomic and idempotent for the canonical key.
- EventSink.emit(event: str, payload: Mapping[str, Any]) records only the allow-listed lifecycle event names and safe IDs/statuses/counts/timings; the coordinator emits no prompts, reports, completions, or reasoning text.

The public cycle value used by the CLI and tests is:

~~~python
@dataclass(frozen=True, slots=True)
class WatchCycleResult:
    lifecycle_status: str
    observed_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    skip_reasons: tuple[str, ...]
    active_run_id: str | None
    evaluation_status: str | None
    evaluation_due_pending: bool
    recovery_gap_count: int
    error_code: str | None
~~~

- [ ] Step 1: Write failing normal-flow tests

~~~python
def test_normal_poll_claims_one_current_opportunity_and_persists_decision(tmp_path):
    harness = WatcherHarness(tmp_path, runner_result=_complete_run_result())
    harness.start()

    cycle = harness.poll(_utc("2026-09-09T12:15:31Z"))
    assert cycle.lifecycle_status == "ANALYZING"
    assert harness.runner.calls == ["EURUSD"]
    assert harness.store.list_opportunities()[0].status == "RUNNING"

    harness.complete_runner()
    finished = harness.poll(_utc("2026-09-09T12:15:32Z"))

    assert finished.active_run_id is None
    assert harness.store.list_opportunities()[0].status == "DECISION_SAVED"
    assert harness.store.list_runs()[0].run_status in {"SUCCEEDED", "SUCCEEDED_SLOW"}
    assert harness.store.decision_for_run("run-1").executed is False


def test_new_bar_during_analysis_is_terminally_skipped_and_not_queued(tmp_path):
    harness = WatcherHarness(tmp_path, runner_result=_deferred_run_result())
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))

    cycle = harness.poll(_utc("2026-09-09T12:30:31Z"))

    assert "ANALYSIS_ALREADY_RUNNING" in cycle.skip_reasons
    assert harness.runner.calls == ["EURUSD"]
    assert harness.executor.pending_count == 0
    assert all(row.status == "SKIPPED" for row in harness.store.list_skipped())


def test_duplicate_polls_do_not_create_second_decision(tmp_path):
    harness = WatcherHarness(tmp_path, runner_result=_complete_run_result())
    harness.start()
    now = _utc("2026-09-09T12:15:31Z")
    harness.poll(now)
    harness.complete_runner()
    harness.poll(now + timedelta(seconds=1))
    harness.poll(now + timedelta(seconds=2))

    assert harness.runner.calls == ["EURUSD"]
    assert len(harness.store.list_opportunities()) == 1


def test_market_probe_failure_skips_without_graph_or_llm(tmp_path):
    harness = WatcherHarness(tmp_path, probe_error=RuntimeError("MT5 unavailable"))
    harness.start()

    cycle = harness.poll(_utc("2026-09-09T12:15:31Z"))

    assert cycle.skip_reasons == ("MT5_UNAVAILABLE",)
    assert harness.runner.calls == []
    assert harness.provider_factory.created == 1


def test_lifecycle_events_are_allowlisted_and_redacted(tmp_path):
    harness = WatcherHarness(tmp_path, runner_result=_complete_run_result())
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))

    assert {event["event"] for event in harness.events} <= {
        "WATCHER_STARTED", "LEASE_RECOVERED", "OPPORTUNITY_OBSERVED",
        "OPPORTUNITY_SKIPPED", "ANALYSIS_STARTED", "ANALYSIS_TIMEOUT_OBSERVED",
        "ANALYSIS_FINISHED", "ANALYSIS_FAILED", "EVALUATION_FINISHED",
        "EVALUATION_FAILED", "CIRCUIT_OPENED", "SHUTDOWN_REQUESTED",
        "WATCHER_STOPPED",
    }
    assert all("prompt" not in repr(event).lower() for event in harness.events)
    assert all("completion" not in repr(event).lower() for event in harness.events)


def test_cooldown_selects_only_newest_current_candidate(tmp_path):
    harness = WatcherHarness(tmp_path, symbols=("EURUSD", "USDJPY"))
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))
    harness.complete_runner()
    harness.poll(_utc("2026-09-09T12:16:32Z"))

    assert harness.runner.calls == ["EURUSD"]
    assert "COOLDOWN" in harness.last_cycle.skip_reasons


def test_restart_catchup_is_bounded_and_never_queues_old_buckets(tmp_path):
    harness = WatcherHarness(tmp_path)
    harness.start()

    cycle = harness.poll(_utc("2026-09-09T15:00:31Z"))

    assert len(harness.store.list_opportunities()) <= harness.config.max_recovery_buckets + 1
    assert harness.executor.pending_count == 0
    assert cycle.recovery_gap_count >= 0
~~~

- [ ] Step 2: Run focused coordinator tests to verify they fail

Run:

~~~powershell
pytest tests/test_forex_watcher.py -k "normal_poll or analysis or duplicate or probe_failure or cooldown" -q
~~~

Expected: FAIL because the coordinator and store claim/finalization methods are not implemented.

- [ ] Step 3: Implement the normal coordinator path

Implement run_once() in this order when no active worker exists: renew lease; materialize completed buckets, retaining at most config.max_recovery_buckets and recording one sanitized gap summary for older downtime; insert/deduplicate each opportunity; finalize a completed future if present; run due evaluation only when no future is active; apply circuit/cooldown; choose the newest eligible current row; acquire the gate for probe; probe and release its provider; atomically claim the row; submit exactly one worker; set lifecycle ANALYZING. A gap is metadata, never a queue or an instruction to replay old buckets.

The worker callable must be equivalent to:

~~~python
def _run_claimed(self, run):
    with self.gate.acquire("runner"):
        return self.runner.run(
            symbol=run.requested_symbol,
            analysis_date=run.anchor_timestamp.date(),
            analysis_profile=self.config.analysis_profile,
            analysts=self.config.analysts,
            terminal_path=self.config.terminal_path,
            db_path=self.config.db_path,
            source_run_id=run.source_run_id,
            callbacks=self.callbacks,
        )
~~~

On completion, verify the returned decision ID, source_run_id, persisted row, and executed is False; map metrics/context/normalization/temporal fields into RunEvidence; mark DECISION_SAVED, SUCCEEDED, or SUCCEEDED_SLOW; set cooldown from application completion; and release the slot only after the future is finalized. A provider/probe/runner/DB error becomes a sanitized FAILED/skip reason and never retries the old key. The coordinator must not read report text or parse a recommendation.

- [ ] Step 4: Run focused tests and verify they pass

Run:

~~~powershell
pytest tests/test_forex_watcher.py tests/test_forex_watch_store.py -q
~~~

Expected: PASS for one-slot claims, deduplication, explicit skips, provider shutdown, cooldown, and executed=False verification.

- [ ] Step 5: Commit normal scheduling

~~~powershell
git add tradingagents/forex/watcher.py tradingagents/forex/watch_store.py tests/test_forex_watcher.py tests/test_forex_watch_store.py
git commit -m "feat: add phase 6 coordinator scheduling"
~~~

---

### Task 6: Add deferred evaluation ordering, restart recovery, circuit breakers, and graceful shutdown

**Files:**
- Modify: tradingagents/forex/watcher.py
- Modify: tradingagents/forex/watch_store.py
- Test: tests/test_forex_watcher.py
- Test: tests/test_forex_watch_store.py

**Interfaces:**
- OutcomeCoordinator.evaluate_pending(now: datetime) -> EvaluationRunEvidence calls only the injected ShadowOutcomeEvaluator.evaluate_pending(now=now, terminal_path=terminal_path) and reports zero LLM calls.
- WatcherCoordinator exposes evaluation_due_pending in WatchCycleResult and summary(); the flag is durable in forex_watcher_state.
- CircuitBreakers increments only the configured failure class, opens DEGRADED after its threshold, and allows zero-LLM evaluation when no runner is active.

Define the result value used by that boundary before wiring the coordinator:

~~~python
@dataclass(frozen=True, slots=True)
class EvaluationRunEvidence:
    status: str
    errors: tuple[str, ...]
    metrics: Mapping[str, Any]
~~~

- [ ] Step 1: Write failing tests for serialization, deferred evaluation, leases, and circuits

~~~python
def test_due_evaluation_during_analysis_is_deferred_then_runs_before_next_probe(tmp_path):
    harness = WatcherHarness(tmp_path, runner_result=_deferred_run_result())
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))
    harness.clock.set(_utc("2026-09-09T12:16:32Z"))

    active_cycle = harness.poll()

    assert active_cycle.evaluation_due_pending is True
    assert harness.evaluator.calls == []
    assert harness.probe.calls == ["EURUSD"]
    assert harness.provider_factory.created_while_active == 0

    harness.complete_runner()
    harness.clock.set(_utc("2026-09-09T12:30:31Z"))
    finished = harness.poll()

    assert harness.sequence.index("runner_shutdown") < harness.sequence.index("evaluation")
    assert harness.sequence.index("evaluation") < harness.sequence.index("next_probe")
    assert finished.evaluation_due_pending is False
    assert harness.evaluator.calls == ["evaluate_pending"]


def test_no_probe_or_evaluator_or_provider_creation_while_analyzing(tmp_path):
    harness = WatcherHarness(tmp_path, runner_result=_deferred_run_result())
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))
    harness.poll(_utc("2026-09-09T12:30:31Z"))

    assert harness.probe.calls == ["EURUSD"]
    assert harness.evaluator.calls == []
    assert harness.provider_factory.created_while_active == 0
    assert harness.gate.overlaps == []


def test_nonexpired_lease_returns_already_running_without_mt5(tmp_path):
    first = WatcherHarness(tmp_path)
    first.start()
    second = WatcherHarness(tmp_path)

    result = second.start()

    assert result.status.name == "WATCHER_ALREADY_RUNNING"
    assert second.provider_factory.created == 0
    assert second.process_inspector.calls == []


def test_expired_exact_live_process_requires_operator_review(tmp_path):
    first = WatcherHarness(tmp_path)
    first.start()
    first.store.force_expiry(_utc("2026-09-09T12:20:00Z"))
    second = WatcherHarness(tmp_path, process_alive=True)

    result = second.start(now=_utc("2026-09-09T12:20:01Z"))

    assert result.status.name == "WATCHER_OPERATOR_REVIEW_REQUIRED"
    assert second.provider_factory.created == 0


def test_expired_dead_owner_reconciles_without_resubmitting_old_bucket(tmp_path):
    first = WatcherHarness(tmp_path, runner_result=_deferred_run_result())
    first.start()
    first.poll(_utc("2026-09-09T12:15:31Z"))
    first.crash_without_finalizing()
    second = WatcherHarness(tmp_path, process_alive=False)

    result = second.start(now=_utc("2026-09-09T14:00:00Z"))

    assert result.status.name == "ACQUIRED"
    assert second.store.list_runs()[0].run_status == "ABANDONED"
    assert second.runner.calls == []


def test_soft_timeout_marks_slow_without_killing_runner(tmp_path):
    harness = WatcherHarness(tmp_path, runner_result=_deferred_run_result())
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))
    harness.heartbeat(monotonic=7201.0)

    assert harness.store.list_runs()[0].runtime_alert_at is not None
    assert harness.executor.active is True
    harness.complete_runner()
    harness.poll(_utc("2026-09-09T12:15:32Z"))
    assert harness.store.list_runs()[0].run_status == "SUCCEEDED_SLOW"
~~~

- [ ] Step 2: Run focused tests to verify they fail

Run:

~~~powershell
pytest tests/test_forex_watcher.py -k "evaluation or analyzing or lease or timeout" -q
~~~

Expected: FAIL because the coordinator would probe/evaluate without a durable deferred flag and has no recovery/circuit ordering.

- [ ] Step 3: Implement the active-run branch and immediate post-run evaluator

When a future is not done, run_once() must do only heartbeat renewal, bucket materialization, SKIPPED / ANALYSIS_ALREADY_RUNNING writes, and mark_evaluation_due() when the interval elapsed. It must not call ReadOnlyMarketProbe, ShadowOutcomeEvaluator, or any provider factory. When a future becomes done, finalize it and shut down the runner-owned provider before acquiring the gate for evaluation; invoke the evaluator once, clear the flag only after return/error persistence, and only then apply cooldown and consider the next probe.

Use this guard structure so the ordering is explicit:

~~~python
if self._analysis_future is not None and not self._analysis_future.done():
    self._observe_and_skip_busy_buckets(now)
    if self._evaluation_is_due(now):
        self.store.mark_evaluation_due(self.owner_token, now)
    return self._cycle("ANALYZING")

if self._analysis_future is not None:
    self._finalize_completed_future(now)

if self._evaluation_is_due(now) and self.config.evaluation_enabled:
    with self.gate.acquire("evaluator"):
        evaluation = self.outcomes.evaluate_pending(now=now)
    self.store.clear_evaluation_due(self.owner_token, now, evaluation.status)

return self._maybe_start_current_analysis(now)
~~~

The single gate wraps probe, runner, and evaluator calls. The coordinator treats any gate contention as a failed closed cycle, never waits behind an active runner, and never creates a compensating provider/session.

- [ ] Step 4: Implement lease startup/recovery, circuits, and shutdown

start() must return WATCHER_ALREADY_RUNNING for any non-expired owner without probing or reconciling; return WATCHER_OPERATOR_REVIEW_REQUIRED when an expired same-host exact process is proven alive; otherwise reconcile stale runs under the newly acquired token before IDLE. Counters and circuit reasons are updated in the singleton state. SIGINT/SIGTERM sets a stop event, stops new claims, waits for the current future by default, performs one serialized best-effort evaluation, releases the lease, and persists STOPPED; forced termination is recovered only on the next expired-lease startup.

- [ ] Step 5: Run all watcher tests and commit

Run:

~~~powershell
pytest tests/test_forex_watcher.py tests/test_forex_watch_store.py -q
~~~

Expected: PASS, including strict lease behavior, no MT5 overlap, deferred evaluation ordering, recovery, circuit, and slow-run tests.

~~~powershell
git add tradingagents/forex/watcher.py tradingagents/forex/watch_store.py tests/test_forex_watcher.py tests/test_forex_watch_store.py
git commit -m "feat: add watcher recovery and serialized evaluation"
~~~

---

### Task 7: Add the separate forex-watch CLI and stable status output

**Files:**
- Create: cli/forex_watch.py
- Modify: pyproject.toml
- Test: tests/test_forex_watch_cli.py

**Interfaces:**
- build_parser() -> argparse.ArgumentParser exposes subcommands run, once, status, and evaluate only.
- main(argv: Sequence[str] | None = None, *, coordinator_factory=None, store_factory=None) -> int prints the safety banner before external connections and maps lease/config/provider errors to nonzero exits.
- status reads SQLite only by default; --probe is an explicit optional read-only probe.
- run and once accept only non-secret schedule/runtime options; provider/model selection remains in the existing DEFAULT_CONFIG/TRADINGAGENTS_* environment system.

- [ ] Step 1: Write failing CLI tests

~~~python
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from cli.forex_watch import build_parser, main
from tradingagents.forex.watch_store import LeaseStatus, WatcherStore


def test_forex_watch_defaults_and_subcommands():
    parser = build_parser()
    args = parser.parse_args(["once"])
    assert args.command == "once"
    assert args.symbols == "EURUSD"
    assert args.analysis_profile == "INTRADAY"
    assert args.analysts == "market,news"
    assert args.schedule_timeframe == "M15"


def test_forex_watch_has_no_execution_or_credential_options():
    parser = build_parser()
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }
    assert "--order-send" not in option_strings
    assert "--api-key" not in option_strings
    assert "--llm-provider" not in option_strings
    assert "--deep-model" not in option_strings


def test_once_prints_shadow_collector_banner(capsys, monkeypatch, tmp_path):
    class FakeCoordinator:
        def __init__(self, **kwargs):
            pass

        def start(self):
            return SimpleNamespace(status=LeaseStatus.ACQUIRED)

        def run_once(self, now=None):
            return SimpleNamespace(
                lifecycle_status="IDLE", active_run_id=None, error_code=None
            )

        def shutdown(self):
            pass

    monkeypatch.setattr("cli.forex_watch.WatcherCoordinator", FakeCoordinator)
    assert main(["once", "--db-path", str(tmp_path / "watch.db")]) == 0
    output = capsys.readouterr().out
    assert "MT5 FOREX — SHADOW COLLECTOR" in output
    assert "NO ORDER WILL BE SENT" in output


def test_status_does_not_construct_mt5_or_llm(capsys, tmp_path):
    assert main(["status", "--db-path", str(tmp_path / "watch.db")]) == 0
    output = capsys.readouterr().out
    assert "WATCHER STATUS" in output
    assert "LLM CALLS" in output


def test_status_probe_refuses_while_watcher_lease_is_valid(tmp_path, capsys):
    store = WatcherStore(tmp_path / "watch.db")
    store.acquire_lease(_owner(), datetime.now(timezone.utc))

    result = main(["status", "--probe", "--db-path", str(tmp_path / "watch.db")])

    assert result == 1
    assert "WATCHER_ALREADY_RUNNING" in capsys.readouterr().err


def test_pyproject_registers_only_new_forex_watch_script():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'forex-watch = "cli.forex_watch:main"' in text
    assert 'tradingagents = "cli.main:app"' in text
~~~

- [ ] Step 2: Run focused CLI tests to verify they fail

Run:

~~~powershell
pytest tests/test_forex_watch_cli.py -q
~~~

Expected: FAIL because the forex-watch module and console script do not exist.

- [ ] Step 3: Implement parser, banners, and dispatch

Use an argparse subparser tree with these non-secret options: --db-path, --terminal-path, --symbols, --analysis-profile, --analysts, --schedule-timeframe, --poll-interval-seconds, --cooldown-seconds, --analysis-timeout-seconds, --evaluation-interval-seconds, --no-evaluate, and --json-logs for run; once shares the schedule options; status accepts --db-path, --json, and --probe; evaluate accepts --db-path and --terminal-path only. Do not add provider/model/credential/order flags.

Print before constructing the coordinator or probe:

~~~text
MT5 FOREX — SHADOW COLLECTOR
NO ORDER WILL BE SENT
~~~

once starts/acquires the lease, performs one current scheduling cycle, waits for its one active worker to finish, performs the immediate due evaluator if configured, prints the safe summary, and releases the lease. run loops using the configured poll interval and handles shutdown. status uses WatcherStore.summary() only and emits stable JSON keys when --json is present. status --probe first checks the active lease and returns WATCHER_ALREADY_RUNNING without opening MT5 when the watcher owns the database. evaluate delegates to the existing evaluator through the same lease guard in Task 8. Production construction merges the parsed watcher settings with DEFAULT_CONFIG and passes that mapping to ForexShadowRunner; it never asks for a provider/model secret on the command line.

- [ ] Step 4: Register and test the console script

Add exactly one line under [project.scripts]:

~~~toml
forex-watch = "cli.forex_watch:main"
~~~

Run:

~~~powershell
pytest tests/test_forex_watch_cli.py -q
~~~

Expected: PASS; existing stock entry-point assertions remain unchanged.

- [ ] Step 5: Commit the CLI

~~~powershell
git add cli/forex_watch.py pyproject.toml tests/test_forex_watch_cli.py
git commit -m "feat: add forex watch shadow cli"
~~~

---

### Task 8: Fence the existing evaluator command and document operation/lease semantics

**Files:**
- Modify: cli/forex_evaluate.py
- Modify: docs/forex-shadow.md
- Modify: tradingagents/forex/__init__.py
- Test: tests/test_forex_evaluate_cli.py
- Test: tests/test_forex_watch_cli.py

**Interfaces:**
- forex-evaluate must perform a read-only WatcherStore active-lease check before constructing ShadowOutcomeEvaluator or an MT5 provider.
- Public exports may include WatcherConfig, WatcherCoordinator, WatcherStore, ReadOnlyMarketProbe, ScheduledOpportunity, WatchCycleResult, LeaseResult, RunEvidence, and OutcomeCoordinator; no stock import or execution API is exported.

- [ ] Step 1: Write failing active-lease and documentation-contract tests

~~~python
from datetime import datetime, timezone


def test_forex_evaluate_refuses_nonexpired_watcher_lease(capsys, tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    now = datetime.now(timezone.utc)
    store.acquire_lease(_owner(), now)

    class MustNotConstruct:
        def __init__(self, **kwargs):
            raise AssertionError("evaluator must not be constructed")

    result = main(
        ["--pending", "--db-path", str(tmp_path / "watch.db")],
        evaluator_factory=MustNotConstruct,
    )

    assert result == 1
    assert "WATCHER_ALREADY_RUNNING" in capsys.readouterr().err


def test_forex_shadow_docs_describe_serialized_collector_and_no_execution():
    text = Path("docs/forex-shadow.md").read_text(encoding="utf-8")
    assert "forex-watch" in text
    assert "WATCHER_ALREADY_RUNNING" in text
    assert "NO ORDER WILL BE SENT" in text
    assert "evaluation_due_pending" in text
~~~

- [ ] Step 2: Run focused tests to verify they fail

Run:

~~~powershell
pytest tests/test_forex_evaluate_cli.py tests/test_forex_watch_cli.py -k "lease or docs" -q
~~~

Expected: FAIL because the existing evaluator command does not know the watcher lease and the documentation has no collector contract.

- [ ] Step 3: Add the lease guard without changing Phase 5 evaluation

Before building EvaluationConfig, ShadowOutcomeEvaluator, or any provider, instantiate WatcherStore(args.db_path), initialize only its idempotent schema, and call active_lease(now=utc_now()). If lease_expires_at > now, print FOREX EVALUATION ERROR: WATCHER_ALREADY_RUNNING and return 1. If no lease or an expired lease exists, continue through the existing evaluator code unchanged. Keep the existing zero-LLM and read-only checks.

- [ ] Step 4: Update docs and exports

Add sections to docs/forex-shadow.md covering:

- forex-watch run, once, status, and evaluate examples;
- the MT5 FOREX — SHADOW COLLECTOR / NO ORDER WILL BE SENT banner;
- M15 completed-bar anchors, 30-second settle, one active runner, no queue, and explicit busy skips;
- strict non-expired lease ownership, expired exact-process operator review, and stale-run reconciliation;
- no probe/evaluator/provider creation during ANALYZING, deferred evaluation ordering, and at-most-one-MT5-operation invariant;
- status --probe and forex-evaluate lease guards that refuse a valid active watcher without opening MT5;
- Phase 5 basis/status meanings, safe provenance, no training labels, and no execution.

Re-export only the new watcher contracts needed by callers; leave existing forex and stock exports intact.

- [ ] Step 5: Run focused tests and commit

Run:

~~~powershell
pytest tests/test_forex_evaluate_cli.py tests/test_forex_watch_cli.py -q
~~~

Expected: PASS, with the evaluator refusing an active lease and continuing to work after the watcher is stopped.

~~~powershell
git add cli/forex_evaluate.py docs/forex-shadow.md tradingagents/forex/__init__.py tests/test_forex_evaluate_cli.py tests/test_forex_watch_cli.py
git commit -m "docs: fence evaluator beside forex watcher"
~~~

---

### Task 9: Add deterministic end-to-end safety, provenance, and stock-regression coverage

**Files:**
- Create: tests/test_forex_watch_integration.py
- Modify: tests/test_forex_watcher.py if harness helpers are shared
- Modify: tests/test_forex_shadow_integration.py only if a reusable forbidden-method tuple is needed; preserve existing assertions

**Interfaces:**
- The mocked integration harness must use fake clock, fake provider/probe, fake runner result, fake evaluator, fake gate, and a temporary SQLite file.
- The optional real test is marked integration, gated by RUN_MT5_INTEGRATION=1, and must compare positions/orders before and after without invoking a graph or LLM.

- [ ] Step 1: Write failing integration/safety tests

~~~python
FORBIDDEN = {
    "order_send", "buy", "sell", "open_position", "close_position",
    "modify_position", "modify_order", "place_order", "cancel_order",
}


def test_mocked_collector_trace_has_one_mt5_operation_at_a_time(tmp_path):
    harness = WatcherHarness(tmp_path, runner_result=_complete_run_result())
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))
    harness.complete_runner()
    harness.poll(_utc("2026-09-09T12:15:32Z"))

    assert harness.gate.events == [
        "probe:start", "probe:end", "runner:start", "runner:end"
    ]
    assert harness.evaluator.calls == [] or harness.sequence.index("runner:end") < harness.sequence.index("evaluation")
    decision = harness.store.list_decisions()[0]
    assert decision.executed is False


def test_phase6_sources_define_no_forbidden_mutation_methods():
    root = Path(__file__).resolve().parents[1]
    for path in (
        root / "tradingagents" / "forex" / "watcher.py",
        root / "tradingagents" / "forex" / "watch_store.py",
        root / "cli" / "forex_watch.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {
            node.name.casefold()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert FORBIDDEN.isdisjoint(defined), path


def test_run_summary_keeps_temporal_bases_and_staleness_separate(tmp_path):
    harness = WatcherHarness(tmp_path, runner_result=_complete_run_result())
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))
    harness.complete_runner(completed_at=_utc("2026-09-09T13:00:00Z"))
    harness.poll(_utc("2026-09-09T13:00:01Z"))

    run = harness.store.list_runs()[0]
    assert run.analysis_snapshot_timestamp < run.decision_completed_timestamp
    assert run.decision_reference_timestamp >= run.decision_completed_timestamp
    assert run.analysis_latency_seconds > 0
    assert run.stale_by_completion is True
    safe_config = run.safe_config_json
    assert "API_KEY" not in safe_config.upper()
    assert "secret" not in safe_config.lower()


@pytest.mark.integration
def test_optional_local_mt5_probe_is_read_only():
    if os.getenv("RUN_MT5_INTEGRATION") != "1":
        pytest.skip("set RUN_MT5_INTEGRATION=1")
    pytest.importorskip("MetaTrader5")
    terminal_path = os.getenv("MT5_TERMINAL_PATH")
    requested = os.getenv("MT5_TEST_SYMBOL", "EURUSD")

    def account_state(symbol):
        account_provider = MT5Provider(terminal_path=terminal_path)
        try:
            account_provider.initialize()
            return (
                account_provider.get_positions(symbol),
                account_provider.get_orders(symbol),
            )
        finally:
            account_provider.shutdown()

    before_positions, before_orders = account_state(requested)
    result = ReadOnlyMarketProbe(
        lambda terminal_path=None: MT5Provider(terminal_path=terminal_path),
        terminal_path=terminal_path,
    ).probe(requested)
    after_positions, after_orders = account_state(result.resolved_symbol)
    assert result.timestamp.tzinfo == timezone.utc
    assert result.ask >= result.bid > 0
    assert before_positions == after_positions
    assert before_orders == after_orders
    assert all(not hasattr(MT5Provider, name) for name in FORBIDDEN)
~~~

- [ ] Step 2: Run mocked tests to verify they fail

Run:

~~~powershell
pytest tests/test_forex_watch_integration.py -q
~~~

Expected: FAIL until the final gate trace, provenance join, and safety scan are wired.

- [ ] Step 3: Implement harness assertions and safe summaries

Ensure the mocked run exposes only IDs, statuses, counts, sizes, timings, action, context/normalization status, and executed=False. Assert that the watcher’s SQLite run row links source_run_id, decision ID, resolved symbol, analysis/reference timestamps, latency, safe model/provider identifiers, runtime, LLM/tool counts, and no raw PM/debate text. Assert status summaries keep context completeness separate from terminal outcome status and never emit training_eligible=1.

- [ ] Step 4: Run mocked integration and stock regression tests

Run:

~~~powershell
pytest tests/test_forex_watch_integration.py tests/test_forex_shadow_integration.py tests/test_forex_shadow_cli.py tests/test_forex_evaluate_cli.py -q
~~~

Expected: PASS with the stock CLI entry-point string unchanged and all existing provider/runner mutation guards still green. The optional local test remains skipped unless explicitly enabled.

- [ ] Step 5: Commit safety coverage

~~~powershell
git add tests/test_forex_watch_integration.py tests/test_forex_watcher.py tests/test_forex_shadow_integration.py
git commit -m "test: verify forex watcher shadow safety"
~~~

---

### Task 10: Run the complete verification gate and prepare the Phase 6 handoff

**Files:**
- Modify: none unless a verification failure identifies a concrete defect in the preceding task; any fix must add a regression test in the owning task’s test file.

**Interfaces:**
- Verification covers every required deterministic behavior in spec section 22: normal decision, busy skip, bounded backlog, dedup, failures, MT5 unavailability, incomplete/complete decisions, evaluation maturity/recovery, clean restart, stale-run recovery, status, defaults/provenance, zero execution, stock CLI preservation, Phase 5 reuse, serialized MT5 access, and no real-time sleep.

- [ ] Step 1: Run the focused Phase 6 suite

~~~powershell
pytest tests/test_forex_watch_store.py tests/test_forex_watcher.py tests/test_forex_watch_cli.py tests/test_forex_watch_integration.py -q
~~~

Expected: all deterministic watcher tests pass; only explicitly gated MT5 tests may skip.

- [ ] Step 2: Run the full repository test suite

~~~powershell
pytest -q
~~~

Expected: existing Phase 3/4/5 and stock-mode tests remain green; no required test performs a 40-minute LLM run.

- [ ] Step 3: Run lint, compile, and diff checks

~~~powershell
ruff check tradingagents/forex/watch_store.py tradingagents/forex/watcher.py cli/forex_watch.py cli/forex_evaluate.py tests/test_forex_watch_store.py tests/test_forex_watcher.py tests/test_forex_watch_cli.py tests/test_forex_watch_integration.py
python -m compileall -q tradingagents cli
git diff --check 9e4b56c..HEAD
~~~

Expected: Ruff, compileall, and whitespace checks pass with no generated files or unrelated stock CLI changes.

- [ ] Step 4: Run the opt-in bounded MT5 read-only check when configured

~~~powershell
$env:RUN_MT5_INTEGRATION = "1"
pytest tests/test_forex_watch_integration.py -m integration -q
~~~

If the package, terminal, account, or symbol is unavailable, report the typed provider/configuration failure or skip reason; do not substitute Yahoo data, fabricate a quote/decision, or invoke execution.

- [ ] Step 5: Inspect the final scope and report evidence

~~~powershell
git status --short
git diff --stat 9e4b56c..HEAD
git log --oneline --decorate -12
~~~

The handoff must state: final commit range from corrected design baseline 9e4b56c; files changed; test/lint/compile results; default config; lease behavior; one-active-run/zero-queue behavior; serialized MT5/evaluation ordering; skipped-opportunity semantics; crash/restart result; provider/model provenance safety; stock CLI unchanged; optional MT5 verification result; and explicit executed=False/no-order guarantee. Do not claim profitability, live execution quality, training eligibility, or a completed real LLM analysis from this collector implementation.

This plan stops after Phase 6 verification. It does not authorize Phase 7, training, execution, or changes to the TradingAgents/LangGraph workflow.
