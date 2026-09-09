from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import pytest

from tradingagents.forex.watcher import (
    CompletedBarSchedule,
    FrozenClock,
    LeaseHeartbeat,
    Mt5OperationBusy,
    ReadOnlyMarketProbe,
    SerializedMt5OperationGate,
    SingleSlotAnalysisExecutor,
    WatcherCoordinator,
    WatcherConfig,
    canonical_opportunity_key,
    safe_effective_config,
)
from tradingagents.forex.shadow import ShadowDecisionStore, ShadowTradeDecision
from tradingagents.forex.watch_store import WatcherStore

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


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
    safe = safe_effective_config(
        {"llm_provider": "openai", "deep_think_llm": "gpt-5.6", "api_key": "secret"}
    )

    assert safe["llm_provider"] == "openai"
    assert "api_key" not in safe
    assert "secret" not in repr(safe)


def test_operation_gate_rejects_nested_or_concurrent_operation():
    gate = SerializedMt5OperationGate()
    with gate.acquire("runner"):
        with pytest.raises(Mt5OperationBusy):
            with gate.acquire("evaluator"):
                pass


class FakeSubmitter:
    def __init__(self):
        self.futures = []

    def __call__(self, fn):
        from concurrent.futures import Future

        future = Future()
        self.futures.append((future, fn))
        return future

    def complete(self, future, value):
        future.set_result(value)


def test_single_slot_executor_has_zero_capacity_queue():
    submitter = FakeSubmitter()
    executor = SingleSlotAnalysisExecutor(submitter=submitter)
    first = executor.submit(lambda: "done")
    with pytest.raises(RuntimeError, match="analysis slot is busy"):
        executor.submit(lambda: "must-not-queue")
    submitter.complete(first, "done")
    assert first.result() == "done"
    executor.shutdown(wait=True)


class FakeProbeProvider:
    def __init__(self):
        self.initialize_calls = 0
        self.shutdown_calls = 0

    def initialize(self):
        self.initialize_calls += 1
        return True

    def ensure_symbol(self, symbol):
        return "EURUSDm"

    def get_tick(self, symbol):
        return SimpleNamespace(
            symbol=symbol,
            timestamp=NOW,
            bid=1.1,
            ask=1.1002,
        )

    def shutdown(self):
        self.shutdown_calls += 1


def test_probe_shuts_down_provider_and_never_exposes_mutation_api():
    provider = FakeProbeProvider()
    probe = ReadOnlyMarketProbe(lambda terminal_path=None: provider, terminal_path=None)

    result = probe.probe("EURUSD")

    assert result.resolved_symbol == "EURUSDm"
    assert provider.initialize_calls == 1
    assert provider.shutdown_calls == 1
    for name in ("order_send", "buy", "sell", "close_position", "modify_position"):
        assert not hasattr(probe, name)


class FakeHeartbeatStore:
    def __init__(self):
        self.renewals = 0
        self.runtime_alerts = 0

    def heartbeat(self, owner_token, now, run_id=None):
        self.renewals += 1

    def mark_runtime_alert(self, owner_token, run_id, now):
        self.runtime_alerts += 1


def test_heartbeat_only_renews_and_marks_soft_timeout():
    store = FakeHeartbeatStore()
    heartbeat = LeaseHeartbeat(
        store=store,
        owner_token="owner",
        run_id="run",
        timeout_seconds=7200,
        sequence=None,
    )
    heartbeat.tick(NOW, 0.0)
    heartbeat.tick(NOW.replace(hour=13), 7201.0)

    assert store.renewals == 2
    assert store.runtime_alerts == 1
    assert heartbeat.runner_future_still_owned is True


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _decision(source_run_id: str) -> ShadowTradeDecision:
    from datetime import date

    return ShadowTradeDecision(
        decision_id=f"decision-{source_run_id}",
        created_at=NOW,
        snapshot_timestamp=NOW,
        analysis_date=date(2026, 9, 9),
        requested_symbol="EURUSD",
        resolved_symbol="EURUSDm",
        action="BUY",
        raw_portfolio_manager_result={"rating": "Buy"},
        normalization_status="NORMALIZED",
        normalization_error=None,
        confidence=None,
        reference_bid=1.1,
        reference_ask=1.1002,
        reference_mid=1.1001,
        spread=0.0002,
        spread_points=2.0,
        analysis_timeframe="M15",
        trader_summary="trader",
        portfolio_manager_summary="pm",
        source_run_id=source_run_id,
        decision_context_status="COMPLETE",
        executed=False,
    )


class _Probe:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def probe(self, symbol):
        self.calls.append(symbol)
        if self.error:
            raise self.error
        return SimpleNamespace(
            requested_symbol=symbol,
            resolved_symbol=f"{symbol}m",
            timestamp=NOW,
            bid=1.1,
            ask=1.1002,
        )


class _Runner:
    def __init__(self, store, result=None):
        self.store = store
        self.result = result
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        result = self.result
        if callable(result):
            result = result(kwargs)
        decision = result.decision
        if decision.source_run_id != kwargs["source_run_id"]:
            decision = replace(decision, source_run_id=kwargs["source_run_id"])
            result = replace(result, decision=decision) if hasattr(result, "__dataclass_fields__") else SimpleNamespace(
                decision=decision, elapsed_seconds=result.elapsed_seconds, metrics=result.metrics
            )
        if not self.store.path.exists() or not ShadowDecisionStore(self.store.path).find_by_source_run_id(
            decision.source_run_id
        ):
            ShadowDecisionStore(self.store.path).record(decision)
        return result


class _Evaluator:
    def __init__(self):
        self.calls = []

    def evaluate_pending(self, **kwargs):
        self.calls.append("evaluate_pending")
        return SimpleNamespace(status="OK", errors=(), metrics={"llm_calls": 0})


class _Harness:
    def __init__(self, tmp_path: Path, runner_result=None, probe_error=None, symbols=("EURUSD",)):
        from concurrent.futures import Future

        self.clock = FrozenClock(NOW)
        self.config = WatcherConfig(
            db_path=tmp_path / "watch.db",
            symbols=symbols,
            evaluation_interval_seconds=0,
        )
        self.store = WatcherStore(self.config.db_path, lease_ttl_seconds=self.config.lease_ttl_seconds)
        self.probe = _Probe(probe_error)
        self.provider_factory = SimpleNamespace(created=0, created_while_active=0)
        self.deferred = runner_result
        self.runner = _Runner(self.store, runner_result or _complete_run_result())
        self.evaluator = _Evaluator()
        self.submitter_futures = []

        def submit(fn):
            future = Future()
            # Invoke the fake worker immediately so calls/arguments are
            # observable, but keep the Future pending until the test releases
            # it.  No wall-clock thread or model work is involved.
            result = fn()
            self.submitter_futures.append((future, result))
            return future

        self.executor = SingleSlotAnalysisExecutor(submitter=submit)
        self.gate = SerializedMt5OperationGate()
        self.events = []
        self.sequence = []
        schedule = CompletedBarSchedule(
            timeframe="M15",
            settle_seconds=30,
            config_fingerprint=self.config.safe_fingerprint,
            requested_symbols=symbols,
        )
        self.coordinator = WatcherCoordinator(
            config=self.config,
            store=self.store,
            clock=self.clock,
            schedule=schedule,
            probe=self.probe,
            runner=self.runner,
            evaluator=self.evaluator,
            executor=self.executor,
            gate=self.gate,
            events=self.events,
        )

    def start(self, now=None):
        return self.coordinator.start(now=now or NOW)

    def poll(self, now=None):
        return self.coordinator.run_once(now=now)

    def complete_runner(self, completed_at=None):
        if not self.submitter_futures:
            raise AssertionError("runner was not submitted")
        future, result = self.submitter_futures[-1]
        future.set_result(result)

    def shutdown(self):
        self.coordinator.shutdown()


def _complete_run_result():
    return SimpleNamespace(
        decision=_decision("run-1"),
        elapsed_seconds=1.0,
        metrics={"llm_calls": 1, "tool_calls": 1},
    )


def test_normal_poll_claims_one_current_opportunity_and_persists_decision(tmp_path):
    harness = _Harness(tmp_path, runner_result=_complete_run_result())
    harness.start()

    cycle = harness.poll(_utc("2026-09-09T12:15:31Z"))
    assert cycle.lifecycle_status == "ANALYZING"
    assert [call["symbol"] for call in harness.runner.calls] == ["EURUSD"]
    assert harness.store.list_opportunities()[0].status == "RUNNING"

    harness.complete_runner()
    finished = harness.poll(_utc("2026-09-09T12:15:32Z"))

    assert finished.active_run_id is None
    assert harness.store.list_opportunities()[0].status == "DECISION_SAVED"
    assert harness.store.list_runs()[0].run_status in {"SUCCEEDED", "SUCCEEDED_SLOW"}
    assert harness.store.decision_for_run(harness.store.list_runs()[0].run_id).executed is False


def test_new_bar_during_analysis_is_terminally_skipped_and_not_queued(tmp_path):
    harness = _Harness(tmp_path, runner_result=_complete_run_result())
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))

    cycle = harness.poll(_utc("2026-09-09T12:30:31Z"))

    assert "ANALYSIS_ALREADY_RUNNING" in cycle.skip_reasons
    assert len(harness.runner.calls) == 1
    assert harness.executor.pending_count == 1
    assert all(row.status == "SKIPPED" for row in harness.store.list_skipped())


def test_duplicate_polls_do_not_create_second_decision(tmp_path):
    harness = _Harness(tmp_path, runner_result=_complete_run_result())
    harness.start()
    now = _utc("2026-09-09T12:15:31Z")
    harness.poll(now)
    harness.complete_runner()
    harness.poll(now + timedelta(seconds=1))
    harness.poll(now + timedelta(seconds=2))

    assert len(harness.runner.calls) == 1
    assert len(harness.store.list_opportunities()) == 1


def test_market_probe_failure_skips_without_graph_or_llm(tmp_path):
    harness = _Harness(tmp_path, probe_error=RuntimeError("MT5 unavailable"))
    harness.start()

    cycle = harness.poll(_utc("2026-09-09T12:15:31Z"))

    assert cycle.skip_reasons == ("MT5_UNAVAILABLE",)
    assert harness.runner.calls == []
    assert harness.probe.calls == ["EURUSD"]


def test_worker_forwards_existing_runner_iso_analysis_date_contract(tmp_path):
    harness = _Harness(tmp_path, runner_result=_complete_run_result())
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))

    assert harness.runner.calls[0]["analysis_date"] == "2026-09-09"
    assert isinstance(harness.runner.calls[0]["analysis_date"], str)


def test_lifecycle_events_are_allowlisted_and_redacted(tmp_path):
    harness = _Harness(tmp_path, runner_result=_complete_run_result())
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
    harness = _Harness(tmp_path, symbols=("EURUSD", "USDJPY"))
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))
    harness.complete_runner()
    cycle = harness.poll(_utc("2026-09-09T12:16:32Z"))

    assert [call["symbol"] for call in harness.runner.calls] == ["EURUSD"]
    assert "COOLDOWN" in cycle.skip_reasons


def test_restart_catchup_is_bounded_and_never_queues_old_buckets(tmp_path):
    harness = _Harness(tmp_path)
    harness.start()

    cycle = harness.poll(_utc("2026-09-09T15:00:31Z"))

    assert len(harness.store.list_opportunities()) <= harness.config.max_recovery_buckets + 1
    assert harness.executor.pending_count == 1
    assert cycle.recovery_gap_count >= 0
