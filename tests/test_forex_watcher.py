from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tradingagents.forex.watcher import (
    CompletedBarSchedule,
    LeaseHeartbeat,
    Mt5OperationBusy,
    ReadOnlyMarketProbe,
    SerializedMt5OperationGate,
    SingleSlotAnalysisExecutor,
    WatcherConfig,
    canonical_opportunity_key,
    safe_effective_config,
)

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
