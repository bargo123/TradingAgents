from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tradingagents.forex.watcher import (
    CompletedBarSchedule,
    WatcherConfig,
    canonical_opportunity_key,
    safe_effective_config,
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
    safe = safe_effective_config(
        {"llm_provider": "openai", "deep_think_llm": "gpt-5.6", "api_key": "secret"}
    )

    assert safe["llm_provider"] == "openai"
    assert "api_key" not in safe
    assert "secret" not in repr(safe)
