from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tradingagents.forex.hft.market_calendar import BrokerSessionCalendar, CalendarStatus
from tradingagents.forex.hft.market_lifecycle import (
    MarketLifecycleController,
    MarketReason,
    MarketState,
)


def _calendar_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "calendar_version": "fixture-2026a",
        "broker_server": "Example-Demo",
        "symbol": "EURUSD.a",
        "timezone": "Europe/London",
        "valid_from": "2026-01-01",
        "valid_through": "2026-12-31",
        "weekly_sessions": {
            "mon": [{"open": "09:00", "close": "17:00"}],
            "tue": [{"open": "09:00", "close": "17:00"}],
            "wed": [{"open": "09:00", "close": "17:00"}],
            "thu": [{"open": "09:00", "close": "17:00"}],
            "fri": [{"open": "09:00", "close": "17:00"}],
        },
        "holidays": ["2026-05-25"],
        "date_overrides": {"2026-12-24": [{"open": "09:00", "close": "12:00"}]},
    }


def _load(tmp_path, data: dict[str, object] | None = None) -> BrokerSessionCalendar:
    path = tmp_path / "calendar.json"
    path.write_text(json.dumps(_calendar_data() if data is None else data), encoding="utf-8")
    return BrokerSessionCalendar.from_json(path)


def test_calendar_matches_exact_broker_and_resolved_symbol(tmp_path) -> None:
    calendar = _load(tmp_path)
    at = datetime(2026, 6, 1, 10, tzinfo=UTC)

    assert calendar.status_at(at, broker_server="Example-Demo", symbol="EURUSD.a") is CalendarStatus.OPEN
    assert calendar.status_at(at, broker_server="example-demo", symbol="EURUSD.a") is CalendarStatus.UNKNOWN
    assert calendar.status_at(at, broker_server="Example-Demo", symbol="EURUSD") is CalendarStatus.UNKNOWN


def test_calendar_requires_valid_coverage_and_aware_utc_time(tmp_path) -> None:
    calendar = _load(tmp_path)

    assert calendar.status_at(
        datetime(2027, 1, 1, tzinfo=UTC), broker_server="Example-Demo", symbol="EURUSD.a"
    ) is CalendarStatus.UNKNOWN
    assert calendar.status_at(
        datetime(2026, 6, 1, 10), broker_server="Example-Demo", symbol="EURUSD.a"
    ) is CalendarStatus.UNKNOWN


def test_weekly_sessions_and_closed_interval_boundaries(tmp_path) -> None:
    calendar = _load(tmp_path)

    assert calendar.status_at(
        datetime(2026, 6, 1, 9, tzinfo=UTC), broker_server="Example-Demo", symbol="EURUSD.a"
    ) is CalendarStatus.OPEN
    assert calendar.status_at(
        datetime(2026, 6, 1, 17, tzinfo=UTC), broker_server="Example-Demo", symbol="EURUSD.a"
    ) is CalendarStatus.CLOSED
    assert calendar.status_at(
        datetime(2026, 6, 6, 10, tzinfo=UTC), broker_server="Example-Demo", symbol="EURUSD.a"
    ) is CalendarStatus.CLOSED


def test_full_date_holiday_and_early_close_override(tmp_path) -> None:
    calendar = _load(tmp_path)

    assert calendar.status_at(
        datetime(2026, 5, 25, 10, tzinfo=UTC), broker_server="Example-Demo", symbol="EURUSD.a"
    ) is CalendarStatus.CLOSED
    assert calendar.status_at(
        datetime(2026, 12, 24, 11, tzinfo=UTC), broker_server="Example-Demo", symbol="EURUSD.a"
    ) is CalendarStatus.OPEN
    assert calendar.status_at(
        datetime(2026, 12, 24, 13, tzinfo=UTC), broker_server="Example-Demo", symbol="EURUSD.a"
    ) is CalendarStatus.CLOSED


def test_dst_utc_instants_map_to_the_correct_local_session(tmp_path) -> None:
    data = _calendar_data()
    data["weekly_sessions"] = {"sun": [{"open": "01:00", "close": "02:00"}]}
    calendar = _load(tmp_path, data)

    # Europe/London repeats 01:00 local during the autumn transition. The two
    # UTC instants are distinct and are evaluated without guessing a fold.
    first = calendar.status_at(
        datetime(2026, 10, 25, 0, 30, tzinfo=UTC), broker_server="Example-Demo", symbol="EURUSD.a"
    )
    second = calendar.status_at(
        datetime(2026, 10, 25, 1, 30, tzinfo=UTC), broker_server="Example-Demo", symbol="EURUSD.a"
    )
    assert first is CalendarStatus.OPEN
    assert second is CalendarStatus.OPEN


def test_dst_local_ambiguity_is_not_accepted_as_calendar_input(tmp_path) -> None:
    calendar = _load(tmp_path)
    ambiguous = datetime(2026, 10, 25, 1, 30, tzinfo=ZoneInfo("Europe/London"), fold=0)

    assert calendar.status_at(
        ambiguous, broker_server="Example-Demo", symbol="EURUSD.a"
    ) is CalendarStatus.UNKNOWN


def test_versioned_metaquotes_demo_eurusd_calendar_matches_observed_week_and_dst() -> None:
    path = Path(__file__).parents[1] / "config" / "broker_sessions" / "metaquotes-demo-eurusd.json"
    calendar = BrokerSessionCalendar.from_json(path)

    assert calendar.calendar_version == "metaquotes-demo-eurusd-new-york-fx-week-v1-observed-2026-10-04"
    assert calendar.timezone.key == "America/New_York"
    assert calendar.status_at(
        datetime(2026, 10, 4, 21, 0, tzinfo=UTC),
        broker_server="MetaQuotes-Demo",
        symbol="EURUSD",
    ) is CalendarStatus.OPEN
    assert calendar.status_at(
        datetime(2026, 10, 2, 21, 0, tzinfo=UTC),
        broker_server="MetaQuotes-Demo",
        symbol="EURUSD",
    ) is CalendarStatus.CLOSED
    assert calendar.status_at(
        datetime(2026, 11, 1, 22, 0, tzinfo=UTC),
        broker_server="MetaQuotes-Demo",
        symbol="EURUSD",
    ) is CalendarStatus.OPEN
    assert calendar.status_at(
        datetime(2026, 10, 3, 15, 0, tzinfo=UTC),
        broker_server="MetaQuotes-Demo",
        symbol="EURUSD",
    ) is CalendarStatus.CLOSED


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update(timezone="Not/AZone"),
        lambda data: data["weekly_sessions"].update(mon=[{"open": "17:00", "close": "09:00"}]),
        lambda data: data["weekly_sessions"].update(mon=[{"open": "09:00", "close": "12:00"}, {"open": "11:00", "close": "13:00"}]),
        lambda data: data.update(schema_version=2),
        lambda data: data.update(valid_from="2026-99-99"),
    ],
)
def test_malformed_or_overlapping_calendar_is_rejected(tmp_path, mutate) -> None:
    data = _calendar_data()
    mutate(data)

    with pytest.raises(ValueError):
        _load(tmp_path, data)


def test_healthy_calendar_closed_does_not_require_a_fresh_tick() -> None:
    observation = MarketLifecycleController().observe(
        CalendarStatus.CLOSED,
        terminal_healthy=True,
        account_healthy=True,
        tick_fresh=False,
    )

    assert observation.state is MarketState.MARKET_CLOSED
    assert observation.reason is None


def test_calendar_open_without_a_fresh_quote_is_not_market_closed() -> None:
    observation = MarketLifecycleController().observe(
        CalendarStatus.OPEN,
        terminal_healthy=True,
        account_healthy=True,
        tick_fresh=False,
    )

    assert observation.state is MarketState.MARKET_UNKNOWN
    assert observation.reason is MarketReason.STALE_DATA


def test_broker_tick_against_closed_calendar_enters_opening_validation() -> None:
    observation = MarketLifecycleController().observe(
        CalendarStatus.CLOSED,
        terminal_healthy=True,
        account_healthy=True,
        tick_fresh=True,
    )

    assert observation.state is MarketState.MARKET_OPENING_VALIDATION
    assert observation.reason is MarketReason.CALENDAR_BROKER_MISMATCH


@pytest.mark.parametrize(
    ("terminal_healthy", "account_healthy", "expected_reason"),
    [
        (False, True, MarketReason.BROKER_DISCONNECTED),
        (True, False, MarketReason.DATA_FAILURE),
    ],
)
def test_unhealthy_connectivity_never_counts_as_market_closed(
    terminal_healthy, account_healthy, expected_reason
) -> None:
    observation = MarketLifecycleController().observe(
        CalendarStatus.CLOSED,
        terminal_healthy=terminal_healthy,
        account_healthy=account_healthy,
        tick_fresh=False,
    )

    assert observation.state is MarketState.MARKET_DATA_ERROR
    assert observation.reason is expected_reason


def test_unknown_calendar_is_fail_closed_even_when_a_quote_is_seen() -> None:
    observation = MarketLifecycleController().observe(
        CalendarStatus.UNKNOWN,
        terminal_healthy=True,
        account_healthy=True,
        tick_fresh=True,
    )

    assert observation.state is MarketState.MARKET_UNKNOWN


def test_failed_reopen_validation_remains_no_trade() -> None:
    controller = MarketLifecycleController()
    opening = controller.observe(
        CalendarStatus.OPEN,
        terminal_healthy=True,
        account_healthy=True,
        tick_fresh=True,
    )

    rejected = controller.complete_opening_validation(False, reason=MarketReason.BROKER_CLOCK_ERROR)

    assert opening.state is MarketState.MARKET_OPENING_VALIDATION
    assert rejected.state is MarketState.MARKET_DATA_ERROR
    assert rejected.reason is MarketReason.BROKER_CLOCK_ERROR


def test_validated_broker_tick_preserves_calendar_mismatch_diagnostic() -> None:
    controller = MarketLifecycleController()
    controller.observe(
        CalendarStatus.CLOSED,
        terminal_healthy=True,
        account_healthy=True,
        tick_fresh=True,
    )

    opened = controller.complete_opening_validation(
        True, reason=MarketReason.CALENDAR_BROKER_MISMATCH
    )

    assert opened.state is MarketState.MARKET_OPEN
    assert opened.permits_strategy_evaluation is True
    assert opened.reason is MarketReason.CALENDAR_BROKER_MISMATCH
