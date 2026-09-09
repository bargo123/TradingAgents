from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.dataflows.mt5.clock import (
    BrokerClockConfig,
    BrokerClockSample,
    Mt5BrokerClock,
    calibrate_broker_clock,
    decode_mt5_epoch,
)
from tradingagents.dataflows.mt5.errors import Mt5BrokerClockError

UTC = timezone.utc
BASE = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)


def _sample(offset_seconds: int | float, *, raw_age_seconds: float = 0.5) -> BrokerClockSample:
    before = BASE - timedelta(milliseconds=100)
    after = BASE + timedelta(milliseconds=100)
    raw = BASE + timedelta(seconds=offset_seconds - raw_age_seconds)
    return BrokerClockSample(
        observed_before_utc=before,
        observed_after_utc=after,
        raw_timestamp_utc=raw,
    )


def test_decode_mt5_epoch_seconds_is_exact_utc() -> None:
    assert decode_mt5_epoch(1_700_000_000) == datetime(
        2023, 11, 14, 22, 13, 20, tzinfo=UTC
    )


def test_decode_mt5_epoch_milliseconds_preserves_precision() -> None:
    assert decode_mt5_epoch(1_700_000_000_123, milliseconds=True) == datetime(
        2023, 11, 14, 22, 13, 20, 123_000, tzinfo=UTC
    )


@pytest.mark.parametrize("offset_seconds", [3 * 3600, 2 * 3600, 5 * 3600 + 45 * 60])
def test_calibration_supports_global_quarter_hour_offsets(offset_seconds: int) -> None:
    clock = calibrate_broker_clock(
        [_sample(offset_seconds), _sample(offset_seconds, raw_age_seconds=1.0)],
        server="Test-Demo",
        symbol="EURUSD",
    )

    assert clock.status == "CALIBRATED"
    assert clock.offset_seconds == offset_seconds
    assert clock.sample_count == 2
    assert clock.max_residual_seconds <= 1.0


def test_calibration_is_independent_of_computer_local_timezone() -> None:
    clock = calibrate_broker_clock([_sample(3 * 3600)], server="Test", symbol="EURUSD")
    normalized = clock.normalize_epoch(
        decode_mt5_epoch(1_700_000_000, milliseconds=False).timestamp()
    )

    assert normalized.tzinfo == UTC
    assert decode_mt5_epoch(1_700_000_000).timestamp() == 1_700_000_000


def test_calibration_fails_closed_for_unavailable_samples() -> None:
    clock = calibrate_broker_clock([], server="Test", symbol="EURUSD")

    assert clock.status == "UNAVAILABLE"
    assert clock.offset_seconds is None


def test_calibration_fails_closed_for_ambiguous_candidates() -> None:
    config = BrokerClockConfig(max_tick_age_seconds=1_400)
    clock = calibrate_broker_clock(
        [_sample(0, raw_age_seconds=450)],
        server="Test",
        symbol="EURUSD",
        config=config,
    )

    assert clock.status == "AMBIGUOUS"
    assert clock.offset_seconds is None


def test_calibration_fails_closed_for_stale_market() -> None:
    config = BrokerClockConfig(max_tick_age_seconds=120)
    clock = calibrate_broker_clock(
        [_sample(0, raw_age_seconds=14 * 3600 + 600)],
        server="Test",
        symbol="EURUSD",
        config=config,
    )

    assert clock.status == "STALE_MARKET"
    assert clock.offset_seconds is None


def test_zero_offset_is_a_valid_calibration() -> None:
    clock = calibrate_broker_clock([_sample(0)], server="Test", symbol="EURUSD")

    assert clock.status == "CALIBRATED"
    assert clock.offset_seconds == 0


def test_millisecond_normalization_and_inverse_request_translation() -> None:
    clock = calibrate_broker_clock([_sample(3 * 3600)], server="Test", symbol="EURUSD")
    true_utc = datetime(2026, 9, 9, 12, 0, 0, 123_000, tzinfo=UTC)
    broker_time = clock.to_broker_datetime(true_utc)

    assert broker_time == datetime(2026, 9, 9, 15, 0, 0, 123_000, tzinfo=UTC)
    assert clock.normalize_epoch(broker_time.timestamp(), milliseconds=False) == true_utc
    assert clock.normalize_epoch(broker_time.timestamp() * 1000, milliseconds=True) == true_utc


def test_stale_calibration_is_rejected() -> None:
    clock = Mt5BrokerClock(
        offset_seconds=0,
        status="CALIBRATED",
        calibrated_at_utc=BASE - timedelta(hours=2),
        server="Test",
        symbol="EURUSD",
        sample_count=1,
        max_residual_seconds=0.1,
        source="TEST",
    )

    with pytest.raises(Mt5BrokerClockError, match="stale"):
        clock.ensure_fresh(BASE, max_age_seconds=3600)


def test_production_clock_source_has_no_local_or_broker_specific_offset() -> None:
    source = Path(__file__).parents[1].joinpath(
        "tradingagents", "dataflows", "mt5", "clock.py"
    )
    text = source.read_text(encoding="utf-8").lower()
    assert "jordan" not in text
    assert "metaquotes" not in text
    assert "-3 hours" not in text
    assert "-10800" not in text


def test_forex_shadow_docs_describe_broker_clock_limits() -> None:
    docs = Path(__file__).parents[1].joinpath("docs", "forex-shadow.md")
    text = docs.read_text(encoding="utf-8").lower()
    assert "broker clock" in text
    assert "fail closed" in text
    assert "computer timezone" in text or "local timezone" in text
    assert "dst" in text
    assert "historical" in text
