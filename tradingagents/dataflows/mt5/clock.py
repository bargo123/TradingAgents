"""Calibrated, read-only normalization for broker-clock MT5 timestamps.

MetaTrader terminals can expose epoch fields whose numeric values follow the
broker's wall clock rather than the application's true UTC clock.  This module
keeps literal epoch decoding separate from the explicitly calibrated broker
offset so the provider never consults the computer's timezone or guesses an
offset from a single unvalidated sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from .errors import Mt5BrokerClockError

UTC = timezone.utc
BrokerClockStatus = Literal["CALIBRATED", "UNAVAILABLE", "AMBIGUOUS", "STALE_MARKET"]


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must be UTC")
    return value.astimezone(UTC)


def decode_mt5_epoch(value: object, *, milliseconds: bool = False) -> datetime:
    """Decode a literal MT5 epoch without consulting local timezone state."""

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("MT5 epoch must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError("MT5 epoch must be finite")
    if milliseconds:
        numeric /= 1000.0
    try:
        return datetime.fromtimestamp(numeric, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("MT5 epoch is outside the supported datetime range") from exc


@dataclass(frozen=True, slots=True)
class BrokerClockConfig:
    """Explicit safety limits for live broker-clock calibration."""

    offset_min_seconds: int = -(14 * 60 * 60)
    offset_max_seconds: int = 14 * 60 * 60
    offset_step_seconds: int = 15 * 60
    sample_count: int = 3
    max_tick_age_seconds: float = 120.0
    max_future_skew_seconds: float = 2.0
    max_calibration_age_seconds: float = 3_600.0

    def __post_init__(self) -> None:
        if self.offset_min_seconds > self.offset_max_seconds:
            raise ValueError("offset_min_seconds must not exceed offset_max_seconds")
        if self.offset_step_seconds <= 0:
            raise ValueError("offset_step_seconds must be positive")
        if (self.offset_max_seconds - self.offset_min_seconds) % self.offset_step_seconds:
            raise ValueError("offset range must align to offset_step_seconds")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        for value, name in (
            (self.max_tick_age_seconds, "max_tick_age_seconds"),
            (self.max_future_skew_seconds, "max_future_skew_seconds"),
            (self.max_calibration_age_seconds, "max_calibration_age_seconds"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BrokerClockSample:
    """One bounded live observation used to calibrate the broker clock."""

    observed_before_utc: datetime
    observed_after_utc: datetime
    raw_timestamp_utc: datetime

    def __post_init__(self) -> None:
        before = _utc(self.observed_before_utc, "observed_before_utc")
        after = _utc(self.observed_after_utc, "observed_after_utc")
        raw = _utc(self.raw_timestamp_utc, "raw_timestamp_utc")
        if after < before:
            raise ValueError("observed_after_utc must not precede observed_before_utc")
        object.__setattr__(self, "observed_before_utc", before)
        object.__setattr__(self, "observed_after_utc", after)
        object.__setattr__(self, "raw_timestamp_utc", raw)


@dataclass(frozen=True, slots=True)
class Mt5BrokerClock:
    """Immutable calibration provenance and broker-clock transformation."""

    offset_seconds: float | None
    status: BrokerClockStatus
    calibrated_at_utc: datetime | None = None
    server: str | None = None
    symbol: str | None = None
    sample_count: int = 0
    max_residual_seconds: float | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"CALIBRATED", "UNAVAILABLE", "AMBIGUOUS", "STALE_MARKET"}:
            raise ValueError(f"unsupported broker clock status: {self.status!r}")
        if self.offset_seconds is not None and not math.isfinite(float(self.offset_seconds)):
            raise ValueError("offset_seconds must be finite")
        if self.status == "CALIBRATED":
            if self.offset_seconds is None or self.calibrated_at_utc is None:
                raise ValueError("calibrated broker clock requires offset and calibrated_at_utc")
            object.__setattr__(
                self, "calibrated_at_utc", _utc(self.calibrated_at_utc, "calibrated_at_utc")
            )
        elif self.offset_seconds is not None:
            raise ValueError("uncalibrated broker clock must not expose an offset")
        elif self.calibrated_at_utc is not None:
            object.__setattr__(
                self, "calibrated_at_utc", _utc(self.calibrated_at_utc, "calibrated_at_utc")
            )
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if self.max_residual_seconds is not None:
            residual = float(self.max_residual_seconds)
            if not math.isfinite(residual) or residual < 0:
                raise ValueError("max_residual_seconds must be finite and non-negative")
            object.__setattr__(self, "max_residual_seconds", residual)

    @property
    def is_calibrated(self) -> bool:
        return self.status == "CALIBRATED" and self.offset_seconds is not None

    def ensure_fresh(self, now_utc: datetime, *, max_age_seconds: float) -> None:
        """Reject an unavailable, ambiguous, or aged calibration."""

        now = _utc(now_utc, "now_utc")
        if not self.is_calibrated or self.calibrated_at_utc is None:
            raise Mt5BrokerClockError(
                f"broker clock calibration is {self.status.lower()}"
            )
        age = (now - self.calibrated_at_utc).total_seconds()
        if age < 0 or age > float(max_age_seconds):
            raise Mt5BrokerClockError(
                f"broker clock calibration is stale (age={age:.3f}s)"
            )

    def normalize_decoded(self, raw_timestamp_utc: datetime) -> datetime:
        if not self.is_calibrated or self.offset_seconds is None:
            raise Mt5BrokerClockError(
                f"broker clock calibration is {self.status.lower()}"
            )
        raw = _utc(raw_timestamp_utc, "raw_timestamp_utc")
        return raw - timedelta(seconds=float(self.offset_seconds))

    def normalize_epoch(self, value: object, *, milliseconds: bool = False) -> datetime:
        return self.normalize_decoded(
            decode_mt5_epoch(value, milliseconds=milliseconds)
        )

    def to_broker_datetime(self, true_utc: datetime) -> datetime:
        if not self.is_calibrated or self.offset_seconds is None:
            raise Mt5BrokerClockError(
                f"broker clock calibration is {self.status.lower()}"
            )
        return _utc(true_utc, "true_utc") + timedelta(seconds=float(self.offset_seconds))

    def as_dict(self) -> dict[str, object]:
        return {
            "offset_seconds": self.offset_seconds,
            "status": self.status,
            "calibrated_at_utc": (
                None
                if self.calibrated_at_utc is None
                else self.calibrated_at_utc.isoformat().replace("+00:00", "Z")
            ),
            "server": self.server,
            "symbol": self.symbol,
            "sample_count": self.sample_count,
            "max_residual_seconds": self.max_residual_seconds,
            "source": self.source,
        }


def _candidate_offsets(config: BrokerClockConfig) -> tuple[int, ...]:
    return tuple(
        range(
            config.offset_min_seconds,
            config.offset_max_seconds + config.offset_step_seconds,
            config.offset_step_seconds,
        )
    )


def _sample_candidate_valid(
    sample: BrokerClockSample,
    offset_seconds: int,
    config: BrokerClockConfig,
) -> tuple[bool, float]:
    midpoint = sample.observed_before_utc + (
        sample.observed_after_utc - sample.observed_before_utc
    ) / 2
    normalized = sample.raw_timestamp_utc - timedelta(seconds=offset_seconds)
    age = (sample.observed_after_utc - normalized).total_seconds()
    future = (normalized - sample.observed_after_utc).total_seconds()
    residual = abs((normalized - midpoint).total_seconds())
    valid = (
        future <= config.max_future_skew_seconds
        and age <= config.max_tick_age_seconds
    )
    return valid, residual


def calibrate_broker_clock(
    samples: tuple[BrokerClockSample, ...] | list[BrokerClockSample],
    *,
    server: str | None,
    symbol: str | None,
    config: BrokerClockConfig | None = None,
    calibrated_at_utc: datetime | None = None,
    source: str = "LIVE_TICK_MIDPOINT",
) -> Mt5BrokerClock:
    """Select one validated broker offset from bounded live observations."""

    resolved_config = config or BrokerClockConfig()
    samples = tuple(samples)
    if not samples:
        return Mt5BrokerClock(
            offset_seconds=None,
            status="UNAVAILABLE",
            server=server,
            symbol=symbol,
            source=source,
        )

    candidates = _candidate_offsets(resolved_config)
    valid_by_sample: list[set[int]] = []
    residuals_by_offset: dict[int, list[float]] = {offset: [] for offset in candidates}
    for sample in samples:
        valid_for_sample: set[int] = set()
        for offset in candidates:
            valid, residual = _sample_candidate_valid(sample, offset, resolved_config)
            if valid:
                valid_for_sample.add(offset)
                residuals_by_offset[offset].append(residual)
        valid_by_sample.append(valid_for_sample)

    valid_all = set.intersection(*valid_by_sample) if valid_by_sample else set()
    if len(valid_all) != 1:
        if len(valid_all) > 1 or any(valid_by_sample):
            status: BrokerClockStatus = "AMBIGUOUS"
        else:
            status = "STALE_MARKET"
        return Mt5BrokerClock(
            offset_seconds=None,
            status=status,
            server=server,
            symbol=symbol,
            sample_count=len(samples),
            source=source,
        )

    offset = next(iter(valid_all))
    residuals = residuals_by_offset[offset]
    calibration_time = calibrated_at_utc or max(sample.observed_after_utc for sample in samples)
    return Mt5BrokerClock(
        offset_seconds=float(offset),
        status="CALIBRATED",
        calibrated_at_utc=calibration_time,
        server=server,
        symbol=symbol,
        sample_count=len(samples),
        max_residual_seconds=max(residuals, default=0.0),
        source=source,
    )
