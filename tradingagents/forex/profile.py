"""Forex-only analysis profiles and deterministic market feature helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from tradingagents.dataflows.mt5.models import Mt5Bar

MACRO_EVENT_UNAVAILABLE = "MACRO/EVENT DATA UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ForexAnalysisProfile:
    """A bounded decision horizon for a forex shadow run."""

    name: str
    horizon_label: str
    valid_for_seconds: int
    default_bar_count: int


INTRADAY_PROFILE = ForexAnalysisProfile(
    name="INTRADAY",
    horizon_label="minutes to hours",
    valid_for_seconds=3_600,
    default_bar_count=100,
)

_PROFILES = {INTRADAY_PROFILE.name: INTRADAY_PROFILE}


def resolve_forex_profile(name: str | ForexAnalysisProfile | None = None) -> ForexAnalysisProfile:
    """Resolve a supported profile and reject unknown values explicitly."""
    if isinstance(name, ForexAnalysisProfile):
        profile = _PROFILES.get(name.name.upper())
        if profile == name:
            return profile
        raise ValueError(f"unknown forex analysis profile: {name.name!r}")
    key = "INTRADAY" if name is None else str(name).strip().upper()
    profile = _PROFILES.get(key)
    if profile is None:
        raise ValueError(f"unknown forex analysis profile: {name!r}")
    return profile


def build_forex_profile_context(
    profile: str | ForexAnalysisProfile | None = None,
) -> str:
    """Render profile constraints for injection into forex prompts."""
    resolved = resolve_forex_profile(profile)
    return "\n".join(
        (
            f"FOREX ANALYSIS PROFILE: {resolved.name}",
            f"DECISION HORIZON: {resolved.horizon_label}",
            f"VALID FOR SECONDS: {resolved.valid_for_seconds}",
            "Use minutes/hours and current-session conditions; never use "
            "months, years, equity-investment, or issuer-valuation horizons.",
        )
    )


def _true_range(bar: Mt5Bar, previous_close: float | None) -> float:
    ranges = [bar.high - bar.low]
    if previous_close is not None:
        ranges.extend((abs(bar.high - previous_close), abs(bar.low - previous_close)))
    return max(ranges)


def calculate_timeframe_features(bars: Sequence[Mt5Bar]) -> dict[str, Any]:
    """Calculate compact, deterministic features from one timeframe's bars.

    A single candle still gets OHLC/range statistics, but it cannot establish a
    multi-bar return or direction. Those fields are explicitly marked as
    insufficient so a prompt cannot turn one candle into a trend claim.
    """
    series = tuple(bars)
    if not series:
        return {
            "candle_count": 0,
            "latest": None,
            "return_over_bars": None,
            "recent_high": None,
            "recent_low": None,
            "range": None,
            "direction": "INSUFFICIENT_DATA",
            "average_true_range": None,
            "range_pct": None,
            "close_position": None,
        }

    latest = series[-1]
    recent_high = max(bar.high for bar in series)
    recent_low = min(bar.low for bar in series)
    recent_range = recent_high - recent_low
    first_open = series[0].open
    multi_bar_return = None
    direction = "INSUFFICIENT_DATA"
    if len(series) >= 2:
        if first_open:
            multi_bar_return = (latest.close - first_open) / first_open
        direction = (
            "UP" if latest.close > first_open
            else "DOWN" if latest.close < first_open
            else "FLAT"
        )

    true_ranges = []
    previous_close = None
    for bar in series:
        true_ranges.append(_true_range(bar, previous_close))
        previous_close = bar.close
    average_true_range = sum(true_ranges) / len(true_ranges)
    range_pct = recent_range / first_open if first_open else None
    close_position = (
        (latest.close - recent_low) / recent_range
        if recent_range
        else None
    )

    return {
        "candle_count": len(series),
        "latest": {
            "timestamp": latest.timestamp,
            "open": latest.open,
            "high": latest.high,
            "low": latest.low,
            "close": latest.close,
        },
        "return_over_bars": multi_bar_return,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "range": recent_range,
        "direction": direction,
        "average_true_range": average_true_range,
        "range_pct": range_pct,
        "close_position": close_position,
    }


def get_forex_profile_from_state(state: dict[str, Any]) -> ForexAnalysisProfile:
    """Resolve the profile carried by a forex graph state."""
    return resolve_forex_profile(state.get("forex_analysis_profile"))
