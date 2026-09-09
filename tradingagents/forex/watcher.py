"""Autonomous, read-only forex shadow collector contracts.

The watcher deliberately lives beside the existing forex runner rather than in
the TradingAgents graph.  The first part of this module contains pure schedule
and configuration contracts; the operational components are added below these
contracts as the collector is wired.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from tradingagents.default_config import DEFAULT_CONFIG

_UTC = timezone.utc
_TIMEFRAME_SECONDS = {"M5": 5 * 60, "M15": 15 * 60, "H1": 60 * 60}
_FOREX_ANALYSTS = frozenset({"market", "news"})
_ANALYSIS_CONTRACT_VERSION = "forex-shadow.v1"


def _require_aware_utc(value: datetime, name: str = "timestamp") -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != _UTC.utcoffset(value):
        raise ValueError(f"{name} must be UTC")
    return value.astimezone(_UTC)


def _iso(value: datetime) -> str:
    return _require_aware_utc(value).isoformat().replace("+00:00", "Z")


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_choice(value: Any, choices: set[str], name: str) -> None:
    if not isinstance(value, str) or value.strip().upper() not in choices:
        raise ValueError(f"{name} must be one of {', '.join(sorted(choices))}")


def _normalize_symbol_tuple(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        values = (values,)  # type: ignore[assignment]
    try:
        result = tuple(str(value).strip().upper() for value in values)
    except TypeError as exc:
        raise ValueError("symbols must be a sequence") from exc
    if not result or any(not value for value in result):
        raise ValueError("symbols must contain at least one non-empty symbol")
    if len(set(result)) != len(result):
        raise ValueError("symbols must not contain duplicates")
    return result


def _normalize_analyst_tuple(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        values = tuple(part.strip() for part in str(values).split(","))
    try:
        result = tuple(str(value).strip().lower() for value in values)
    except TypeError as exc:
        raise ValueError("analysts must be a sequence") from exc
    if not result or any(not value for value in result):
        raise ValueError("analysts must contain at least one name")
    if len(set(result)) != len(result):
        raise ValueError("analysts must not contain duplicates")
    invalid = sorted(set(result) - _FOREX_ANALYSTS)
    if invalid:
        raise ValueError(
            "forex shadow mode only supports market/news analysts; "
            f"unsupported: {', '.join(invalid)}"
        )
    return result


_SAFE_CONFIG_KEYS = frozenset(
    {
        "llm_provider",
        "quick_think_llm",
        "deep_think_llm",
        "backend_url",
        "output_language",
        "max_debate_rounds",
        "max_risk_discuss_rounds",
        "max_recur_limit",
        "temperature",
        "max_tokens",
        "llm_max_retries",
        "google_thinking_level",
        "openai_reasoning_effort",
        "anthropic_effort",
        "forex_quick_reasoning_effort",
        "forex_deep_reasoning_effort",
        "forex_quick_thinking_level",
        "forex_deep_thinking_level",
        "forex_quick_effort",
        "forex_deep_effort",
        "forex_quick_thinking",
        "forex_deep_thinking",
        "analysis_profile",
        "analysts",
        "schedule_timeframe",
        "analysis_contract_version",
        "collector_contract_version",
        "prompt_config_version",
        "application_version",
        "freshness_budget_seconds",
        "evaluation_interval_seconds",
        "evaluation_enabled",
    }
)
_CREDENTIAL_WORDS = ("key", "token", "secret", "password", "authorization", "credential")


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_json_value(item)
            for key, item in value.items()
            if not any(word in str(key).casefold() for word in _CREDENTIAL_WORDS)
        }
    if isinstance(value, (tuple, list)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_effective_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return an allow-listed, JSON-safe config without credentials."""

    merged: dict[str, Any] = dict(DEFAULT_CONFIG)
    if config:
        merged.update(dict(config))
    safe: dict[str, Any] = {}
    for key, value in merged.items():
        if key not in _SAFE_CONFIG_KEYS:
            continue
        if any(word in key.casefold() for word in _CREDENTIAL_WORDS):
            continue
        if key == "backend_url" and isinstance(value, str):
            # A URL query may carry a bearer/API credential.  Preserve only
            # the endpoint path and discard query/fragment data.
            value = value.split("?", 1)[0].split("#", 1)[0]
        safe[key] = _safe_json_value(value)
    return safe


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
        object.__setattr__(self, "symbols", _normalize_symbol_tuple(self.symbols))
        object.__setattr__(self, "analysts", _normalize_analyst_tuple(self.analysts))
        timeframe = self.schedule_timeframe.strip().upper()
        _require_choice(timeframe, set(_TIMEFRAME_SECONDS), "schedule_timeframe")
        object.__setattr__(self, "schedule_timeframe", timeframe)
        _require_choice(self.analysis_profile, {"INTRADAY"}, "analysis_profile")
        object.__setattr__(self, "analysis_profile", self.analysis_profile.upper())
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
        object.__setattr__(self, "db_path", Path(self.db_path))
        if self.terminal_path is not None and not isinstance(self.terminal_path, str):
            raise ValueError("terminal_path must be a string or None")

    @property
    def safe_fingerprint(self) -> str:
        payload = safe_effective_config(
            {
                **asdict(self),
                "analysis_contract_version": _ANALYSIS_CONTRACT_VERSION,
            }
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class Clock(Protocol):
    def now(self) -> datetime:
        ...

    def monotonic(self) -> float:
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(_UTC)

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass
class FrozenClock:
    current: datetime
    current_monotonic: float = 0.0

    def __post_init__(self) -> None:
        self.current = _require_aware_utc(self.current, "current")

    def now(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.current_monotonic

    def set(self, value: datetime) -> None:
        self.current = _require_aware_utc(value, "current")


@dataclass(frozen=True, slots=True)
class ScheduledOpportunity:
    requested_symbol: str
    analysis_profile: str
    schedule_timeframe: str
    anchor_timestamp: datetime
    bar_close_timestamp: datetime
    eligible_after: datetime
    config_fingerprint: str
    opportunity_key: str = field(default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_symbol", self.requested_symbol.strip().upper())
        object.__setattr__(self, "analysis_profile", self.analysis_profile.strip().upper())
        object.__setattr__(self, "schedule_timeframe", self.schedule_timeframe.strip().upper())
        for name in ("anchor_timestamp", "bar_close_timestamp", "eligible_after"):
            object.__setattr__(self, name, _require_aware_utc(getattr(self, name), name))
        if not self.requested_symbol or not self.config_fingerprint:
            raise ValueError("scheduled opportunity requires symbol and config fingerprint")
        expected_key = canonical_opportunity_key(self)
        if self.opportunity_key and self.opportunity_key != expected_key:
            raise ValueError("opportunity_key does not match canonical opportunity identity")
        object.__setattr__(self, "opportunity_key", expected_key)


def canonical_opportunity_key(opportunity: ScheduledOpportunity) -> str:
    payload = {
        "requested_symbol": opportunity.requested_symbol.strip().upper(),
        "analysis_profile": opportunity.analysis_profile.strip().upper(),
        "schedule_timeframe": opportunity.schedule_timeframe.strip().upper(),
        "anchor_timestamp": _iso(opportunity.anchor_timestamp),
        "analysis_config_version": _ANALYSIS_CONTRACT_VERSION,
        "config_fingerprint": opportunity.config_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class SchedulePolicy(Protocol):
    def opportunities_between(
        self, previous: datetime | None, now: datetime
    ) -> tuple[ScheduledOpportunity, ...]:
        ...


@dataclass(frozen=True, slots=True)
class CompletedBarSchedule:
    timeframe: str = "M15"
    settle_seconds: int = 30
    config_fingerprint: str = ""
    requested_symbols: tuple[str, ...] = ("EURUSD",)
    analysis_profile: str = "INTRADAY"

    def __post_init__(self) -> None:
        timeframe = self.timeframe.strip().upper()
        _require_choice(timeframe, set(_TIMEFRAME_SECONDS), "timeframe")
        object.__setattr__(self, "timeframe", timeframe)
        _require_nonnegative_int(self.settle_seconds, "settle_seconds")
        if not self.config_fingerprint:
            raise ValueError("config_fingerprint must be non-empty")
        object.__setattr__(self, "requested_symbols", _normalize_symbol_tuple(self.requested_symbols))
        object.__setattr__(self, "analysis_profile", self.analysis_profile.strip().upper())

    def _floor(self, value: datetime) -> datetime:
        value = _require_aware_utc(value)
        seconds = _TIMEFRAME_SECONDS[self.timeframe]
        epoch = int(value.timestamp())
        return datetime.fromtimestamp(epoch - epoch % seconds, tz=_UTC)

    def opportunities_between(
        self, previous: datetime | None, now: datetime
    ) -> tuple[ScheduledOpportunity, ...]:
        now = _require_aware_utc(now, "now")
        step = timedelta(seconds=_TIMEFRAME_SECONDS[self.timeframe])
        latest_anchor = self._floor(now - timedelta(seconds=self.settle_seconds)) - step
        if previous is None:
            anchors = (latest_anchor,)
        else:
            previous = _require_aware_utc(previous, "previous")
            previous_latest = (
                self._floor(previous - timedelta(seconds=self.settle_seconds)) - step
            )
            cursor = previous_latest + step
            values: list[datetime] = []
            while cursor <= latest_anchor:
                values.append(cursor)
                cursor += step
            anchors = tuple(values)
        result: list[ScheduledOpportunity] = []
        for anchor in anchors:
            close = anchor + step
            result.extend(
                ScheduledOpportunity(
                    requested_symbol=symbol,
                    analysis_profile=self.analysis_profile,
                    schedule_timeframe=self.timeframe,
                    anchor_timestamp=anchor,
                    bar_close_timestamp=close,
                    eligible_after=close + timedelta(seconds=self.settle_seconds),
                    config_fingerprint=self.config_fingerprint,
                )
                for symbol in self.requested_symbols
            )
        return tuple(result)


__all__ = [
    "WatcherConfig",
    "Clock",
    "SystemClock",
    "FrozenClock",
    "ScheduledOpportunity",
    "SchedulePolicy",
    "CompletedBarSchedule",
    "canonical_opportunity_key",
    "safe_effective_config",
]
