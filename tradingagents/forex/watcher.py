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


@dataclass(frozen=True, slots=True)
class MarketProbeResult:
    requested_symbol: str
    resolved_symbol: str
    timestamp: datetime
    bid: float
    ask: float

    @classmethod
    def from_tick(
        cls, requested_symbol: str, resolved_symbol: str, tick: Any
    ) -> "MarketProbeResult":
        timestamp = getattr(tick, "timestamp", None)
        timestamp = _require_aware_utc(timestamp, "broker tick timestamp")
        if not isinstance(resolved_symbol, str) or not resolved_symbol.strip():
            raise ValueError("resolved symbol must be non-empty")
        try:
            bid = float(getattr(tick, "bid"))
            ask = float(getattr(tick, "ask"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("probe quote must contain numeric bid and ask") from exc
        import math

        if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask < bid:
            raise ValueError("probe quote must be finite, positive, and ask >= bid")
        return cls(
            requested_symbol=str(requested_symbol).strip().upper(),
            resolved_symbol=resolved_symbol.strip(),
            timestamp=timestamp,
            bid=bid,
            ask=ask,
        )


class ReadOnlyMarketProbe:
    """One short read-only provider lifecycle used as an analysis preflight."""

    def __init__(self, provider_factory: Callable[..., Any], terminal_path: str | None = None):
        self.provider_factory = provider_factory
        self.terminal_path = terminal_path
        self.calls: list[str] = []

    def probe(self, requested_symbol: str) -> MarketProbeResult:
        self.calls.append(requested_symbol)
        provider: Any | None = None
        try:
            provider = self.provider_factory(terminal_path=self.terminal_path)
            if not callable(getattr(provider, "initialize", None)):
                raise RuntimeError("MT5 provider does not expose initialize()")
            if not provider.initialize():
                raise RuntimeError("MT5 provider initialization failed")
            resolved = provider.ensure_symbol(requested_symbol)
            tick = provider.get_tick(resolved)
            return MarketProbeResult.from_tick(requested_symbol, resolved, tick)
        finally:
            shutdown = getattr(provider, "shutdown", None) if provider is not None else None
            if callable(shutdown):
                shutdown()


class Mt5OperationBusy(RuntimeError):
    """Raised when a second MT5 operation attempts to overlap the first."""


class SerializedMt5OperationGate:
    """Non-reentrant, non-blocking guard around every MT5 operation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._operation: str | None = None
        self.events: list[str] = []
        self.overlaps: list[tuple[str, str]] = []

    @contextmanager
    def acquire(self, operation: str):
        name = str(operation).strip() or "unknown"
        if not self._lock.acquire(blocking=False):
            self.overlaps.append((self._operation or "unknown", name))
            raise Mt5OperationBusy(f"MT5 operation busy: {name}")
        self._operation = name
        self.events.append(f"{name}:start")
        try:
            yield
        finally:
            self.events.append(f"{name}:end")
            self._operation = None
            self._lock.release()


class SingleSlotAnalysisExecutor:
    """One active future and no application-level pending queue."""

    def __init__(
        self,
        *,
        submitter: Callable[[Callable[[], Any]], Future[Any]] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._future: Future[Any] | None = None
        self._executor: ThreadPoolExecutor | None = None
        if submitter is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forex-shadow")
            self._submitter = self._executor.submit
        else:
            self._submitter = submitter

    def submit(self, fn: Callable[[], Any]) -> Future[Any]:
        with self._lock:
            if self._future is not None and not self._future.done():
                raise RuntimeError("analysis slot is busy")
            future = self._submitter(fn)
            if not isinstance(future, Future):
                raise TypeError("analysis submitter must return concurrent.futures.Future")
            self._future = future

            def clear(done: Future[Any]) -> None:
                with self._lock:
                    if self._future is done:
                        self._future = None

            future.add_done_callback(clear)
            return future

    @property
    def future(self) -> Future[Any] | None:
        with self._lock:
            return self._future

    @property
    def active(self) -> bool:
        future = self.future
        return future is not None and not future.done()

    @property
    def pending_count(self) -> int:
        # There is intentionally no pending queue; only the active slot exists.
        return 1 if self.active else 0

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait)


class LeaseHeartbeat:
    """Renew a lease and record a single soft timeout, never kill work."""

    def __init__(
        self,
        *,
        store: Any,
        owner_token: str,
        run_id: str | None,
        timeout_seconds: float,
        sequence: list[str] | None = None,
    ) -> None:
        self.store = store
        self.owner_token = owner_token
        self.run_id = run_id
        self.timeout_seconds = float(timeout_seconds)
        self.sequence = sequence
        self._runtime_alerted = False
        self.runner_future_still_owned = True

    def tick(self, now: datetime, monotonic_now: float) -> None:
        renew = getattr(self.store, "heartbeat", None) or getattr(self.store, "renew", None)
        if not callable(renew):
            raise RuntimeError("watcher store does not expose a lease heartbeat")
        try:
            renew(self.owner_token, now, run_id=self.run_id)
        except TypeError:
            renew(self.owner_token, now)
        if self.sequence is not None:
            self.sequence.append("heartbeat")
        if monotonic_now > self.timeout_seconds and not self._runtime_alerted and self.run_id:
            mark = getattr(self.store, "mark_runtime_alert", None)
            if callable(mark):
                mark(self.owner_token, self.run_id, now)
            self._runtime_alerted = True

    @property
    def runtime_alerted(self) -> bool:
        return self._runtime_alerted


@dataclass(frozen=True, slots=True)
class EvaluationRunEvidence:
    status: str
    errors: tuple[str, ...]
    metrics: Mapping[str, Any]


class EventSink:
    """Allow-listed operational event collector with no prompt/report fields."""

    ALLOWED = frozenset(
        {
            "WATCHER_STARTED",
            "LEASE_RECOVERED",
            "OPPORTUNITY_OBSERVED",
            "OPPORTUNITY_SKIPPED",
            "ANALYSIS_STARTED",
            "ANALYSIS_TIMEOUT_OBSERVED",
            "ANALYSIS_FINISHED",
            "ANALYSIS_FAILED",
            "EVALUATION_FINISHED",
            "EVALUATION_FAILED",
            "CIRCUIT_OPENED",
            "SHUTDOWN_REQUESTED",
            "WATCHER_STOPPED",
        }
    )

    def __init__(self, target: list[dict[str, Any]] | None = None) -> None:
        self.target = target if target is not None else []

    def emit(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        if event not in self.ALLOWED:
            raise ValueError(f"event is not allow-listed: {event}")
        safe = {"event": event}
        for key, value in dict(payload or {}).items():
            if key.casefold() in {"prompt", "completion", "report", "reasoning", "prose"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
        self.target.append(safe)


@dataclass(frozen=True, slots=True)
class WatchCycleResult:
    lifecycle_status: str
    observed_keys: tuple[str, ...] = ()
    skipped_keys: tuple[str, ...] = ()
    skip_reasons: tuple[str, ...] = ()
    active_run_id: str | None = None
    evaluation_status: str | None = None
    evaluation_due_pending: bool = False
    recovery_gap_count: int = 0
    error_code: str | None = None


class OutcomeCoordinator:
    """Adapter around the existing zero-LLM Phase 5 evaluator."""

    def __init__(self, evaluator: Any, *, terminal_path: str | None = None) -> None:
        self.evaluator = evaluator
        self.terminal_path = terminal_path

    def evaluate_pending(self, now: datetime) -> EvaluationRunEvidence:
        try:
            result = self.evaluator.evaluate_pending(now=now, terminal_path=self.terminal_path)
        except TypeError:
            result = self.evaluator.evaluate_pending(now=now)
        errors = tuple(str(item) for item in getattr(result, "errors", ()) or ())
        metrics = getattr(result, "metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        metrics = dict(metrics)
        metrics["llm_calls"] = 0
        status = "ERROR" if errors else "OK"
        return EvaluationRunEvidence(status=status, errors=errors, metrics=metrics)


class CircuitBreakers:
    """Small in-memory view of persisted failure-class thresholds."""

    def __init__(self, config: WatcherConfig) -> None:
        self.config = config
        self.counts: dict[str, int] = {
            "mt5": 0,
            "analysis": 0,
            "incomplete": 0,
            "normalization": 0,
            "runtime": 0,
        }
        self.open_reason: str | None = None

    def record(self, failure_class: str) -> bool:
        if failure_class not in self.counts:
            raise ValueError(f"unknown circuit failure class: {failure_class}")
        self.counts[failure_class] += 1
        threshold = {
            "mt5": self.config.max_consecutive_mt5_failures,
            "analysis": self.config.max_consecutive_analysis_failures,
            "incomplete": self.config.max_consecutive_incomplete,
            "normalization": self.config.max_consecutive_normalization_failures,
            "runtime": self.config.max_consecutive_runtime_exceeded,
        }[failure_class]
        if self.counts[failure_class] >= threshold:
            self.open_reason = f"{failure_class.upper()}_FAILURES"
        return self.open_reason is not None

    def reset(self, failure_class: str) -> None:
        if failure_class not in self.counts:
            raise ValueError(f"unknown circuit failure class: {failure_class}")
        self.counts[failure_class] = 0
        if self.open_reason and self.open_reason.startswith(failure_class.upper()):
            self.open_reason = None

    @property
    def is_open(self) -> bool:
        return self.open_reason is not None


class WatcherCoordinator:
    """Foreground scheduler that composes the existing read-only forex runner."""

    def __init__(
        self,
        config: WatcherConfig,
        store: Any,
        clock: Clock,
        schedule: SchedulePolicy,
        probe: Any,
        runner: Any,
        evaluator: Any,
        executor: SingleSlotAnalysisExecutor,
        gate: SerializedMt5OperationGate,
        heartbeat: LeaseHeartbeat | None = None,
        process_inspector: Any | None = None,
        events: list[dict[str, Any]] | EventSink | None = None,
        callbacks: Sequence[Any] = (),
    ) -> None:
        self.config = config
        self.store = store
        self.clock = clock
        self.schedule = schedule
        self.probe = probe
        self.runner = runner
        self.outcomes = evaluator if isinstance(evaluator, OutcomeCoordinator) else OutcomeCoordinator(evaluator, terminal_path=config.terminal_path)
        self.executor = executor
        self.gate = gate
        self.process_inspector = process_inspector
        self.callbacks = tuple(callbacks)
        self.events = events if isinstance(events, EventSink) else EventSink(events)
        self.owner_token: str | None = None
        self._owner = None
        self._analysis_future: Future[Any] | None = None
        self._active_run_id: str | None = None
        self._last_poll: datetime | None = None
        self._next_eligible_at: datetime | None = None
        self._last_evaluation_at: datetime | None = None
        self._stop = threading.Event()
        self._runtime_alerted = False
        self.circuits = CircuitBreakers(config)
        self.heartbeat = heartbeat

    def start(self, now: datetime | None = None):
        from tradingagents.forex.watch_store import LeaseOwner, LeaseStatus

        now = _require_aware_utc(now or self.clock.now(), "now")
        self.store.initialize()
        owner = LeaseOwner(
            owner_token=str(uuid.uuid4()),
            pid=os.getpid(),
            host=socket.gethostname(),
            process_started_at=now,
        )
        result = self.store.acquire_lease(owner, now, self.process_inspector)
        if result.status is not LeaseStatus.ACQUIRED:
            return result
        self.owner_token = owner.owner_token
        self._owner = owner
        self.store.set_lifecycle(owner.owner_token, "IDLE", now)
        if result.recovered_expired:
            try:
                from tradingagents.forex.shadow import ShadowDecisionStore

                self.store.reconcile_stale_runs(
                    owner.owner_token, now, ShadowDecisionStore(self.config.db_path)
                )
            except Exception as exc:  # recovery is fail-closed but startup can report it
                self.store.set_error(owner.owner_token, "RECONCILIATION_FAILED", str(exc), now)
        self.events.emit("LEASE_RECOVERED" if result.recovered_expired else "WATCHER_STARTED", {})
        self._last_poll = None
        self._last_evaluation_at = now
        if self.heartbeat is None:
            self.heartbeat = LeaseHeartbeat(
                store=self.store,
                owner_token=owner.owner_token,
                run_id=None,
                timeout_seconds=self.config.analysis_timeout_seconds,
            )
        return result

    def _require_started(self, now: datetime) -> str:
        if self.owner_token is None:
            raise RuntimeError("watcher is not started")
        self.store.heartbeat(self.owner_token, now)
        return self.owner_token

    def _observe(self, now: datetime) -> tuple[list[Any], int]:
        items = self.schedule.opportunities_between(self._last_poll, now)
        self._last_poll = now
        observed: list[Any] = []
        for item in items:
            row = self.store.observe_opportunity(item, now)
            observed.append(row)
            self.events.emit("OPPORTUNITY_OBSERVED", {"opportunity_key": item.opportunity_key})
        return observed, 0

    def _skip_busy(self, rows: Sequence[Any], now: datetime, owner_token: str) -> tuple[list[str], list[str]]:
        keys: list[str] = []
        reasons: list[str] = []
        for row in rows:
            if row.status != "ELIGIBLE":
                continue
            self.store.record_skip(owner_token, row.opportunity_key, "ANALYSIS_ALREADY_RUNNING", now)
            keys.append(row.opportunity_key)
            reasons.append("ANALYSIS_ALREADY_RUNNING")
            self.events.emit(
                "OPPORTUNITY_SKIPPED",
                {"opportunity_key": row.opportunity_key, "reason": "ANALYSIS_ALREADY_RUNNING"},
            )
        return keys, reasons

    def _run_claimed(self, run: Any):
        opportunity = self.store.get_opportunity(run.opportunity_key)
        with self.gate.acquire("runner"):
            return self.runner.run(
                symbol=run.requested_symbol,
                analysis_date=opportunity.anchor_timestamp.date().isoformat(),
                analysis_profile=self.config.analysis_profile,
                analysts=self.config.analysts,
                terminal_path=self.config.terminal_path,
                db_path=self.config.db_path,
                source_run_id=run.source_run_id,
                callbacks=self.callbacks,
            )

    def _evidence_from_result(self, run: Any, result: Any) -> Any:
        from tradingagents.forex.watch_store import RunEvidence

        decision = getattr(result, "decision", None)
        if decision is None:
            raise ValueError("runner result did not contain a decision")
        if getattr(decision, "executed", True) is not False:
            raise ValueError("shadow decision executed invariant was violated")
        if getattr(decision, "source_run_id", None) != run.source_run_id:
            raise ValueError("runner source_run_id does not match watcher run")
        metrics = getattr(result, "metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        safe_config = json.dumps(safe_effective_config(self.config.__dict__ if hasattr(self.config, "__dict__") else asdict(self.config)), sort_keys=True)
        return RunEvidence(
            run_id=run.run_id,
            source_run_id=run.source_run_id,
            decision_id=getattr(decision, "decision_id", None),
            requested_symbol=decision.requested_symbol,
            resolved_symbol=decision.resolved_symbol,
            decision_context_status=getattr(decision, "decision_context_status", None),
            normalization_status=getattr(decision, "normalization_status", None),
            normalized_action=getattr(decision, "action", None),
            analysis_snapshot_timestamp=getattr(decision, "analysis_snapshot_timestamp", None),
            decision_completed_timestamp=getattr(decision, "decision_completed_timestamp", None),
            analysis_latency_seconds=getattr(decision, "analysis_latency_seconds", None),
            decision_reference_timestamp=getattr(decision, "decision_reference_timestamp", None),
            decision_reference_delay_seconds=getattr(decision, "decision_reference_delay_seconds", None),
            stale_by_completion=(
                None
                if getattr(decision, "decision_completed_timestamp", None) is None
                else getattr(decision, "decision_completed_timestamp")
                - getattr(decision, "analysis_snapshot_timestamp", getattr(decision, "snapshot_timestamp"))
                > timedelta(seconds=self.config.freshness_budget_seconds)
            ),
            runtime_seconds=getattr(result, "elapsed_seconds", None),
            llm_calls=metrics.get("llm_calls"),
            tool_calls=metrics.get("tool_calls"),
            tokens_in=metrics.get("tokens_in"),
            tokens_out=metrics.get("tokens_out"),
            reasoning_tokens=metrics.get("reasoning_tokens"),
            llm_provider=getattr(decision, "llm_provider", None),
            quick_model=getattr(decision, "quick_model", None),
            deep_model=getattr(decision, "deep_model", None),
            executed=False,
            decision_reference_status=getattr(decision, "decision_reference_status", None),
            freshness_budget_seconds=self.config.freshness_budget_seconds,
            safe_config_json=safe_config,
            metrics_json=json.dumps(dict(metrics), default=str, sort_keys=True),
        )

    def _finalize_completed(self, now: datetime) -> tuple[str | None, str | None]:
        future = self._analysis_future
        run_id = self._active_run_id
        self._analysis_future = None
        self._active_run_id = None
        if future is None or run_id is None:
            return None, None
        run = self.store.get_run(run_id)
        try:
            result = future.result()
            evidence = self._evidence_from_result(run, result)
            slow = bool(
                (self.heartbeat is not None and self.heartbeat.runtime_alerted)
                or (
                    evidence.runtime_seconds is not None
                    and evidence.runtime_seconds > self.config.analysis_timeout_seconds
                )
            )
            status = "SUCCEEDED_SLOW" if slow else "SUCCEEDED"
            self.store.finalize_run(self.owner_token, run_id, now, status=status, evidence=evidence)
            self._next_eligible_at = now + timedelta(seconds=self.config.cooldown_seconds)
            self.events.emit(
                "ANALYSIS_FINISHED",
                {"run_id": run_id, "decision_id": evidence.decision_id, "status": status},
            )
            return evidence.decision_id, None
        except Exception as exc:
            self.store.finalize_run(
                self.owner_token,
                run_id,
                now,
                status="FAILED",
                failure_code="ANALYSIS_FAILED",
                failure_detail=str(exc),
            )
            self.events.emit("ANALYSIS_FAILED", {"run_id": run_id, "error_code": "ANALYSIS_FAILED"})
            return None, "ANALYSIS_FAILED"

    def _maybe_evaluate(self, now: datetime) -> str | None:
        if not self.config.evaluation_enabled:
            return None
        summary = self.store.summary(now)
        due = bool(summary.get("evaluation_due_pending"))
        last = self._last_evaluation_at
        if not due and self.config.evaluation_interval_seconds:
            due = last is None or (now - last).total_seconds() >= self.config.evaluation_interval_seconds
        if not due:
            return None
        with self.gate.acquire("evaluator"):
            evidence = self.outcomes.evaluate_pending(now)
        self.store.clear_evaluation_due(self.owner_token, now, evidence.status)
        self._last_evaluation_at = now
        event = "EVALUATION_FAILED" if evidence.errors else "EVALUATION_FINISHED"
        self.events.emit(event, {"status": evidence.status, "llm_calls": 0})
        return evidence.status

    def run_once(self, now: datetime | None = None) -> WatchCycleResult:
        now = _require_aware_utc(now or self.clock.now(), "now")
        owner_token = self._require_started(now)
        observed, gap = self._observe(now)
        observed_keys = tuple(row.opportunity_key for row in observed)
        if self._analysis_future is not None and not self._analysis_future.done():
            skipped_keys, skip_reasons = self._skip_busy(observed, now, owner_token)
            summary = self.store.summary(now)
            if (
                self.config.evaluation_interval_seconds == 0
                or self._last_evaluation_at is None
                or (
                    now - self._last_evaluation_at
                ).total_seconds()
                >= self.config.evaluation_interval_seconds
            ):
                self.store.mark_evaluation_due(owner_token, now)
            self.store.set_lifecycle(owner_token, "ANALYZING", now)
            if self.heartbeat is not None:
                self.heartbeat.tick(now, self.clock.monotonic())
            return WatchCycleResult(
                lifecycle_status="ANALYZING",
                observed_keys=observed_keys,
                skipped_keys=tuple(skipped_keys),
                skip_reasons=tuple(skip_reasons),
                active_run_id=self._active_run_id,
                evaluation_due_pending=True,
                recovery_gap_count=gap,
            )

        error_code = None
        evaluation_status = None
        if self._analysis_future is not None:
            _, error_code = self._finalize_completed(now)
            if error_code is None:
                try:
                    evaluation_status = self._maybe_evaluate(now)
                except Exception as exc:
                    error_code = "EVALUATION_FAILED"
                    self.store.set_error(owner_token, error_code, str(exc), now)
        else:
            try:
                evaluation_status = self._maybe_evaluate(now)
            except Exception as exc:
                error_code = "EVALUATION_FAILED"
                self.store.set_error(owner_token, error_code, str(exc), now)

        if error_code is not None:
            return WatchCycleResult(
                lifecycle_status="IDLE",
                observed_keys=observed_keys,
                evaluation_status=evaluation_status,
                evaluation_due_pending=bool(self.store.summary(now).get("evaluation_due_pending")),
                recovery_gap_count=gap,
                error_code=error_code,
            )

        candidates = [
            row
            for row in self.store.list_opportunities()
            if row.status == "ELIGIBLE" and row.eligible_after <= now
        ]
        if self._next_eligible_at is not None and now < self._next_eligible_at:
            for row in candidates:
                self.store.record_skip(owner_token, row.opportunity_key, "COOLDOWN", now)
            return WatchCycleResult(
                lifecycle_status="IDLE",
                observed_keys=observed_keys,
                skipped_keys=tuple(row.opportunity_key for row in candidates),
                skip_reasons=("COOLDOWN",) if not candidates else tuple("COOLDOWN" for _ in candidates),
                evaluation_status=evaluation_status,
                evaluation_due_pending=False,
                recovery_gap_count=gap,
            )
        if not candidates:
            self.store.set_lifecycle(owner_token, "IDLE", now)
            return WatchCycleResult(
                lifecycle_status="IDLE",
                observed_keys=observed_keys,
                evaluation_status=evaluation_status,
                skip_reasons=("COOLDOWN",) if self._next_eligible_at is not None and now < self._next_eligible_at else (),
                evaluation_due_pending=False,
                recovery_gap_count=gap,
            )

        candidate = max(candidates, key=lambda row: row.anchor_timestamp)
        skipped_keys: list[str] = []
        skip_reasons: list[str] = []
        for row in candidates:
            if row.opportunity_key == candidate.opportunity_key:
                continue
            self.store.record_skip(owner_token, row.opportunity_key, "ANALYSIS_ALREADY_RUNNING", now)
            skipped_keys.append(row.opportunity_key)
            skip_reasons.append("ANALYSIS_ALREADY_RUNNING")
        try:
            with self.gate.acquire("probe"):
                probe_result = self.probe.probe(candidate.requested_symbol)
        except Exception as exc:
            reason = "MT5_UNAVAILABLE"
            self.store.record_skip(owner_token, candidate.opportunity_key, reason, now, str(exc))
            skipped_keys.append(candidate.opportunity_key)
            skip_reasons.append(reason)
            return WatchCycleResult(
                lifecycle_status="IDLE",
                observed_keys=observed_keys,
                skipped_keys=tuple(skipped_keys),
                skip_reasons=tuple(skip_reasons),
                evaluation_status=evaluation_status,
                evaluation_due_pending=False,
                recovery_gap_count=gap,
                error_code=reason,
            )
        try:
            run = self.store.claim_opportunity(
                owner_token,
                candidate.opportunity_key,
                now,
                resolved_symbol=probe_result.resolved_symbol,
            )
            future = self.executor.submit(lambda run=run: self._run_claimed(run))
        except Exception as exc:
            self.store.record_skip(owner_token, candidate.opportunity_key, "DB_LOCKED", now, str(exc))
            return WatchCycleResult(
                lifecycle_status="IDLE",
                observed_keys=observed_keys,
                skipped_keys=tuple(skipped_keys),
                skip_reasons=tuple(skip_reasons + ["DB_LOCKED"]),
                evaluation_status=evaluation_status,
                evaluation_due_pending=False,
                recovery_gap_count=gap,
                error_code="DB_LOCKED",
            )
        self._analysis_future = future
        self._active_run_id = run.run_id
        if self.heartbeat is not None:
            self.heartbeat.run_id = run.run_id
        self.store.set_lifecycle(owner_token, "ANALYZING", now)
        self.events.emit("ANALYSIS_STARTED", {"run_id": run.run_id, "opportunity_key": candidate.opportunity_key})
        return WatchCycleResult(
            lifecycle_status="ANALYZING",
            observed_keys=observed_keys,
            skipped_keys=tuple(skipped_keys),
            skip_reasons=tuple(skip_reasons),
            active_run_id=run.run_id,
            evaluation_status=evaluation_status,
            evaluation_due_pending=False,
            recovery_gap_count=gap,
        )

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.config.poll_interval_seconds)

    def request_shutdown(self) -> None:
        self.events.emit("SHUTDOWN_REQUESTED", {})
        self._stop.set()

    def shutdown(self) -> None:
        now = _require_aware_utc(self.clock.now(), "now")
        if self.owner_token is not None:
            token = self.owner_token
            if self._analysis_future is not None and not self._analysis_future.done():
                self._analysis_future.result(timeout=self.config.analysis_timeout_seconds)
                self.run_once(now)
            try:
                self._maybe_evaluate(now)
            except Exception:
                pass
            self.store.release_lease(token, now)
            self.events.emit("WATCHER_STOPPED", {})
            self.owner_token = None
        self.executor.shutdown(wait=True)


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
    "MarketProbeResult",
    "ReadOnlyMarketProbe",
    "Mt5OperationBusy",
    "SerializedMt5OperationGate",
    "SingleSlotAnalysisExecutor",
    "LeaseHeartbeat",
    "EvaluationRunEvidence",
    "EventSink",
    "WatchCycleResult",
    "OutcomeCoordinator",
    "WatcherCoordinator",
]
