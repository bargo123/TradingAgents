"""SQLite state, lease, and provenance storage for the forex shadow watcher."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from tradingagents.forex.shadow import ShadowDecisionStore, ShadowTradeDecision
    from tradingagents.forex.watcher import ScheduledOpportunity

_UTC = timezone.utc


def _utc(value: datetime, name: str = "timestamp") -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != _UTC.utcoffset(value):
        raise ValueError(f"{name} must be UTC")
    return value.astimezone(_UTC)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _parse(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return _utc(datetime.fromisoformat(text))


def _safe_text(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\x00", " ").strip()
    return text[:limit] if text else None


class LeaseStatus(Enum):
    ACQUIRED = "ACQUIRED"
    WATCHER_ALREADY_RUNNING = "WATCHER_ALREADY_RUNNING"
    WATCHER_OPERATOR_REVIEW_REQUIRED = "WATCHER_OPERATOR_REVIEW_REQUIRED"


class LeaseLostError(RuntimeError):
    """Raised when a stale owner attempts a fenced write."""


@dataclass(frozen=True, slots=True)
class LeaseOwner:
    owner_token: str
    pid: int
    host: str
    process_started_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.owner_token, str) or not self.owner_token.strip():
            raise ValueError("owner_token must be non-empty")
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid <= 0:
            raise ValueError("pid must be a positive integer")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be non-empty")
        object.__setattr__(self, "process_started_at", _utc(self.process_started_at))


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    host: str
    pid: int
    process_started_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "process_started_at", _utc(self.process_started_at))


class ProcessInspector(Protocol):
    def proves_alive(self, owner: LeaseOwner) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class LeaseRecord:
    owner_token: str
    pid: int
    host: str
    process_started_at: datetime
    lease_acquired_at: datetime
    heartbeat_at: datetime
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class LeaseResult:
    status: LeaseStatus
    owner_token: str | None = None
    recovered_expired: bool = False
    previous_owner_token: str | None = None


@dataclass(frozen=True, slots=True)
class WatchRun:
    run_id: str
    opportunity_key: str
    attempt_number: int
    owner_token: str
    run_status: str
    requested_symbol: str
    resolved_symbol: str | None
    started_at: datetime
    heartbeat_at: datetime | None
    runtime_alert_at: datetime | None
    completed_at: datetime | None
    decision_id: str | None
    source_run_id: str
    failure_code: str | None
    failure_detail: str | None
    decision_context_status: str | None
    normalization_status: str | None
    normalized_action: str | None
    analysis_snapshot_timestamp: datetime | None
    decision_completed_timestamp: datetime | None
    analysis_latency_seconds: float | None
    decision_reference_timestamp: datetime | None
    decision_reference_status: str | None
    decision_reference_delay_seconds: float | None
    stale_by_completion: bool | None
    freshness_budget_seconds: int | None
    llm_provider: str | None
    quick_model: str | None
    deep_model: str | None
    analyst_set_json: str
    analysis_profile: str
    prompt_config_version: str
    application_version: str
    git_commit: str | None
    collector_contract_version: str
    config_fingerprint: str
    safe_config_json: str
    runtime_seconds: float | None
    llm_calls: int | None
    tool_calls: int | None
    tokens_in: int | None
    tokens_out: int | None
    reasoning_tokens: int | None
    metrics_json: str | None


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
    decision_reference_status: str | None = None
    freshness_budget_seconds: int | None = None
    safe_config_json: str = "{}"
    metrics_json: str | None = None


class SkipReason(str, Enum):
    ANALYSIS_ALREADY_RUNNING = "ANALYSIS_ALREADY_RUNNING"
    COOLDOWN = "COOLDOWN"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    MT5_UNAVAILABLE = "MT5_UNAVAILABLE"
    SYMBOL_NOT_FOUND = "SYMBOL_NOT_FOUND"
    STALE_BUCKET = "STALE_BUCKET"
    WATCHER_NOT_RUNNING = "WATCHER_NOT_RUNNING"
    DB_LOCKED = "DB_LOCKED"
    INVALID_QUOTE = "INVALID_QUOTE"


@dataclass(frozen=True, slots=True)
class WatchOpportunity:
    opportunity_key: str
    requested_symbol: str
    analysis_profile: str
    analyst_set_json: str
    schedule_timeframe: str
    anchor_timestamp: datetime
    bar_close_timestamp: datetime
    eligible_after: datetime
    config_fingerprint: str
    status: str
    skip_reason: str | None
    skip_detail: str | None
    run_id: str | None
    decision_id: str | None
    attempt_count: int
    first_seen_at: datetime
    updated_at: datetime


class WatcherStore:
    """Small fenced SQLite store; every operation opens its own connection."""

    def __init__(
        self,
        path: str | Path,
        *,
        lease_ttl_seconds: int = 9000,
        busy_timeout_seconds: int = 5,
    ) -> None:
        self.path = Path(path)
        self.lease_ttl_seconds = int(lease_ttl_seconds)
        self.busy_timeout_seconds = int(busy_timeout_seconds)
        if self.lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        if self.busy_timeout_seconds <= 0:
            raise ValueError("busy_timeout_seconds must be positive")

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=self.busy_timeout_seconds)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_seconds * 1000}")
        return conn

    @contextmanager
    def _transaction(self, *, immediate: bool = False):
        conn = self._connect()
        try:
            if immediate:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._transaction(immediate=True) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS forex_watcher_state (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    lifecycle_status TEXT NOT NULL CHECK (
                        lifecycle_status IN ('STOPPED','STARTING','IDLE','CHECK_MARKET',
                                             'START_ANALYSIS','ANALYZING','DECISION_SAVED',
                                             'DEGRADED','STOPPING')
                    ),
                    owner_token TEXT,
                    owner_pid INTEGER,
                    owner_host TEXT,
                    process_started_at TEXT,
                    lease_acquired_at TEXT,
                    heartbeat_at TEXT,
                    lease_expires_at TEXT,
                    current_run_id TEXT,
                    current_opportunity_key TEXT,
                    last_loop_at TEXT,
                    last_analysis_completed_at TEXT,
                    next_eligible_at TEXT,
                    last_evaluation_at TEXT,
                    last_evaluation_status TEXT,
                    evaluation_due_pending INTEGER NOT NULL DEFAULT 0 CHECK (evaluation_due_pending IN (0,1)),
                    last_error_code TEXT,
                    last_error TEXT,
                    circuit_reason TEXT,
                    circuit_opened_at TEXT,
                    consecutive_mt5_failures INTEGER NOT NULL DEFAULT 0,
                    consecutive_analysis_failures INTEGER NOT NULL DEFAULT 0,
                    consecutive_incomplete_decisions INTEGER NOT NULL DEFAULT 0,
                    consecutive_normalization_failures INTEGER NOT NULL DEFAULT 0,
                    consecutive_runtime_exceeded INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forex_watch_opportunities (
                    opportunity_key TEXT PRIMARY KEY,
                    requested_symbol TEXT NOT NULL,
                    analysis_profile TEXT NOT NULL,
                    analyst_set_json TEXT NOT NULL,
                    schedule_timeframe TEXT NOT NULL CHECK (schedule_timeframe IN ('M5','M15','H1')),
                    anchor_timestamp TEXT NOT NULL,
                    bar_close_timestamp TEXT NOT NULL,
                    eligible_after TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('ELIGIBLE','RUNNING','DECISION_SAVED','SKIPPED','FAILED','ABANDONED')),
                    skip_reason TEXT,
                    skip_detail TEXT,
                    run_id TEXT,
                    decision_id TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                    first_seen_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (requested_symbol, analysis_profile, schedule_timeframe, anchor_timestamp, config_fingerprint)
                );
                CREATE TABLE IF NOT EXISTS forex_watch_runs (
                    run_id TEXT PRIMARY KEY,
                    opportunity_key TEXT NOT NULL REFERENCES forex_watch_opportunities(opportunity_key),
                    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
                    owner_token TEXT NOT NULL,
                    run_status TEXT NOT NULL CHECK (run_status IN ('RUNNING','SUCCEEDED','SUCCEEDED_SLOW','FAILED','ABANDONED')),
                    requested_symbol TEXT NOT NULL,
                    resolved_symbol TEXT,
                    started_at TEXT NOT NULL,
                    heartbeat_at TEXT,
                    runtime_alert_at TEXT,
                    completed_at TEXT,
                    decision_id TEXT,
                    source_run_id TEXT NOT NULL UNIQUE,
                    failure_code TEXT,
                    failure_detail TEXT,
                    decision_context_status TEXT CHECK (decision_context_status IS NULL OR decision_context_status IN ('COMPLETE','INCOMPLETE')),
                    normalization_status TEXT CHECK (normalization_status IS NULL OR normalization_status IN ('NORMALIZED','FAILED')),
                    normalized_action TEXT CHECK (normalized_action IS NULL OR normalized_action IN ('BUY','SELL','HOLD')),
                    analysis_snapshot_timestamp TEXT,
                    decision_completed_timestamp TEXT,
                    analysis_latency_seconds REAL,
                    decision_reference_timestamp TEXT,
                    decision_reference_status TEXT,
                    decision_reference_delay_seconds REAL,
                    stale_by_completion INTEGER CHECK (stale_by_completion IS NULL OR stale_by_completion IN (0,1)),
                    freshness_budget_seconds INTEGER,
                    llm_provider TEXT,
                    quick_model TEXT,
                    deep_model TEXT,
                    analyst_set_json TEXT NOT NULL,
                    analysis_profile TEXT NOT NULL,
                    prompt_config_version TEXT NOT NULL,
                    application_version TEXT NOT NULL,
                    git_commit TEXT,
                    collector_contract_version TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL,
                    safe_config_json TEXT NOT NULL,
                    runtime_seconds REAL,
                    llm_calls INTEGER,
                    tool_calls INTEGER,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    reasoning_tokens INTEGER,
                    metrics_json TEXT,
                    UNIQUE (opportunity_key, attempt_number)
                );
                CREATE INDEX IF NOT EXISTS idx_forex_watch_opportunities_status
                    ON forex_watch_opportunities(status, eligible_after);
                CREATE INDEX IF NOT EXISTS idx_forex_watch_runs_status
                    ON forex_watch_runs(run_status, started_at);
                """
            )
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(forex_watcher_state)").fetchall()
            }
            if "evaluation_due_pending" not in columns:
                conn.execute(
                    "ALTER TABLE forex_watcher_state ADD COLUMN evaluation_due_pending INTEGER NOT NULL DEFAULT 0"
                )
            now = _iso(datetime.now(_UTC))
            conn.execute(
                """
                INSERT OR IGNORE INTO forex_watcher_state
                    (singleton_id, lifecycle_status, updated_at, evaluation_due_pending)
                VALUES (1, 'STOPPED', ?, 0)
                """,
                (now,),
            )

    def table_names(self) -> tuple[str, ...]:
        self.initialize()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return tuple(row[0] for row in rows)

    def _read_lease_row(self, conn: sqlite3.Connection) -> LeaseRecord | None:
        row = conn.execute(
            "SELECT owner_token, owner_pid, owner_host, process_started_at, lease_acquired_at, heartbeat_at, lease_expires_at FROM forex_watcher_state WHERE singleton_id = 1"
        ).fetchone()
        if row is None or not row[0] or not row[6]:
            return None
        values = tuple(_parse(row[index]) for index in (3, 4, 5, 6))
        if any(value is None for value in values):
            return None
        return LeaseRecord(
            owner_token=row[0],
            pid=int(row[1]),
            host=row[2] or "",
            process_started_at=values[0],  # type: ignore[arg-type]
            lease_acquired_at=values[1],  # type: ignore[arg-type]
            heartbeat_at=values[2],  # type: ignore[arg-type]
            lease_expires_at=values[3],  # type: ignore[arg-type]
        )

    def active_lease(self, now: datetime | None = None) -> LeaseRecord | None:
        """Read the lease without renewing, taking, or probing MT5."""

        del now  # retained for callers that want an explicit observation time
        self.initialize()
        with self._connect() as conn:
            return self._read_lease_row(conn)

    def acquire_lease(
        self,
        owner: LeaseOwner,
        now: datetime,
        inspector: ProcessInspector | None = None,
    ) -> LeaseResult:
        now = _utc(now, "now")
        self.initialize()
        with self._transaction(immediate=True) as conn:
            current = self._read_lease_row(conn)
            # Expiry is authoritative.  Do not call the inspector while the
            # existing lease is still valid.
            if current is not None and current.lease_expires_at > now:
                return LeaseResult(
                    LeaseStatus.WATCHER_ALREADY_RUNNING,
                    owner_token=current.owner_token,
                    previous_owner_token=current.owner_token,
                )
            if current is not None and inspector is not None:
                old_owner = LeaseOwner(
                    owner_token=current.owner_token,
                    pid=current.pid,
                    host=current.host,
                    process_started_at=current.process_started_at,
                )
                if inspector.proves_alive(old_owner):
                    return LeaseResult(
                        LeaseStatus.WATCHER_OPERATOR_REVIEW_REQUIRED,
                        owner_token=current.owner_token,
                        previous_owner_token=current.owner_token,
                    )
            expires = now + timedelta(seconds=self.lease_ttl_seconds)
            conn.execute(
                """
                UPDATE forex_watcher_state SET
                    lifecycle_status='STARTING', owner_token=?, owner_pid=?, owner_host=?,
                    process_started_at=?, lease_acquired_at=?, heartbeat_at=?, lease_expires_at=?,
                    current_run_id=NULL, current_opportunity_key=NULL, last_error_code=NULL,
                    last_error=NULL, updated_at=?
                WHERE singleton_id=1
                """,
                (
                    owner.owner_token,
                    owner.pid,
                    owner.host,
                    _iso(owner.process_started_at),
                    _iso(now),
                    _iso(now),
                    _iso(expires),
                    _iso(now),
                ),
            )
            return LeaseResult(
                LeaseStatus.ACQUIRED,
                owner_token=owner.owner_token,
                recovered_expired=current is not None,
                previous_owner_token=None if current is None else current.owner_token,
            )

    def _require_owner(self, conn: sqlite3.Connection, owner_token: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM forex_watcher_state WHERE singleton_id=1"
        ).fetchone()
        if row is None or row["owner_token"] != owner_token:
            raise LeaseLostError("watcher lease is no longer owned by this process")
        return row

    def heartbeat(self, owner_token: str, now: datetime, *, run_id: str | None = None) -> None:
        now = _utc(now, "now")
        self.initialize()
        with self._transaction(immediate=True) as conn:
            self._require_owner(conn, owner_token)
            expires = now + timedelta(seconds=self.lease_ttl_seconds)
            if run_id is None:
                conn.execute(
                    "UPDATE forex_watcher_state SET heartbeat_at=?, lease_expires_at=?, last_loop_at=?, updated_at=? WHERE singleton_id=1",
                    (_iso(now), _iso(expires), _iso(now), _iso(now)),
                )
            else:
                changed = conn.execute(
                    "UPDATE forex_watch_runs SET heartbeat_at=? WHERE run_id=? AND owner_token=? AND run_status='RUNNING'",
                    (_iso(now), run_id, owner_token),
                ).rowcount
                if changed != 1:
                    raise LeaseLostError("active watcher run is no longer owned")
                conn.execute(
                    "UPDATE forex_watcher_state SET heartbeat_at=?, lease_expires_at=?, last_loop_at=?, updated_at=? WHERE singleton_id=1",
                    (_iso(now), _iso(expires), _iso(now), _iso(now)),
                )

    renew = heartbeat

    def set_lifecycle(self, owner_token: str, status: str, now: datetime) -> None:
        now = _utc(now, "now")
        self.initialize()
        with self._transaction(immediate=True) as conn:
            self._require_owner(conn, owner_token)
            conn.execute(
                "UPDATE forex_watcher_state SET lifecycle_status=?, updated_at=? WHERE singleton_id=1",
                (status, _iso(now)),
            )

    def release_lease(self, owner_token: str, now: datetime, *, status: str = "STOPPED") -> None:
        now = _utc(now, "now")
        self.initialize()
        with self._transaction(immediate=True) as conn:
            self._require_owner(conn, owner_token)
            conn.execute(
                """
                UPDATE forex_watcher_state SET lifecycle_status=?, owner_token=NULL,
                    owner_pid=NULL, owner_host=NULL, process_started_at=NULL,
                    lease_acquired_at=NULL, heartbeat_at=NULL, lease_expires_at=NULL,
                    current_run_id=NULL, current_opportunity_key=NULL, updated_at=?
                WHERE singleton_id=1
                """,
                (status, _iso(now)),
            )

    def mark_evaluation_due(self, owner_token: str, now: datetime) -> None:
        now = _utc(now, "now")
        self.initialize()
        with self._transaction(immediate=True) as conn:
            self._require_owner(conn, owner_token)
            conn.execute(
                "UPDATE forex_watcher_state SET evaluation_due_pending=1, updated_at=? WHERE singleton_id=1",
                (_iso(now),),
            )

    def clear_evaluation_due(self, owner_token: str, now: datetime, status: str) -> None:
        now = _utc(now, "now")
        self.initialize()
        with self._transaction(immediate=True) as conn:
            self._require_owner(conn, owner_token)
            conn.execute(
                "UPDATE forex_watcher_state SET evaluation_due_pending=0, last_evaluation_at=?, last_evaluation_status=?, updated_at=? WHERE singleton_id=1",
                (_iso(now), _safe_text(status, 80), _iso(now)),
            )

    def observe_opportunity(self, opportunity: ScheduledOpportunity, now: datetime) -> WatchOpportunity:
        now = _utc(now, "now")
        self.initialize()
        analyst_json = json.dumps([], separators=(",", ":"))
        with self._transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO forex_watch_opportunities
                    (opportunity_key, requested_symbol, analysis_profile, analyst_set_json,
                     schedule_timeframe, anchor_timestamp, bar_close_timestamp, eligible_after,
                     config_fingerprint, status, first_seen_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ELIGIBLE', ?, ?)
                """,
                (
                    opportunity.opportunity_key,
                    opportunity.requested_symbol,
                    opportunity.analysis_profile,
                    analyst_json,
                    opportunity.schedule_timeframe,
                    _iso(opportunity.anchor_timestamp),
                    _iso(opportunity.bar_close_timestamp),
                    _iso(opportunity.eligible_after),
                    opportunity.config_fingerprint,
                    _iso(now),
                    _iso(now),
                ),
            )
        return self.get_opportunity(opportunity.opportunity_key)

    def get_opportunity(self, opportunity_key: str) -> WatchOpportunity:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM forex_watch_opportunities WHERE opportunity_key=?",
                (opportunity_key,),
            ).fetchone()
        if row is None:
            raise KeyError(opportunity_key)
        return self._row_to_opportunity(row)

    def list_opportunities(self) -> list[WatchOpportunity]:
        self.initialize()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forex_watch_opportunities ORDER BY anchor_timestamp, requested_symbol, opportunity_key"
            ).fetchall()
        return [self._row_to_opportunity(row) for row in rows]

    def list_skipped(self) -> list[WatchOpportunity]:
        return [row for row in self.list_opportunities() if row.status == "SKIPPED"]

    def record_skip(
        self,
        owner_token: str,
        opportunity_key: str,
        reason: str | SkipReason,
        now: datetime,
        detail: str | None = None,
    ) -> None:
        now = _utc(now, "now")
        reason_value = reason.value if isinstance(reason, SkipReason) else str(reason)
        self.initialize()
        with self._transaction(immediate=True) as conn:
            self._require_owner(conn, owner_token)
            conn.execute(
                """
                UPDATE forex_watch_opportunities
                   SET status='SKIPPED', skip_reason=?, skip_detail=?, updated_at=?
                 WHERE opportunity_key=? AND status='ELIGIBLE'
                """,
                (reason_value, _safe_text(detail), _iso(now), opportunity_key),
            )

    def claim_opportunity(
        self,
        owner_token: str,
        opportunity_key: str,
        now: datetime,
        *,
        source_run_id: str | None = None,
        resolved_symbol: str | None = None,
    ) -> WatchRun:
        now = _utc(now, "now")
        source_run_id = source_run_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        self.initialize()
        with self._transaction(immediate=True) as conn:
            self._require_owner(conn, owner_token)
            row = conn.execute(
                "SELECT * FROM forex_watch_opportunities WHERE opportunity_key=?",
                (opportunity_key,),
            ).fetchone()
            if row is None:
                raise KeyError(opportunity_key)
            if row["status"] != "ELIGIBLE":
                raise RuntimeError(f"opportunity is not eligible: {row['status']}")
            attempt = int(row["attempt_count"]) + 1
            conn.execute(
                "UPDATE forex_watch_opportunities SET status='RUNNING', attempt_count=?, run_id=?, updated_at=? WHERE opportunity_key=? AND status='ELIGIBLE'",
                (attempt, run_id, _iso(now), opportunity_key),
            )
            conn.execute(
                "INSERT INTO forex_watch_runs (run_id, opportunity_key, attempt_number, owner_token, run_status, requested_symbol, resolved_symbol, started_at, source_run_id, analyst_set_json, analysis_profile, prompt_config_version, application_version, collector_contract_version, config_fingerprint, safe_config_json) VALUES (?, ?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    opportunity_key,
                    attempt,
                    owner_token,
                    row["requested_symbol"],
                    resolved_symbol,
                    _iso(now),
                    source_run_id,
                    row["analyst_set_json"],
                    row["analysis_profile"],
                    "forex-shadow.v1",
                    "unknown",
                    "forex-watch.v1",
                    row["config_fingerprint"],
                    "{}",
                ),
            )
            conn.execute(
                "UPDATE forex_watcher_state SET lifecycle_status='START_ANALYSIS', current_run_id=?, current_opportunity_key=?, updated_at=? WHERE singleton_id=1",
                (run_id, opportunity_key, _iso(now)),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> WatchRun:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM forex_watch_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._row_to_run(row)

    def list_runs(self) -> list[WatchRun]:
        self.initialize()
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM forex_watch_runs ORDER BY started_at, run_id").fetchall()
        return [self._row_to_run(row) for row in rows]

    def finalize_run(
        self,
        owner_token: str,
        run_id: str,
        now: datetime,
        *,
        status: str,
        evidence: RunEvidence | None = None,
        failure_code: str | None = None,
        failure_detail: str | None = None,
    ) -> WatchRun:
        now = _utc(now, "now")
        self.initialize()
        with self._transaction(immediate=True) as conn:
            self._require_owner(conn, owner_token)
            row = conn.execute("SELECT * FROM forex_watch_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["run_status"] != "RUNNING":
                return self._row_to_run(row)
            decision_id = None if evidence is None else evidence.decision_id
            run_status = status
            sets = [
                "run_status=?", "completed_at=?", "decision_id=?", "failure_code=?", "failure_detail=?"
            ]
            values: list[Any] = [
                run_status,
                _iso(now),
                decision_id,
                _safe_text(failure_code, 100),
                _safe_text(failure_detail),
            ]
            if evidence is not None:
                updates = {
                    "resolved_symbol": evidence.resolved_symbol,
                    "decision_context_status": evidence.decision_context_status,
                    "normalization_status": evidence.normalization_status,
                    "normalized_action": evidence.normalized_action,
                    "analysis_snapshot_timestamp": None if evidence.analysis_snapshot_timestamp is None else _iso(evidence.analysis_snapshot_timestamp),
                    "decision_completed_timestamp": None if evidence.decision_completed_timestamp is None else _iso(evidence.decision_completed_timestamp),
                    "analysis_latency_seconds": evidence.analysis_latency_seconds,
                    "decision_reference_timestamp": None if evidence.decision_reference_timestamp is None else _iso(evidence.decision_reference_timestamp),
                    "decision_reference_status": evidence.decision_reference_status,
                    "decision_reference_delay_seconds": evidence.decision_reference_delay_seconds,
                    "stale_by_completion": None if evidence.stale_by_completion is None else int(evidence.stale_by_completion),
                    "freshness_budget_seconds": evidence.freshness_budget_seconds,
                    "llm_provider": evidence.llm_provider,
                    "quick_model": evidence.quick_model,
                    "deep_model": evidence.deep_model,
                    "safe_config_json": evidence.safe_config_json,
                    "runtime_seconds": evidence.runtime_seconds,
                    "llm_calls": evidence.llm_calls,
                    "tool_calls": evidence.tool_calls,
                    "tokens_in": evidence.tokens_in,
                    "tokens_out": evidence.tokens_out,
                    "reasoning_tokens": evidence.reasoning_tokens,
                    "metrics_json": evidence.metrics_json,
                }
                for key, value in updates.items():
                    sets.append(f"{key}=?")
                    values.append(value)
            values.append(run_id)
            conn.execute(
                f"UPDATE forex_watch_runs SET {', '.join(sets)} WHERE run_id=? AND run_status='RUNNING'",
                values,
            )
            opportunity_status = "DECISION_SAVED" if decision_id else ("ABANDONED" if status == "ABANDONED" else "FAILED")
            conn.execute(
                "UPDATE forex_watch_opportunities SET status=?, decision_id=?, updated_at=? WHERE opportunity_key=?",
                (opportunity_status, decision_id, _iso(now), row["opportunity_key"]),
            )
            conn.execute(
                "UPDATE forex_watcher_state SET lifecycle_status=?, current_run_id=NULL, current_opportunity_key=NULL, last_analysis_completed_at=?, updated_at=? WHERE singleton_id=1",
                ("DECISION_SAVED" if decision_id else "IDLE", _iso(now), _iso(now)),
            )
        return self.get_run(run_id)

    def mark_runtime_alert(self, owner_token: str, run_id: str, now: datetime) -> None:
        now = _utc(now, "now")
        self.initialize()
        with self._transaction(immediate=True) as conn:
            self._require_owner(conn, owner_token)
            conn.execute(
                "UPDATE forex_watch_runs SET runtime_alert_at=COALESCE(runtime_alert_at, ?) WHERE run_id=? AND run_status='RUNNING'",
                (_iso(now), run_id),
            )

    def set_error(self, owner_token: str, code: str, detail: str | None, now: datetime) -> None:
        now = _utc(now, "now")
        self.initialize()
        with self._transaction(immediate=True) as conn:
            self._require_owner(conn, owner_token)
            conn.execute(
                "UPDATE forex_watcher_state SET last_error_code=?, last_error=?, updated_at=? WHERE singleton_id=1",
                (_safe_text(code, 100), _safe_text(detail), _iso(now)),
            )

    def summary(self, now: datetime | None = None) -> dict[str, Any]:
        del now
        self.initialize()
        with self._connect() as conn:
            state = conn.execute("SELECT * FROM forex_watcher_state WHERE singleton_id=1").fetchone()
            counts = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM forex_watch_opportunities GROUP BY status"
                ).fetchall()
            }
            run_counts = {
                row["run_status"]: row["count"]
                for row in conn.execute(
                    "SELECT run_status, COUNT(*) AS count FROM forex_watch_runs GROUP BY run_status"
                ).fetchall()
            }
        result = {key: state[key] for key in state.keys()} if state is not None else {}
        result["evaluation_due_pending"] = bool(result.get("evaluation_due_pending", 0))
        result["opportunity_counts"] = counts
        result["run_counts"] = run_counts
        result["database_path"] = str(self.path)
        return result

    def force_expiry(self, expiry: datetime) -> None:
        self.initialize()
        with self._transaction(immediate=True) as conn:
            conn.execute(
                "UPDATE forex_watcher_state SET lease_expires_at=?, updated_at=? WHERE singleton_id=1",
                (_iso(expiry), _iso(expiry)),
            )

    def circuit_reason(self) -> str | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT circuit_reason FROM forex_watcher_state WHERE singleton_id=1").fetchone()
        return None if row is None else row[0]

    def list_decisions(self) -> list[ShadowTradeDecision]:
        from tradingagents.forex.shadow import ShadowDecisionStore

        store = ShadowDecisionStore(self.path)
        store.initialize()
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT decision_id FROM shadow_decisions ORDER BY created_at, decision_id").fetchall()
        return [store.get(row[0]) for row in rows]

    def decision_for_run(self, run_id: str) -> ShadowTradeDecision:
        run = self.get_run(run_id)
        if not run.decision_id:
            raise KeyError(run_id)
        from tradingagents.forex.shadow import ShadowDecisionStore

        return ShadowDecisionStore(self.path).get(run.decision_id)

    def _row_to_opportunity(self, row: sqlite3.Row) -> WatchOpportunity:
        return WatchOpportunity(
            opportunity_key=row["opportunity_key"],
            requested_symbol=row["requested_symbol"],
            analysis_profile=row["analysis_profile"],
            analyst_set_json=row["analyst_set_json"],
            schedule_timeframe=row["schedule_timeframe"],
            anchor_timestamp=_parse(row["anchor_timestamp"]),  # type: ignore[arg-type]
            bar_close_timestamp=_parse(row["bar_close_timestamp"]),  # type: ignore[arg-type]
            eligible_after=_parse(row["eligible_after"]),  # type: ignore[arg-type]
            config_fingerprint=row["config_fingerprint"],
            status=row["status"],
            skip_reason=row["skip_reason"],
            skip_detail=row["skip_detail"],
            run_id=row["run_id"],
            decision_id=row["decision_id"],
            attempt_count=int(row["attempt_count"]),
            first_seen_at=_parse(row["first_seen_at"]),  # type: ignore[arg-type]
            updated_at=_parse(row["updated_at"]),  # type: ignore[arg-type]
        )

    def _row_to_run(self, row: sqlite3.Row) -> WatchRun:
        dt_fields = (
            "started_at", "heartbeat_at", "runtime_alert_at", "completed_at",
            "analysis_snapshot_timestamp", "decision_completed_timestamp",
            "decision_reference_timestamp",
        )
        values = {field: _parse(row[field]) for field in dt_fields}
        return WatchRun(
            run_id=row["run_id"], opportunity_key=row["opportunity_key"],
            attempt_number=int(row["attempt_number"]), owner_token=row["owner_token"],
            run_status=row["run_status"], requested_symbol=row["requested_symbol"],
            resolved_symbol=row["resolved_symbol"], started_at=values["started_at"],
            heartbeat_at=values["heartbeat_at"], runtime_alert_at=values["runtime_alert_at"],
            completed_at=values["completed_at"], decision_id=row["decision_id"],
            source_run_id=row["source_run_id"], failure_code=row["failure_code"],
            failure_detail=row["failure_detail"], decision_context_status=row["decision_context_status"],
            normalization_status=row["normalization_status"], normalized_action=row["normalized_action"],
            analysis_snapshot_timestamp=values["analysis_snapshot_timestamp"],
            decision_completed_timestamp=values["decision_completed_timestamp"],
            analysis_latency_seconds=row["analysis_latency_seconds"],
            decision_reference_timestamp=values["decision_reference_timestamp"],
            decision_reference_status=row["decision_reference_status"],
            decision_reference_delay_seconds=row["decision_reference_delay_seconds"],
            stale_by_completion=None if row["stale_by_completion"] is None else bool(row["stale_by_completion"]),
            freshness_budget_seconds=row["freshness_budget_seconds"],
            llm_provider=row["llm_provider"], quick_model=row["quick_model"], deep_model=row["deep_model"],
            analyst_set_json=row["analyst_set_json"], analysis_profile=row["analysis_profile"],
            prompt_config_version=row["prompt_config_version"], application_version=row["application_version"],
            git_commit=row["git_commit"], collector_contract_version=row["collector_contract_version"],
            config_fingerprint=row["config_fingerprint"], safe_config_json=row["safe_config_json"],
            runtime_seconds=row["runtime_seconds"], llm_calls=row["llm_calls"], tool_calls=row["tool_calls"],
            tokens_in=row["tokens_in"], tokens_out=row["tokens_out"], reasoning_tokens=row["reasoning_tokens"],
            metrics_json=row["metrics_json"],
        )


__all__ = [
    "LeaseOwner", "LeaseStatus", "LeaseResult", "LeaseRecord", "ProcessIdentity",
    "ProcessInspector", "LeaseLostError", "WatchRun", "RunEvidence", "WatchOpportunity",
    "SkipReason", "WatcherStore",
]
