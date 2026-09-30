"""Versioned SQLite persistence for Phase 12 shadow/replay artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .features import TickFeatures
from .models import FastAction, Tick

HFT_SCHEMA_VERSION = 1


def _utc(value: datetime, name: str = "timestamp") -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be UTC")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


class HftLeaseStatus(str, Enum):
    ACQUIRED = "ACQUIRED"
    HFT_ALREADY_RUNNING = "HFT_ALREADY_RUNNING"


@dataclass(frozen=True, slots=True)
class HftLeaseOwner:
    owner_token: str
    pid: int
    host: str
    process_started_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.owner_token, str) or not self.owner_token.strip():
            raise ValueError("owner_token must be non-empty")
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid <= 0:
            raise ValueError("pid must be positive")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be non-empty")
        object.__setattr__(self, "process_started_at", _utc(self.process_started_at, "process_started_at"))


@dataclass(frozen=True, slots=True)
class HftLeaseRecord:
    owner_token: str
    pid: int
    host: str
    process_started_at: datetime
    lease_acquired_at: datetime
    heartbeat_at: datetime
    lease_expires_at: datetime

    def __post_init__(self) -> None:
        for name in ("process_started_at", "lease_acquired_at", "heartbeat_at", "lease_expires_at"):
            object.__setattr__(self, name, _utc(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class HftLeaseResult:
    status: HftLeaseStatus
    owner_token: str | None = None
    recovered_expired: bool = False


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


class HftShadowStore:
    def __init__(self, path: str | Path, *, lease_ttl_seconds: int = 30) -> None:
        self.path = Path(path)
        if self.path.name in {"", ".", ".."}:
            raise ValueError("HFT artifact path must be a file")
        if isinstance(lease_ttl_seconds, bool) or not isinstance(lease_ttl_seconds, int) or lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be a positive integer")
        self.lease_ttl_seconds = lease_ttl_seconds

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS hft_meta (
                    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hft_lease (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    owner_token TEXT NOT NULL,
                    owner_pid INTEGER NOT NULL,
                    owner_host TEXT NOT NULL,
                    process_started_at TEXT NOT NULL,
                    lease_acquired_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    lease_expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hft_runs (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL CHECK (mode IN ('REPLAY','SHADOW')),
                    source_fingerprint TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0)
                );
                CREATE TABLE IF NOT EXISTS hft_plans (
                    plan_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES hft_runs(run_id),
                    symbol TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0)
                );
                CREATE TABLE IF NOT EXISTS hft_ticks (
                    run_id TEXT NOT NULL REFERENCES hft_runs(run_id),
                    tick_key TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    features_json TEXT NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0),
                    PRIMARY KEY (run_id, tick_key)
                );
                CREATE TABLE IF NOT EXISTS hft_actions (
                    action_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES hft_runs(run_id),
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    processing_ms REAL NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0)
                );
                CREATE TABLE IF NOT EXISTS hft_risk (
                    action_id TEXT PRIMARY KEY REFERENCES hft_actions(action_id),
                    accepted INTEGER NOT NULL CHECK (accepted IN (0,1)),
                    reason_code TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    risk_fraction REAL NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0)
                );
                CREATE TABLE IF NOT EXISTS hft_fills (
                    fill_id TEXT PRIMARY KEY,
                    action_id TEXT,
                    run_id TEXT NOT NULL REFERENCES hft_runs(run_id),
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    slippage_points REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0)
                );
                CREATE TABLE IF NOT EXISTS hft_positions (
                    position_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES hft_runs(run_id),
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0)
                );
                CREATE TABLE IF NOT EXISTS hft_account (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES hft_runs(run_id),
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0)
                );
                CREATE TABLE IF NOT EXISTS hft_run_health (
                    run_id TEXT PRIMARY KEY REFERENCES hft_runs(run_id),
                    dropped_ticks INTEGER NOT NULL DEFAULT 0,
                    stale_ticks INTEGER NOT NULL DEFAULT 0,
                    out_of_order_ticks INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    observed_at TEXT NOT NULL
                );
                """
            )
            row = db.execute("SELECT schema_version FROM hft_meta LIMIT 1").fetchone()
            if row is None:
                db.execute("INSERT INTO hft_meta VALUES (?, ?)", (HFT_SCHEMA_VERSION, _iso(datetime.now(timezone.utc))))
            elif row[0] != HFT_SCHEMA_VERSION:
                raise ValueError("unsupported HFT schema version")

    def _read_lease(self, db: sqlite3.Connection) -> HftLeaseRecord | None:
        row = db.execute(
            "SELECT owner_token, owner_pid, owner_host, process_started_at, lease_acquired_at, heartbeat_at, lease_expires_at FROM hft_lease WHERE singleton_id=1"
        ).fetchone()
        if row is None:
            return None
        return HftLeaseRecord(
            owner_token=row[0],
            pid=row[1],
            host=row[2],
            process_started_at=_parse(row[3]),
            lease_acquired_at=_parse(row[4]),
            heartbeat_at=_parse(row[5]),
            lease_expires_at=_parse(row[6]),
        )

    def read_only_active_lease(self, now: datetime | None = None) -> HftLeaseRecord | None:
        """Read an existing HFT lease without creating or mutating the artifact."""

        del now
        resolved = self.path.expanduser().resolve()
        if not resolved.is_file():
            return None
        uri = f"file:{resolved.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=0) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "hft_lease" not in tables:
                return None
            return self._read_lease(db)

    def acquire_lease(self, owner: HftLeaseOwner, now: datetime) -> HftLeaseResult:
        now = _utc(now, "now")
        self.initialize()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._read_lease(db)
            if current is not None and current.lease_expires_at > now:
                db.rollback()
                return HftLeaseResult(HftLeaseStatus.HFT_ALREADY_RUNNING, current.owner_token)
            expires = now + timedelta(seconds=self.lease_ttl_seconds)
            db.execute(
                "INSERT OR REPLACE INTO hft_lease(singleton_id,owner_token,owner_pid,owner_host,process_started_at,lease_acquired_at,heartbeat_at,lease_expires_at) VALUES(1,?,?,?,?,?,?,?)",
                (owner.owner_token, owner.pid, owner.host, _iso(owner.process_started_at), _iso(now), _iso(now), _iso(expires)),
            )
            db.commit()
            return HftLeaseResult(HftLeaseStatus.ACQUIRED, owner.owner_token, current is not None)

    def heartbeat_lease(self, owner_token: str, now: datetime) -> None:
        now = _utc(now, "now")
        expires = now + timedelta(seconds=self.lease_ttl_seconds)
        with self._connect() as db:
            updated = db.execute(
                "UPDATE hft_lease SET heartbeat_at=?, lease_expires_at=? WHERE singleton_id=1 AND owner_token=?",
                (_iso(now), _iso(expires), owner_token),
            ).rowcount
            if updated != 1:
                raise RuntimeError("HFT lease lost")

    def release_lease(self, owner_token: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM hft_lease WHERE singleton_id=1 AND owner_token=?", (owner_token,))

    def start_run(self, run_id: str, *, mode: str, source_fingerprint: str) -> None:
        if mode not in {"REPLAY", "SHADOW"} or not run_id or not source_fingerprint:
            raise ValueError("run metadata is invalid")
        self.initialize()
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO hft_runs(run_id,mode,source_fingerprint,started_at,status) VALUES(?,?,?,?,?)",
                (run_id, mode, source_fingerprint, _iso(datetime.now(timezone.utc)), "RUNNING"),
            )

    def close_run(self, run_id: str, *, status: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE hft_runs SET ended_at=?, status=? WHERE run_id=?", (_iso(datetime.now(timezone.utc)), status, run_id))

    def record_plan(self, run_id: str, plan: Mapping[str, Any]) -> None:
        if plan.get("executed", False) is not False:
            raise ValueError("plans must be shadow-only")
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO hft_plans(plan_id,run_id,symbol,created_at,expires_at,payload_json) VALUES(?,?,?,?,?,?)", (plan["plan_id"], run_id, plan["symbol"], plan["created_at"], plan["expires_at"], _json(dict(plan))))

    def record_tick(self, run_id: str, tick: Tick, *, features: Mapping[str, Any] | TickFeatures) -> None:
        if isinstance(features, TickFeatures):
            feature_payload = {name: getattr(features, name) for name in features.__dataclass_fields__}
        elif isinstance(features, Mapping):
            feature_payload = dict(features)
        else:
            raise TypeError("features must be TickFeatures or mapping")
        key = str(tick.sequence if tick.sequence is not None else tick.timestamp.isoformat())
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO hft_ticks(run_id,tick_key,symbol,timestamp,bid,ask,features_json) VALUES(?,?,?,?,?,?,?)", (run_id, key, tick.symbol, _iso(tick.timestamp), tick.bid, tick.ask, _json(feature_payload)))

    def record_action(self, run_id: str, action_id: str, tick: Tick, action: FastAction, reason: str, *, processing_ms: float) -> None:
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO hft_actions(action_id,run_id,timestamp,action,reason,processing_ms) VALUES(?,?,?,?,?,?)", (action_id, run_id, _iso(tick.timestamp), FastAction(action).value, str(reason), float(processing_ms)))

    def record_risk(self, action_id: str, risk: Mapping[str, Any]) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO hft_risk(action_id,accepted,reason_code,reason,risk_fraction) VALUES(?,?,?,?,?)", (action_id, int(bool(risk["accepted"])), risk["reason_code"], risk["reason"], float(risk.get("risk_fraction", 0.0))))

    def record_fill(self, run_id: str, fill: Mapping[str, Any]) -> None:
        if fill.get("executed", False) is not False:
            raise ValueError("fills must be shadow-only")
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO hft_fills(fill_id,action_id,run_id,symbol,timestamp,action,price,size,slippage_points,latency_ms) VALUES(?,?,?,?,?,?,?,?,?,?)", (fill["fill_id"], fill.get("action_id"), run_id, fill["symbol"], fill["timestamp"], fill["action"], fill["price"], fill["size"], fill.get("slippage_points", 0.0), fill.get("latency_ms", 0.0)))

    def record_position(self, run_id: str, position: Mapping[str, Any]) -> None:
        if position.get("executed", False) is not False:
            raise ValueError("positions must be shadow-only")
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO hft_positions(position_id,run_id,state,payload_json) VALUES(?,?,?,?)", (position["position_id"], run_id, position["state"], _json(dict(position))))

    def record_account(self, run_id: str, timestamp: datetime, payload: Mapping[str, Any]) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO hft_account(run_id,timestamp,payload_json) VALUES(?,?,?)", (run_id, _iso(timestamp), _json(dict(payload))))

    def record_run_health(
        self,
        run_id: str,
        *,
        dropped_ticks: int,
        stale_ticks: int,
        out_of_order_ticks: int,
        error_code: str | None = None,
    ) -> None:
        counters = (dropped_ticks, stale_ticks, out_of_order_ticks)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
            raise ValueError("tick-quality counters must be non-negative integers")
        if error_code is not None:
            error_code = str(error_code).strip().upper()[:80] or None
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO hft_run_health(
                    run_id,dropped_ticks,stale_ticks,out_of_order_ticks,error_code,observed_at
                ) VALUES(?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    dropped_ticks=excluded.dropped_ticks,
                    stale_ticks=excluded.stale_ticks,
                    out_of_order_ticks=excluded.out_of_order_ticks,
                    error_code=excluded.error_code,
                    observed_at=excluded.observed_at
                """,
                (
                    run_id,
                    dropped_ticks,
                    stale_ticks,
                    out_of_order_ticks,
                    error_code,
                    _iso(datetime.now(timezone.utc)),
                ),
            )

    def recover_open_positions(self) -> tuple[dict[str, Any], ...]:
        if not self.path.is_file():
            return ()
        with self._connect() as db:
            rows = db.execute("SELECT payload_json FROM hft_positions WHERE state IN ('LONG','SHORT','PENDING_ENTRY','PENDING_EXIT') ORDER BY position_id").fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def snapshot(self) -> dict[str, int]:
        self.initialize()
        with self._connect() as db:
            tables = ("runs", "ticks", "actions", "risk", "fills", "positions", "account")
            return {
                "schema_version": HFT_SCHEMA_VERSION,
                **{
                    name: int(db.execute(f"SELECT COUNT(*) FROM hft_{name}").fetchone()[0])
                    for name in tables
                },
            }


__all__ = [
    "HFT_SCHEMA_VERSION",
    "HftLeaseOwner",
    "HftLeaseRecord",
    "HftLeaseResult",
    "HftLeaseStatus",
    "HftShadowStore",
]
