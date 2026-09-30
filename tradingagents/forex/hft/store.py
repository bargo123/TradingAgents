"""Versioned SQLite persistence for Phase 12 shadow/replay artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .features import TickFeatures
from .models import FastAction, Tick

HFT_SCHEMA_VERSION = 1


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("timestamps must be timezone-aware UTC")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


class HftShadowStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.name in {"", ".", ".."}:
            raise ValueError("HFT artifact path must be a file")

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
                """
            )
            row = db.execute("SELECT schema_version FROM hft_meta LIMIT 1").fetchone()
            if row is None:
                db.execute("INSERT INTO hft_meta VALUES (?, ?)", (HFT_SCHEMA_VERSION, _iso(datetime.now(timezone.utc))))
            elif row[0] != HFT_SCHEMA_VERSION:
                raise ValueError("unsupported HFT schema version")

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


__all__ = ["HFT_SCHEMA_VERSION", "HftShadowStore"]
