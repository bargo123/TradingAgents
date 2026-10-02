"""Separate transactional SQLite ledger for Phase 12D DEMO execution."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .demo_models import DemoAccountSnapshot, DemoOrderIntent, ExecutionMode

DEMO_SCHEMA_VERSION = 1


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _iso(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))


class DemoExecutionStore:
    """Own only DEMO execution artifacts; never opens the shadow database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        return db

    def initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS demo_meta(
                    singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
                    schema_version INTEGER NOT NULL CHECK(schema_version=1),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS demo_runs(
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    git_commit TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                CREATE TABLE IF NOT EXISTS demo_account_snapshots(
                    snapshot_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES demo_runs(run_id),
                    observed_at TEXT NOT NULL,
                    login INTEGER,
                    server TEXT,
                    company TEXT,
                    currency TEXT,
                    trade_mode INTEGER,
                    balance REAL,
                    equity REAL,
                    margin REAL,
                    free_margin REAL,
                    margin_level REAL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                CREATE TABLE IF NOT EXISTS demo_order_intents(
                    intent_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES demo_runs(run_id),
                    plan_id TEXT NOT NULL,
                    source_decision_id TEXT NOT NULL,
                    source_run_id TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('LONG','SHORT')),
                    volume REAL NOT NULL CHECK(volume>0),
                    requested_price REAL NOT NULL CHECK(requested_price>0),
                    stop_loss REAL NOT NULL CHECK(stop_loss>0),
                    take_profit REAL NOT NULL CHECK(take_profit>0),
                    deviation_points INTEGER NOT NULL CHECK(deviation_points>=0),
                    created_at TEXT NOT NULL,
                    git_commit TEXT NOT NULL,
                    account_trade_mode INTEGER NOT NULL,
                    broker_order_sent INTEGER NOT NULL DEFAULT 0 CHECK(broker_order_sent IN (0,1)),
                    request_payload TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                CREATE TABLE IF NOT EXISTS demo_signals(
                    signal_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES demo_runs(run_id),
                    tick_timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    plan_id TEXT,
                    payload_json TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                CREATE TABLE IF NOT EXISTS demo_plans(
                    plan_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES demo_runs(run_id),
                    symbol TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                CREATE TABLE IF NOT EXISTS demo_risk_decisions(
                    action_id TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL REFERENCES demo_signals(signal_id),
                    accepted INTEGER NOT NULL CHECK(accepted IN (0,1)),
                    reason_code TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    risk_fraction REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                CREATE TABLE IF NOT EXISTS demo_order_results(
                    intent_id TEXT PRIMARY KEY REFERENCES demo_order_intents(intent_id),
                    classification TEXT NOT NULL,
                    retcode INTEGER,
                    order_ticket INTEGER,
                    deal_ticket INTEGER,
                    fill_price REAL,
                    fill_volume REAL,
                    broker_comment TEXT,
                    broker_timestamp TEXT,
                    request_payload TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    broker_order_sent INTEGER NOT NULL CHECK(broker_order_sent IN (0,1)),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                CREATE TABLE IF NOT EXISTS demo_positions(
                    ticket INTEGER PRIMARY KEY,
                    intent_id TEXT NOT NULL REFERENCES demo_order_intents(intent_id),
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('LONG','SHORT')),
                    volume REAL NOT NULL CHECK(volume>0),
                    price_open REAL NOT NULL CHECK(price_open>0),
                    stop_loss REAL,
                    take_profit REAL,
                    state TEXT NOT NULL,
                    owned INTEGER NOT NULL CHECK(owned=1),
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                CREATE TABLE IF NOT EXISTS demo_exits(
                    exit_id TEXT PRIMARY KEY,
                    ticket INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    intent_id TEXT,
                    classification TEXT,
                    retcode INTEGER,
                    deal_ticket INTEGER,
                    fill_price REAL,
                    fill_volume REAL,
                    realized_pnl REAL,
                    payload_json TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                CREATE TABLE IF NOT EXISTS demo_reconciliation(
                    event_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                CREATE TABLE IF NOT EXISTS demo_control_requests(
                    request_id TEXT PRIMARY KEY,
                    command TEXT NOT NULL CHECK(command='RECONCILE_DEMO_POSITION'),
                    ticket INTEGER NOT NULL CHECK(ticket>0),
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('PENDING','CLAIMED','COMPLETED','FAILED')),
                    requested_at TEXT NOT NULL,
                    claimed_at TEXT,
                    claimed_by TEXT,
                    completed_at TEXT,
                    result_json TEXT
                );
                CREATE TABLE IF NOT EXISTS demo_circuit_state(
                    singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
                    status TEXT NOT NULL,
                    daily_start_equity REAL,
                    daily_loss_fraction REAL NOT NULL DEFAULT 0,
                    consecutive_losses INTEGER NOT NULL DEFAULT 0,
                    reason TEXT,
                    updated_at TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK(execution_mode='DEMO'),
                    real_money INTEGER NOT NULL CHECK(real_money=0)
                );
                """
            )
            db.execute(
                "INSERT OR IGNORE INTO demo_meta(singleton_id,schema_version,created_at) VALUES(1,?,?)",
                (DEMO_SCHEMA_VERSION, _iso(datetime.now(timezone.utc))),
            )
            db.execute(
                """
                INSERT OR IGNORE INTO demo_circuit_state(
                    singleton_id,status,updated_at,execution_mode,real_money
                ) VALUES(1,'READY',?,'DEMO',0)
                """,
                (_iso(datetime.now(timezone.utc)),),
            )

    def integrity_check(self) -> str:
        self.initialize()
        with self._connect() as db:
            return str(db.execute("PRAGMA integrity_check").fetchone()[0])

    def start_run(self, run_id: str, *, git_commit: str) -> None:
        self.initialize()
        with self._connect() as db:
            db.execute(
                """
                INSERT OR IGNORE INTO demo_runs(
                    run_id,started_at,status,git_commit,execution_mode,real_money
                ) VALUES(?,?,? ,?,'DEMO',0)
                """,
                (str(run_id), _iso(datetime.now(timezone.utc)), "RUNNING", str(git_commit)),
            )

    def close_run(self, run_id: str, *, status: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE demo_runs SET ended_at=?,status=? WHERE run_id=?",
                (_iso(datetime.now(timezone.utc)), str(status), str(run_id)),
            )

    def record_account_snapshot(self, run_id: str, account: DemoAccountSnapshot) -> str:
        if not isinstance(account, DemoAccountSnapshot):
            raise TypeError("account must be DemoAccountSnapshot")
        if account.trade_mode is None:
            raise ValueError("DEMO account snapshot requires authoritative trade_mode")
        snapshot_id = str(uuid.uuid4())
        observed = account.observed_at or datetime.now(timezone.utc)
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO demo_account_snapshots(
                    snapshot_id,run_id,observed_at,login,server,company,currency,
                    trade_mode,balance,equity,margin,free_margin,margin_level,
                    execution_mode,real_money
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'DEMO',0)
                """,
                (
                    snapshot_id,
                    str(run_id),
                    _iso(observed),
                    account.login,
                    account.server,
                    account.company,
                    account.currency,
                    account.trade_mode,
                    account.balance,
                    account.equity,
                    account.margin,
                    account.free_margin,
                    account.margin_level,
                ),
            )
        return snapshot_id

    def record_order_intent(self, run_id: str, intent: DemoOrderIntent) -> None:
        if not isinstance(intent, DemoOrderIntent):
            raise TypeError("intent must be DemoOrderIntent")
        if intent.execution_mode is not ExecutionMode.DEMO:
            raise ValueError("DEMO ledger accepts only execution_mode=DEMO")
        if intent.account_trade_mode is None:
            raise ValueError("DEMO order intent requires authoritative account trade mode")
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO demo_order_intents(
                    intent_id,run_id,plan_id,source_decision_id,source_run_id,
                    strategy_id,symbol,direction,volume,requested_price,stop_loss,
                    take_profit,deviation_points,created_at,git_commit,
                    account_trade_mode,request_payload,execution_mode,real_money
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'DEMO',0)
                ON CONFLICT(intent_id) DO NOTHING
                """,
                (
                    intent.intent_id,
                    str(run_id),
                    intent.plan_id,
                    intent.source_decision_id,
                    intent.source_run_id,
                    intent.strategy_id,
                    intent.symbol,
                    intent.direction,
                    intent.volume,
                    intent.requested_price,
                    intent.stop_loss,
                    intent.take_profit,
                    intent.deviation_points,
                    _iso(intent.created_at),
                    intent.git_commit,
                    intent.account_trade_mode,
                    _json(intent),
                ),
            )

    def record_signal(
        self,
        *,
        run_id: str,
        signal_id: str,
        tick_timestamp: datetime,
        symbol: str,
        action: str,
        reason: str,
        plan_id: str | None,
        payload: Mapping[str, Any],
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT OR IGNORE INTO demo_signals(
                    signal_id,run_id,tick_timestamp,symbol,action,reason,plan_id,
                    payload_json,execution_mode,real_money
                ) VALUES(?,?,?,?,?,?,?,?, 'DEMO',0)
                """,
                (
                    str(signal_id),
                    str(run_id),
                    _iso(tick_timestamp),
                    str(symbol).upper(),
                    str(action),
                    str(reason),
                    None if plan_id is None else str(plan_id),
                    _json(payload),
                ),
            )

    def record_plan(self, run_id: str, plan: Any) -> None:
        """Persist the validated plan snapshot separately from shadow data."""

        plan_id = str(getattr(plan, "plan_id", "")).strip()
        symbol = str(getattr(plan, "symbol", "")).strip().upper()
        created_at = getattr(plan, "created_at", None)
        expires_at = getattr(plan, "expires_at", None)
        if not plan_id or not symbol or not isinstance(created_at, datetime) or not isinstance(expires_at, datetime):
            raise ValueError("DEMO plan metadata is invalid")
        with self._connect() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO demo_plans(
                    plan_id,run_id,symbol,created_at,expires_at,payload_json,execution_mode,real_money
                ) VALUES(?,?,?,?,?,?, 'DEMO',0)
                """,
                (plan_id, str(run_id), symbol, _iso(created_at), _iso(expires_at), _json(plan)),
            )

    def record_risk_decision(
        self,
        *,
        action_id: str,
        signal_id: str,
        accepted: bool,
        reason_code: str,
        reason: str,
        risk_fraction: float,
        payload: Mapping[str, Any],
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO demo_risk_decisions(
                    action_id,signal_id,accepted,reason_code,reason,risk_fraction,
                    payload_json,execution_mode,real_money
                ) VALUES(?,?,?,?,?,?,?,'DEMO',0)
                """,
                (
                    str(action_id),
                    str(signal_id),
                    int(bool(accepted)),
                    str(reason_code),
                    str(reason),
                    float(risk_fraction),
                    _json(payload),
                ),
            )

    def record_order_result(
        self,
        *,
        intent_id: str,
        classification: str,
        retcode: int | None,
        order_ticket: int | None,
        deal_ticket: int | None,
        fill_price: float | None,
        fill_volume: float | None,
        broker_comment: str | None,
        broker_timestamp: datetime | None,
        request_payload: Mapping[str, Any],
        broker_order_sent: bool = True,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO demo_order_results(
                    intent_id,classification,retcode,order_ticket,deal_ticket,
                    fill_price,fill_volume,broker_comment,broker_timestamp,
                    request_payload,observed_at,execution_mode,broker_order_sent,real_money
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,'DEMO',?,0)
                """,
                (
                    str(intent_id),
                    str(classification).upper(),
                    retcode,
                    order_ticket,
                    deal_ticket,
                    fill_price,
                    fill_volume,
                    None if broker_comment is None else str(broker_comment)[:500],
                    None if broker_timestamp is None else _iso(broker_timestamp),
                    _json(request_payload),
                    _iso(datetime.now(timezone.utc)),
                    int(bool(broker_order_sent)),
                ),
            )
            db.execute(
                "UPDATE demo_order_intents SET broker_order_sent=? WHERE intent_id=?",
                (int(bool(broker_order_sent)), str(intent_id)),
            )

    def read_order_intent(self, intent_id: str) -> dict[str, Any] | None:
        """Read one local intent for owner-side broker reconciliation."""

        self.initialize()
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM demo_order_intents WHERE intent_id=?",
                (str(intent_id),),
            ).fetchone()
            return None if row is None else dict(row)

    def read_order_results(self, intent_id: str) -> tuple[dict[str, Any], ...]:
        """Read broker result identities associated with one local intent."""

        self.initialize()
        with self._connect() as db:
            return tuple(
                dict(row)
                for row in db.execute(
                    "SELECT * FROM demo_order_results WHERE intent_id=?",
                    (str(intent_id),),
                ).fetchall()
            )

    def record_position(self, position: Mapping[str, Any]) -> None:
        required = ("ticket", "intent_id", "symbol", "direction", "volume", "price_open", "state")
        if any(key not in position for key in required):
            raise ValueError("position is missing required fields")
        if str(position["direction"]).upper() not in {"LONG", "SHORT"}:
            raise ValueError("position direction is invalid")
        with self._connect() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO demo_positions(
                    ticket,intent_id,symbol,direction,volume,price_open,stop_loss,
                    take_profit,state,owned,payload_json,updated_at,execution_mode,real_money
                ) VALUES(?,?,?,?,?,?,?,?,?,1,?,?,'DEMO',0)
                """,
                (
                    int(position["ticket"]),
                    str(position["intent_id"]),
                    str(position["symbol"]).upper(),
                    str(position["direction"]).upper(),
                    float(position["volume"]),
                    float(position["price_open"]),
                    position.get("stop_loss"),
                    position.get("take_profit"),
                    str(position["state"]).upper(),
                    _json(dict(position)),
                    _iso(datetime.now(timezone.utc)),
                ),
            )

    def read_owned_positions(self, symbol: str | None = None) -> tuple[dict[str, Any], ...]:
        self.initialize()
        query = "SELECT * FROM demo_positions WHERE owned=1"
        params: tuple[Any, ...] = ()
        if symbol is not None:
            query += " AND symbol=?"
            params = (str(symbol).upper(),)
        query += " ORDER BY ticket"
        with self._connect() as db:
            return tuple(dict(row) for row in db.execute(query, params).fetchall())

    def record_exit(self, *, ticket: int, reason: str, payload: Mapping[str, Any], realized_pnl: float | None = None) -> str:
        exit_id = str(uuid.uuid4())
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO demo_exits(
                    exit_id,ticket,reason,realized_pnl,payload_json,observed_at,execution_mode,real_money
                ) VALUES(?,?,?,?,?,?,'DEMO',0)
                """,
                (exit_id, int(ticket), str(reason), realized_pnl, _json(payload), _iso(datetime.now(timezone.utc))),
            )
        return exit_id

    def record_reconciliation(self, *, status: str, reason: str, details: Mapping[str, Any]) -> str:
        event_id = str(uuid.uuid4())
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO demo_reconciliation(
                    event_id,status,reason,details_json,observed_at,execution_mode,real_money
                ) VALUES(?,?,?,?,?,'DEMO',0)
                """,
                (event_id, str(status), str(reason), _json(details), _iso(datetime.now(timezone.utc))),
            )
        return event_id

    def request_reconciliation(self, *, ticket: int, symbol: str) -> str:
        """Enqueue an owner-side reconciliation request without touching MT5."""

        if isinstance(ticket, bool) or not isinstance(ticket, int) or ticket <= 0:
            raise ValueError("ticket must be a positive integer")
        symbol = str(symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol must be non-empty")
        request_id = str(uuid.uuid4())
        self.initialize()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO demo_control_requests(
                    request_id,command,ticket,symbol,status,requested_at
                ) VALUES(?,?,?,?,'PENDING',?)
                """,
                (request_id, "RECONCILE_DEMO_POSITION", ticket, symbol, _iso(datetime.now(timezone.utc))),
            )
        return request_id

    def claim_control_request(self, *, claimed_by: str) -> dict[str, Any] | None:
        """Atomically claim the oldest pending request for the owner process."""

        claimed_by = str(claimed_by).strip()
        if not claimed_by:
            raise ValueError("claimed_by must be non-empty")
        self.initialize()
        now = _iso(datetime.now(timezone.utc))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT * FROM demo_control_requests
                 WHERE status='PENDING'
                 ORDER BY requested_at, request_id
                 LIMIT 1
                """
            ).fetchone()
            if row is None:
                db.rollback()
                return None
            updated = db.execute(
                """
                UPDATE demo_control_requests
                   SET status='CLAIMED',claimed_at=?,claimed_by=?
                 WHERE request_id=? AND status='PENDING'
                """,
                (now, claimed_by, row["request_id"]),
            ).rowcount
            if updated != 1:
                db.rollback()
                return None
            claimed = dict(row)
            claimed.update(status="CLAIMED", claimed_at=now, claimed_by=claimed_by)
            db.commit()
            return claimed

    def complete_control_request(
        self,
        request_id: str,
        *,
        claimed_by: str,
        status: str,
        result: Mapping[str, Any],
    ) -> None:
        claimed_by = str(claimed_by).strip()
        if not claimed_by:
            raise ValueError("claimed_by must be non-empty")
        status = str(status).strip().upper()
        if status not in {"COMPLETED", "FAILED"}:
            raise ValueError("control request status is invalid")
        with self._connect() as db:
            db.execute(
                """
                UPDATE demo_control_requests
                   SET status=?,completed_at=?,result_json=?
                 WHERE request_id=? AND status='CLAIMED' AND claimed_by=?
                """,
                (status, _iso(datetime.now(timezone.utc)), _json(result), str(request_id), claimed_by),
            )

    def read_control_requests(self) -> tuple[dict[str, Any], ...]:
        self.initialize()
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM demo_control_requests ORDER BY requested_at, request_id"
            ).fetchall()
            return tuple(dict(row) for row in rows)

    def read_reconciliation_state(self) -> dict[str, Any]:
        self.initialize()
        with self._connect() as db:
            row = db.execute(
                """
                SELECT status,reason,details_json,observed_at
                  FROM demo_reconciliation
                 ORDER BY observed_at DESC,event_id DESC
                 LIMIT 1
                """
            ).fetchone()
        if row is None:
            return {"status": None, "reason": None, "details": None, "observed_at": None}
        try:
            details = json.loads(row[2])
        except (TypeError, ValueError):
            details = None
        return {
            "status": str(row[0]),
            "reason": str(row[1]),
            "details": details,
            "observed_at": row[3],
        }

    def read_circuit_state(self) -> dict[str, Any]:
        self.initialize()
        with self._connect() as db:
            row = db.execute("SELECT * FROM demo_circuit_state WHERE singleton_id=1").fetchone()
            return {} if row is None else dict(row)

    def set_circuit_state(
        self,
        *,
        status: str,
        daily_start_equity: float | None = None,
        daily_loss_fraction: float = 0.0,
        consecutive_losses: int = 0,
        reason: str | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO demo_circuit_state(
                    singleton_id,status,daily_start_equity,daily_loss_fraction,
                    consecutive_losses,reason,updated_at,execution_mode,real_money
                ) VALUES(1,?,?,?,?,?,?,'DEMO',0)
                """,
                (
                    str(status),
                    daily_start_equity,
                    float(daily_loss_fraction),
                    int(consecutive_losses),
                    None if reason is None else str(reason),
                    _iso(datetime.now(timezone.utc)),
                ),
            )

    def snapshot(self) -> dict[str, int]:
        self.initialize()
        tables = {
            "runs": "demo_runs",
            "account_snapshots": "demo_account_snapshots",
            "plans": "demo_plans",
            "signals": "demo_signals",
            "risk_decisions": "demo_risk_decisions",
            "orders": "demo_order_intents",
            "results": "demo_order_results",
            "positions": "demo_positions",
            "exits": "demo_exits",
            "reconciliation": "demo_reconciliation",
        }
        with self._connect() as db:
            return {name: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for name, table in tables.items()}

    def read_only_snapshot(self) -> dict[str, int]:
        """Read DEMO counts without creating or mutating the ledger."""

        resolved = self.path.expanduser().resolve()
        if not resolved.is_file():
            return {}
        uri = f"file:{resolved.as_posix()}?mode=ro"
        tables = {
            "runs": "demo_runs",
            "account_snapshots": "demo_account_snapshots",
            "plans": "demo_plans",
            "signals": "demo_signals",
            "risk_decisions": "demo_risk_decisions",
            "orders": "demo_order_intents",
            "results": "demo_order_results",
            "positions": "demo_positions",
            "exits": "demo_exits",
            "reconciliation": "demo_reconciliation",
        }
        with sqlite3.connect(uri, uri=True, timeout=0) as db:
            available = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not set(tables.values()).issubset(available):
                return {}
            return {name: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for name, table in tables.items()}

    def read_only_circuit_state(self) -> dict[str, Any]:
        resolved = self.path.expanduser().resolve()
        if not resolved.is_file():
            return {}
        uri = f"file:{resolved.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=0) as db:
            db.row_factory = sqlite3.Row
            try:
                row = db.execute("SELECT * FROM demo_circuit_state WHERE singleton_id=1").fetchone()
            except sqlite3.Error:
                return {}
            return {} if row is None else dict(row)

    def read_only_reconciliation_state(self) -> dict[str, Any]:
        """Read only the latest reconciliation event without creating tables."""

        resolved = self.path.expanduser().resolve()
        if not resolved.is_file():
            return {}
        uri = f"file:{resolved.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=0) as db:
            db.row_factory = sqlite3.Row
            try:
                row = db.execute(
                    """
                    SELECT status,reason,details_json,observed_at
                      FROM demo_reconciliation
                     ORDER BY observed_at DESC,event_id DESC
                     LIMIT 1
                    """
                ).fetchone()
            except sqlite3.Error:
                return {}
        if row is None:
            return {"status": None, "reason": None, "details": None, "observed_at": None}
        try:
            details = json.loads(row[2])
        except (TypeError, ValueError):
            details = None
        return {
            "status": str(row[0]),
            "reason": str(row[1]),
            "details": details,
            "observed_at": row[3],
        }


__all__ = ["DEMO_SCHEMA_VERSION", "DemoExecutionStore"]
