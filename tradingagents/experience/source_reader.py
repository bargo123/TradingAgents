"""Read-only adapter for Phase 5/6 SQLite evidence."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .errors import (
    SourceDatabaseUnavailableError,
    SourceSchemaIncompatibleError,
    SourceSnapshotChangedError,
)
from .identity import source_snapshot_fingerprint

_DECISION_COLUMNS = {
    "decision_id",
    "created_at",
    "analysis_date",
    "source_run_id",
    "requested_symbol",
    "resolved_symbol",
    "analysis_profile",
    "analysis_timeframe",
    "valid_for_seconds",
    "valid_until",
    "action",
    "normalization_status",
    "decision_context_status",
    "llm_provider",
    "quick_model",
    "deep_model",
    "snapshot_timestamp",
    "analysis_snapshot_timestamp",
    "analysis_snapshot_bid",
    "analysis_snapshot_ask",
    "analysis_snapshot_spread",
    "analysis_snapshot_spread_points",
    "decision_completed_timestamp",
    "decision_reference_timestamp",
    "decision_reference_status",
    "snapshot_json",
    "executed",
    "normalization_error",
    "raw_portfolio_manager_result",
    "trader_summary",
    "portfolio_manager_summary",
    "bull_summary",
    "bear_summary",
    "reference_bid",
    "reference_ask",
    "reference_mid",
    "spread",
    "spread_points",
}
_EVALUATION_COLUMNS = {
    "decision_id",
    "resolved_symbol",
    "evaluation_basis",
    "horizon_seconds",
    "evaluation_version",
    "market_data_source",
    "source_context_eligible",
    "training_eligible",
    "training_eligibility_reason",
    "target_timestamp",
    "observation_timestamp",
    "observation_lag_ms",
    "entry_timestamp",
    "entry_bid",
    "entry_ask",
    "entry_spread",
    "entry_spread_points",
    "future_bid",
    "future_ask",
    "future_spread",
    "future_spread_points",
    "point",
    "digits",
    "buy_net_price",
    "buy_net_points",
    "sell_net_price",
    "sell_net_points",
    "selected_action",
    "selected_action_net_price",
    "selected_action_net_points",
    "best_counterfactual_action",
    "best_counterfactual_net_points",
    "hold_opportunity_cost_points",
    "buy_mfe_price",
    "buy_mfe_points",
    "buy_mae_price",
    "buy_mae_points",
    "sell_mfe_price",
    "sell_mfe_points",
    "sell_mae_price",
    "sell_mae_points",
    "evaluation_status",
    "unavailable_reason",
    "created_at",
    "evaluated_at",
    "recovered_from_unavailable_at",
    "previous_unavailable_reason",
}
_WATCH_RUN_COLUMNS = {"run_id", "source_run_id"}
_WATCH_OPPORTUNITY_COLUMNS = {"opportunity_key", "run_id", "decision_id"}
_WATCH_STATE_COLUMNS = {
    "singleton_id",
    "lifecycle_status",
    "owner_token",
    "owner_pid",
    "owner_host",
    "process_started_at",
    "lease_acquired_at",
    "heartbeat_at",
    "lease_expires_at",
    "current_run_id",
    "current_opportunity_key",
    "last_loop_at",
    "last_analysis_completed_at",
    "next_eligible_at",
    "last_evaluation_at",
    "last_evaluation_status",
    "evaluation_due_pending",
    "last_error_code",
    "last_error",
    "circuit_reason",
    "circuit_opened_at",
    "consecutive_mt5_failures",
    "consecutive_analysis_failures",
    "consecutive_incomplete_decisions",
    "consecutive_normalization_failures",
    "consecutive_runtime_exceeded",
    "updated_at",
}


@dataclass(frozen=True, slots=True)
class ReadonlySourceSnapshot:
    source_schema: dict[str, Any]
    file_fingerprint: dict[str, Any]
    wal_fingerprint: dict[str, Any]
    decisions: tuple[dict[str, Any], ...]
    evaluations: tuple[dict[str, Any], ...]
    watcher_runs: tuple[dict[str, Any], ...]
    watcher_opportunities: tuple[dict[str, Any], ...]
    watcher_state: tuple[dict[str, Any], ...]
    sqlite_uri: str
    canonical_path: str
    observed_at: datetime
    source_snapshot_fingerprint: str
    query_only: bool = True


def _utc(value: Any, field: str) -> Any:
    if not isinstance(value, str) or "T" not in value:
        return value
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return parsed


class ReadonlySourceReader:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve()
        self._connection: sqlite3.Connection | None = None

    @property
    def sqlite_uri(self) -> str:
        # quote the path while retaining drive-colon syntax accepted by SQLite URI mode.
        return "file:" + quote(self.path.as_posix(), safe="/:\\") + "?mode=ro"

    def _file_fingerprint(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise SourceDatabaseUnavailableError(f"source database is unavailable: {self.path}")
        digest = hashlib.sha256()
        with self.path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        stat = self.path.stat()
        return {"sha256": digest.hexdigest(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _wal_fingerprint(self) -> dict[str, Any]:
        wal = Path(str(self.path) + "-wal")
        if not wal.is_file():
            return {"sha256": None, "size": 0, "mtime_ns": None}
        digest = hashlib.sha256(wal.read_bytes()).hexdigest()
        stat = wal.stat()
        return {"sha256": digest, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _open(self) -> sqlite3.Connection:
        if self._connection is None:
            try:
                self._connection = sqlite3.connect(self.sqlite_uri, uri=True)
                self._connection.execute("PRAGMA query_only=ON")
            except sqlite3.Error as exc:
                raise SourceDatabaseUnavailableError(str(exc)) from exc
        return self._connection

    @staticmethod
    def _schema(connection: sqlite3.Connection) -> dict[str, Any]:
        tables: dict[str, Any] = {}
        names = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (name,) in names:
            columns = connection.execute(
                f'PRAGMA table_info("{name.replace(chr(34), chr(34) * 2)}")'
            ).fetchall()
            tables[name] = tuple({"name": row[1], "type": row[2], "pk": row[5]} for row in columns)
        return {"source_schema": "phase5-6", "source_schema_version": "v1", "tables": tables}

    @staticmethod
    def _require(
        schema: dict[str, Any], table: str, columns: set[str], *, required: bool = True
    ) -> None:
        actual = schema["tables"].get(table)
        if actual is None:
            if required:
                raise SourceSchemaIncompatibleError(f"required table missing: {table}")
            return
        names = {item["name"] for item in actual}
        missing = sorted(columns - names)
        if missing:
            raise SourceSchemaIncompatibleError(f"{table} missing columns: {', '.join(missing)}")

    def _rows(
        self, connection: sqlite3.Connection, table: str, order: str
    ) -> tuple[dict[str, Any], ...]:
        cursor = connection.execute(f'SELECT * FROM "{table}" ORDER BY {order}')
        fields = tuple(item[0] for item in cursor.description)
        rows = []
        for row in cursor.fetchall():
            item = dict(zip(fields, row, strict=False))
            for key, value in tuple(item.items()):
                if value is not None and (key.endswith("_at") or key.endswith("_timestamp")):
                    item[key] = _utc(value, key)
            rows.append(item)
        return tuple(rows)

    def read_snapshot(self) -> ReadonlySourceSnapshot:
        before_file, before_wal = self._file_fingerprint(), self._wal_fingerprint()
        connection = self._open()
        schema = self._schema(connection)
        self._require(schema, "shadow_decisions", _DECISION_COLUMNS)
        self._require(schema, "shadow_decision_evaluations", _EVALUATION_COLUMNS, required=False)
        self._require(schema, "forex_watch_runs", _WATCH_RUN_COLUMNS, required=False)
        self._require(
            schema, "forex_watch_opportunities", _WATCH_OPPORTUNITY_COLUMNS, required=False
        )
        self._require(schema, "forex_watcher_state", _WATCH_STATE_COLUMNS, required=False)
        decisions = self._rows(connection, "shadow_decisions", '"decision_id"')
        evaluations = (
            self._rows(
                connection,
                "shadow_decision_evaluations",
                "decision_id, evaluation_basis, horizon_seconds",
            )
            if "shadow_decision_evaluations" in schema["tables"]
            else ()
        )
        runs = (
            self._rows(connection, "forex_watch_runs", '"run_id"')
            if "forex_watch_runs" in schema["tables"]
            else ()
        )
        opportunities = (
            self._rows(connection, "forex_watch_opportunities", '"opportunity_key"')
            if "forex_watch_opportunities" in schema["tables"]
            else ()
        )
        state = (
            self._rows(connection, "forex_watcher_state", '"singleton_id"')
            if "forex_watcher_state" in schema["tables"]
            else ()
        )
        after_file, after_wal = self._file_fingerprint(), self._wal_fingerprint()
        if before_file != after_file or before_wal != after_wal:
            raise SourceSnapshotChangedError("source SQLite or WAL changed during read")
        observed = datetime.now(timezone.utc)
        metadata = {
            "schema": schema,
            "decisions": decisions,
            "evaluations": evaluations,
            "watcher_runs": runs,
            "watcher_opportunities": opportunities,
            "watcher_state": state,
        }
        return ReadonlySourceSnapshot(
            schema,
            before_file,
            before_wal,
            decisions,
            evaluations,
            runs,
            opportunities,
            state,
            self.sqlite_uri,
            str(self.path),
            observed,
            source_snapshot_fingerprint(metadata),
        )

    def execute_for_test(self, sql: str) -> Any:
        return self._open().execute(sql)


__all__ = ["ReadonlySourceReader", "ReadonlySourceSnapshot"]
