"""Read-only importer from validated DEMO/HFT ledgers into Phase 14."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ExecutionMode, ExperienceTrade
from .quality import audit_hft_ticks
from .store import SelfEnhancementStore


def _read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _row_value(row: sqlite3.Row, name: str, default: Any = None) -> Any:
    return dict(row).get(name, default)


@dataclass(frozen=True, slots=True)
class ImportReport:
    accepted: int
    quarantined: int
    source_unchanged: bool
    quality: dict[str, Any]
    quarantine_reasons: dict[str, int]
    reconciliation_uncertainty: bool = False


def import_verified_demo(
    hft_path: str | Path,
    demo_path: str | Path,
    store: SelfEnhancementStore,
    *,
    source_run_ids: set[str] | None = None,
) -> ImportReport:
    hft = Path(hft_path).expanduser().resolve()
    demo = Path(demo_path).expanduser().resolve()
    before = (_file_fingerprint(hft), _file_fingerprint(demo))
    demo_fingerprint = before[1]
    quality = audit_hft_ticks(hft, "EURUSD")
    reasons: dict[str, int] = {}
    accepted = quarantined = 0

    def reject(source_id: str, reason: str, detail: dict[str, Any]) -> None:
        nonlocal quarantined
        quarantined += 1
        reasons[reason] = reasons.get(reason, 0) + 1
        store.quarantine(source_id, reason, detail)

    reconciliation_uncertainty = False
    with _read_only(demo) as db:
        tables = {str(row[0]) for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"demo_positions", "demo_exits"}
        if not required.issubset(tables):
            raise ValueError("DEMO source is missing required tables")
        reconciliation_by_ticket: dict[int, str] = {}
        if "demo_reconciliation" in tables:
            for row in db.execute(
                "SELECT status,details_json FROM demo_reconciliation ORDER BY observed_at,rowid"
            ):
                details = _payload(row["details_json"])
                ticket = details.get("ticket")
                if isinstance(ticket, (int, str)) and str(ticket).isdigit():
                    reconciliation_by_ticket[int(ticket)] = str(row["status"] or "").upper()
        positions = db.execute("SELECT * FROM demo_positions WHERE owned=1 ORDER BY updated_at,ticket").fetchall()
        exit_rows = db.execute("SELECT * FROM demo_exits ORDER BY observed_at").fetchall()
        exits_by_ticket: dict[int, list[sqlite3.Row]] = {}
        for row in exit_rows:
            exits_by_ticket.setdefault(int(row["ticket"]), []).append(row)
        intent_rows: dict[str, sqlite3.Row] = {}
        if "demo_order_intents" in tables:
            for row in db.execute("SELECT * FROM demo_order_intents"):
                intent_rows[str(row["intent_id"])] = row

        for position in positions:
            ticket = int(position["ticket"])
            payload = _payload(position["payload_json"])
            mode = str(payload.get("execution_mode", _row_value(position, "execution_mode", "")) or "").upper()
            if mode != ExecutionMode.DEMO.value or bool(_row_value(position, "real_money", 0)):
                reject(str(ticket), "NON_DEMO_OR_REAL_MONEY", {"ticket": ticket})
                continue
            if payload.get("synthetic") is True or payload.get("test_only") is True:
                reject(str(ticket), "TEST_ONLY_OR_SYNTHETIC", {"ticket": ticket})
                continue
            if reconciliation_by_ticket.get(ticket) not in {None, "", "RECONCILED", "CLEAN", "OK"}:
                reconciliation_uncertainty = True
                reject(str(ticket), "RECONCILIATION_UNCERTAINTY", {"ticket": ticket})
                continue
            intent_id = str(payload.get("intent_id", _row_value(position, "intent_id", "")) or "")
            intent = intent_rows.get(intent_id)
            if source_run_ids and intent is not None and str(intent["run_id"]) not in source_run_ids:
                continue
            matching = []
            for exit_row in exits_by_ticket.get(ticket, []):
                exit_payload = _payload(exit_row["payload_json"])
                if str(exit_payload.get("classification", _row_value(exit_row, "classification", "")) or "").upper() == "FILLED":
                    matching.append((exit_row, exit_payload))
            if not matching:
                reject(str(ticket), "NO_VERIFIED_FILLED_EXIT", {"ticket": ticket})
                continue
            exit_row, exit_payload = matching[-1]
            required = ("opened_at", "entry_bid", "entry_ask", "entry_fill", "expected_move_points", "mfe_points", "mae_points")
            if any(key not in payload for key in required):
                reject(str(ticket), "INCOMPLETE_ENTRY_PROVENANCE", {"ticket": ticket})
                continue
            if exit_row["realized_pnl"] is None or "fill_price" not in exit_payload:
                reject(str(ticket), "INCOMPLETE_EXIT_PROVENANCE", {"ticket": ticket})
                continue
            try:
                entry_timestamp = _parse_time(payload["opened_at"])
                exit_timestamp = _parse_time(exit_row["observed_at"])
                exit_fill = float(exit_payload["fill_price"])
                exit_bid = float(exit_payload.get("exit_bid", exit_fill))
                exit_ask = float(exit_payload.get("exit_ask", exit_fill))
                point = float(payload.get("point", 0.00001))
                spread_points = float(payload.get("spread_points", (float(payload["entry_ask"]) - float(payload["entry_bid"])) / point))
                strategy_id = str(payload.get("strategy_id", _row_value(position, "symbol", "UNKNOWN")))
                source_commit = str(payload.get("source_git_commit", intent["git_commit"] if intent is not None else "UNKNOWN"))
                trade = ExperienceTrade(
                    experience_id=f"demo-{ticket}",
                    source_database_id=f"demo:{demo_fingerprint[:16]}",
                    source_position_id=str(ticket),
                    strategy_id=strategy_id,
                    strategy_version=str(payload.get("strategy_version", "UNKNOWN")),
                    config_version=str(payload.get("config_version", "UNKNOWN")),
                    symbol=str(_row_value(position, "symbol", payload.get("symbol", "UNKNOWN"))),
                    direction=str(_row_value(position, "direction", payload.get("direction", "UNKNOWN"))),
                    entry_timestamp=entry_timestamp,
                    exit_timestamp=exit_timestamp,
                    entry_bid=float(payload["entry_bid"]),
                    entry_ask=float(payload["entry_ask"]),
                    entry_fill=float(payload["entry_fill"]),
                    exit_bid=exit_bid,
                    exit_ask=exit_ask,
                    exit_fill=exit_fill,
                    spread_points=spread_points,
                    slippage_points=float(payload.get("slippage_points", 0.0)),
                    feature_snapshot=payload.get("feature_snapshot", {}),
                    regime=str(payload.get("regime", "UNKNOWN")),
                    expected_move_points=float(payload["expected_move_points"]),
                    volume=float(_row_value(position, "volume", payload.get("volume", 0.0))),
                    risk=float(payload.get("risk", 0.0)),
                    mfe_points=float(payload["mfe_points"]),
                    mae_points=float(payload["mae_points"]),
                        exit_reason=str(exit_payload.get("exit_reason", _row_value(exit_row, "reason", "UNKNOWN")) or "UNKNOWN"),
                    broker_execution_latency_ms=float(payload.get("broker_execution_latency_ms", 0.0)),
                    gross_result=float(exit_row["realized_pnl"]),
                    net_known_result=float(exit_row["realized_pnl"]),
                    commission_known=bool(payload.get("commission_known", False)),
                    profit_to_loss_flip=bool(payload.get("profit_to_loss_flip", False)),
                    session=str(payload.get("session", "UNKNOWN")),
                    volatility_state=str(payload.get("volatility_state", "UNKNOWN")),
                    data_quality_state=quality.quality_status,
                    source_git_commit=source_commit,
                    execution_mode=ExecutionMode.DEMO,
                    source_fingerprint=f"{demo_fingerprint[:16]}:{ticket}",
                    mfe_capture_ratio=(float(exit_row["realized_pnl"]) / float(payload["mfe_points"]) if float(payload["mfe_points"]) > 0 else None),
                    signal_strength=(None if payload.get("signal_strength") is None else float(payload["signal_strength"])),
                )
            except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
                reject(str(ticket), "INVALID_TRADE_PROVENANCE", {"ticket": ticket, "error_type": type(exc).__name__})
                continue
            store.record_experience(trade)
            accepted += 1
    after = (_file_fingerprint(hft), _file_fingerprint(demo))
    return ImportReport(
        accepted,
        quarantined,
        before == after,
        quality.to_dict(),
        reasons,
        reconciliation_uncertainty,
    )


def _parse_time(value: Any):
    from datetime import datetime, timezone

    if not isinstance(value, str):
        raise ValueError("timestamp must be text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp must be UTC")
    return parsed.astimezone(timezone.utc)


__all__ = ["ImportReport", "import_verified_demo"]
