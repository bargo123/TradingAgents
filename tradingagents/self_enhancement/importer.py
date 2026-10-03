"""Read-only importer from validated DEMO/HFT ledgers into Phase 14.

The importer deliberately joins persisted evidence only. It never opens MT5,
rewrites a source ledger, or treats a requested price as a broker fill.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ExecutionMode, ExperienceEvidenceTier, ExperienceTrade
from .quality import audit_hft_ticks
from .store import SelfEnhancementStore

UTC = timezone.utc


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


def _row_value(row: sqlite3.Row | None, name: str, default: Any = None) -> Any:
    return default if row is None else dict(row).get(name, default)


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("timestamp must be UTC")
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ImportReport:
    accepted: int
    quarantined: int
    source_unchanged: bool
    quality: dict[str, Any]
    quarantine_reasons: dict[str, int]
    reconciliation_uncertainty: bool = False
    evidence_tiers: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class _TickEvidence:
    timestamp: datetime
    bid: float
    ask: float
    point: float
    features: dict[str, Any]
    source_run_id: str
    source_tick_key: str


def _load_tick_evidence(hft: sqlite3.Connection, symbol: str) -> dict[str, list[_TickEvidence]]:
    tables = {str(row[0]) for row in hft.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "hft_ticks" not in tables:
        return {}
    by_run: dict[str, list[_TickEvidence]] = {}
    for row in hft.execute(
        "SELECT run_id,tick_key,symbol,timestamp,bid,ask,features_json FROM hft_ticks WHERE symbol=? ORDER BY rowid",
        (symbol.upper(),),
    ):
        try:
            timestamp = _parse_time(row["timestamp"])
            bid = float(row["bid"])
            ask = float(row["ask"])
            if bid <= 0 or ask < bid:
                continue
            features = _payload(row["features_json"])
            point = float(features.get("point", 0.00001) or 0.00001)
            if point <= 0:
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        by_run.setdefault(str(row["run_id"] or "UNKNOWN"), []).append(
            _TickEvidence(timestamp, bid, ask, point, features, str(row["run_id"] or "UNKNOWN"), str(row["tick_key"]))
        )
    for values in by_run.values():
        values.sort(key=lambda item: (item.timestamp, item.source_tick_key))
    return by_run


def _at_or_before(values: list[_TickEvidence], timestamp: datetime, *, max_age_seconds: float = 5.0) -> _TickEvidence | None:
    positions = [item.timestamp for item in values]
    index = bisect.bisect_right(positions, timestamp) - 1
    if index < 0:
        return None
    item = values[index]
    return item if (timestamp - item.timestamp).total_seconds() <= max_age_seconds else None


def _mfe_mae(values: list[_TickEvidence], entry: datetime, exit_at: datetime, fill: float, direction: str, point: float) -> tuple[float, float]:
    path = [item for item in values if entry <= item.timestamp <= exit_at]
    if not path:
        raise ValueError("causal entry/exit path is unavailable")
    marks = [((item.bid if direction == "LONG" else item.ask) - fill) / point * (1.0 if direction == "LONG" else -1.0) for item in path]
    return max(0.0, max(marks)), min(0.0, min(marks))


def _filled_exit(exit_row: sqlite3.Row, exit_payload: dict[str, Any]) -> bool:
    classification = str(exit_payload.get("classification", _row_value(exit_row, "classification", "")) or "").upper()
    if classification == "FILLED":
        return True
    return bool(exit_payload.get("history_order_count", 0) and exit_payload.get("deal_tickets"))


def import_verified_demo(
    hft_path: str | Path,
    demo_path: str | Path,
    store: SelfEnhancementStore,
    *,
    source_run_ids: set[str] | None = None,
) -> ImportReport:
    """Import complete, non-synthetic DEMO executions from persisted evidence."""

    hft = Path(hft_path).expanduser().resolve()
    demo = Path(demo_path).expanduser().resolve()
    before = (_file_fingerprint(hft), _file_fingerprint(demo))
    demo_fingerprint = before[1]
    quality = audit_hft_ticks(hft, "EURUSD")
    reasons: dict[str, int] = {}
    tiers: dict[str, int] = {}
    accepted = quarantined = 0

    def reject(source_id: str, reason: str, detail: dict[str, Any]) -> None:
        nonlocal quarantined
        quarantined += 1
        reasons[reason] = reasons.get(reason, 0) + 1
        store.quarantine(source_id, reason, detail)

    reconciliation_uncertainty = False
    with _read_only(hft) as hft_db, _read_only(demo) as db:
        tick_by_run = _load_tick_evidence(hft_db, "EURUSD")
        fill_by_intent: dict[str, sqlite3.Row] = {}
        risk_by_action: dict[str, sqlite3.Row] = {}
        tables_hft = {str(row[0]) for row in hft_db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "hft_fills" in tables_hft:
            for row in hft_db.execute("SELECT * FROM hft_fills WHERE symbol='EURUSD'"):
                fill_id = str(row["fill_id"])
                if fill_id.startswith("demo-shadow-"):
                    fill_by_intent[fill_id[len("demo-shadow-"):]] = row
        if "hft_risk" in tables_hft:
            for row in hft_db.execute("SELECT * FROM hft_risk"):
                risk_by_action[str(row["action_id"])] = row

        tables = {str(row[0]) for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"demo_positions", "demo_exits"}.issubset(tables):
            raise ValueError("DEMO source is missing required tables")
        reconciliation_by_ticket: dict[int, str] = {}
        if "demo_reconciliation" in tables:
            for row in db.execute("SELECT status,details_json FROM demo_reconciliation ORDER BY observed_at,rowid"):
                details = _payload(row["details_json"])
                ticket = details.get("ticket")
                if isinstance(ticket, (int, str)) and str(ticket).isdigit():
                    reconciliation_by_ticket[int(ticket)] = str(row["status"] or "").upper()
        positions = db.execute("SELECT * FROM demo_positions WHERE owned=1 ORDER BY updated_at,ticket").fetchall()
        exits_by_ticket: dict[int, list[sqlite3.Row]] = {}
        for row in db.execute("SELECT * FROM demo_exits ORDER BY observed_at"):
            exits_by_ticket.setdefault(int(row["ticket"]), []).append(row)
        intent_rows = {str(row["intent_id"]): row for row in db.execute("SELECT * FROM demo_order_intents")} if "demo_order_intents" in tables else {}
        result_rows = {str(row["intent_id"]): row for row in db.execute("SELECT * FROM demo_order_results")} if "demo_order_results" in tables else {}

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
            result = result_rows.get(intent_id)
            if result is not None and str(result["classification"] or "").upper() not in {"FILLED", "PARTIAL"}:
                result = None
            exits = exits_by_ticket.get(ticket, [])
            matching = [(row, _payload(row["payload_json"])) for row in exits if _filled_exit(row, _payload(row["payload_json"]))]
            if not matching:
                reject(str(ticket), "NO_VERIFIED_FILLED_EXIT", {"ticket": ticket})
                continue
            exit_row, exit_payload = matching[-1]
            legacy_required = ("opened_at", "entry_bid", "entry_ask", "entry_fill", "expected_move_points", "mfe_points", "mae_points")
            legacy_complete = all(key in payload for key in legacy_required) and "fill_price" in exit_payload and _row_value(exit_row, "realized_pnl") is not None
            entry_fill_row = fill_by_intent.get(intent_id)
            exit_intent_id = str(exit_payload.get("intent_id", "") or "")
            exit_fill_row = fill_by_intent.get(exit_intent_id)
            if result is None and not legacy_complete:
                reject(str(ticket), "INCOMPLETE_ENTRY_PROVENANCE", {"ticket": ticket})
                continue
            if entry_fill_row is None and not legacy_complete:
                reject(str(ticket), "INCOMPLETE_ENTRY_PROVENANCE", {"ticket": ticket})
                continue
            if exit_fill_row is None and not legacy_complete:
                reject(str(ticket), "NO_VERIFIED_FILLED_EXIT", {"ticket": ticket})
                continue
            try:
                if legacy_complete:
                    entry_timestamp = _parse_time(payload["opened_at"])
                    exit_timestamp = _parse_time(exit_row["observed_at"])
                    entry_bid = float(payload["entry_bid"])
                    entry_ask = float(payload["entry_ask"])
                    entry_fill = float(payload["entry_fill"])
                    exit_fill = float(exit_payload["fill_price"])
                    exit_bid = float(exit_payload.get("exit_bid", exit_fill))
                    exit_ask = float(exit_payload.get("exit_ask", exit_fill))
                    point = float(payload.get("point", 0.00001))
                    features = dict(payload.get("feature_snapshot", {}))
                    mfe_points = float(payload["mfe_points"])
                    mae_points = float(payload["mae_points"])
                    latency_ms = float(payload.get("broker_execution_latency_ms", 0.0))
                    pnl = float(_row_value(exit_row, "realized_pnl"))
                else:
                    run_id = str(entry_fill_row["run_id"])
                    entry_timestamp = _parse_time(entry_fill_row["timestamp"])
                    exit_timestamp = _parse_time(exit_fill_row["timestamp"])
                    if exit_timestamp <= entry_timestamp:
                        raise ValueError("exit precedes entry")
                    values = tick_by_run.get(run_id, [])
                    entry_tick = _at_or_before(values, entry_timestamp)
                    exit_tick = _at_or_before(values, exit_timestamp)
                    if entry_tick is None or exit_tick is None:
                        raise ValueError("causal entry or exit context is unavailable")
                    entry_bid, entry_ask, point = entry_tick.bid, entry_tick.ask, entry_tick.point
                    exit_bid, exit_ask = exit_tick.bid, exit_tick.ask
                    entry_fill = float(result["fill_price"])
                    exit_fill = float(exit_fill_row["price"])
                    direction_for_metrics = str(_row_value(position, "direction", payload.get("direction", ""))).upper()
                    mfe_points, mae_points = _mfe_mae(values, entry_timestamp, exit_timestamp, entry_fill, direction_for_metrics, point)
                    features = dict(entry_tick.features)
                    features.update({"source_run_id": run_id, "source_tick_key": entry_tick.source_tick_key})
                    latency_ms = float(entry_fill_row["latency_ms"] or 0.0)
                    volume_for_pnl = float(result["fill_volume"] or _row_value(position, "volume", 0.0))
                    pnl = ((exit_fill - entry_fill) / point if direction_for_metrics == "LONG" else (entry_fill - exit_fill) / point) * volume_for_pnl
                direction = str(_row_value(position, "direction", payload.get("direction", ""))).upper()
                symbol = str(_row_value(position, "symbol", payload.get("symbol", "EURUSD")) or "EURUSD").upper()
                if direction not in {"LONG", "SHORT"}:
                    raise ValueError("direction is missing")
                spread_points = float(payload.get("spread_points", (entry_ask - entry_bid) / point))
                strategy_id = str(payload.get("strategy_id", _row_value(intent, "strategy_id", "UNKNOWN")) or "UNKNOWN")
                intent_payload = _payload(_row_value(intent, "request_payload", "{}"))
                provenance = intent_payload.get("provenance") if isinstance(intent_payload.get("provenance"), dict) else {}
                strategy_version = str(payload.get("strategy_version", provenance.get("strategy_version", "UNKNOWN")) or "UNKNOWN")
                config_version = str(payload.get("config_version", provenance.get("config_version", "UNKNOWN")) or "UNKNOWN")
                expected_move = payload.get("expected_move_points")
                if expected_move is None and _row_value(intent, "take_profit") is not None:
                    target = float(intent["take_profit"])
                    expected_move = ((target - entry_fill) if direction == "LONG" else (entry_fill - target)) / point
                if expected_move is None:
                    raise ValueError("expected move is missing")
                risk = payload.get("risk")
                if risk is None and entry_fill_row is not None:
                    risk_row = risk_by_action.get(str(entry_fill_row["action_id"]))
                    risk = None if risk_row is None else risk_row["risk_fraction"]
                tier = ExperienceEvidenceTier.FULLY_VERIFIED.value if strategy_version != "UNKNOWN" and config_version != "UNKNOWN" else ExperienceEvidenceTier.VERIFIED_EXECUTION.value
                source_commit = str(payload.get("source_git_commit", _row_value(intent, "git_commit", "UNKNOWN")) or "UNKNOWN")
                trade = ExperienceTrade(
                    experience_id=f"demo-{ticket}",
                    source_database_id=f"demo:{demo_fingerprint[:16]}",
                    source_position_id=str(ticket),
                    strategy_id=strategy_id,
                    strategy_version=strategy_version,
                    config_version=config_version,
                    symbol=symbol,
                    direction=direction,
                    entry_timestamp=entry_timestamp,
                    exit_timestamp=exit_timestamp,
                    entry_bid=entry_bid,
                    entry_ask=entry_ask,
                    entry_fill=entry_fill,
                    exit_bid=exit_bid,
                    exit_ask=exit_ask,
                    exit_fill=exit_fill,
                    spread_points=spread_points,
                    slippage_points=float(payload.get("slippage_points", abs(entry_fill - float(_row_value(intent, "requested_price", entry_fill))) / point)),
                    feature_snapshot=features,
                    regime=str(payload.get("regime", features.get("regime", "UNKNOWN"))),
                    expected_move_points=float(expected_move),
                    volume=float(payload.get("volume", _row_value(result, "fill_volume", _row_value(position, "volume", 0.0)))),
                    risk=float(risk or 0.0),
                    mfe_points=mfe_points,
                    mae_points=mae_points,
                    exit_reason=str(exit_payload.get("exit_reason", _row_value(exit_row, "reason", "UNKNOWN")) or "UNKNOWN"),
                    broker_execution_latency_ms=latency_ms,
                    gross_result=pnl,
                    net_known_result=pnl,
                    commission_known=bool(payload.get("commission_known", False)),
                    profit_to_loss_flip=bool(payload.get("profit_to_loss_flip", False)),
                    session=str(payload.get("session", features.get("session", "UNKNOWN"))),
                    volatility_state=str(payload.get("volatility_state", features.get("volatility", "UNKNOWN"))),
                    data_quality_state=quality.quality_status,
                    source_git_commit=source_commit,
                    execution_mode=ExecutionMode.DEMO,
                    source_fingerprint=f"{demo_fingerprint[:16]}:{ticket}",
                    mfe_capture_ratio=(pnl / mfe_points if mfe_points > 0 else None),
                    signal_strength=(None if payload.get("signal_strength") is None else float(payload["signal_strength"])),
                    evidence_tier=tier,
                )
            except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
                reject(str(ticket), "INVALID_TRADE_PROVENANCE", {"ticket": ticket, "error_type": type(exc).__name__})
                continue
            store.record_experience(trade)
            tiers[tier] = tiers.get(tier, 0) + 1
            accepted += 1
    after = (_file_fingerprint(hft), _file_fingerprint(demo))
    return ImportReport(accepted, quarantined, before == after, quality.to_dict(), reasons, reconciliation_uncertainty, tiers)


__all__ = ["ImportReport", "import_verified_demo"]
