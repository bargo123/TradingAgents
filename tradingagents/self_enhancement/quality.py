"""Read-only data-quality checks for Phase 14 source ledgers."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return None
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    source_path: str
    symbol: str
    total_ticks: int
    unique_ticks: int
    duplicate_ticks: int
    invalid_ticks: int
    stale_ticks: int
    out_of_order_ticks: int
    zero_spread_ticks: int
    gap_count: int
    max_gap_seconds: float | None
    source_fingerprint: str
    quality_status: str
    broker_disconnect_windows: int = 0
    missing_costs: bool = True
    reconciliation_uncertainty: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def audit_hft_ticks(path: str | Path, symbol: str) -> DataQualityReport:
    source = Path(path).expanduser().resolve()
    wanted = str(symbol).strip().upper()
    if not wanted:
        raise ValueError("symbol must be non-empty")
    rows: list[sqlite3.Row]
    broker_disconnect_windows = 0
    with _read_only(source) as db:
        tables = {str(row[0]) for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "hft_ticks" not in tables:
            raise ValueError("source does not contain hft_ticks")
        rows = list(
            db.execute(
                "SELECT tick_key,symbol,timestamp,bid,ask,features_json FROM hft_ticks WHERE symbol=? ORDER BY timestamp,rowid",
                (wanted,),
            )
        )
        if "hft_run_health" in tables:
            broker_disconnect_windows = int(
                db.execute(
                    "SELECT COUNT(*) FROM hft_run_health WHERE error_code IS NOT NULL"
                ).fetchone()[0]
            )
        if "hft_runtime_state" in tables:
            broker_disconnect_windows += int(
                db.execute(
                    "SELECT COUNT(*) FROM hft_runtime_state WHERE status NOT IN ('RUNNING','READY','STOPPED')"
                ).fetchone()[0]
            )
    seen: set[str] = set()
    timestamps: list[datetime] = []
    invalid = stale = out_of_order = zero_spread = gap_count = 0
    canonical_rows: list[dict[str, Any]] = []
    for row in rows:
        tick_key = str(row["tick_key"])
        if tick_key in seen:
            stale += 1
        seen.add(tick_key)
        timestamp = _timestamp(row["timestamp"])
        try:
            bid = float(row["bid"])
            ask = float(row["ask"])
            valid_quote = math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask >= bid
        except (TypeError, ValueError, OverflowError):
            valid_quote = False
            bid = ask = 0.0
        if timestamp is None or not valid_quote:
            invalid += 1
        else:
            if ask == bid:
                zero_spread += 1
            if timestamps and timestamp <= timestamps[-1]:
                out_of_order += int(timestamp < timestamps[-1])
                stale += int(timestamp == timestamps[-1])
            if timestamps:
                gap = (timestamp - timestamps[-1]).total_seconds()
                if gap > 300:
                    gap_count += 1
            timestamps.append(timestamp)
        canonical_rows.append({
            "tick_key": tick_key,
            "timestamp": row["timestamp"],
            "bid": bid,
            "ask": ask,
        })
    gaps = [
        (right - left).total_seconds()
        for left, right in zip(timestamps, timestamps[1:], strict=False)
        if (right - left).total_seconds() > 300
    ]
    fingerprint = hashlib.sha256(
        json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    status = "VALID" if not any((invalid, stale, out_of_order, zero_spread, gap_count, broker_disconnect_windows)) else "QUALITY_WARNINGS"
    return DataQualityReport(
        source_path=str(source),
        symbol=wanted,
        total_ticks=len(rows),
        unique_ticks=len(seen),
        duplicate_ticks=len(rows) - len(seen),
        invalid_ticks=invalid,
        stale_ticks=stale,
        out_of_order_ticks=out_of_order,
        zero_spread_ticks=zero_spread,
        gap_count=gap_count,
        max_gap_seconds=max(gaps) if gaps else None,
        source_fingerprint=fingerprint,
        quality_status=status,
        broker_disconnect_windows=broker_disconnect_windows,
        # HFT source rows contain bid/ask/slippage observations but no
        # authoritative commission field.  Keep this explicit rather than
        # treating missing commission as zero cost.
        missing_costs=True,
    )


__all__ = ["DataQualityReport", "audit_hft_ticks"]
