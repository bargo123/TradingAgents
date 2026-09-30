"""Read-only quality and coverage reports for the durable real-tick ledger."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

UTC = timezone.utc


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return None
    return parsed.astimezone(UTC)


def _session(timestamp: datetime) -> str:
    hour = timestamp.hour
    if hour < 7:
        return "ASIA"
    if hour < 13:
        return "LONDON"
    if hour < 21:
        return "NEW_YORK"
    return "OFF_HOURS"


@dataclass(frozen=True, slots=True)
class TickDatasetReport:
    """Scalar-only audit of persisted ticks; source rows are never rewritten."""

    database_path: str
    symbol: str | None
    total_ticks: int
    unique_ticks: int
    duplicate_ticks: int
    valid_ticks: int
    invalid_ticks: int
    first_timestamp: str | None
    last_timestamp: str | None
    duration_seconds: float | None
    ticks_per_second: float | None
    stale_timestamps: int
    out_of_order_timestamps: int
    invalid_quote_ticks: int
    zero_spread_ticks: int
    negative_spread_ticks: int
    large_gap_count: int
    max_gap_seconds: float | None
    executed_rows: int
    days: tuple[str, ...]
    sessions: tuple[str, ...]
    quality_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "database_path": self.database_path,
            "symbol": self.symbol,
            "total_ticks": self.total_ticks,
            "unique_ticks": self.unique_ticks,
            "duplicate_ticks": self.duplicate_ticks,
            "valid_ticks": self.valid_ticks,
            "invalid_ticks": self.invalid_ticks,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "duration_seconds": self.duration_seconds,
            "ticks_per_second": self.ticks_per_second,
            "stale_timestamps": self.stale_timestamps,
            "out_of_order_timestamps": self.out_of_order_timestamps,
            "invalid_quote_ticks": self.invalid_quote_ticks,
            "zero_spread_ticks": self.zero_spread_ticks,
            "negative_spread_ticks": self.negative_spread_ticks,
            "large_gap_count": self.large_gap_count,
            "max_gap_seconds": self.max_gap_seconds,
            "executed_rows": self.executed_rows,
            "days": list(self.days),
            "sessions": list(self.sessions),
            "quality_status": self.quality_status,
        }


def _empty(path: Path, symbol: str | None, status: str) -> TickDatasetReport:
    return TickDatasetReport(
        database_path=str(path.expanduser().resolve()),
        symbol=symbol,
        total_ticks=0,
        unique_ticks=0,
        duplicate_ticks=0,
        valid_ticks=0,
        invalid_ticks=0,
        first_timestamp=None,
        last_timestamp=None,
        duration_seconds=None,
        ticks_per_second=None,
        stale_timestamps=0,
        out_of_order_timestamps=0,
        invalid_quote_ticks=0,
        zero_spread_ticks=0,
        negative_spread_ticks=0,
        large_gap_count=0,
        max_gap_seconds=None,
        executed_rows=0,
        days=(),
        sessions=(),
        quality_status=status,
    )


def read_tick_dataset(
    path: str | Path,
    *,
    symbol: str | None = None,
    large_gap_seconds: float = 300.0,
) -> TickDatasetReport:
    """Audit the persisted HFT tick ledger through a query-only connection.

    ``hft_ticks`` remains the immutable capture ledger.  Duplicate rows are
    reported, not deleted, so replay can choose an explicit deterministic
    policy without losing source provenance.
    """

    try:
        gap_limit = float(large_gap_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("large_gap_seconds must be finite and positive") from exc
    if not math.isfinite(gap_limit) or gap_limit <= 0:
        raise ValueError("large_gap_seconds must be finite and positive")
    source = Path(path).expanduser()
    if not source.is_file():
        return _empty(source, symbol, "NOT_INITIALIZED")
    uri = "file:" + quote(str(source.resolve()).replace("\\", "/"), safe="/:\\") + "?mode=ro"
    normalized_symbol = None if symbol is None else str(symbol).strip().upper()
    if normalized_symbol == "":
        raise ValueError("symbol must be non-empty when provided")

    total = unique = invalid = valid = 0
    stale = out_of_order = invalid_quotes = zero_spread = negative_spread = 0
    large_gaps = 0
    max_gap: float | None = None
    executed_rows = 0
    identities: set[tuple[object, ...]] = set()
    previous_by_symbol: dict[str, datetime] = {}
    previous_valid_by_symbol: dict[str, datetime] = {}
    timestamps: list[datetime] = []
    days: set[str] = set()
    sessions: set[str] = set()
    with sqlite3.connect(uri, uri=True, timeout=5) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "hft_ticks" not in tables:
            return _empty(source, normalized_symbol, "NOT_INITIALIZED")
        query = "SELECT symbol,timestamp,bid,ask,executed FROM hft_ticks"
        params: tuple[object, ...] = ()
        if normalized_symbol is not None:
            query += " WHERE upper(symbol)=?"
            params = (normalized_symbol,)
        query += " ORDER BY rowid"
        for row in db.execute(query, params):
            total += 1
            row_symbol = str(row[0]).strip().upper()
            timestamp = _parse_timestamp(row[1])
            bid = row[2]
            ask = row[3]
            if row[4] not in (0, False, None):
                executed_rows += 1
            identity = (
                row_symbol,
                None if timestamp is None else timestamp,
                bid,
                ask,
            )
            if identity in identities:
                # Exact byte-level quote/timestamp duplicates are retained in
                # the source ledger but excluded from unique replay counts.
                pass
            else:
                identities.add(identity)
                unique += 1
            previous = previous_by_symbol.get(row_symbol)
            if timestamp is None:
                invalid += 1
                continue
            if previous is not None:
                if timestamp == previous:
                    stale += 1
                elif timestamp < previous:
                    out_of_order += 1
            previous_by_symbol[row_symbol] = timestamp
            timestamps.append(timestamp)
            days.add(timestamp.date().isoformat())
            sessions.add(_session(timestamp))
            previous_valid = previous_valid_by_symbol.get(row_symbol)
            if previous_valid is not None:
                gap = (timestamp - previous_valid).total_seconds()
                if gap > gap_limit:
                    large_gaps += 1
                if max_gap is None or gap > max_gap:
                    max_gap = gap
            previous_valid_by_symbol[row_symbol] = timestamp
            try:
                bid_value = float(bid)
                ask_value = float(ask)
                quote_valid = (
                    math.isfinite(bid_value)
                    and math.isfinite(ask_value)
                    and bid_value > 0
                    and ask_value > 0
                    and ask_value >= bid_value
                )
            except (TypeError, ValueError, OverflowError):
                quote_valid = False
                bid_value = ask_value = 0.0
            spread = ask_value - bid_value
            if spread < 0:
                negative_spread += 1
            elif spread == 0:
                zero_spread += 1
            if quote_valid:
                valid += 1
            else:
                invalid += 1
                invalid_quotes += 1

    if not timestamps and total == 0:
        return _empty(source, normalized_symbol, "EMPTY")
    if not timestamps:
        return TickDatasetReport(
            database_path=str(source.resolve()),
            symbol=normalized_symbol,
            total_ticks=total,
            unique_ticks=unique,
            duplicate_ticks=total - unique,
            valid_ticks=valid,
            invalid_ticks=invalid,
            first_timestamp=None,
            last_timestamp=None,
            duration_seconds=None,
            ticks_per_second=None,
            stale_timestamps=stale,
            out_of_order_timestamps=out_of_order,
            invalid_quote_ticks=invalid_quotes,
            zero_spread_ticks=zero_spread,
            negative_spread_ticks=negative_spread,
            large_gap_count=large_gaps,
            max_gap_seconds=max_gap,
            executed_rows=executed_rows,
            days=(),
            sessions=(),
            quality_status="FLAGGED",
        )
    first = min(timestamps)
    last = max(timestamps)
    duration = max(0.0, (last - first).total_seconds())
    quality_flags = (
        invalid
        or stale
        or out_of_order
        or zero_spread
        or negative_spread
        or large_gaps
        or executed_rows
    )
    return TickDatasetReport(
        database_path=str(source.resolve()),
        symbol=normalized_symbol,
        total_ticks=total,
        unique_ticks=unique,
        duplicate_ticks=total - unique,
        valid_ticks=valid,
        invalid_ticks=invalid,
        first_timestamp=first.isoformat().replace("+00:00", "Z"),
        last_timestamp=last.isoformat().replace("+00:00", "Z"),
        duration_seconds=duration,
        ticks_per_second=(total / duration) if duration > 0 else None,
        stale_timestamps=stale,
        out_of_order_timestamps=out_of_order,
        invalid_quote_ticks=invalid_quotes,
        zero_spread_ticks=zero_spread,
        negative_spread_ticks=negative_spread,
        large_gap_count=large_gaps,
        max_gap_seconds=max_gap,
        executed_rows=executed_rows,
        days=tuple(sorted(days)),
        sessions=tuple(sorted(sessions)),
        quality_status="FLAGGED" if quality_flags else "VALID",
    )


__all__ = ["TickDatasetReport", "read_tick_dataset"]
