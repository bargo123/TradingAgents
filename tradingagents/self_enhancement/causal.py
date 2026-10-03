"""Immutable, read-only causal views over the Phase 12D tick ledger.

The raw ``hft_ticks`` table is an append-only observation log.  This module
never rewrites it; it builds an explicit derived view that orders events by
their authoritative broker timestamp within each runtime and starts a new
replay segment at a large discontinuity.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.forex.hft.models import Tick

UTC = timezone.utc


class CausalDatasetError(ValueError):
    """Raised when a causal tick view cannot be constructed safely."""


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise CausalDatasetError("tick timestamp must be text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CausalDatasetError("tick timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise CausalDatasetError("tick timestamp must be timezone-aware UTC")
    parsed = parsed.astimezone(UTC)
    if parsed > datetime.now(UTC):
        raise CausalDatasetError("future tick is not replayable")
    return parsed


@dataclass(frozen=True, slots=True)
class CausalTick:
    """One derived tick retaining its raw source identity."""

    source_run_id: str
    source_tick_key: str
    source_row_id: int
    tick: Tick
    features: dict[str, Any]

    @property
    def timestamp(self) -> datetime:
        return self.tick.timestamp


@dataclass(frozen=True, slots=True)
class CausalSegment:
    segment_id: str
    source_run_id: str
    ticks: tuple[CausalTick, ...]
    gap_before_seconds: float | None = None

    @property
    def start(self) -> datetime:
        return self.ticks[0].timestamp

    @property
    def end(self) -> datetime:
        return self.ticks[-1].timestamp


@dataclass(frozen=True, slots=True)
class CausalTickDataset:
    symbol: str
    segments: tuple[CausalSegment, ...]
    source_fingerprint: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    out_of_order_rows: int
    timestamp_collisions: int
    zero_spread_ticks: int
    gap_count: int
    max_gap_seconds: float | None

    @property
    def ticks(self) -> tuple[Tick, ...]:
        return tuple(item.tick for segment in self.segments for item in segment.ticks)

    @property
    def quality_status(self) -> str:
        return "VALID_DERIVED" if self.valid_rows and not self.invalid_rows else "QUALITY_WARNINGS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source_fingerprint": self.source_fingerprint,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "out_of_order_rows": self.out_of_order_rows,
            "timestamp_collisions": self.timestamp_collisions,
            "zero_spread_ticks": self.zero_spread_ticks,
            "gap_count": self.gap_count,
            "max_gap_seconds": self.max_gap_seconds,
            "segment_count": len(self.segments),
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "source_run_id": segment.source_run_id,
                    "tick_count": len(segment.ticks),
                    "start": segment.start.isoformat(),
                    "end": segment.end.isoformat(),
                    "gap_before_seconds": segment.gap_before_seconds,
                }
                for segment in self.segments
            ],
            "commission_status": "UNKNOWN",
        }


def load_causal_tick_dataset(
    path: str | Path,
    *,
    symbol: str = "EURUSD",
    gap_seconds: float = 300.0,
) -> CausalTickDataset:
    """Build a deterministic causal view without mutating ``path``."""

    source = Path(path).expanduser().resolve()
    wanted = str(symbol).strip().upper()
    if not source.is_file() or not wanted:
        raise CausalDatasetError("causal tick source is unavailable")
    if not math.isfinite(float(gap_seconds)) or gap_seconds <= 0:
        raise ValueError("gap_seconds must be positive and finite")
    uri = f"file:{source.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "hft_ticks" not in tables:
            raise CausalDatasetError("source does not contain hft_ticks")
        rows = db.execute(
            """
            SELECT rowid AS source_row_id,run_id,tick_key,symbol,timestamp,bid,ask,features_json
              FROM hft_ticks
             WHERE symbol=?
             ORDER BY rowid
            """,
            (wanted,),
        ).fetchall()

    by_run: dict[str, list[CausalTick]] = {}
    canonical: list[dict[str, Any]] = []
    invalid_rows = out_of_order = collisions = zero_spread = gap_count = 0
    max_gap: float | None = None
    previous_by_run: dict[str, datetime] = {}
    for row in rows:
        run_id = str(row["run_id"] or "UNKNOWN")
        try:
            timestamp = _timestamp(row["timestamp"])
            bid = float(row["bid"])
            ask = float(row["ask"])
            if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask < bid:
                raise CausalDatasetError("tick quote is invalid")
            features = json.loads(row["features_json"] or "{}")
            if not isinstance(features, dict):
                features = {}
        except (CausalDatasetError, TypeError, ValueError, OverflowError, json.JSONDecodeError):
            invalid_rows += 1
            continue
        if run_id in previous_by_run and timestamp < previous_by_run[run_id]:
            out_of_order += 1
        previous_by_run[run_id] = timestamp
        point = float(features.get("point", 0.00001) or 0.00001)
        if not math.isfinite(point) or point <= 0:
            invalid_rows += 1
            continue
        if ask == bid:
            zero_spread += 1
        item = CausalTick(
            source_run_id=run_id,
            source_tick_key=str(row["tick_key"]),
            source_row_id=int(row["source_row_id"]),
            tick=Tick(wanted, timestamp, bid, ask, point, int(row["source_row_id"])),
            features=dict(features),
        )
        by_run.setdefault(run_id, []).append(item)
        canonical.append(
            {
                "run_id": run_id,
                "row_id": int(row["source_row_id"]),
                "tick_key": str(row["tick_key"]),
                "timestamp": timestamp.isoformat(),
                "bid": bid,
                "ask": ask,
                "features": features,
            }
        )

    segments: list[CausalSegment] = []
    for run_id, values in sorted(by_run.items(), key=lambda item: item[0]):
        ordered = sorted(values, key=lambda item: (item.timestamp, item.source_row_id, item.source_tick_key))
        current: list[CausalTick] = []
        gap_before: float | None = None
        seen_timestamps: set[datetime] = set()
        for item in ordered:
            if item.timestamp in seen_timestamps:
                collisions += 1
                continue
            if current:
                gap = (item.timestamp - current[-1].timestamp).total_seconds()
                if gap > gap_seconds:
                    gap_count += 1
                    max_gap = gap if max_gap is None else max(max_gap, gap)
                    segments.append(CausalSegment(f"{run_id}:{len(segments)}", run_id, tuple(current), gap_before))
                    current = []
                    gap_before = gap
            current.append(item)
            seen_timestamps.add(item.timestamp)
        if current:
            segments.append(CausalSegment(f"{run_id}:{len(segments)}", run_id, tuple(current), gap_before))

    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CausalTickDataset(
        symbol=wanted,
        segments=tuple(segments),
        source_fingerprint=fingerprint,
        total_rows=len(rows),
        valid_rows=sum(len(segment.ticks) for segment in segments),
        invalid_rows=invalid_rows,
        out_of_order_rows=out_of_order,
        timestamp_collisions=collisions,
        zero_spread_ticks=zero_spread,
        gap_count=gap_count,
        max_gap_seconds=max_gap,
    )


__all__ = ["CausalDatasetError", "CausalTick", "CausalSegment", "CausalTickDataset", "load_causal_tick_dataset"]
