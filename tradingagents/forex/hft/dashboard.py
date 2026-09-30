"""Read-only scalar dashboard metrics for the Phase 12 ledger."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


@dataclass(frozen=True, slots=True)
class HftDashboardSnapshot:
    path: str
    status: str
    runs: int
    ticks: int
    actions: int
    entries: int
    exits: int
    open_positions: int
    p50_processing_ms: float | None
    p95_processing_ms: float | None
    balance: float | None
    equity: float | None
    drawdown: float | None
    executed: bool = False


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(len(values) * fraction) - 1))
    return values[index]


def read_hft_dashboard(path: str | Path) -> HftDashboardSnapshot:
    source = Path(path)
    empty = dict(path=str(source), status="NOT_INITIALIZED", runs=0, ticks=0, actions=0, entries=0, exits=0, open_positions=0, p50_processing_ms=None, p95_processing_ms=None, balance=None, equity=None, drawdown=None, executed=False)
    if not source.is_file():
        return HftDashboardSnapshot(**empty)
    uri = f"file:{quote(str(source.resolve()).replace('\\', '/'), safe='/:' )}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=0) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "hft_runs" not in tables:
                return HftDashboardSnapshot(**empty)
            counts = {name: int(db.execute(f"SELECT COUNT(*) FROM hft_{name}").fetchone()[0]) for name in ("runs", "ticks", "actions")}
            entries = int(db.execute("SELECT COUNT(*) FROM hft_actions WHERE action IN ('ENTER_LONG','ENTER_SHORT')").fetchone()[0])
            exits = int(db.execute("SELECT COUNT(*) FROM hft_actions WHERE action IN ('EXIT','REDUCE')").fetchone()[0])
            open_positions = int(db.execute("SELECT COUNT(*) FROM hft_positions WHERE state IN ('LONG','SHORT','PENDING_ENTRY','PENDING_EXIT')").fetchone()[0])
            timings = [float(row[0]) for row in db.execute("SELECT processing_ms FROM hft_actions").fetchall() if isinstance(row[0], (int, float)) and math.isfinite(float(row[0]))]
            balance = equity = drawdown = None
            if "hft_account" in tables:
                row = db.execute("SELECT payload_json FROM hft_account ORDER BY snapshot_id DESC LIMIT 1").fetchone()
                if row:
                    import json
                    payload = json.loads(row[0])
                    balance, equity, drawdown = payload.get("balance"), payload.get("equity"), payload.get("max_drawdown")
            return HftDashboardSnapshot(path=str(source), status="HEALTHY", runs=counts["runs"], ticks=counts["ticks"], actions=counts["actions"], entries=entries, exits=exits, open_positions=open_positions, p50_processing_ms=_percentile(timings, 0.5), p95_processing_ms=_percentile(timings, 0.95), balance=balance, equity=equity, drawdown=drawdown, executed=False)
    except sqlite3.Error:
        return HftDashboardSnapshot(**{**empty, "status": "UNAVAILABLE"})


__all__ = ["HftDashboardSnapshot", "read_hft_dashboard"]
