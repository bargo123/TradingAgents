from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5Bar, Mt5SymbolInfo
from tradingagents.forex.context import snapshot_to_dict
from tradingagents.forex.performance import (
    BenchmarkConfig,
    ReplayCase,
    load_replay_case,
    run_benchmark,
)
from tradingagents.forex.telemetry import critical_path_from_intervals


def _snapshot() -> ForexMarketSnapshot:
    timestamp = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    bar = Mt5Bar(
        timestamp=timestamp,
        open=1.1,
        high=1.101,
        low=1.099,
        close=1.1005,
        tick_volume=10,
        spread=1,
        real_volume=10,
    )
    return ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSD",
        bid=1.1000,
        ask=1.1002,
        spread=0.0002,
        spread_points=2,
        m1_candles=(bar,),
        m5_candles=(bar,),
        m15_candles=(bar,),
        h1_candles=(bar,),
        account=None,
        positions=(),
        symbol_info=Mt5SymbolInfo(name="EURUSD", digits=5, point=0.0001),
    )


def _source_db(path: Path, snapshot: ForexMarketSnapshot) -> bytes:
    encoded = json.dumps(snapshot_to_dict(snapshot), sort_keys=True)
    with sqlite3.connect(path) as db:
        db.execute(
            """
            CREATE TABLE forex_watch_runs (
                run_id TEXT PRIMARY KEY,
                run_status TEXT,
                requested_symbol TEXT,
                resolved_symbol TEXT,
                analysis_snapshot_timestamp TEXT,
                analysis_snapshot_bid REAL,
                analysis_snapshot_ask REAL,
                analysis_snapshot_spread REAL,
                analysis_snapshot_spread_points REAL,
                snapshot_json TEXT
            )
            """
        )
        db.execute(
            """
            INSERT INTO forex_watch_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-1",
                "SUCCEEDED",
                "EURUSD",
                "EURUSD",
                snapshot.timestamp.isoformat().replace("+00:00", "Z"),
                snapshot.bid,
                snapshot.ask,
                snapshot.spread,
                snapshot.spread_points,
                encoded,
            ),
        )
    return path.read_bytes()


def test_load_replay_case_decodes_one_causal_saved_snapshot(tmp_path: Path) -> None:
    snapshot = _snapshot()
    database = tmp_path / "source.sqlite3"
    before = _source_db(database, snapshot)

    case = load_replay_case(database, "run-1")

    assert isinstance(case, ReplayCase)
    assert case.source_run_id == "run-1"
    assert case.snapshot == snapshot
    assert case.snapshot_bytes == json.dumps(
        snapshot_to_dict(snapshot), sort_keys=True
    ).encode()
    assert case.case_fingerprint == hashlib.sha256(case.snapshot_bytes).hexdigest()
    assert database.read_bytes() == before


def test_run_benchmark_is_replay_only_and_does_not_write_source(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot()
    database = tmp_path / "source.sqlite3"
    before_bytes = _source_db(database, snapshot)
    before_rows = sqlite3.connect(database).execute(
        "SELECT run_id, run_status FROM forex_watch_runs"
    ).fetchall()

    class FakeRunner:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def analyze(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                normalized_action="HOLD",
                normalization_status="NORMALIZED",
                decision_context_status="COMPLETE",
                metrics={
                    "elapsed_seconds": 1.25,
                    "llm_calls": 0,
                    "tool_calls": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "stage_timings": {},
                },
            )

    fake = FakeRunner()
    output = tmp_path / "bench.json"
    report = run_benchmark(
        BenchmarkConfig(
            source_database_path=database,
            source_run_id="run-1",
            output_path=output,
            runner_factory=lambda **_: fake,
        )
    )

    assert report.run_kind == "REPLAY"
    assert report.source_run_id == "run-1"
    assert report.replay_status == "COMPLETE"
    assert output.exists()
    assert fake.calls[0]["persist"] is False
    assert fake.calls[0]["snapshot_bytes"] == _snapshot_bytes(snapshot)
    assert database.read_bytes() == before_bytes
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT run_id, run_status FROM forex_watch_runs").fetchall() == before_rows


def _snapshot_bytes(snapshot: ForexMarketSnapshot) -> bytes:
    return json.dumps(snapshot_to_dict(snapshot), sort_keys=True).encode()


def test_critical_path_accounts_for_overlap_and_dependency_wait() -> None:
    intervals = [
        {"node": "Market Analyst", "started": 0.0, "finished": 4.0},
        {"node": "News Analyst", "started": 0.0, "finished": 3.0},
        {"node": "Research Manager", "started": 5.0, "finished": 7.0},
        {"node": "Trader", "started": 7.0, "finished": 9.0},
    ]
    report = critical_path_from_intervals(
        intervals,
        (("Market Analyst", "Research Manager"), ("News Analyst", "Research Manager"), ("Research Manager", "Trader")),
    )

    assert report["wall_seconds"] == 9.0
    assert report["total_node_seconds"] == 11.0
    assert report["overlap_seconds"] == 2.0
    assert report["idle_wait_seconds"] == 1.0
    assert report["critical_path_seconds"] == 9.0
    assert report["critical_path_nodes"] == ["Market Analyst", "Research Manager", "Trader"]


def test_critical_path_ignores_malformed_intervals_without_inventing_timings() -> None:
    report = critical_path_from_intervals(
        [
            {"node": "valid", "started": 1.0, "finished": 2.0},
            {"node": "missing", "started": 2.0},
            {"node": "negative", "started": 4.0, "finished": 3.0},
        ],
        (),
    )

    assert report["wall_seconds"] == 1.0
    assert report["total_node_seconds"] == 1.0
    assert report["critical_path_seconds"] == 1.0
    assert report["unknown_intervals"] == 2
