import json
import sqlite3
from datetime import datetime, timedelta, timezone

from tradingagents.forex.hft.dataset import read_tick_dataset
from tradingagents.forex.hft.models import Tick
from tradingagents.forex.hft.store import HftShadowStore

UTC = timezone.utc


def _insert_raw(db_path, run_id, key, timestamp, bid, ask):
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO hft_ticks(run_id,tick_key,symbol,timestamp,bid,ask,features_json) VALUES(?,?,?,?,?,?,?)",
            (run_id, key, "EURUSD", timestamp.isoformat(), bid, ask, "{}"),
        )
        db.commit()


def test_tick_dataset_reports_deduplication_quality_and_provenance(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run-1", mode="SHADOW", source_fingerprint="MT5_READ_ONLY")
    store.start_run("run-2", mode="SHADOW", source_fingerprint="MT5_READ_ONLY")
    start = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
    store.record_tick("run-1", Tick("EURUSD", start, 1.1, 1.1001), features={})
    _insert_raw(path, "run-2", "duplicate", start, 1.1, 1.1001)
    _insert_raw(path, "run-1", "out-of-order", start - timedelta(seconds=1), 1.1, 1.1001)
    _insert_raw(path, "run-1", "gap", start + timedelta(seconds=601), 1.1, 1.1001)
    _insert_raw(path, "run-1", "zero-spread", start + timedelta(seconds=602), 1.1, 1.1)
    _insert_raw(path, "run-1", "bad-quote", start + timedelta(seconds=603), 1.1002, 1.1)

    report = read_tick_dataset(path, symbol="EURUSD", large_gap_seconds=300)

    assert report.total_ticks == 6
    assert report.unique_ticks == 5
    assert report.duplicate_ticks == 1
    assert report.valid_ticks == 5
    assert report.invalid_ticks == 1
    assert report.out_of_order_timestamps == 1
    assert report.stale_timestamps == 1
    assert report.zero_spread_ticks == 1
    assert report.negative_spread_ticks == 1
    assert report.large_gap_count == 1
    assert report.days == ("2026-01-01",)
    assert report.sessions == ("ASIA",)
    assert report.executed_rows == 0
    assert report.quality_status == "FLAGGED"


def test_tick_dataset_empty_root_is_explicit(tmp_path):
    report = read_tick_dataset(tmp_path / "missing.sqlite3")
    assert report.total_ticks == 0
    assert report.quality_status == "NOT_INITIALIZED"


def test_tick_dataset_does_not_hide_rows_with_invalid_timestamps(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run-1", mode="SHADOW", source_fingerprint="MT5_READ_ONLY")
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO hft_ticks(run_id,tick_key,symbol,timestamp,bid,ask,features_json) VALUES(?,?,?,?,?,?,?)",
            ("run-1", "bad-time", "EURUSD", "not-a-timestamp", 1.1, 1.1001, "{}"),
        )
        db.commit()

    report = read_tick_dataset(path, symbol="EURUSD")

    assert report.total_ticks == 1
    assert report.invalid_ticks == 1
    assert report.first_timestamp is None
    assert report.quality_status == "FLAGGED"


def test_tick_dataset_flags_zero_spread_without_rewriting_source(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run-1", mode="SHADOW", source_fingerprint="MT5_READ_ONLY")
    timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    _insert_raw(path, "run-1", "zero", timestamp, 1.1, 1.1)

    report = read_tick_dataset(path, symbol="EURUSD")

    assert report.invalid_ticks == 0
    assert report.zero_spread_ticks == 1
    assert report.quality_status == "FLAGGED"


def test_tick_dataset_cli_is_read_only_and_supports_json(tmp_path, capsys):
    from cli.forex_tick_dataset import main

    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run-1", mode="SHADOW", source_fingerprint="MT5_READ_ONLY")
    store.record_tick(
        "run-1",
        Tick("EURUSD", datetime(2026, 1, 1, 12, 0, tzinfo=UTC), 1.1, 1.1001),
        features={},
    )
    before = path.read_bytes()

    assert main(["audit", "--db-path", str(path), "--symbol", "EURUSD", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quality_status"] == "VALID"
    assert payload["unique_ticks"] == 1
    assert payload["executed_rows"] == 0
    assert path.read_bytes() == before
