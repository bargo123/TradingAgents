import sqlite3
from datetime import datetime, timezone

from tradingagents.forex.hft.models import FastAction, Tick
from tradingagents.forex.hft.store import HFT_SCHEMA_VERSION, HftShadowStore

UTC = timezone.utc


def _tick():
    return Tick("EURUSD", datetime(2026, 1, 1, 12, 0, tzinfo=UTC), 1.1, 1.1001, sequence=1)


def test_hft_store_initializes_versioned_isolated_schema_and_idempotent_rows(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run-1", mode="REPLAY", source_fingerprint="abc")
    store.record_tick("run-1", _tick(), features={"momentum": 0.1})
    store.record_action("run-1", "action-1", _tick(), FastAction.NO_ACTION, "test", processing_ms=0.2)
    store.record_action("run-1", "action-1", _tick(), FastAction.NO_ACTION, "test", processing_ms=0.2)
    summary = store.snapshot()
    assert summary["schema_version"] == HFT_SCHEMA_VERSION
    assert summary["ticks"] == 1
    assert summary["actions"] == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT executed FROM hft_actions").fetchone()[0] == 0
        assert db.execute("SELECT executed FROM hft_runs").fetchone()[0] == 0


def test_hft_store_recovers_open_positions_without_mutating_existing_rows(tmp_path):
    store = HftShadowStore(tmp_path / "hft.sqlite3")
    store.initialize()
    store.start_run("run-1", mode="SHADOW", source_fingerprint="abc")
    store.record_position("run-1", {"position_id": "p1", "state": "LONG", "symbol": "EURUSD", "executed": False})
    assert store.recover_open_positions() == ({"position_id": "p1", "state": "LONG", "symbol": "EURUSD", "executed": False},)
    store.close_run("run-1", status="STOPPED")
    assert store.snapshot()["runs"] == 1

