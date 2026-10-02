import json
import sqlite3
from datetime import timezone

from tradingagents.self_enhancement.importer import import_verified_demo
from tradingagents.self_enhancement.quality import audit_hft_ticks
from tradingagents.self_enhancement.store import SelfEnhancementStore

UTC = timezone.utc


def _dbs(tmp_path):
    hft = tmp_path / "hft.sqlite3"
    demo = tmp_path / "demo.sqlite3"
    db = sqlite3.connect(hft)
    db.execute("CREATE TABLE hft_ticks(run_id TEXT,tick_key TEXT,symbol TEXT,timestamp TEXT,bid REAL,ask REAL,features_json TEXT,executed INTEGER)")
    rows = [
        ("r", "1", "EURUSD", "2026-01-01T10:00:00+00:00", 1.1, 1.1001, "{}", 0),
        ("r", "2", "EURUSD", "2026-01-01T10:00:01+00:00", 1.1001, 1.1002, "{}", 0),
        ("r", "2", "EURUSD", "2026-01-01T10:00:01+00:00", 1.1001, 1.1002, "{}", 0),
    ]
    db.executemany("INSERT INTO hft_ticks VALUES(?,?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()
    db = sqlite3.connect(demo)
    db.execute("CREATE TABLE demo_positions(ticket INTEGER,state TEXT,owned INTEGER,payload_json TEXT,updated_at TEXT,execution_mode TEXT,real_money INTEGER,symbol TEXT,direction TEXT,volume REAL,price_open REAL)")
    db.execute("CREATE TABLE demo_exits(ticket INTEGER,payload_json TEXT,observed_at TEXT,realized_pnl REAL)")
    valid = {
        "strategy_id": "range_rejection",
        "strategy_version": "v1",
        "config_version": "cfg1",
        "opened_at": "2026-01-01T10:00:00+00:00",
        "entry_bid": 1.1,
        "entry_ask": 1.1001,
        "entry_fill": 1.1001,
        "expected_move_points": 8,
        "mfe_points": 12,
        "mae_points": -2,
        "regime": "NEUTRAL",
        "session": "LONDON",
        "volatility_state": "NORMAL",
        "source_git_commit": "abc",
        "execution_mode": "DEMO",
        "real_money": False,
        "synthetic": False,
    }
    canary = {**valid, "synthetic": True, "execution_mode": "TEST_ONLY"}
    db.execute("INSERT INTO demo_positions VALUES(?,?,?,?,?,?,?,?,?,?,?)", (1, "OPEN", 1, json.dumps(valid), "2026-01-01T10:00:00+00:00", "DEMO", 0, "EURUSD", "LONG", .01, 1.1001))
    db.execute("INSERT INTO demo_positions VALUES(?,?,?,?,?,?,?,?,?,?,?)", (2, "CLOSED", 1, json.dumps(canary), "2026-01-01T10:00:00+00:00", "TEST_ONLY", 0, "EURUSD", "LONG", .01, 1.1001))
    db.execute("INSERT INTO demo_exits VALUES(?,?,?,?)", (1, json.dumps({"classification": "FILLED", "fill_price": 1.1003, "exit_bid": 1.1003, "exit_ask": 1.1004, "exit_reason": "TAKE_PROFIT", "commission_known": False}), "2026-01-01T10:00:10+00:00", 2.0))
    db.commit()
    db.close()
    return hft, demo


def test_tick_quality_reports_duplicates_without_repairing_source(tmp_path):
    hft, _ = _dbs(tmp_path)
    before = hft.read_bytes()
    report = audit_hft_ticks(hft, "EURUSD")
    assert report.duplicate_ticks == 1
    assert report.quality_status == "QUALITY_WARNINGS"
    assert hft.read_bytes() == before


def test_import_accepts_only_complete_verified_demo_observations(tmp_path):
    hft, demo = _dbs(tmp_path)
    store = SelfEnhancementStore(tmp_path / "phase14.sqlite3")
    store.initialize()
    report = import_verified_demo(hft, demo, store)
    assert report.accepted == 1
    assert report.quarantined == 1
    assert report.source_unchanged is True
    assert store.snapshot()["experience"] == 1


def test_import_quarantines_unresolved_reconciliation_state(tmp_path):
    hft, demo = _dbs(tmp_path)
    db = sqlite3.connect(demo)
    db.execute(
        "CREATE TABLE demo_reconciliation(status TEXT, details_json TEXT, observed_at TEXT)"
    )
    db.execute(
        "INSERT INTO demo_reconciliation VALUES(?,?,?)",
        ("RECONCILIATION_REQUIRED", '{"ticket": 1}', "2026-01-01T10:00:05+00:00"),
    )
    db.commit()
    db.close()
    store = SelfEnhancementStore(tmp_path / "phase14.sqlite3")
    store.initialize()
    report = import_verified_demo(hft, demo, store)
    assert report.accepted == 0
    assert report.reconciliation_uncertainty is True
    assert report.quarantine_reasons["RECONCILIATION_UNCERTAINTY"] == 1
