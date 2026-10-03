import json
import sqlite3
from datetime import timezone

from tradingagents.self_enhancement.importer import import_verified_demo
from tradingagents.self_enhancement.store import SelfEnhancementStore

UTC = timezone.utc


def _make_recoverable_sources(tmp_path, *, include_exit_fill=True):
    hft = tmp_path / "hft.sqlite3"
    demo = tmp_path / "demo.sqlite3"
    ticks = [
        ("run-1", "tick-1", "EURUSD", "2026-01-01T10:00:00.000000Z", 1.1000, 1.1001, json.dumps({"point": 0.00001, "session": "LONDON", "volatility": 0.1}), 0),
        ("run-1", "tick-2", "EURUSD", "2026-01-01T10:00:01.000000Z", 1.1002, 1.1003, json.dumps({"point": 0.00001, "session": "LONDON", "volatility": 0.2}), 0),
        ("run-1", "tick-3", "EURUSD", "2026-01-01T10:00:02.000000Z", 1.1001, 1.1002, json.dumps({"point": 0.00001, "session": "LONDON", "volatility": 0.2}), 0),
    ]
    with sqlite3.connect(hft) as db:
        db.execute("CREATE TABLE hft_ticks(run_id TEXT,tick_key TEXT,symbol TEXT,timestamp TEXT,bid REAL,ask REAL,features_json TEXT,executed INTEGER)")
        db.executemany("INSERT INTO hft_ticks VALUES(?,?,?,?,?,?,?,?)", ticks)
        db.execute("CREATE TABLE hft_fills(fill_id TEXT,action_id TEXT,run_id TEXT,symbol TEXT,timestamp TEXT,action TEXT,price REAL,size REAL,slippage_points REAL,latency_ms REAL,executed INTEGER)")
        db.executemany("INSERT INTO hft_fills VALUES(?,?,?,?,?,?,?,?,?,?,?)", [
            ("demo-shadow-entry-1", "action-entry", "run-1", "EURUSD", "2026-01-01T10:00:00.000000+00:00", "ENTER_LONG", 1.1001, 0.01, 0.0, 12.0, 0),
            *([("demo-shadow-exit-1", "action-exit", "run-1", "EURUSD", "2026-01-01T10:00:02.000000+00:00", "EXIT", 1.1001, 0.01, 0.0, 14.0, 0)] if include_exit_fill else []),
        ])
    with sqlite3.connect(demo) as db:
        db.execute("CREATE TABLE demo_positions(ticket INTEGER,owned INTEGER,payload_json TEXT,execution_mode TEXT,real_money INTEGER,symbol TEXT,direction TEXT,volume REAL,price_open REAL,state TEXT,updated_at TEXT,intent_id TEXT)")
        db.execute("CREATE TABLE demo_exits(ticket INTEGER,payload_json TEXT,observed_at TEXT,realized_pnl REAL)")
        db.execute("CREATE TABLE demo_order_intents(intent_id TEXT,run_id TEXT,plan_id TEXT,source_decision_id TEXT,source_run_id TEXT,strategy_id TEXT,symbol TEXT,direction TEXT,volume REAL,requested_price REAL,stop_loss REAL,take_profit REAL,deviation_points INTEGER,created_at TEXT,git_commit TEXT,account_trade_mode INTEGER,request_payload TEXT,execution_mode TEXT,real_money INTEGER)")
        db.execute("CREATE TABLE demo_order_results(intent_id TEXT,classification TEXT,retcode INTEGER,order_ticket INTEGER,deal_ticket INTEGER,fill_price REAL,fill_volume REAL,broker_comment TEXT,broker_timestamp TEXT,request_payload TEXT,observed_at TEXT,execution_mode TEXT,broker_order_sent INTEGER,real_money INTEGER)")
        position_payload = {"execution_mode": "DEMO", "intent_id": "entry-1", "direction": "LONG", "symbol": "EURUSD", "volume": 0.01, "price_open": 1.1001, "state": "CLOSED"}
        db.execute("INSERT INTO demo_positions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (1001, 1, json.dumps(position_payload), "DEMO", 0, "EURUSD", "LONG", 0.01, 1.1001, "CLOSED", "2026-01-01T10:00:02Z", "entry-1"))
        db.execute("INSERT INTO demo_order_intents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("entry-1", "run-1", "plan-1", "decision-1", "source-1", "range_rejection", "EURUSD", "LONG", 0.01, 1.1001, 1.0999, 1.1004, 20, "2026-01-01T10:00:00Z", "commit-1", 0, "{}", "DEMO", 0))
        db.execute("INSERT INTO demo_order_results VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("entry-1", "FILLED", 10009, 1001, 2001, 1.1001, 0.01, "filled", "2026-01-01T10:00:00Z", "{}", "2026-01-01T10:00:00Z", "DEMO", 1, 0))
        exit_payload = {"classification": "FILLED", "intent_id": "exit-1", "deal_ticket": 2002}
        db.execute("INSERT INTO demo_exits VALUES(?,?,?,?)", (1001, json.dumps(exit_payload), "2026-01-01T10:00:02Z", None))
    return hft, demo


def test_import_recovers_complete_trade_from_persisted_execution_chain(tmp_path):
    hft, demo = _make_recoverable_sources(tmp_path)
    store = SelfEnhancementStore(tmp_path / "phase14.sqlite3")
    store.initialize()

    report = import_verified_demo(hft, demo, store)

    assert report.accepted == 1
    assert report.quarantined == 0
    assert report.source_unchanged is True
    assert store.snapshot()["experience"] == 1


def test_import_keeps_trade_quarantined_when_exit_fill_is_missing(tmp_path):
    hft, demo = _make_recoverable_sources(tmp_path, include_exit_fill=False)
    store = SelfEnhancementStore(tmp_path / "phase14.sqlite3")
    store.initialize()

    report = import_verified_demo(hft, demo, store)

    assert report.accepted == 0
    assert report.quarantine_reasons["NO_VERIFIED_FILLED_EXIT"] == 1
