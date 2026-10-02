import sqlite3

from tradingagents.self_enhancement.orchestrator import SelfEnhancementOrchestrator
from tradingagents.self_enhancement.store import SelfEnhancementStore


def _sources(tmp_path):
    hft = tmp_path / "hft.sqlite3"
    db = sqlite3.connect(hft)
    db.execute("CREATE TABLE hft_ticks(run_id TEXT,tick_key TEXT,symbol TEXT,timestamp TEXT,bid REAL,ask REAL,features_json TEXT,executed INTEGER)")
    for i in range(16):
        db.execute("INSERT INTO hft_ticks VALUES(?,?,?,?,?,?,?,?)", ("r", str(i), "EURUSD", f"2026-01-01T10:00:{i:02d}+00:00", 1.1 + i * .00001, 1.10001 + i * .00001, "{}", 0))
    db.commit()
    db.close()
    demo = tmp_path / "demo.sqlite3"
    db = sqlite3.connect(demo)
    db.execute("CREATE TABLE demo_positions(ticket INTEGER,owned INTEGER,payload_json TEXT,execution_mode TEXT,real_money INTEGER,symbol TEXT,direction TEXT,volume REAL,price_open REAL,state TEXT,updated_at TEXT)")
    db.execute("CREATE TABLE demo_exits(ticket INTEGER,payload_json TEXT,observed_at TEXT,realized_pnl REAL)")
    db.commit()
    db.close()
    return hft, demo


def test_orchestrator_persists_no_experiment_when_verified_sample_is_insufficient(tmp_path):
    hft, demo = _sources(tmp_path)
    root = tmp_path / "phase14"
    report = SelfEnhancementOrchestrator(root).run_once(
        hft_path=hft,
        demo_path=demo,
        source_commit="abc123",
        minimum_verified_trades=2,
    )
    assert report.status == "NO_EXPERIMENT"
    assert report.candidate_count == 0
    assert SelfEnhancementStore(root / "phase14.sqlite3").snapshot()["experiments"] == 1


def test_orchestrator_does_not_construct_mt5_or_llm(tmp_path):
    hft, demo = _sources(tmp_path)
    report = SelfEnhancementOrchestrator(tmp_path / "phase14").run_once(hft_path=hft, demo_path=demo, source_commit="abc")
    assert report.llm_calls == 0
    assert report.mt5_calls == 0


def test_orchestrator_blocks_non_monotonic_source_without_repairing_it(tmp_path):
    hft, demo = _sources(tmp_path)
    db = sqlite3.connect(hft)
    db.execute("INSERT INTO hft_ticks VALUES(?,?,?,?,?,?,?,?)", ("r", "duplicate", "EURUSD", "2026-01-01T10:00:05+00:00", 1.1, 1.10001, "{}", 0))
    db.commit()
    db.close()
    before = hft.read_bytes()
    report = SelfEnhancementOrchestrator(tmp_path / "phase14").run_once(hft_path=hft, demo_path=demo, source_commit="abc")
    assert report.status == "NO_EXPERIMENT"
    assert report.candidate_count == 0
    assert hft.read_bytes() == before
