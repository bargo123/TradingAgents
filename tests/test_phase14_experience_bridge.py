import sqlite3

from tradingagents.self_enhancement.experience_bridge import import_phase8_observations
from tradingagents.self_enhancement.store import SelfEnhancementStore


def test_phase8_observations_are_imported_read_only_and_provenance_preserved(tmp_path):
    phase8 = tmp_path / "phase8"
    phase8.mkdir()
    db_path = phase8 / "catalog.sqlite3"
    db = sqlite3.connect(db_path)
    db.execute(
        "CREATE TABLE experience_records(experience_id TEXT,source_database_id TEXT,source_decision_id TEXT,source_decision_fingerprint TEXT,symbol TEXT,trust TEXT,tombstoned INTEGER)"
    )
    db.execute(
        "INSERT INTO experience_records VALUES(?,?,?,?,?,?,?)",
        ("exp1", "db1", "decision1", "fingerprint1", "EURUSD", "TIER_A_HIGH_TRUST", 0),
    )
    db.execute(
        "INSERT INTO experience_records VALUES(?,?,?,?,?,?,?)",
        ("exp2", "db1", "decision2", "fingerprint2", "EURUSD", "TIER_C_DIAGNOSTIC_ONLY", 0),
    )
    db.commit()
    db.close()
    before = db_path.read_bytes()
    store = SelfEnhancementStore(tmp_path / "phase14.sqlite3")
    store.initialize()
    assert import_phase8_observations(phase8, store) == 1
    assert db_path.read_bytes() == before
    assert store.snapshot()["findings"] == 1
