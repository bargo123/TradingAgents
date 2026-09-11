import hashlib
import sqlite3
from pathlib import Path

import pytest

from tests.fixtures.experience_source_db import create_source_db
from tradingagents.experience.errors import SourceSchemaIncompatibleError, SourceSnapshotChangedError
from tradingagents.experience.source_reader import ReadonlySourceReader


@pytest.fixture()
def source_db(tmp_path: Path) -> Path:
    return create_source_db(tmp_path / "source.sqlite3")


def test_reader_uses_mode_ro_and_query_only(source_db: Path) -> None:
    snapshot = ReadonlySourceReader(source_db).read_snapshot()
    assert snapshot.query_only is True
    assert snapshot.sqlite_uri.endswith("mode=ro")
    assert snapshot.decisions[0]["decision_id"] == "d1"
    assert snapshot.evaluations[0]["horizon_seconds"] == 300
    assert snapshot.watcher_runs[0]["run_id"] == "wr1"


def test_reader_does_not_mutate_source(source_db: Path) -> None:
    before = hashlib.sha256(source_db.read_bytes()).hexdigest()
    ReadonlySourceReader(source_db).read_snapshot()
    assert hashlib.sha256(source_db.read_bytes()).hexdigest() == before


def test_mutation_sql_is_rejected(source_db: Path) -> None:
    with pytest.raises(sqlite3.OperationalError):
        ReadonlySourceReader(source_db).execute_for_test("CREATE TABLE forbidden(x INTEGER)")


def test_incompatible_schema_is_visible(source_db: Path) -> None:
    with sqlite3.connect(source_db) as connection:
        connection.execute("ALTER TABLE shadow_decisions RENAME TO wrong")
    with pytest.raises(SourceSchemaIncompatibleError):
        ReadonlySourceReader(source_db).read_snapshot()


def test_snapshot_change_is_typed(monkeypatch, source_db: Path) -> None:
    reader = ReadonlySourceReader(source_db)
    original = reader._file_fingerprint
    calls = iter([original(), {**original(), "size": original()["size"] + 1}])
    monkeypatch.setattr(reader, "_file_fingerprint", lambda: next(calls))
    with pytest.raises(SourceSnapshotChangedError):
        reader.read_snapshot()


def test_evaluation_schema_requires_resolved_symbol(source_db: Path) -> None:
    with sqlite3.connect(source_db) as connection:
        connection.execute("ALTER TABLE shadow_decision_evaluations RENAME TO old_evaluations")
        connection.execute("CREATE TABLE shadow_decision_evaluations (decision_id TEXT, evaluation_basis TEXT, horizon_seconds INTEGER, evaluation_status TEXT)")
    with pytest.raises(SourceSchemaIncompatibleError, match="resolved_symbol"):
        ReadonlySourceReader(source_db).read_snapshot()


def test_watcher_state_schema_and_singleton_id_are_enforced(source_db: Path) -> None:
    with sqlite3.connect(source_db) as connection:
        connection.execute("ALTER TABLE forex_watcher_state RENAME TO old_state")
        connection.execute("CREATE TABLE forex_watcher_state (singleton_id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO forex_watcher_state VALUES (1)")
    with pytest.raises(SourceSchemaIncompatibleError, match="lifecycle_status"):
        ReadonlySourceReader(source_db).read_snapshot()


def test_wal_mtime_change_is_typed(monkeypatch, source_db: Path) -> None:
    reader = ReadonlySourceReader(source_db)
    original = reader._wal_fingerprint
    first = original()
    calls = iter([first, {**first, "mtime_ns": 1}])
    monkeypatch.setattr(reader, "_wal_fingerprint", lambda: next(calls))
    with pytest.raises(SourceSnapshotChangedError):
        reader.read_snapshot()
