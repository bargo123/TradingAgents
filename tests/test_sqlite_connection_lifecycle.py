from __future__ import annotations

import sqlite3

from tradingagents.forex import watch_store
from tradingagents.forex.watch_store import WatcherStore


def _is_closed(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


def test_watcher_read_connections_are_closed(monkeypatch, tmp_path):
    store = WatcherStore(tmp_path / "watch.db")
    store.initialize()
    opened: list[sqlite3.Connection] = []
    original_connect = watch_store.sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(watch_store.sqlite3, "connect", tracking_connect)

    store.table_names()
    store.active_lease()
    store.list_opportunities()
    store.list_runs()
    store.summary()
    store.circuit_reason()

    assert opened
    assert all(_is_closed(connection) for connection in opened)
