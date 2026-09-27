from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import pytest


class _FailingConnection:
    def __init__(self) -> None:
        self.closed = False
        self.row_factory = None

    def execute(self, *_args, **_kwargs):
        raise sqlite3.OperationalError("synthetic open failure")

    def close(self) -> None:
        self.closed = True


def _connect_failure(module, monkeypatch):
    connections: list[_FailingConnection] = []

    def connect(*_args, **_kwargs):
        connection = _FailingConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(module.sqlite3, "connect", connect)
    return connections


@pytest.mark.parametrize(
    ("module_name", "function_name", "error_name"),
    [
        ("tradingagents.forex.research_signal_audit", "_readonly_connection", "ResearchSignalAuditReadError"),
        ("tradingagents.forex.decision_path_audit", "_readonly_connection", "DecisionPathAuditReadError"),
        ("tradingagents.forex.hold_audit", "_readonly_connection", "HoldAuditReadError"),
        ("tradingagents.forex.dashboard", "_read_only_connection", "DashboardReadError"),
        ("tradingagents.forex.revision_validation", "_connect", "RevisionValidationError"),
    ],
)
def test_read_only_open_failure_closes_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    function_name: str,
    error_name: str,
) -> None:
    module = importlib.import_module(module_name)
    path = tmp_path / "source.sqlite3"
    path.touch()
    connections = _connect_failure(module, monkeypatch)
    function = getattr(module, function_name)
    error = getattr(module, error_name)

    with pytest.raises(error):
        function(path, 0.01) if function_name != "_connect" else function(path)

    assert connections
    assert all(connection.closed for connection in connections)


def test_dataset_source_open_failure_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("tradingagents.datasets.sources")
    path = tmp_path / "source.sqlite3"
    path.touch()
    connections = _connect_failure(module, monkeypatch)

    with pytest.raises(module.SourceDatabaseUnavailableError):
        module.ReadonlyPhase56Source(path)._open()

    assert connections and all(connection.closed for connection in connections)


def test_experience_source_open_failure_clears_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("tradingagents.experience.source_reader")
    reader = module.ReadonlySourceReader(tmp_path / "source.sqlite3")
    connections = _connect_failure(module, monkeypatch)

    with pytest.raises(module.SourceDatabaseUnavailableError):
        reader._open()

    assert reader._connection is None
    assert connections and all(connection.closed for connection in connections)


def test_knowledge_catalog_open_failure_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("tradingagents.knowledge.catalog")
    connections = _connect_failure(module, monkeypatch)
    catalog = module.KnowledgeCatalog(tmp_path / "catalog.sqlite3")

    with pytest.raises(sqlite3.OperationalError):
        catalog._connect()

    assert connections and all(connection.closed for connection in connections)


@pytest.mark.parametrize("mode", ("standalone", "transaction"))
def test_experience_catalog_open_failure_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    module = importlib.import_module("tradingagents.experience.catalog")
    connections = _connect_failure(module, monkeypatch)
    catalog = module.ExperienceCatalog.__new__(module.ExperienceCatalog)
    catalog.artifact_root = tmp_path
    catalog.database_path = tmp_path / "catalog.sqlite3"
    catalog._transaction_connection = None

    context = catalog._connect() if mode == "standalone" else catalog.transaction()
    with pytest.raises(sqlite3.OperationalError), context:
        pass

    assert connections and all(connection.closed for connection in connections)


@pytest.mark.parametrize("method_name", ("_connect", "_read_only_connect"))
def test_watch_store_open_failure_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method_name: str
) -> None:
    module = importlib.import_module("tradingagents.forex.watch_store")
    path = tmp_path / "watch.sqlite3"
    path.touch()
    connections = _connect_failure(module, monkeypatch)
    store = module.WatcherStore(path)

    with pytest.raises(sqlite3.OperationalError):
        getattr(store, method_name)()

    assert connections and all(connection.closed for connection in connections)


def test_watch_store_lease_probe_open_failure_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("tradingagents.forex.watch_store")
    path = tmp_path / "watch.db"
    path.touch()
    connections = _connect_failure(module, monkeypatch)
    store = module.WatcherStore(path)

    with pytest.raises(sqlite3.OperationalError):
        store.read_only_active_lease()

    assert connections and all(connection.closed for connection in connections)
