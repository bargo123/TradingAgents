from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from cli.forex_supervisor import main
from tradingagents.forex.ollama_runtime import OllamaHealth
from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig
from tradingagents.forex.supervisor import ForexSupervisor

UTC = timezone.utc


def test_supervisor_refuses_active_watcher_before_ollama_start():
    now = datetime.now(UTC)
    store = SimpleNamespace(
        active_lease=lambda _now: SimpleNamespace(lease_expires_at=now + timedelta(minutes=5))
    )
    constructed = []

    def runtime_factory(_config):
        constructed.append(True)
        raise AssertionError("dedicated Ollama must not start under an active lease")

    supervisor = ForexSupervisor(
        ForexShadowRuntimeConfig(), runtime_factory=runtime_factory, store_factory=lambda _path: store
    )

    assert supervisor.run(db_path="watch.db", watch_main=lambda *a, **k: 0) == 1
    assert constructed == []


def test_supervisor_lease_guard_prefers_read_only_probe():
    now = datetime.now(UTC)

    class Store:
        def read_only_active_lease(self, _now):
            return SimpleNamespace(lease_expires_at=now + timedelta(minutes=5))

        def active_lease(self, _now):
            raise AssertionError("startup guard must not use writer-initializing lease read")

    supervisor = ForexSupervisor(
        ForexShadowRuntimeConfig(),
        runtime_factory=lambda _config: (_ for _ in ()).throw(
            AssertionError("runtime must not be constructed")
        ),
        store_factory=lambda _path: Store(),
    )

    assert supervisor.run(db_path="watch.db", watch_main=lambda *a, **k: 0) == 1


def test_supervisor_uses_bounded_runtime_then_existing_watcher():
    store = SimpleNamespace(active_lease=lambda _now: None)
    calls = []

    class Runtime:
        def ensure_healthy(self):
            calls.append("ensure")
            return OllamaHealth("HEALTHY", "http://127.0.0.1:11435", "v", ("qwen3.5:2b", "qwen3.5:4b"), 16384)

        def prewarm(self):
            calls.append("prewarm")
            return {}

        def health(self):
            calls.append("health")
            return OllamaHealth("HEALTHY", "http://127.0.0.1:11435", "v", ("qwen3.5:2b", "qwen3.5:4b"), 16384)

    def watch_main(args, *, runtime_config):
        calls.append((args, runtime_config.backend_url))
        return 0

    supervisor = ForexSupervisor(
        ForexShadowRuntimeConfig(), runtime_factory=lambda _config: Runtime(), store_factory=lambda _path: store
    )

    assert supervisor.run(db_path="watch.db", watch_main=watch_main) == 0
    assert calls[:2] == ["ensure", "prewarm"]
    assert calls[-1][0][:2] == ["run", "--db-path"]


def test_supervisor_never_starts_collector_before_ollama_is_healthy():
    store = SimpleNamespace(active_lease=lambda _now: None)
    watched = []

    class Runtime:
        def ensure_healthy(self):
            return OllamaHealth(
                "DEGRADED",
                "http://127.0.0.1:11435",
                "v",
                ("qwen3.5:2b", "qwen3.5:4b"),
                None,
                "CONTEXT_NOT_VERIFIED",
            )

    supervisor = ForexSupervisor(
        runtime_factory=lambda _config: Runtime(), store_factory=lambda _path: store
    )

    assert supervisor.run(db_path="watch.db", watch_main=lambda *a, **k: watched.append(True)) == 1
    assert watched == []


def test_supervisor_status_separates_operational_health_from_strategy_distribution():
    watcher = {"lifecycle_status": "IDLE", "circuit_reason": None}
    store = SimpleNamespace(
        summary=lambda _now: watcher,
        active_lease=lambda _now: None,
    )

    class Runtime:
        def health(self):
            return OllamaHealth("HEALTHY", "http://127.0.0.1:11435", "v", (), 16384)

    supervisor = ForexSupervisor(
        runtime_factory=lambda _config: Runtime(), store_factory=lambda _path: store
    )
    report = supervisor.status("watch.db")

    assert report["health_level"] == "HEALTHY"
    assert report["executed"] is False


def test_supervisor_status_exposes_dedicated_runtime_and_watcher_fields():
    watcher = {
        "lifecycle_status": "ANALYZING",
        "owner_pid": 4321,
        "current_run_id": "run-1",
        "last_error_code": None,
        "circuit_reason": None,
    }
    store = SimpleNamespace(
        summary=lambda _now: watcher,
        active_lease=lambda _now: None,
    )

    class Runtime:
        def health(self):
            return OllamaHealth(
                "HEALTHY",
                "http://127.0.0.1:11435",
                "v",
                ("qwen3.5:2b", "qwen3.5:4b"),
                16384,
                None,
                True,
                ("qwen3.5:4b",),
                True,
                True,
                16384,
                True,
                999,
                0,
            )

    supervisor = ForexSupervisor(
        runtime_factory=lambda _config: Runtime(), store_factory=lambda _path: store
    )
    report = supervisor.status("watch.db")

    assert report["supervisor_status"] == "HEALTHY"
    assert report["dedicated_ollama_pid"] == 999
    assert report["dedicated_ollama_port"] == 11435
    assert report["ollama_server_healthy"] is True
    assert report["ollama_models_available"] == ["qwen3.5:2b", "qwen3.5:4b"]
    assert report["quick_context_verified"] is True
    assert report["deep_context_verified"] is True
    assert report["verified_context_length"] == 16384
    assert report["openai_probe_ok"] is True
    assert report["watcher_status"] == "ANALYZING"
    assert report["watcher_pid"] == 4321
    assert report["current_run_id"] == "run-1"
    assert report["recovery_attempts"] == 0


def test_supervisor_status_prefers_read_only_lease_probe():
    watcher = {"lifecycle_status": "IDLE", "circuit_reason": None}

    class Store:
        def summary(self, _now):
            return watcher

        def read_only_active_lease(self, _now):
            return None

        def active_lease(self, _now):
            raise AssertionError("status must not use writer-initializing lease read")

    class Runtime:
        def health(self):
            return OllamaHealth("HEALTHY", "http://127.0.0.1:11435", "v", (), 16384)

    report = ForexSupervisor(
        runtime_factory=lambda _config: Runtime(), store_factory=lambda _path: Store()
    ).status("watch.db")

    assert report["watcher_lease_status"] == "INACTIVE"


def test_supervisor_status_prefers_read_only_summary():
    watcher = {"lifecycle_status": "IDLE", "circuit_reason": None}

    class Store:
        def read_only_summary(self, _now):
            return watcher

        def summary(self, _now):
            raise AssertionError("status must not initialize the writer store")

        def read_only_active_lease(self, _now):
            return None

    class Runtime:
        def health(self):
            return OllamaHealth("HEALTHY", "http://127.0.0.1:11435", "v", (), 16384)

    report = ForexSupervisor(
        runtime_factory=lambda _config: Runtime(), store_factory=lambda _path: Store()
    ).status("watch.db")

    assert report["watcher_status"] == "IDLE"


def test_supervisor_status_reports_database_error_without_traceback(capsys):
    class Supervisor:
        def status(self, _db_path):
            raise sqlite3.DatabaseError("file is not a database")

    assert main(
        ["status", "--db-path", "broken.db"],
        supervisor_factory=lambda _config: Supervisor(),
    ) == 1
    captured = capsys.readouterr()
    assert "FOREX SUPERVISOR ERROR:" in captured.err
    assert "Traceback" not in captured.err
