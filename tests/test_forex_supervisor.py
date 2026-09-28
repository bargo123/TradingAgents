from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli.forex_supervisor import main
from tradingagents.forex.ollama_runtime import OllamaHealth
from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig
from tradingagents.forex.supervisor import ForexSupervisor

UTC = timezone.utc


def test_supervisor_rejects_invalid_runtime_config() -> None:
    with pytest.raises(TypeError, match="runtime_config"):
        ForexSupervisor(runtime_config=False)


@pytest.mark.parametrize("field", ["runtime_factory", "store_factory"])
def test_supervisor_rejects_invalid_factories(field) -> None:
    with pytest.raises(TypeError, match=field):
        ForexSupervisor(**{field: False})


def test_supervisor_rejects_invalid_restart_controls_before_runtime_start():
    constructed = []

    class Runtime:
        def ensure_healthy(self):
            raise AssertionError("runtime must not start for invalid controls")

    supervisor = ForexSupervisor(
        runtime_factory=lambda _config: constructed.append(True) or Runtime(),
        store_factory=lambda _path: SimpleNamespace(active_lease=lambda _now: None),
    )

    for kwargs in (
        {"max_restarts": True},
        {"max_restarts": 1.0},
        {"max_restarts": -1},
        {"restart_backoff_seconds": True},
        {"restart_backoff_seconds": "1"},
        {"restart_backoff_seconds": math.inf},
        {"restart_backoff_seconds": -1.0},
    ):
        with pytest.raises(ValueError):
            supervisor.run(db_path="watch.db", watch_main=lambda *a, **k: 0, **kwargs)

    assert constructed == []


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


def test_supervisor_shuts_down_owned_runtime_when_watcher_exits():
    store = SimpleNamespace(active_lease=lambda _now: None)
    calls = []

    class Runtime:
        def ensure_healthy(self):
            calls.append("ensure")
            return OllamaHealth(
                "HEALTHY",
                "http://127.0.0.1:11435",
                "v",
                ("qwen3.5:2b", "qwen3.5:4b"),
                16384,
            )

        def prewarm(self):
            calls.append("prewarm")
            return {}

        def health(self):
            calls.append("health")
            return OllamaHealth(
                "HEALTHY",
                "http://127.0.0.1:11435",
                "v",
                ("qwen3.5:2b", "qwen3.5:4b"),
                16384,
            )

        def shutdown(self):
            calls.append("shutdown")

    runtime = Runtime()
    supervisor = ForexSupervisor(
        ForexShadowRuntimeConfig(),
        runtime_factory=lambda _config: runtime,
        store_factory=lambda _path: store,
    )

    assert supervisor.run(db_path="watch.db", watch_main=lambda *_a, **_k: 0) == 0
    assert calls[-1] == "shutdown"


def test_supervisor_preserves_watcher_failure_when_runtime_shutdown_fails():
    store = SimpleNamespace(active_lease=lambda _now: None)

    class Runtime:
        def ensure_healthy(self):
            return OllamaHealth(
                "HEALTHY",
                "http://127.0.0.1:11435",
                "v",
                ("qwen3.5:2b", "qwen3.5:4b"),
                16384,
            )

        def prewarm(self):
            return {}

        def health(self):
            return OllamaHealth(
                "HEALTHY",
                "http://127.0.0.1:11435",
                "v",
                ("qwen3.5:2b", "qwen3.5:4b"),
                16384,
            )

        def shutdown(self):
            raise RuntimeError("shutdown failure")

    def watch_main(*_args, **_kwargs):
        raise RuntimeError("watch failure")

    supervisor = ForexSupervisor(
        ForexShadowRuntimeConfig(),
        runtime_factory=lambda _config: Runtime(),
        store_factory=lambda _path: store,
    )

    with pytest.raises(RuntimeError, match="watch failure"):
        supervisor.run(db_path="watch.db", watch_main=watch_main)


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


def test_supervisor_status_degrades_when_evaluation_failed_while_idle():
    watcher = {
        "lifecycle_status": "IDLE",
        "last_evaluation_status": "ERROR",
        "last_error_code": "EVALUATION_FAILED",
        "circuit_reason": None,
    }
    store = SimpleNamespace(
        summary=lambda _now: watcher,
        active_lease=lambda _now: None,
    )

    class Runtime:
        def health(self):
            return OllamaHealth("HEALTHY", "http://127.0.0.1:11435", "v", (), 16384)

    report = ForexSupervisor(
        runtime_factory=lambda _config: Runtime(), store_factory=lambda _path: store
    ).status("watch.db")

    assert report["supervisor_status"] == "DEGRADED"
    assert report["health_reason"] == "EVALUATION_FAILED"


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


def test_supervisor_status_uses_verified_health_snapshot_for_active_owner(tmp_path):
    db_path = tmp_path / "watch.db"
    db_path.touch()
    config = ForexShadowRuntimeConfig()
    snapshot_path = Path(f"{db_path}.ollama-health.json")
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observed_at": "2026-09-28T20:00:00+00:00",
                "supervisor_pid": 4321,
                "config_fingerprint": config.fingerprint,
                "endpoint": config.ollama_base_url,
                "quick_model": config.quick_model,
                "deep_model": config.deep_model,
                "context_length": config.context_length,
                "status": "HEALTHY",
                "version": "ollama version 0.34.2",
                "models": [config.quick_model, config.deep_model],
                "loaded_models": [config.deep_model],
                "quick_context_verified": True,
                "deep_context_verified": True,
                "verified_context_length": config.context_length,
                "openai_probe_ok": True,
                "dedicated_pid": 8640,
                "recovery_attempts": 0,
            }
        ),
        encoding="utf-8",
    )
    now = datetime.now(UTC)
    watcher = {
        "lifecycle_status": "IDLE",
        "owner_pid": 4321,
        "circuit_reason": None,
    }
    lease = SimpleNamespace(lease_expires_at=now + timedelta(minutes=5), pid=4321)

    class Store:
        def read_only_summary(self, _now):
            return watcher

        def read_only_active_lease(self, _now):
            return lease

    class Runtime:
        def health(self):
            return OllamaHealth(
                "DEGRADED",
                config.ollama_base_url,
                "ollama version 0.34.2",
                (config.quick_model, config.deep_model),
                None,
                "CONTEXT_NOT_VERIFIED",
                True,
            )

    report = ForexSupervisor(
        config,
        runtime_factory=lambda _config: Runtime(),
        store_factory=lambda _path: Store(),
    ).status(db_path)

    assert report["health_level"] == "HEALTHY"
    assert report["ollama"]["verified_context_length"] == 16384
    assert report["ollama"]["openai_probe_ok"] is True


def test_supervisor_status_does_not_reuse_snapshot_after_live_context_regresses(tmp_path):
    db_path = tmp_path / "watch.db"
    db_path.touch()
    config = ForexShadowRuntimeConfig()
    Path(f"{db_path}.ollama-health.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observed_at": "2026-09-28T20:00:00+00:00",
                "supervisor_pid": 4321,
                "config_fingerprint": config.fingerprint,
                "endpoint": config.ollama_base_url,
                "quick_model": config.quick_model,
                "deep_model": config.deep_model,
                "context_length": config.context_length,
                "status": "HEALTHY",
                "models": [config.quick_model, config.deep_model],
                "quick_context_verified": True,
                "deep_context_verified": True,
                "verified_context_length": config.context_length,
                "openai_probe_ok": True,
            }
        ),
        encoding="utf-8",
    )
    now = datetime.now(UTC)
    watcher = {"lifecycle_status": "IDLE", "owner_pid": 4321, "circuit_reason": None}
    lease = SimpleNamespace(lease_expires_at=now + timedelta(minutes=5), pid=4321)

    class Store:
        def read_only_summary(self, _now):
            return watcher

        def read_only_active_lease(self, _now):
            return lease

    class Runtime:
        def health(self):
            return OllamaHealth(
                "DEGRADED",
                config.ollama_base_url,
                "v",
                (config.quick_model, config.deep_model),
                8192,
                "CONTEXT_TOO_SMALL",
                True,
                (config.deep_model,),
            )

    report = ForexSupervisor(
        config,
        runtime_factory=lambda _config: Runtime(),
        store_factory=lambda _path: Store(),
    ).status(db_path)

    assert report["health_level"] == "DEGRADED"
    assert report["health_reason"] == "CONTEXT_TOO_SMALL"


def test_supervisor_status_marks_unverified_live_context_unknown():
    watcher = {"lifecycle_status": "IDLE", "circuit_reason": None}

    class Store:
        def read_only_summary(self, _now):
            return watcher

        def read_only_active_lease(self, _now):
            return None

    class Runtime:
        def health(self):
            return OllamaHealth(
                "DEGRADED",
                "http://127.0.0.1:11435",
                "v",
                ("qwen3.5:2b", "qwen3.5:4b"),
                16384,
                "CONTEXT_NOT_VERIFIED",
                True,
            )

    report = ForexSupervisor(
        runtime_factory=lambda _config: Runtime(),
        store_factory=lambda _path: Store(),
    ).status("missing-watch.db")

    assert report["health_level"] == "UNKNOWN"
    assert report["health_reason"] == "OLLAMA_CONTEXT_NOT_VERIFIED"


def test_supervisor_run_persists_scalar_health_snapshot(tmp_path):
    db_path = tmp_path / "watch.db"
    db_path.touch()
    store = SimpleNamespace(active_lease=lambda _now: None)
    config = ForexShadowRuntimeConfig()
    probe_calls = []

    class Runtime:
        def ensure_healthy(self):
            return OllamaHealth(
                "HEALTHY",
                config.ollama_base_url,
                "v",
                (config.quick_model, config.deep_model),
                config.context_length,
                None,
                True,
                (config.deep_model,),
                True,
                True,
                config.context_length,
                None,
                8640,
                0,
            )

        def prewarm(self):
            return {}

        def health(self):
            return self.ensure_healthy()

        def probe_openai_compatible(self):
            probe_calls.append(True)
            return None

        def shutdown(self):
            return None

    supervisor = ForexSupervisor(
        config,
        runtime_factory=lambda _config: Runtime(),
        store_factory=lambda _path: store,
    )

    assert supervisor.run(db_path=db_path, watch_main=lambda *_a, **_k: 0) == 0
    assert probe_calls == [True]
    snapshot = json.loads(Path(f"{db_path}.ollama-health.json").read_text(encoding="utf-8"))
    assert snapshot["status"] == "HEALTHY"
    assert snapshot["quick_context_verified"] is True
    assert snapshot["deep_context_verified"] is True
    assert snapshot["openai_probe_ok"] is True
    assert "prompt" not in snapshot
    assert "completion" not in snapshot


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


def test_supervisor_run_reports_startup_error_without_traceback(capsys):
    class Supervisor:
        def run(self, **_kwargs):
            raise sqlite3.DatabaseError("file is not a database")

    assert main(
        ["run", "--db-path", "broken.db"],
        supervisor_factory=lambda _config: Supervisor(),
        watch_main=lambda *_args, **_kwargs: 0,
    ) == 1
    captured = capsys.readouterr()
    assert "FOREX SUPERVISOR ERROR:" in captured.err
    assert "Traceback" not in captured.err


def test_supervisor_factory_error_is_reported_without_traceback(capsys):
    def broken_factory(_config):
        raise sqlite3.DatabaseError("runtime construction failed")

    assert main(
        ["run", "--db-path", "broken.db"],
        supervisor_factory=broken_factory,
        watch_main=lambda *_args, **_kwargs: 0,
    ) == 1
    captured = capsys.readouterr()
    assert "FOREX SUPERVISOR ERROR:" in captured.err
    assert "Traceback" not in captured.err
