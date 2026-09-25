from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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
