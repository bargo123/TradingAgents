from datetime import datetime, timezone
from types import SimpleNamespace

from tradingagents.forex.hft.store import HftLeaseOwner, HftShadowStore
from tradingagents.forex.ollama_runtime import OllamaHealth
from tradingagents.forex.supervisor import ForexSupervisor

UTC = timezone.utc


def test_supervisor_hft_status_is_scalar_and_shadow_only(tmp_path):
    report = ForexSupervisor().hft_status(tmp_path / "hft.sqlite3")
    assert report["status"] == "NOT_INITIALIZED"
    assert report["unique_ticks"] == 0
    assert report["tick_quality_status"] == "NOT_INITIALIZED"
    assert report["executed"] is False


def test_supervisor_health_degrades_when_hft_worker_requires_review(tmp_path):
    hft_path = tmp_path / "hft.sqlite3"
    hft_store = HftShadowStore(hft_path)
    hft_store.initialize()
    hft_store.set_runtime_state(
        "OPERATOR_REVIEW_REQUIRED",
        error_code="MT5ACCOUNTDISCONNECTEDERROR",
        recovery_count=3,
        last_recovery_result="OPERATOR_REVIEW_REQUIRED",
    )

    class Runtime:
        def health(self):
            return OllamaHealth(
                "HEALTHY", "http://127.0.0.1:11435", "v", ("qwen3.5:2b",), 16384
            )

    supervisor = ForexSupervisor(
        runtime_factory=lambda _config: Runtime(),
        store_factory=lambda _path: SimpleNamespace(
            read_only_active_lease=lambda _now: None,
            read_only_summary=lambda _now: {"lifecycle_status": "IDLE"},
        ),
    )

    report = supervisor.status(tmp_path / "strategic.db", hft_db_path=hft_path)

    assert report["hft_engine_health"] == "OPERATOR_REVIEW_REQUIRED"
    assert report["health_level"] == "OPERATOR_REVIEW_REQUIRED"
    assert report["health_reason"] == "MT5ACCOUNTDISCONNECTEDERROR"


def test_supervisor_hft_mode_forwards_one_context_to_existing_watcher(tmp_path):
    calls = []

    class Runtime:
        def ensure_healthy(self):
            return OllamaHealth(
                "HEALTHY", "http://127.0.0.1:11435", "v", ("qwen3.5:2b",), 16384
            )

        def prewarm(self):
            return None

        def health(self):
            return self.ensure_healthy()

        def shutdown(self):
            calls.append("runtime_shutdown")

    def watch_main(args, *, runtime_config, hft_context):
        calls.append((args, runtime_config, hft_context))
        assert "--hft-shadow" in args
        assert "--hft-db-path" in args
        assert hft_context.worker is not None
        return 0

    db_path = tmp_path / "strategic.db"
    hft_path = tmp_path / "hft.db"
    supervisor = ForexSupervisor(
        runtime_factory=lambda _config: Runtime(),
        store_factory=lambda _path: SimpleNamespace(read_only_active_lease=lambda _now: None),
    )

    assert supervisor.run(
        db_path=db_path,
        watch_main=watch_main,
        hft_shadow=True,
        hft_db_path=hft_path,
        hft_max_ticks=1,
    ) == 0
    assert calls and calls[0][2].worker.running is False
    assert calls[-1] == "runtime_shutdown"


def test_supervisor_hft_mode_refuses_active_hft_lease_before_runtime(tmp_path):
    hft_path = tmp_path / "hft.db"
    store = HftShadowStore(hft_path)
    now = datetime.now(UTC)
    result = store.acquire_lease(
        HftLeaseOwner("active", 123, "host", now), now
    )
    assert result.status.value == "ACQUIRED"
    constructed = []

    supervisor = ForexSupervisor(
        runtime_factory=lambda _config: constructed.append(True)
        or (_ for _ in ()).throw(AssertionError("runtime must not start")),
        store_factory=lambda _path: SimpleNamespace(read_only_active_lease=lambda _now: None),
    )

    assert supervisor.run(
        db_path=tmp_path / "strategic.db",
        watch_main=lambda *args, **kwargs: 0,
        hft_shadow=True,
        hft_db_path=hft_path,
    ) == 1
    assert constructed == []
