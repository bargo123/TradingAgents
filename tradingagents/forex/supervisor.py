"""Operational health policy for the standalone forex shadow collector."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .ollama_runtime import DedicatedOllamaRuntime, OllamaHealth
from .runtime_config import ForexShadowRuntimeConfig
from .watch_store import WatcherStore

UTC = timezone.utc


def _health_level(ollama: OllamaHealth, watcher: Mapping[str, Any]) -> tuple[str, str | None]:
    if ollama.status == "UNAVAILABLE":
        return "OPERATOR_REVIEW_REQUIRED", ollama.error_code or "OLLAMA_UNAVAILABLE"
    if ollama.status != "HEALTHY":
        return "DEGRADED", ollama.error_code or "OLLAMA_DEGRADED"
    if watcher.get("circuit_reason"):
        return "OPERATOR_REVIEW_REQUIRED", str(watcher["circuit_reason"])
    lifecycle = str(watcher.get("lifecycle_status") or "STOPPED")
    if lifecycle == "DEGRADED":
        return "DEGRADED", str(watcher.get("last_error_code") or "WATCHER_DEGRADED")
    if lifecycle == "STOPPED":
        return "DEGRADED", "WATCHER_NOT_RUNNING"
    return "HEALTHY", None


def _read_active_lease(store: Any, now: datetime) -> Any:
    reader = getattr(store, "read_only_active_lease", None)
    if callable(reader):
        return reader(now)
    return store.active_lease(now)


class ForexSupervisor:
    """Compose bounded Ollama health and existing fenced watcher lifecycle."""

    def __init__(
        self,
        runtime_config: ForexShadowRuntimeConfig | None = None,
        *,
        runtime_factory: Callable[..., DedicatedOllamaRuntime] = DedicatedOllamaRuntime,
        store_factory: Callable[..., WatcherStore] = WatcherStore,
    ) -> None:
        self.runtime_config = runtime_config or ForexShadowRuntimeConfig()
        self.runtime_factory = runtime_factory
        self.store_factory = store_factory

    def _runtime(self) -> DedicatedOllamaRuntime:
        return self.runtime_factory(self.runtime_config)

    def status(self, db_path: str | Path) -> dict[str, Any]:
        runtime = self._runtime()
        health = runtime.health()
        store = self.store_factory(Path(db_path))
        now = datetime.now(UTC)
        watcher = store.summary(now)
        lease = store.active_lease(now)
        level, reason = _health_level(health, watcher)
        lease_active = bool(lease is not None and lease.lease_expires_at > now)
        watcher_status = watcher.get("lifecycle_status")
        watcher_pid = watcher.get("owner_pid")
        endpoint = urlparse(health.endpoint)
        port = endpoint.port
        nested_ollama = {
            "status": health.status,
            "endpoint": health.endpoint,
            "version": health.version,
            "models": list(health.models),
            "context_length": health.context_length,
            "error_code": health.error_code,
            "server_healthy": health.server_healthy,
            "loaded_models": list(health.loaded_models),
            "quick_context_verified": health.quick_context_verified,
            "deep_context_verified": health.deep_context_verified,
            "verified_context_length": health.verified_context_length,
            "openai_probe_ok": health.openai_probe_ok,
            "dedicated_pid": health.dedicated_pid,
            "recovery_attempts": health.recovery_attempts,
        }
        return {
            "supervisor_status": level,
            "health_level": level,
            "health_reason": reason,
            "ollama": nested_ollama,
            "dedicated_ollama_pid": health.dedicated_pid,
            "dedicated_ollama_port": port,
            "ollama_server_healthy": health.server_healthy,
            "ollama_models_available": list(health.models),
            "quick_model": self.runtime_config.quick_model,
            "deep_model": self.runtime_config.deep_model,
            "quick_context_verified": health.quick_context_verified,
            "deep_context_verified": health.deep_context_verified,
            "verified_context_length": health.verified_context_length,
            "openai_probe_ok": health.openai_probe_ok,
            "watcher": watcher,
            "watcher_status": watcher_status,
            "watcher_pid": watcher_pid,
            "watcher_lease_status": "ACTIVE" if lease_active else "INACTIVE",
            "current_run_id": watcher.get("current_run_id"),
            "last_error_code": watcher.get("last_error_code"),
            "recovery_attempts": health.recovery_attempts,
            "lease_active": lease_active,
            "database_path": str(Path(db_path)),
            "executed": False,
        }

    def run(
        self,
        *,
        db_path: str | Path,
        watch_main: Callable[..., int],
        terminal_path: str | None = None,
        prewarm: bool = True,
        max_restarts: int = 3,
        restart_backoff_seconds: float = 5.0,
    ) -> int:
        """Start the dedicated runtime then delegate to the existing watcher.

        The lease is checked before any Ollama startup so an active collector
        cannot be disturbed or duplicated.
        """

        store = self.store_factory(Path(db_path))
        now = datetime.now(UTC)
        lease = _read_active_lease(store, now)
        if lease is not None and lease.lease_expires_at > now:
            print("FOREX SUPERVISOR: WATCHER_ALREADY_RUNNING")
            return 1

        runtime = self._runtime()
        health = runtime.ensure_healthy()
        if not health.healthy:
            print(f"FOREX SUPERVISOR: OLLAMA_{health.error_code or health.status}")
            return 1
        if prewarm:
            runtime.prewarm()
            health = runtime.health()
            if not health.healthy:
                print(f"FOREX SUPERVISOR: OLLAMA_{health.error_code or health.status}")
                return 1

        args = ["run", "--db-path", str(db_path)]
        if terminal_path:
            args.extend(["--terminal-path", terminal_path])
        print("FOREX SUPERVISOR: HEALTHY")
        print(f"PROVIDER: {self.runtime_config.provider}")
        print(f"BACKEND: {self.runtime_config.backend_url}")
        print(f"QUICK MODEL: {self.runtime_config.quick_model}")
        print(f"DEEP MODEL: {self.runtime_config.deep_model}")
        print(f"OLLAMA CONTEXT: {self.runtime_config.context_length}")
        print("NO ORDER WILL BE SENT")
        if isinstance(max_restarts, bool) or max_restarts < 0:
            raise ValueError("max_restarts must be non-negative")
        result = 0
        for attempt in range(max_restarts + 1):
            result = int(watch_main(args, runtime_config=self.runtime_config))
            if result == 0 or attempt >= max_restarts:
                return result
            # A non-zero foreground exit is a bounded crash recovery signal.
            # The watcher itself performs lease/process-identity reconciliation
            # before it can claim another run; this loop never deletes rows or
            # kills an active analysis.
            time.sleep(max(0.0, min(float(restart_backoff_seconds), 60.0)))
            health = runtime.health()
            if not health.healthy:
                health = runtime.ensure_healthy()
                if not health.healthy:
                    print("FOREX SUPERVISOR: OPERATOR_REVIEW_REQUIRED")
                    return 1
        return result


__all__ = ["ForexSupervisor"]
