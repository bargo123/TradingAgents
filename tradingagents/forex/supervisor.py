"""Operational health policy for the standalone forex shadow collector."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .ollama_runtime import DedicatedOllamaRuntime, OllamaHealth
from .runtime_config import ForexShadowRuntimeConfig
from .watch_store import WatcherStore

UTC = timezone.utc
_HEALTH_SNAPSHOT_SCHEMA_VERSION = 1


def _health_snapshot_path(db_path: str | Path) -> Path:
    """Return the supervisor-owned, scalar-only health sidecar path."""

    return Path(f"{Path(db_path)}.ollama-health.json")


def _health_snapshot_payload(
    config: ForexShadowRuntimeConfig,
    health: OllamaHealth,
    *,
    supervisor_pid: int,
    observed_at: datetime,
) -> dict[str, Any]:
    """Serialize only bounded runtime facts; never prompts or model output."""

    return {
        "schema_version": _HEALTH_SNAPSHOT_SCHEMA_VERSION,
        "observed_at": observed_at.astimezone(UTC).isoformat(),
        "supervisor_pid": supervisor_pid,
        "config_fingerprint": config.fingerprint,
        "endpoint": health.endpoint,
        "quick_model": config.quick_model,
        "deep_model": config.deep_model,
        "context_length": config.context_length,
        "status": health.status,
        "version": health.version,
        "models": list(health.models),
        "loaded_models": list(health.loaded_models),
        "quick_context_verified": health.quick_context_verified,
        "deep_context_verified": health.deep_context_verified,
        "verified_context_length": health.verified_context_length,
        "openai_probe_ok": health.openai_probe_ok,
        "dedicated_pid": health.dedicated_pid,
        "recovery_attempts": health.recovery_attempts,
    }


def _write_health_snapshot(
    path: Path,
    config: ForexShadowRuntimeConfig,
    health: OllamaHealth,
    *,
    supervisor_pid: int,
    observed_at: datetime | None = None,
) -> bool:
    """Atomically publish a safe verification snapshot when the parent exists."""

    if not path.parent.is_dir():
        return False
    observed = datetime.now(UTC) if observed_at is None else observed_at
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    temporary_name: str | None = None
    try:
        payload = _health_snapshot_payload(
            config,
            health,
            supervisor_pid=supervisor_pid,
            observed_at=observed,
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        return True
    except (OSError, TypeError, ValueError):
        if temporary_name:
            with suppress(OSError):
                Path(temporary_name).unlink()
        return False


def _read_health_snapshot(path: Path) -> tuple[dict[str, Any], datetime] | None:
    """Read and minimally validate a persisted scalar health snapshot."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != _HEALTH_SNAPSHOT_SCHEMA_VERSION:
            return None
        observed = datetime.fromisoformat(str(payload["observed_at"]))
        if observed.tzinfo is None:
            return None
        if not isinstance(payload.get("supervisor_pid"), int) or isinstance(
            payload["supervisor_pid"], bool
        ):
            return None
        return payload, observed.astimezone(UTC)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _verified_snapshot_matches(
    payload: Mapping[str, Any],
    config: ForexShadowRuntimeConfig,
    health: OllamaHealth,
    lease: Any,
) -> bool:
    """Accept cached verification only for the active supervisor/config."""

    required = {config.quick_model, config.deep_model}
    lease_pid = getattr(lease, "pid", None)
    models = payload.get("models")
    verified_length = payload.get("verified_context_length")
    return bool(
        health.server_healthy
        and payload.get("status") == "HEALTHY"
        and payload.get("supervisor_pid") == lease_pid
        and payload.get("config_fingerprint") == config.fingerprint
        and payload.get("endpoint") == config.ollama_base_url
        and payload.get("quick_model") == config.quick_model
        and payload.get("deep_model") == config.deep_model
        and isinstance(models, list)
        and required.issubset({item for item in models if isinstance(item, str)})
        and payload.get("quick_context_verified") is True
        and payload.get("deep_context_verified") is True
        and isinstance(verified_length, int)
        and not isinstance(verified_length, bool)
        and verified_length >= config.context_length
        and payload.get("openai_probe_ok") is True
    )


def _health_level(ollama: OllamaHealth, watcher: Mapping[str, Any]) -> tuple[str, str | None]:
    if ollama.status == "UNAVAILABLE":
        return "OPERATOR_REVIEW_REQUIRED", ollama.error_code or "OLLAMA_UNAVAILABLE"
    if ollama.status != "HEALTHY":
        if ollama.server_healthy and ollama.error_code == "CONTEXT_NOT_VERIFIED":
            return "UNKNOWN", "OLLAMA_CONTEXT_NOT_VERIFIED"
        return "DEGRADED", ollama.error_code or "OLLAMA_DEGRADED"
    if watcher.get("circuit_reason"):
        return "OPERATOR_REVIEW_REQUIRED", str(watcher["circuit_reason"])
    if watcher.get("last_evaluation_status") == "ERROR":
        return "DEGRADED", str(watcher.get("last_error_code") or "EVALUATION_FAILED")
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


def _read_summary(store: Any, now: datetime) -> Mapping[str, Any]:
    reader = getattr(store, "read_only_summary", None)
    if callable(reader):
        return reader(now)
    return store.summary(now)


class ForexSupervisor:
    """Compose bounded Ollama health and existing fenced watcher lifecycle."""

    def __init__(
        self,
        runtime_config: ForexShadowRuntimeConfig | None = None,
        *,
        runtime_factory: Callable[..., DedicatedOllamaRuntime] = DedicatedOllamaRuntime,
        store_factory: Callable[..., WatcherStore] = WatcherStore,
    ) -> None:
        for name, factory in (
            ("runtime_factory", runtime_factory),
            ("store_factory", store_factory),
        ):
            if not callable(factory):
                raise TypeError(f"{name} must be callable")
        self.runtime_config = (
            ForexShadowRuntimeConfig() if runtime_config is None else runtime_config
        )
        if not isinstance(self.runtime_config, ForexShadowRuntimeConfig):
            raise TypeError("runtime_config must be ForexShadowRuntimeConfig")
        self.runtime_factory = runtime_factory
        self.store_factory = store_factory

    def _runtime(self) -> DedicatedOllamaRuntime:
        return self.runtime_factory(self.runtime_config)

    def status(self, db_path: str | Path) -> dict[str, Any]:
        runtime = self._runtime()
        health = runtime.health()
        store = self.store_factory(Path(db_path))
        now = datetime.now(UTC)
        watcher = _read_summary(store, now)
        lease = _read_active_lease(store, now)
        lease_active = bool(lease is not None and lease.lease_expires_at > now)
        verification_source = "LIVE_RUNTIME"
        verification_observed_at: str | None = None
        if lease_active and health.server_healthy:
            snapshot = _read_health_snapshot(_health_snapshot_path(db_path))
            if snapshot is not None:
                payload, observed_at = snapshot
                if _verified_snapshot_matches(payload, self.runtime_config, health, lease):
                    health = health.__class__(
                        status="HEALTHY",
                        endpoint=health.endpoint,
                        version=health.version or payload.get("version"),
                        models=health.models,
                        context_length=health.context_length or payload.get("context_length"),
                        error_code=None,
                        server_healthy=health.server_healthy,
                        loaded_models=health.loaded_models,
                        quick_context_verified=True,
                        deep_context_verified=True,
                        verified_context_length=int(payload["verified_context_length"]),
                        openai_probe_ok=True,
                        dedicated_pid=health.dedicated_pid or payload.get("dedicated_pid"),
                        recovery_attempts=health.recovery_attempts,
                    )
                    verification_source = "PERSISTED_VERIFICATION"
                    verification_observed_at = observed_at.isoformat()
        level, reason = _health_level(health, watcher)
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
            "ollama_verification_source": verification_source,
            "ollama_verification_observed_at": verification_observed_at,
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

        if isinstance(max_restarts, bool) or not isinstance(max_restarts, int) or max_restarts < 0:
            raise ValueError("max_restarts must be a non-negative integer")
        try:
            restart_backoff = float(restart_backoff_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("restart_backoff_seconds must be a finite non-negative number") from exc
        if (
            isinstance(restart_backoff_seconds, bool)
            or not isinstance(restart_backoff_seconds, (int, float))
            or not math.isfinite(restart_backoff)
            or restart_backoff < 0
        ):
            raise ValueError("restart_backoff_seconds must be a finite non-negative number")

        store = self.store_factory(Path(db_path))
        now = datetime.now(UTC)
        lease = _read_active_lease(store, now)
        if lease is not None and lease.lease_expires_at > now:
            print("FOREX SUPERVISOR: WATCHER_ALREADY_RUNNING")
            return 1

        runtime = self._runtime()
        try:
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

            db_file = Path(db_path).expanduser()
            if db_file.is_file():
                _write_health_snapshot(
                    _health_snapshot_path(db_file),
                    self.runtime_config,
                    health,
                    supervisor_pid=os.getpid(),
                )

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
        finally:
            shutdown = getattr(runtime, "shutdown", None)
            if callable(shutdown):
                cleanup_failed_during_primary_error = sys.exc_info()[0] is not None
                try:
                    shutdown()
                except Exception:
                    if not cleanup_failed_during_primary_error:
                        raise


__all__ = ["ForexSupervisor"]
