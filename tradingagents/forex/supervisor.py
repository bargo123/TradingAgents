"""Operational health policy for the standalone forex shadow collector."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from .ollama_runtime import DedicatedOllamaRuntime, OllamaHealth
from .runtime_config import ForexShadowRuntimeConfig
from .watch_store import WatcherStore

UTC = timezone.utc
_HEALTH_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class HftShadowSupervisorContext:
    """Single-process handoff between the strategic watcher and Phase 12C."""

    worker: Any
    gate: Any
    provider_factory: Callable[..., Any]
    plan_store: Any
    hft_db_path: Path
    regime_store: Any | None = None
    provider_session: Any | None = None

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()
        session = self.provider_session
        close = getattr(session, "close", None)
        if callable(close):
            with self.gate.acquire("provider_close"):
                close()

    def handle_decision(self, decision: Any) -> bool:
        return bool(self.worker.handle_decision(decision))


class _SharedReadOnlyMt5Provider:
    """Keep one MT5 module session behind all supervisor-owned consumers."""

    # Consumer wrappers may call ``shutdown`` during their own cleanup, but
    # this provider deliberately keeps the supervisor-owned session alive.
    # The read-only proxy uses this explicit marker to avoid contending for
    # the MT5 operation gate for that no-op cleanup call.
    consumer_shutdown_is_noop = True

    def __init__(self, provider: Any) -> None:
        if provider is None:
            raise TypeError("provider is required")
        self._provider = provider
        self._lock = Lock()
        self._initialized = False

    def initialize(self) -> bool:
        with self._lock:
            if self._initialized:
                return True
            initialized = bool(self._provider.initialize())
            self._initialized = initialized
            return initialized

    def reinitialize(self) -> bool:
        """Refresh the supervisor-owned read-only MT5 session after a disconnect."""

        with self._lock:
            if self._initialized:
                try:
                    self._provider.shutdown()
                finally:
                    self._initialized = False
            initialized = bool(self._provider.initialize())
            self._initialized = initialized
            return initialized

    def shutdown(self) -> None:
        # Consumer lifecycles (runner/probe/evaluator/worker) must not tear
        # down the supervisor-owned session.  `close()` is the sole owner.
        return None

    def close(self) -> None:
        with self._lock:
            if self._initialized:
                try:
                    self._provider.shutdown()
                finally:
                    self._initialized = False

    def ensure_symbol(self, symbol: str) -> Any:
        return self._provider.ensure_symbol(symbol)

    def get_market_snapshot(self, symbol: str, *, count: int = 100) -> Any:
        return self._provider.get_market_snapshot(symbol, count=count)

    def get_tick(self, symbol: str) -> Any:
        return self._provider.get_tick(symbol)

    def get_bars(self, symbol: str, timeframe: str, count: int) -> Any:
        return self._provider.get_bars(symbol, timeframe, count)

    def get_ticks_range(self, symbol: str, start: Any, end: Any) -> Any:
        return self._provider.get_ticks_range(symbol, start, end)

    def get_account_info(self) -> Any:
        return self._provider.get_account_info()

    def get_terminal_info(self) -> Any:
        return self._provider.get_terminal_info()

    def get_symbols(self) -> Any:
        return self._provider.get_symbols()

    def get_positions(self, symbol: str | None = None) -> Any:
        return self._provider.get_positions(symbol)

    def get_spread(self, symbol: str) -> Any:
        return self._provider.get_spread(symbol)

    def demo_api(self) -> Any:
        """Return the initialized native API only for the DEMO gateway seam."""

        api = getattr(self._provider, "_api", None)
        if api is None:
            loader = getattr(self._provider, "_load_api", None)
            if callable(loader):
                api = loader()
        if api is None:
            raise RuntimeError("MT5 DEMO order API unavailable")
        return api


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
    loaded_models = set(health.loaded_models)
    live_context_valid = health.context_length is None or health.context_length >= config.context_length
    return bool(
        health.server_healthy
        and health.error_code != "CONTEXT_TOO_SMALL"
        and live_context_valid
        and loaded_models.issubset(required)
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


def _refresh_openai_probe(runtime: Any, health: OllamaHealth) -> OllamaHealth:
    """Refresh the scalar OpenAI-compatibility probe after model prewarming."""

    probe = getattr(runtime, "probe_openai_compatible", None)
    if not callable(probe) or health.openai_probe_ok is True:
        return health
    try:
        error_code = probe()
    except Exception as exc:  # health reporting remains bounded and sanitized
        error_code = type(exc).__name__.upper()
    if error_code is None:
        return replace(health, openai_probe_ok=True)
    return replace(
        health,
        status="DEGRADED",
        error_code=f"OPENAI_PROBE_{error_code}",
        openai_probe_ok=False,
    )


def _health_level(
    ollama: OllamaHealth,
    watcher: Mapping[str, Any],
    hft: Mapping[str, Any] | None = None,
) -> tuple[str, str | None]:
    if ollama.status == "UNAVAILABLE":
        return "OPERATOR_REVIEW_REQUIRED", ollama.error_code or "OLLAMA_UNAVAILABLE"
    if ollama.status != "HEALTHY":
        if ollama.server_healthy and ollama.error_code == "CONTEXT_NOT_VERIFIED":
            return "UNKNOWN", "OLLAMA_CONTEXT_NOT_VERIFIED"
        return "DEGRADED", ollama.error_code or "OLLAMA_DEGRADED"
    if hft is not None:
        hft_status = str(hft.get("status") or "")
        if hft_status == "OPERATOR_REVIEW_REQUIRED":
            return "OPERATOR_REVIEW_REQUIRED", str(
                hft.get("last_error_code") or "HFT_OPERATOR_REVIEW_REQUIRED"
            )
        if hft_status == "DEGRADED":
            return "DEGRADED", str(hft.get("last_error_code") or "HFT_DEGRADED")
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

    def _hft_context(
        self,
        *,
        terminal_path: str | None,
        source_db_path: str | Path,
        symbol: str,
        hft_db_path: str | Path,
        max_ticks: int,
        poll_interval_seconds: float,
        gate: Any,
        demo_execute: bool = False,
        demo_db_path: str | Path | None = None,
    ) -> HftShadowSupervisorContext:
        from tradingagents.dataflows.mt5.provider import MT5Provider
        from tradingagents.forex.hft.plan_store import AtomicPlanStore
        from tradingagents.forex.hft.regime_store import AtomicRegimeStore
        from tradingagents.forex.hft.runtime import HftShadowConfig
        from tradingagents.forex.hft.store import HftShadowStore
        from tradingagents.forex.hft.supervisor import HftShadowWorker
        from tradingagents.forex.runtime_config import collect_runtime_provenance
        from tradingagents.forex.shadow import ShadowDecisionStore

        resolved_hft_path = Path(hft_db_path).expanduser()

        shared_provider = _SharedReadOnlyMt5Provider(
            MT5Provider(terminal_path=terminal_path)
        )

        def provider_factory(terminal_path: str | None = None) -> Any:
            del terminal_path
            return shared_provider

        plan_store = AtomicPlanStore()
        regime_store = AtomicRegimeStore()
        hft_store = HftShadowStore(resolved_hft_path)
        config = HftShadowConfig(
            execution_mode="DEMO" if demo_execute else "SHADOW",
            symbol=symbol,
            artifact_path=resolved_hft_path,
            max_ticks=max_ticks,
            poll_interval_seconds=poll_interval_seconds,
        )
        provenance = collect_runtime_provenance(self.runtime_config)
        # The plan feed must carry the revision that owns the supervisor.  A
        # missing git executable is fail-closed by retaining a bounded marker;
        # it never affects MT5 or execution behavior.
        git_commit = str(provenance.get("git_commit") or "unknown")
        demo_runtime_factory = None
        if demo_execute:
            from tradingagents.forex.hft.demo_gateway import VerifiedDemoExecutionGateway
            from tradingagents.forex.hft.demo_runtime import DemoHftRuntime, DemoRuntimeConfig
            from tradingagents.forex.hft.demo_store import DemoExecutionStore

            resolved_demo_path = (
                Path(demo_db_path).expanduser()
                if demo_db_path is not None
                else resolved_hft_path.with_suffix(".demo.sqlite3")
            )
            demo_store = DemoExecutionStore(resolved_demo_path)
            demo_run_id = str(uuid.uuid4())

            def demo_runtime_factory(provider: Any) -> Any:
                broker_api = shared_provider.demo_api()
                demo_trade_mode = getattr(broker_api, "ACCOUNT_TRADE_MODE_DEMO", None)
                if isinstance(demo_trade_mode, bool) or not isinstance(demo_trade_mode, int):
                    raise RuntimeError("MT5 DEMO trade-mode constant unavailable")
                with gate.acquire("demo_account_preflight"):
                    account = VerifiedDemoExecutionGateway.read_account(provider)
                VerifiedDemoExecutionGateway.require_demo_account(account, demo_trade_mode)
                gateway = VerifiedDemoExecutionGateway(
                    provider,
                    broker_api,
                    demo_store,
                    run_id=demo_run_id,
                    demo_trade_mode=demo_trade_mode,
                    magic=12012012,
                    comment="TradingAgents-P12D-DEMO",
                    volume_cap=0.01,
                    gate=gate,
                )
                return DemoHftRuntime(
                    provider,
                    plan_store,
                    gateway=gateway,
                    store=demo_store,
                    config=DemoRuntimeConfig(
                        symbol=symbol,
                        artifact_path=resolved_demo_path,
                        max_ticks=max_ticks,
                        poll_interval_seconds=poll_interval_seconds,
                        hft_first=True,
                        no_regime_policy="BOOTSTRAP_NEUTRAL",
                    ),
                    regime_store=regime_store,
                    shadow_store=hft_store,
                    mt5_gate=gate,
                    lease_store=hft_store,
                )

        worker = HftShadowWorker(
            provider_factory,
            plan_store,
            regime_store=regime_store,
            config=config,
            store=hft_store,
            terminal_path=terminal_path,
            mt5_gate=gate,
            git_commit=git_commit,
            decision_validator=ShadowDecisionStore(source_db_path).is_execution_eligible,
            runtime_factory=demo_runtime_factory,
        )
        with suppress(Exception):
            latest = ShadowDecisionStore(source_db_path).latest_eligible(symbol)
            if latest is not None:
                worker.handle_decision(latest)
        return HftShadowSupervisorContext(
            worker=worker,
            gate=gate,
            provider_factory=provider_factory,
            plan_store=plan_store,
            regime_store=regime_store,
            hft_db_path=resolved_hft_path,
            provider_session=shared_provider,
        )

    def status(
        self,
        db_path: str | Path,
        *,
        hft_db_path: str | Path | None = None,
        demo_db_path: str | Path | None = None,
    ) -> dict[str, Any]:
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
        hft_report = self.hft_status(
            hft_db_path
            if hft_db_path is not None
            else Path(db_path).with_suffix(".hft.sqlite3")
        )
        demo_report = self.demo_status(demo_db_path) if demo_db_path is not None else {
            "status": "DISABLED",
            "execution_mode": "DEMO",
            "real_money": False,
            "orders": 0,
            "positions": 0,
            "reconciliation": 0,
        }
        level, reason = _health_level(health, watcher, hft_report)
        if demo_report["status"] in {"DEGRADED", "RECONCILIATION_REQUIRED", "OPERATOR_REVIEW_REQUIRED"}:
            level = "OPERATOR_REVIEW_REQUIRED"
            reason = f"DEMO_{demo_report['status']}"
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
            "strategic_brain_health": level,
            "ollama_health": health.status,
            "hft_engine_health": hft_report["status"],
            "mt5_read_only_health": (
                "ACTIVE" if hft_report["hft_lease_status"] == "ACTIVE" else "INACTIVE"
            ),
            "hft": hft_report,
            "demo": demo_report,
            "demo_execution_health": demo_report["status"],
        }

    def demo_status(self, db_path: str | Path | None) -> dict[str, Any]:
        """Read scalar DEMO ledger health without opening MT5 or Ollama."""

        if db_path is None:
            return {
                "status": "DISABLED",
                "execution_mode": "DEMO",
                "real_money": False,
                "orders": 0,
                "positions": 0,
                "reconciliation": 0,
            }
        from tradingagents.forex.hft.demo_store import DemoExecutionStore

        path = Path(db_path).expanduser()
        if not path.is_file():
            return {
                "status": "DISABLED",
                "execution_mode": "DEMO",
                "real_money": False,
                "orders": 0,
                "positions": 0,
                "reconciliation": 0,
                "database_path": str(path),
            }
        store = DemoExecutionStore(path)
        counts = store.read_only_snapshot()
        circuit = store.read_only_circuit_state()
        if not counts:
            return {
                "status": "OPERATOR_REVIEW_REQUIRED",
                "execution_mode": "DEMO",
                "real_money": False,
                "orders": 0,
                "positions": 0,
                "reconciliation": 0,
                "database_path": str(path),
            }
        status = "READY"
        if counts.get("reconciliation", 0):
            status = "RECONCILIATION_REQUIRED"
        elif str(circuit.get("status", "READY")) not in {"READY", "OPEN"}:
            status = "DEGRADED"
        elif counts.get("positions", 0):
            status = "POSITION_OPEN"
        return {
            "status": status,
            "execution_mode": "DEMO",
            "real_money": False,
            "database_path": str(path),
            **counts,
            "circuit_status": circuit.get("status"),
        }

    def hft_status(self, db_path: str | Path) -> dict[str, Any]:
        """Read HFT shadow health without constructing MT5 or Ollama."""

        from .hft.dashboard import read_hft_dashboard

        snapshot = read_hft_dashboard(db_path)
        return {
            "status": snapshot.status,
            "runs": snapshot.runs,
            "ticks": snapshot.ticks,
            "actions": snapshot.actions,
            "entries": snapshot.entries,
            "exits": snapshot.exits,
            "open_positions": snapshot.open_positions,
            "p50_processing_ms": snapshot.p50_processing_ms,
            "p95_processing_ms": snapshot.p95_processing_ms,
            "p99_processing_ms": snapshot.p99_processing_ms,
            "max_processing_ms": snapshot.max_processing_ms,
            "ticks_per_second": snapshot.ticks_per_second,
            "trades": snapshot.trades,
            "balance": snapshot.balance,
            "equity": snapshot.equity,
            "drawdown": snapshot.drawdown,
            "compound_return": snapshot.compound_return,
            "win_rate": snapshot.win_rate,
            "profit_factor": snapshot.profit_factor,
            "expectancy": snapshot.expectancy,
            "benchmark_target": snapshot.benchmark_target,
            "benchmark_difference": snapshot.benchmark_difference,
            "benchmark_achieved": snapshot.benchmark_achieved,
            "active_plan_id": snapshot.active_plan_id,
            "active_plan_direction": snapshot.active_plan_direction,
            "active_plan_strategy_family": snapshot.active_plan_strategy_family,
            "active_plan_created_at": snapshot.active_plan_created_at,
            "active_plan_expires_at": snapshot.active_plan_expires_at,
            "hft_lease_status": snapshot.hft_lease_status,
            "dropped_ticks": snapshot.dropped_ticks,
            "stale_ticks": snapshot.stale_ticks,
            "out_of_order_ticks": snapshot.out_of_order_ticks,
            "last_error_code": snapshot.last_error_code,
            "runtime_status": snapshot.runtime_status,
            "last_disconnect_at": snapshot.last_disconnect_at,
            "last_recovery_attempt_at": snapshot.last_recovery_attempt_at,
            "recovery_count": snapshot.recovery_count,
            "last_recovery_result": snapshot.last_recovery_result,
            "unique_ticks": snapshot.unique_ticks,
            "tick_quality_status": snapshot.tick_quality_status,
            "dataset_first_timestamp": snapshot.dataset_first_timestamp,
            "dataset_last_timestamp": snapshot.dataset_last_timestamp,
            "dataset_duration_seconds": snapshot.dataset_duration_seconds,
            "dataset_days": snapshot.dataset_days,
            "dataset_sessions": snapshot.dataset_sessions,
            "dataset_invalid_ticks": snapshot.dataset_invalid_ticks,
            "dataset_duplicate_ticks": snapshot.dataset_duplicate_ticks,
            "dataset_large_gap_count": snapshot.dataset_large_gap_count,
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
        hft_shadow: bool = False,
        hft_db_path: str | Path | None = None,
        hft_symbol: str = "EURUSD",
        hft_max_ticks: int = 0,
        hft_poll_interval_seconds: float = 1.0,
        demo_execute: bool = False,
        demo_db_path: str | Path | None = None,
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
        if not isinstance(hft_shadow, bool):
            raise ValueError("hft_shadow must be boolean")
        if not isinstance(demo_execute, bool):
            raise ValueError("demo_execute must be boolean")
        if demo_execute and not hft_shadow:
            raise ValueError("--demo-execute requires --hft-shadow")
        if isinstance(hft_max_ticks, bool) or not isinstance(hft_max_ticks, int) or hft_max_ticks < 0:
            raise ValueError("hft_max_ticks must be a non-negative integer")
        try:
            hft_poll_interval = float(hft_poll_interval_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("hft_poll_interval_seconds must be finite and non-negative") from exc
        if not math.isfinite(hft_poll_interval) or hft_poll_interval < 0:
            raise ValueError("hft_poll_interval_seconds must be finite and non-negative")
        if not isinstance(hft_symbol, str) or not hft_symbol.strip():
            raise ValueError("hft_symbol must be non-empty")

        store = self.store_factory(Path(db_path))
        now = datetime.now(UTC)
        lease = _read_active_lease(store, now)
        if lease is not None and lease.lease_expires_at > now:
            print("FOREX SUPERVISOR: WATCHER_ALREADY_RUNNING")
            return 1

        resolved_hft_path = (
            Path(hft_db_path).expanduser()
            if hft_db_path is not None
            else Path(db_path).with_suffix(".hft.sqlite3")
        )
        if hft_shadow:
            from tradingagents.forex.hft.store import HftShadowStore

            hft_lease = HftShadowStore(resolved_hft_path).read_only_active_lease(now)
            if hft_lease is not None and hft_lease.lease_expires_at > now:
                print("FOREX SUPERVISOR: HFT_ALREADY_RUNNING")
                return 1

        runtime = self._runtime()
        hft_context: HftShadowSupervisorContext | None = None
        shared_gate: Any | None = None
        try:
            health = runtime.ensure_healthy()
            if not health.healthy:
                print(f"FOREX SUPERVISOR: OLLAMA_{health.error_code or health.status}")
                return 1

            if hft_shadow:
                from tradingagents.forex.watcher import SerializedMt5OperationGate

                shared_gate = SerializedMt5OperationGate()
                hft_context = self._hft_context(
                    terminal_path=terminal_path,
                    source_db_path=db_path,
                    symbol=hft_symbol,
                    hft_db_path=resolved_hft_path,
                    max_ticks=hft_max_ticks,
                    poll_interval_seconds=hft_poll_interval,
                    gate=shared_gate,
                    demo_execute=demo_execute,
                    demo_db_path=demo_db_path,
                )
            if prewarm:
                runtime.prewarm()
                health = runtime.health()
                health = _refresh_openai_probe(runtime, health)
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
            if hft_shadow:
                args.extend(
                    [
                        "--hft-shadow",
                        "--hft-db-path",
                        str(resolved_hft_path),
                        "--hft-symbol",
                        hft_symbol,
                        "--hft-max-ticks",
                        str(hft_max_ticks),
                        "--hft-poll-interval-seconds",
                        str(hft_poll_interval),
                    ]
                )
            print("FOREX SUPERVISOR: HEALTHY")
            print(f"PROVIDER: {self.runtime_config.provider}")
            print(f"BACKEND: {self.runtime_config.backend_url}")
            print(f"QUICK MODEL: {self.runtime_config.quick_model}")
            print(f"DEEP MODEL: {self.runtime_config.deep_model}")
            print(f"OLLAMA CONTEXT: {self.runtime_config.context_length}")
            print("MT5 FOREX — DEMO MODE" if demo_execute else "NO ORDER WILL BE SENT")
            if demo_execute:
                print("DEMO ACCOUNT ONLY — NO REAL-MONEY ORDERS")
            result = 0
            for attempt in range(max_restarts + 1):
                if hft_context is None:
                    result = int(watch_main(args, runtime_config=self.runtime_config))
                else:
                    result = int(
                        watch_main(
                            args,
                            runtime_config=self.runtime_config,
                            hft_context=hft_context,
                        )
                    )
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
            if hft_context is not None:
                with suppress(Exception):
                    hft_context.stop()
            shutdown = getattr(runtime, "shutdown", None)
            if callable(shutdown):
                cleanup_failed_during_primary_error = sys.exc_info()[0] is not None
                try:
                    shutdown()
                except Exception:
                    if not cleanup_failed_during_primary_error:
                        raise


__all__ = ["ForexSupervisor", "HftShadowSupervisorContext"]
