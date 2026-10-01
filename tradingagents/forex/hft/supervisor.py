"""Supervisor-owned lifecycle for the read-only Phase 12C HFT worker."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from tradingagents.forex.watcher import SerializedMt5OperationGate

from .plan_feed import build_plan_from_shadow_decision
from .plan_store import AtomicPlanStore, PlanRejectedError
from .regime_feed import build_regime_from_shadow_decision
from .regime_store import AtomicRegimeStore, RegimeRejectedError
from .runtime import (
    HftLeaseBusyError,
    HftShadowConfig,
    HftShadowRuntime,
    MT5ReadOnlyTickSource,
)
from .store import HftShadowStore

_RECOVERABLE_MT5_ERROR_CODES = frozenset(
    {
        "MT5ACCOUNTDISCONNECTEDERROR",
        "MT5BROKERDISCONNECTEDERROR",
        "MT5TERMINALDISCONNECTEDERROR",
        "MT5NOTINITIALIZEDERROR",
        "MT5INITIALIZATIONFAILED",
        "MT5REINITIALIZATIONFAILED",
    }
)


class HftShadowWorker:
    """Own one HFT runtime thread and one read-only provider lifecycle."""

    def __init__(
        self,
        provider_factory: Callable[..., Any],
        plan_store: AtomicPlanStore,
        *,
        regime_store: AtomicRegimeStore | None = None,
        config: HftShadowConfig,
        store: HftShadowStore | None = None,
        terminal_path: str | None = None,
        mt5_gate: SerializedMt5OperationGate | None = None,
        git_commit: str = "unknown",
        decision_validator: Callable[[Any], bool] | None = None,
        runtime_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if not callable(provider_factory):
            raise TypeError("provider_factory must be callable")
        if not isinstance(plan_store, AtomicPlanStore):
            raise TypeError("plan_store must be AtomicPlanStore")
        if regime_store is not None and not isinstance(regime_store, AtomicRegimeStore):
            raise TypeError("regime_store must be AtomicRegimeStore")
        self.provider_factory = provider_factory
        self.plan_store = plan_store
        self.regime_store = regime_store
        self.config = config
        if config.execution_mode == "DEMO" and runtime_factory is None:
            raise ValueError("DEMO worker requires a dedicated runtime factory")
        self.store = store or HftShadowStore(config.artifact_path)
        self.terminal_path = terminal_path
        self.mt5_gate = mt5_gate or SerializedMt5OperationGate()
        self.git_commit = str(git_commit).strip() or "unknown"
        if decision_validator is not None and not callable(decision_validator):
            raise TypeError("decision_validator must be callable")
        self.decision_validator = decision_validator
        if runtime_factory is not None and not callable(runtime_factory):
            raise TypeError("runtime_factory must be callable")
        self.runtime_factory = runtime_factory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: dict[str, object] | None = None
        self._error_code: str | None = None
        self._active_provider: Any | None = None
        self._recovery_count = 0
        self._consecutive_recovery_attempts = 0
        self._last_disconnect_at: datetime | None = None
        self._last_recovery_attempt_at: datetime | None = None
        self._last_recovery_result: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def result(self) -> dict[str, object] | None:
        return None if self._result is None else dict(self._result)

    @property
    def error_code(self) -> str | None:
        return self._error_code

    @property
    def health(self) -> dict[str, Any]:
        """Return scalar worker health without exposing model or market data."""

        return self.store.read_runtime_state()

    def handle_decision(self, decision: Any) -> bool:
        """Atomically replace or clear the active plan from a decision.

        An ineligible or malformed update clears any prior directional plan so
        the fast path fails closed to ``NONE`` rather than continuing to act on
        stale strategic intent.
        """

        if self.decision_validator is not None:
            try:
                if not bool(self.decision_validator(decision)):
                    self.plan_store.clear()
                    if self.regime_store is not None:
                        self.regime_store.clear()
                    return False
            except Exception:
                self.plan_store.clear()
                if self.regime_store is not None:
                    self.regime_store.clear()
                return False
        regime = build_regime_from_shadow_decision(decision, git_commit=self.git_commit) if self.regime_store is not None else None
        if self.regime_store is not None:
            if regime is None:
                self.regime_store.clear()
                self.plan_store.clear()
                return False
            try:
                self.regime_store.replace(regime, now=datetime.now(timezone.utc))
            except RegimeRejectedError:
                self.regime_store.clear()
                self.plan_store.clear()
                return False
        plan = build_plan_from_shadow_decision(decision, git_commit=self.git_commit)
        if plan is None:
            self.plan_store.clear()
            if self.regime_store is not None:
                self.regime_store.clear()
            return False
        now = datetime.now(timezone.utc)
        try:
            self.plan_store.replace(plan, now=now)
        except PlanRejectedError:
            if self.regime_store is not None:
                self.regime_store.clear()
            self.plan_store.clear()
            return False
        return True

    def _create_provider(self) -> Any:
        try:
            parameters = inspect.signature(self.provider_factory).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "terminal_path" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return self.provider_factory(terminal_path=self.terminal_path)
        return self.provider_factory()

    @staticmethod
    def _classify_error(exc: BaseException) -> str:
        message = str(exc).upper()
        for candidate in _RECOVERABLE_MT5_ERROR_CODES:
            if candidate in message:
                return candidate
        return type(exc).__name__.upper()[:80]

    @staticmethod
    def _is_recoverable(code: str) -> bool:
        return code in _RECOVERABLE_MT5_ERROR_CODES

    def _set_health(self, status: str, *, error_code: str | None = None) -> None:
        """Persist only bounded lifecycle/recovery metadata."""

        with suppress(Exception):
            self.store.set_runtime_state(
                status,
                error_code=error_code,
                last_disconnect_at=self._last_disconnect_at,
                last_recovery_attempt_at=self._last_recovery_attempt_at,
                recovery_count=self._recovery_count,
                last_recovery_result=self._last_recovery_result,
            )

    def _initialize_provider(self, provider: Any) -> None:
        with self.mt5_gate.acquire("hft_initialize"):
            if not provider.initialize():
                raise RuntimeError("MT5_INITIALIZATION_FAILED")

    def _mark_healthy_tick(self) -> None:
        """Reset the bounded reconnect budget only after a real tick arrives."""

        self._consecutive_recovery_attempts = 0

    def _reinitialize_provider(self, provider: Any) -> None:
        with self.mt5_gate.acquire("hft_reinitialize"):
            reinitialize = getattr(provider, "reinitialize", None)
            if callable(reinitialize):
                if not reinitialize():
                    raise RuntimeError("MT5_REINITIALIZATION_FAILED")
                return
            shutdown = getattr(provider, "shutdown", None)
            if callable(shutdown):
                shutdown()
            if not provider.initialize():
                raise RuntimeError("MT5_REINITIALIZATION_FAILED")

    def _recover_provider(self, provider: Any, code: str) -> bool:
        """Perform a bounded reconnect sequence for one disconnect incident."""

        self._last_disconnect_at = datetime.now(timezone.utc)
        if self._consecutive_recovery_attempts >= self.config.max_reconnect_attempts:
            self._last_recovery_result = "OPERATOR_REVIEW_REQUIRED"
            self._set_health("OPERATOR_REVIEW_REQUIRED", error_code=code)
            return False
        while self._consecutive_recovery_attempts < self.config.max_reconnect_attempts:
            self._consecutive_recovery_attempts += 1
            self._recovery_count += 1
            self._last_recovery_attempt_at = datetime.now(timezone.utc)
            self._last_recovery_result = "RETRYING"
            self._set_health("DEGRADED", error_code=code)
            delay = min(
                60.0,
                self.config.reconnect_backoff_seconds
                * (2 ** (self._consecutive_recovery_attempts - 1)),
            )
            if self._stop.wait(delay):
                self._last_recovery_result = "STOPPED"
                self._set_health("STOPPED", error_code=code)
                return False
            self._set_health("RECOVERING", error_code=code)
            try:
                self._reinitialize_provider(provider)
            except Exception as exc:
                code = self._classify_error(exc)
                self._error_code = code
                self._last_recovery_result = "RETRYING"
                self._set_health("DEGRADED", error_code=code)
                continue
            self._error_code = None
            self._last_recovery_result = "RECOVERED"
            self._set_health("RUNNING", error_code=None)
            return True
        self._last_recovery_result = "OPERATOR_REVIEW_REQUIRED"
        self._set_health("OPERATOR_REVIEW_REQUIRED", error_code=code)
        return False

    def _run(self) -> None:
        provider = None
        terminal_status = "STOPPED"
        try:
            provider = self._create_provider()
            self._active_provider = provider
            self._recovery_count = 0
            self._last_disconnect_at = None
            self._last_recovery_attempt_at = None
            self._last_recovery_result = None
            self._set_health("STARTING")
            try:
                self._initialize_provider(provider)
            except Exception as exc:
                code = self._classify_error(exc)
                self._error_code = code
                if not self._is_recoverable(code) or not self._recover_provider(provider, code):
                    terminal_status = "OPERATOR_REVIEW_REQUIRED"
                    return
            while not self._stop.is_set():
                if self.runtime_factory is None:
                    runtime = HftShadowRuntime(
                        MT5ReadOnlyTickSource(provider, point=self.config.point),
                        self.plan_store,
                        config=self.config,
                        store=self.store,
                        mt5_gate=self.mt5_gate,
                        on_tick=self._mark_healthy_tick,
                    )
                else:
                    runtime = self.runtime_factory(provider)
                    if runtime is None or not callable(getattr(runtime, "run", None)):
                        raise TypeError("runtime_factory must return a runnable runtime")
                self._set_health("RUNNING", error_code=None)
                try:
                    self._result = runtime.run(stop_event=self._stop)
                    terminal_status = "STOPPED"
                    return
                except HftLeaseBusyError:
                    self._error_code = "HFT_ALREADY_RUNNING"
                    terminal_status = "OPERATOR_REVIEW_REQUIRED"
                    self._last_recovery_result = "OPERATOR_REVIEW_REQUIRED"
                    self._set_health(terminal_status, error_code=self._error_code)
                    return
                except Exception as exc:
                    code = self._classify_error(exc)
                    self._error_code = code
                    if not self._is_recoverable(code):
                        terminal_status = "OPERATOR_REVIEW_REQUIRED"
                        self._last_recovery_result = "OPERATOR_REVIEW_REQUIRED"
                        self._set_health(terminal_status, error_code=code)
                        return
                    if not self._recover_provider(provider, code):
                        terminal_status = "OPERATOR_REVIEW_REQUIRED"
                        return
        except HftLeaseBusyError:
            self._error_code = "HFT_ALREADY_RUNNING"
            terminal_status = "OPERATOR_REVIEW_REQUIRED"
            self._last_recovery_result = "OPERATOR_REVIEW_REQUIRED"
            self._set_health(terminal_status, error_code=self._error_code)
        except Exception as exc:  # persist only a bounded error category
            self._error_code = self._classify_error(exc)
            terminal_status = "OPERATOR_REVIEW_REQUIRED"
            self._last_recovery_result = "OPERATOR_REVIEW_REQUIRED"
            self._set_health(terminal_status, error_code=self._error_code)
        finally:
            if terminal_status == "STOPPED" and self.health.get("status") != "OPERATOR_REVIEW_REQUIRED":
                self._set_health("STOPPED", error_code=None)
            if provider is not None:
                try:
                    with self.mt5_gate.acquire("hft_shutdown"):
                        provider.shutdown()
                except Exception:
                    if self._error_code is None:
                        self._error_code = "MT5_SHUTDOWN_FAILED"
                self._active_provider = None

    def start(self) -> None:
        if self.running:
            raise RuntimeError("HFT worker is already running")
        self._stop.clear()
        self._result = None
        self._error_code = None
        self._recovery_count = 0
        self._consecutive_recovery_attempts = 0
        self._last_disconnect_at = None
        self._last_recovery_attempt_at = None
        self._last_recovery_result = None
        self._thread = threading.Thread(target=self._run, name="phase12-hft-shadow", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self.join(timeout)


__all__ = ["HftShadowWorker"]
