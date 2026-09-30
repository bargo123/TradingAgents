"""Supervisor-owned lifecycle for the read-only Phase 12C HFT worker."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from tradingagents.forex.watcher import SerializedMt5OperationGate

from .plan_feed import build_plan_from_shadow_decision
from .plan_store import AtomicPlanStore, PlanRejectedError
from .runtime import HftLeaseBusyError, HftShadowConfig, HftShadowRuntime, MT5ReadOnlyTickSource
from .store import HftShadowStore


class HftShadowWorker:
    """Own one HFT runtime thread and one read-only provider lifecycle."""

    def __init__(
        self,
        provider_factory: Callable[..., Any],
        plan_store: AtomicPlanStore,
        *,
        config: HftShadowConfig,
        store: HftShadowStore | None = None,
        terminal_path: str | None = None,
        mt5_gate: SerializedMt5OperationGate | None = None,
        git_commit: str = "unknown",
    ) -> None:
        if not callable(provider_factory):
            raise TypeError("provider_factory must be callable")
        if not isinstance(plan_store, AtomicPlanStore):
            raise TypeError("plan_store must be AtomicPlanStore")
        self.provider_factory = provider_factory
        self.plan_store = plan_store
        self.config = config
        self.store = store or HftShadowStore(config.artifact_path)
        self.terminal_path = terminal_path
        self.mt5_gate = mt5_gate or SerializedMt5OperationGate()
        self.git_commit = str(git_commit).strip() or "unknown"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: dict[str, object] | None = None
        self._error_code: str | None = None
        self._active_provider: Any | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def result(self) -> dict[str, object] | None:
        return None if self._result is None else dict(self._result)

    @property
    def error_code(self) -> str | None:
        return self._error_code

    def handle_decision(self, decision: Any) -> bool:
        """Atomically replace the active plan when the decision is eligible."""

        plan = build_plan_from_shadow_decision(decision, git_commit=self.git_commit)
        if plan is None:
            return False
        now = datetime.now(timezone.utc)
        try:
            self.plan_store.replace(plan, now=now)
        except PlanRejectedError:
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

    def _run(self) -> None:
        provider = None
        try:
            provider = self._create_provider()
            self._active_provider = provider
            with self.mt5_gate.acquire("hft_initialize"):
                if not provider.initialize():
                    raise RuntimeError("MT5_INITIALIZATION_FAILED")
            runtime = HftShadowRuntime(
                MT5ReadOnlyTickSource(provider, point=self.config.point),
                self.plan_store,
                config=self.config,
                store=self.store,
                mt5_gate=self.mt5_gate,
            )
            self._result = runtime.run(stop_event=self._stop)
        except HftLeaseBusyError:
            self._error_code = "HFT_ALREADY_RUNNING"
        except Exception as exc:  # persist only a bounded error category
            self._error_code = type(exc).__name__.upper()
        finally:
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
