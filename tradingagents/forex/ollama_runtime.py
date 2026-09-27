"""Bounded lifecycle and health checks for the dedicated local Ollama server.

Only a supervisor-owned child may be terminated by this module. The normal
Ollama instance (normally listening on 11434) is never touched. Health checks
are scalar-only and never persist prompts, completions, or reasoning.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit

import requests

from .runtime_config import ForexShadowRuntimeConfig


@dataclass(frozen=True, slots=True)
class OllamaHealth:
    status: str
    endpoint: str
    version: str | None
    models: tuple[str, ...]
    context_length: int | None
    error_code: str | None = None
    server_healthy: bool = False
    loaded_models: tuple[str, ...] = ()
    quick_context_verified: bool = False
    deep_context_verified: bool = False
    verified_context_length: int | None = None
    openai_probe_ok: bool | None = None
    dedicated_pid: int | None = None
    recovery_attempts: int = 0

    @property
    def healthy(self) -> bool:
        return self.status == "HEALTHY"


def _response_json(response: Any) -> Mapping[str, Any]:
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    payload = response.json() if callable(getattr(response, "json", None)) else response
    return payload if isinstance(payload, Mapping) else {}


def _model_name(item: Any) -> str | None:
    if not isinstance(item, Mapping):
        return None
    for key in ("name", "model"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _context_value(item: Any) -> int | None:
    """Read only documented runtime context fields from an /api/ps item."""

    if not isinstance(item, Mapping):
        return None
    for key in ("context_length", "context", "context_window"):
        value = item.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit() and int(value) > 0:
            return int(value)
    for key in ("details", "options", "model_info"):
        nested = item.get(key)
        value = _context_value(nested)
        if value is not None:
            return value
    return None


class DedicatedOllamaRuntime:
    """Probe/start one explicitly owned local Ollama endpoint."""

    def __init__(
        self,
        config: ForexShadowRuntimeConfig,
        *,
        http: Any = requests,
        process_launcher: Callable[..., Any] = subprocess.Popen,
        version_runner: Callable[..., Any] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
        probe_attempts: int = 12,
        probe_interval_seconds: float = 0.5,
        max_recovery_attempts: int = 2,
        prewarm_timeout_seconds: float = 30.0,
    ) -> None:
        if isinstance(probe_attempts, bool) or not isinstance(probe_attempts, int) or probe_attempts <= 0:
            raise ValueError("probe_attempts must be a positive integer")
        if (
            isinstance(probe_interval_seconds, bool)
            or not isinstance(probe_interval_seconds, (int, float))
            or not math.isfinite(float(probe_interval_seconds))
            or probe_interval_seconds < 0
        ):
            raise ValueError("probe_interval_seconds must be finite and non-negative")
        if isinstance(max_recovery_attempts, bool) or not isinstance(max_recovery_attempts, int) or max_recovery_attempts < 0:
            raise ValueError("max_recovery_attempts must be a non-negative integer")
        if (
            isinstance(prewarm_timeout_seconds, bool)
            or not isinstance(prewarm_timeout_seconds, (int, float))
            or not math.isfinite(float(prewarm_timeout_seconds))
            or prewarm_timeout_seconds <= 0
        ):
            raise ValueError("prewarm_timeout_seconds must be finite and positive")
        self.config = config
        self.http = http
        self.process_launcher = process_launcher
        self.version_runner = version_runner
        self.sleep = sleep
        self.probe_attempts = probe_attempts
        self.probe_interval_seconds = float(probe_interval_seconds)
        self.max_recovery_attempts = max_recovery_attempts
        self.prewarm_timeout_seconds = float(prewarm_timeout_seconds)
        self._owned_process: Any | None = None
        self._verified_contexts: dict[str, int] = {}
        self._recovery_attempts = 0

    @property
    def dedicated_pid(self) -> int | None:
        process = self._owned_process
        value = getattr(process, "pid", None) if process is not None else None
        return value if isinstance(value, int) and value > 0 else None

    def installed_version(self) -> str | None:
        try:
            result = self.version_runner(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if getattr(result, "returncode", 1) != 0:
            return None
        output = str(getattr(result, "stdout", "")).strip()
        return output[:120] if output else None

    def _api_version(self) -> str:
        payload = _response_json(
            self.http.get(f"{self.config.ollama_base_url}/api/version", timeout=3)
        )
        version = payload.get("version")
        if not isinstance(version, str) or not version.strip():
            raise RuntimeError("OLLAMA_VERSION_MISSING")
        return version.strip()[:120]

    def _read_server_state(self) -> tuple[str, tuple[str, ...], dict[str, int | None]]:
        base = self.config.ollama_base_url
        server_version = self._api_version()
        tags = _response_json(self.http.get(f"{base}/api/tags", timeout=3))
        entries = tags.get("models", [])
        models = tuple(sorted(filter(None, (_model_name(item) for item in entries))))
        ps = _response_json(self.http.get(f"{base}/api/ps", timeout=3))
        loaded: dict[str, int | None] = {}
        for item in ps.get("models", []):
            name = _model_name(item)
            if name:
                loaded[name] = _context_value(item)
        return server_version, models, loaded

    def _make_health(
        self,
        *,
        status: str,
        version: str | None,
        models: tuple[str, ...],
        loaded: Mapping[str, int | None],
        context_length: int | None,
        error_code: str | None,
        server_healthy: bool,
    ) -> OllamaHealth:
        required = {self.config.quick_model, self.config.deep_model}
        verified = {
            model: value
            for model, value in self._verified_contexts.items()
            if model in required and value >= self.config.context_length
        }
        current_valid = (
            bool(loaded)
            and set(loaded).issubset(required)
            and all(
                value is not None and value >= self.config.context_length
                for value in loaded.values()
            )
        )
        if status == "CONTEXT_NOT_VERIFIED" and verified.keys() == required and current_valid:
            status = "HEALTHY"
            error_code = None
        return OllamaHealth(
            status="HEALTHY" if status == "HEALTHY" else "DEGRADED",
            endpoint=self.config.ollama_base_url,
            version=version,
            models=models,
            context_length=context_length,
            error_code=error_code,
            server_healthy=server_healthy,
            loaded_models=tuple(sorted(loaded)),
            quick_context_verified=self.config.quick_model in verified,
            deep_context_verified=self.config.deep_model in verified,
            verified_context_length=(
                min(verified.values()) if verified.keys() == required else None
            ),
            dedicated_pid=self.dedicated_pid,
            recovery_attempts=self._recovery_attempts,
        )

    def _observe(self) -> tuple[OllamaHealth, dict[str, int | None]]:
        version = self.installed_version()
        try:
            server_version, models, loaded = self._read_server_state()
            version = version or server_version
            required = {self.config.quick_model, self.config.deep_model}
            missing = sorted(required - set(models))
            if missing:
                return (
                    self._make_health(
                        status="DEGRADED",
                        version=version,
                        models=models,
                        loaded=loaded,
                        context_length=None,
                        error_code="REQUIRED_MODEL_MISSING",
                        server_healthy=True,
                    ),
                    loaded,
                )
            observed = [value for value in loaded.values() if value is not None]
            context_length = max(observed) if observed else None
            if any(
                value is not None and value < self.config.context_length
                for value in loaded.values()
            ):
                error_code = "CONTEXT_TOO_SMALL"
            elif (
                not loaded
                or not set(loaded).issubset(required)
                or any(value is None for value in loaded.values())
                or any(
                    self._verified_contexts.get(model, 0) < self.config.context_length
                    for model in required
                )
            ):
                error_code = "CONTEXT_NOT_VERIFIED"
            else:
                error_code = None
            status = "HEALTHY" if error_code is None else "DEGRADED"
            health = self._make_health(
                status=status,
                version=version,
                models=models,
                loaded=loaded,
                context_length=context_length,
                error_code=error_code,
                server_healthy=True,
            )
            return health, loaded
        except Exception as exc:  # health reporting must remain bounded
            return (
                OllamaHealth(
                    status="UNAVAILABLE",
                    endpoint=self.config.ollama_base_url,
                    version=version,
                    models=(),
                    context_length=None,
                    error_code=type(exc).__name__.upper(),
                    server_healthy=False,
                    dedicated_pid=self.dedicated_pid,
                    recovery_attempts=self._recovery_attempts,
                ),
                {},
            )

    def health(self) -> OllamaHealth:
        return self._observe()[0]

    def _process_alive(self) -> bool:
        process = self._owned_process
        if process is None:
            return False
        poll = getattr(process, "poll", None)
        return not callable(poll) or poll() is None

    def _start_owned_server(self) -> None:
        if self._process_alive():
            return
        self._owned_process = None
        endpoint = urlsplit(self.config.ollama_base_url)
        hostname = endpoint.hostname
        if not hostname or endpoint.port is None:
            raise RuntimeError("OLLAMA_LOCAL_ENDPOINT_INVALID")
        host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
        env = os.environ.copy()
        env.update(
            {
                "OLLAMA_HOST": f"{host}:{endpoint.port}",
                "OLLAMA_CONTEXT_LENGTH": str(self.config.context_length),
                "OLLAMA_MAX_LOADED_MODELS": "1",
                "OLLAMA_NUM_PARALLEL": "1",
            }
        )
        self._owned_process = self.process_launcher(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            # No reader is attached to the child stderr stream.  Keeping a
            # PIPE here can eventually block the long-lived server when its
            # diagnostics fill the OS pipe buffer.
            stderr=subprocess.DEVNULL,
        )
        self._verified_contexts.clear()

    def _stop_owned_server(self) -> None:
        process = self._owned_process
        self._owned_process = None
        if process is None:
            return
        poll = getattr(process, "poll", None)
        if callable(poll) and poll() is not None:
            return
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            with suppress(OSError, subprocess.SubprocessError, TimeoutError):
                terminate()
        wait = getattr(process, "wait", None)
        try:
            if callable(wait):
                wait(timeout=5)
        except (OSError, subprocess.SubprocessError, TimeoutError):
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
            try:
                if callable(wait):
                    wait(timeout=5)
            except (OSError, subprocess.SubprocessError, TimeoutError):
                pass

    def shutdown(self) -> None:
        """Stop only the Ollama process started and owned by this runtime."""

        self._stop_owned_server()

    def _wait_for_server(self) -> OllamaHealth:
        current = self.health()
        for _ in range(self.probe_attempts):
            if current.status != "UNAVAILABLE":
                return current
            self.sleep(self.probe_interval_seconds)
            current = self.health()
        return current

    def _prewarm_model(self, model: str) -> str:
        try:
            response = self.http.post(
                f"{self.config.ollama_base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply OK."}],
                    "stream": False,
                    "think": False,
                    "options": {"num_predict": 1},
                },
                timeout=self.prewarm_timeout_seconds,
            )
            _response_json(response)
            return "OK"
        except Exception as exc:
            return type(exc).__name__.upper()

    def _verify_model(self, model: str) -> OllamaHealth:
        last = self.health()
        for attempt in range(self.probe_attempts):
            last, loaded = self._observe()
            value = loaded.get(model)
            if value is not None:
                if value < self.config.context_length:
                    return last
                self._verified_contexts[model] = value
                return self.health()
            if attempt + 1 < self.probe_attempts:
                self.sleep(self.probe_interval_seconds)
        return last

    def _probe_after_verification(self, health: OllamaHealth) -> OllamaHealth:
        probe_error = self.probe_openai_compatible()
        if probe_error is None:
            return replace(health, openai_probe_ok=True)
        return replace(
            health,
            status="DEGRADED",
            error_code=f"OPENAI_PROBE_{probe_error}",
            openai_probe_ok=False,
        )

    def ensure_healthy(self) -> OllamaHealth:
        """Boundedly start/recover the dedicated server and verify both models."""

        for recovery in range(self.max_recovery_attempts + 1):
            self._recovery_attempts = recovery
            current = self.health()
            if current.status == "UNAVAILABLE":
                if not self._process_alive():
                    self._start_owned_server()
                current = self._wait_for_server()
                if current.status == "UNAVAILABLE":
                    if recovery < self.max_recovery_attempts and self._process_alive():
                        self._stop_owned_server()
                    continue
            if current.error_code == "REQUIRED_MODEL_MISSING":
                return current
            if current.error_code == "CONTEXT_TOO_SMALL":
                if not self._process_alive() and recovery == 0:
                    return current
                if not self._process_alive():
                    self._start_owned_server()
                    continue
                self._stop_owned_server()
                continue

            for model in (self.config.quick_model, self.config.deep_model):
                if self._verified_contexts.get(model, 0) >= self.config.context_length:
                    continue
                self._prewarm_model(model)
                current = self._verify_model(model)
                if current.error_code == "CONTEXT_TOO_SMALL":
                    break
            else:
                current = self.health()
                if current.healthy:
                    return self._probe_after_verification(current)

            # A prewarm can time out while Ollama is still loading a model.
            # Treat an unverified context as a bounded, recoverable state and
            # consume the configured recovery budget before failing closed.
            if current.error_code == "CONTEXT_NOT_VERIFIED" and recovery < self.max_recovery_attempts:
                continue

            if current.error_code == "CONTEXT_TOO_SMALL" and self._process_alive():
                self._stop_owned_server()
                continue
            if current.status == "UNAVAILABLE" and self._process_alive():
                self._stop_owned_server()
                continue
            return current
        return self.health()

    def prewarm(self) -> dict[str, Any]:
        """Warm both models with bounded, content-free health requests."""

        return {
            model: self._prewarm_model(model)
            for model in (self.config.quick_model, self.config.deep_model)
        }

    def probe_openai_compatible(self) -> str | None:
        """Run one bounded scalar-only ``/v1/chat/completions`` probe."""

        try:
            response = self.http.post(
                f"{self.config.backend_url.rstrip('/')}/chat/completions",
                json={
                    "model": self.config.quick_model,
                    "messages": [{"role": "user", "content": "Reply OK."}],
                    "temperature": 0,
                    "max_tokens": 1,
                    "reasoning_effort": "none",
                    "stream": False,
                },
                timeout=10,
            )
            payload = _response_json(response)
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                return "OPENAI_PROBE_EMPTY"
            return None
        except Exception as exc:
            return type(exc).__name__.upper()


__all__ = ["DedicatedOllamaRuntime", "OllamaHealth"]
