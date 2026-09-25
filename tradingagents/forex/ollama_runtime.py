"""Bounded lifecycle and health checks for the dedicated local Ollama server.

Only a supervisor owns the process started by this module.  Existing Ollama
instances on other ports are never terminated or reconfigured.  Health checks
are scalar-only and never persist prompts, completions, or reasoning.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

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
    ) -> None:
        self.config = config
        self.http = http
        self.process_launcher = process_launcher
        self.version_runner = version_runner
        self.sleep = sleep
        self.probe_attempts = max(1, int(probe_attempts))
        self.probe_interval_seconds = max(0.0, float(probe_interval_seconds))
        self._owned_process: Any | None = None

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

    def health(self) -> OllamaHealth:
        base = self.config.ollama_base_url
        version = self.installed_version()
        try:
            tags = _response_json(self.http.get(f"{base}/api/tags", timeout=3))
            entries = tags.get("models", [])
            models = tuple(sorted(filter(None, (_model_name(item) for item in entries))))
            required = {self.config.quick_model, self.config.deep_model}
            missing = sorted(required - set(models))
            if missing:
                return OllamaHealth(
                    "DEGRADED", base, version, models, None, "REQUIRED_MODEL_MISSING"
                )
            ps = _response_json(self.http.get(f"{base}/api/ps", timeout=3))
            context_length = None
            for item in ps.get("models", []):
                if not isinstance(item, Mapping):
                    continue
                for key in ("context_length", "context", "context_window"):
                    value = item.get(key)
                    if isinstance(value, int) and value > 0:
                        context_length = max(context_length or 0, value)
                        break
            if context_length is not None and context_length < self.config.context_length:
                return OllamaHealth(
                    "DEGRADED", base, version, models, context_length, "CONTEXT_TOO_SMALL"
                )
            if context_length is None:
                return OllamaHealth(
                    "DEGRADED", base, version, models, None, "CONTEXT_NOT_VERIFIED"
                )
            return OllamaHealth("HEALTHY", base, version, models, context_length)
        except Exception as exc:  # health reporting must remain bounded
            return OllamaHealth(
                "UNAVAILABLE", base, version, (), None, type(exc).__name__.upper()
            )

    def _start_owned_server(self) -> None:
        if self._owned_process is not None:
            poll = getattr(self._owned_process, "poll", None)
            if callable(poll) and poll() is None:
                return
        env = os.environ.copy()
        env.update(
            {
                "OLLAMA_HOST": "127.0.0.1:11435",
                "OLLAMA_CONTEXT_LENGTH": str(self.config.context_length),
                "OLLAMA_MAX_LOADED_MODELS": "1",
                "OLLAMA_NUM_PARALLEL": "1",
            }
        )
        self._owned_process = self.process_launcher(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def ensure_healthy(self) -> OllamaHealth:
        current = self.health()
        if current.healthy:
            probe_error = self.probe_openai_compatible()
            return (
                current
                if probe_error is None
                else replace(current, status="DEGRADED", error_code=f"OPENAI_PROBE_{probe_error}")
            )
        # A responsive endpoint on the dedicated port is not ours to kill or
        # replace.  If models are present but context is not yet observable,
        # warm them and re-check; a too-small or missing-model endpoint is a
        # visible configuration fault, not permission to launch a second server.
        if current.status != "UNAVAILABLE":
            if (
                current.error_code == "CONTEXT_NOT_VERIFIED"
                and {self.config.quick_model, self.config.deep_model} <= set(current.models)
            ):
                self.prewarm()
                current = self.health()
                if current.healthy:
                    probe_error = self.probe_openai_compatible()
                    if probe_error is not None:
                        return replace(
                            current,
                            status="DEGRADED",
                            error_code=f"OPENAI_PROBE_{probe_error}",
                        )
                return current
            return current
        self._start_owned_server()
        prewarmed = False
        for _ in range(self.probe_attempts):
            self.sleep(self.probe_interval_seconds)
            current = self.health()
            if (
                current.error_code == "CONTEXT_NOT_VERIFIED"
                and {self.config.quick_model, self.config.deep_model} <= set(current.models)
                and not prewarmed
            ):
                self.prewarm()
                prewarmed = True
                continue
            if current.healthy:
                probe_error = self.probe_openai_compatible()
                if probe_error is None:
                    return current
                current = replace(
                    current,
                    status="DEGRADED",
                    error_code=f"OPENAI_PROBE_{probe_error}",
                )
        return current

    def prewarm(self) -> dict[str, Any]:
        """Warm both models with bounded, content-free health prompts."""

        results: dict[str, Any] = {}
        base = self.config.ollama_base_url
        for model in (self.config.quick_model, self.config.deep_model):
            try:
                response = self.http.post(
                    f"{base}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Reply OK."}],
                        "stream": False,
                        "think": False,
                        "options": {"num_predict": 1},
                    },
                    timeout=10,
                )
                _response_json(response)
                results[model] = "OK"
            except Exception as exc:
                results[model] = type(exc).__name__.upper()
        return results

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
