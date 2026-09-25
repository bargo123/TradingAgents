"""Deterministic runtime configuration and provenance for forex shadow mode.

This module is intentionally separate from the stock CLI configuration.  It
contains no credentials and does not start processes or touch MT5.  The
supervisor uses it to construct a complete, reproducible local Ollama
configuration instead of inheriting arbitrary PowerShell environment state.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ForexShadowRuntimeConfig:
    """Validated, non-secret configuration used by the autonomous launcher."""

    provider: str = "ollama"
    backend_url: str = "http://127.0.0.1:11435/v1"
    quick_model: str = "qwen3.5:2b"
    deep_model: str = "qwen3.5:4b"
    temperature: float = 0.0
    max_tokens: int = 1024
    context_length: int = 16384
    quick_thinking: bool = False
    deep_thinking: bool = True
    prompt_config_version: str = "forex-shadow.runtime.v1"
    collector_contract_version: str = "forex-watch.v1"
    application_version: str = "unknown"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider.casefold() != "ollama":
            raise ValueError("forex shadow runtime provider must be ollama")
        for name in ("backend_url", "quick_model", "deep_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.backend_url.rstrip("/").endswith("/v1"):
            raise ValueError("backend_url must point at an OpenAI-compatible /v1 endpoint")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        for name in ("max_tokens", "context_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.quick_thinking, bool) or not isinstance(self.deep_thinking, bool):
            raise ValueError("thinking controls must be bools")

    @property
    def ollama_base_url(self) -> str:
        return self.backend_url.rsplit("/v1", 1)[0].rstrip("/")

    def to_tradingagents_config(self, base: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return a complete graph config while preserving unrelated defaults."""

        from tradingagents.default_config import DEFAULT_CONFIG

        config = dict(DEFAULT_CONFIG)
        if base:
            config.update(dict(base))
        config.update(
            {
                "llm_provider": self.provider,
                "backend_url": self.backend_url,
                "quick_think_llm": self.quick_model,
                "deep_think_llm": self.deep_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                # These bounds are part of the approved forex contract and
                # must not be inherited from arbitrary shell overrides.
                "forex_quick_max_tokens": 512,
                "forex_pm_max_tokens": 2048,
                "forex_quick_thinking": self.quick_thinking,
                "forex_deep_thinking": self.deep_thinking,
                "prompt_config_version": self.prompt_config_version,
                "collector_contract_version": self.collector_contract_version,
                "application_version": self.application_version,
            }
        )
        config.update(dict(self.extra))
        return config

    def safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "backend_url": self.backend_url,
            "quick_model": self.quick_model,
            "deep_model": self.deep_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "quick_max_tokens": 512,
            "pm_max_tokens": 2048,
            "context_length": self.context_length,
            "quick_thinking": self.quick_thinking,
            "deep_thinking": self.deep_thinking,
            "prompt_config_version": self.prompt_config_version,
            "collector_contract_version": self.collector_contract_version,
            "application_version": self.application_version,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.safe_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _git_value(
    args: list[str],
    *,
    repo_root: Path,
    runner: Callable[..., Any] = subprocess.run,
) -> str | None:
    try:
        result = runner(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    value = str(getattr(result, "stdout", "")).strip()
    return value or None


def collect_runtime_provenance(
    runtime: ForexShadowRuntimeConfig | None = None,
    *,
    safe_config: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    git_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Collect safe source provenance without reading prompts or credentials.

    The supervisor supplies a typed runtime config.  Direct ``forex-watch``
    callers may instead provide the already allow-listed effective config so
    their provenance cannot be mislabeled as the supervisor's dedicated
    endpoint.
    """

    if runtime is None and safe_config is None:
        raise ValueError("runtime or safe_config is required")

    root = Path(repo_root or Path(__file__).resolve().parents[2])
    commit = _git_value(["rev-parse", "HEAD"], repo_root=root, runner=git_runner)
    dirty_text = _git_value(
        ["status", "--porcelain", "--untracked-files=no"],
        repo_root=root,
        runner=git_runner,
    )
    try:
        version = importlib.metadata.version("tradingagents")
    except importlib.metadata.PackageNotFoundError:
        version = (
            runtime.application_version
            if runtime is not None
            else str((safe_config or {}).get("application_version") or "unknown")
        )
    if runtime is not None:
        safe_values = runtime.safe_dict()
        prompt_version = runtime.prompt_config_version
        collector_version = runtime.collector_contract_version
        application_version = runtime.application_version
    else:
        safe_values = {str(key): value for key, value in (safe_config or {}).items()}
        prompt_version = str(safe_values.get("prompt_config_version") or "forex-shadow.v1")
        collector_version = str(
            safe_values.get("collector_contract_version") or "forex-watch.v1"
        )
        application_version = str(safe_values.get("application_version") or version)
    safe_config_json = json.dumps(
        safe_values,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return {
        "git_commit": commit,
        "working_tree_dirty": bool(dirty_text),
        "application_version": application_version or version,
        "prompt_config_version": prompt_version,
        "collector_contract_version": collector_version,
        "config_fingerprint": hashlib.sha256(safe_config_json.encode()).hexdigest(),
        "safe_config_json": safe_config_json,
    }


__all__ = [
    "ForexShadowRuntimeConfig",
    "collect_runtime_provenance",
]
