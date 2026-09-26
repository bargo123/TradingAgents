"""Deterministic runtime configuration and provenance for forex shadow mode.

This module is intentionally separate from the stock CLI configuration.  It
contains no credentials and does not start processes or touch MT5.  The
supervisor uses it to construct a complete, reproducible local Ollama
configuration instead of inheriting arbitrary PowerShell environment state.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import ipaddress
import json
import math
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

_RUNTIME_OWNED_CONFIG_KEYS = frozenset(
    {
        "llm_provider",
        "backend_url",
        "quick_think_llm",
        "deep_think_llm",
        "temperature",
        "max_tokens",
        "forex_quick_max_tokens",
        "forex_pm_max_tokens",
        "forex_quick_thinking",
        "forex_deep_thinking",
        "prompt_config_version",
        "collector_contract_version",
        "application_version",
    }
)


def _normalize_extra(item: Any) -> Any:
    if isinstance(item, Mapping):
        if any(not isinstance(key, str) for key in item):
            raise ValueError("extra mapping keys must be strings")
        return {key: _normalize_extra(item[key]) for key in sorted(item)}
    if isinstance(item, (list, tuple)):
        return [_normalize_extra(child) for child in item]
    if item is None or isinstance(item, (str, int, bool)):
        return item
    if isinstance(item, float) and math.isfinite(item):
        return item
    raise ValueError("extra values must be JSON-safe")


def _freeze_extra(item: Any) -> Any:
    if isinstance(item, Mapping):
        return MappingProxyType({key: _freeze_extra(value) for key, value in item.items()})
    if isinstance(item, list):
        return tuple(_freeze_extra(value) for value in item)
    return item


def _thaw_extra(item: Any) -> Any:
    if isinstance(item, Mapping):
        return {key: _thaw_extra(value) for key, value in item.items()}
    if isinstance(item, tuple):
        return [_thaw_extra(value) for value in item]
    return item


def _canonical_extra_json(value: Mapping[str, Any]) -> str:
    """Serialize additive runtime extensions without retaining their values."""

    return json.dumps(
        _normalize_extra(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


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
        if not isinstance(self.provider, str) or self.provider.strip().casefold() != "ollama":
            raise ValueError("forex shadow runtime provider must be ollama")
        provider = self.provider.strip().casefold()
        for name in ("backend_url", "quick_model", "deep_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        backend_url = self.backend_url.strip()
        quick_model = self.quick_model.strip()
        deep_model = self.deep_model.strip()
        try:
            endpoint = urlsplit(backend_url)
            hostname = endpoint.hostname
            is_loopback = hostname == "localhost"
            if hostname and not is_loopback:
                is_loopback = ipaddress.ip_address(hostname).is_loopback
            endpoint_port = endpoint.port
        except ValueError as exc:
            raise ValueError("backend_url must be a valid local HTTP endpoint") from exc
        if (
            endpoint.scheme != "http"
            or not hostname
            or not is_loopback
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or endpoint.path.rstrip("/") != "/v1"
            or endpoint_port is None
        ):
            raise ValueError(
                "backend_url must be a credential-free loopback HTTP /v1 endpoint with an explicit port"
            )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "backend_url", backend_url)
        object.__setattr__(self, "quick_model", quick_model)
        object.__setattr__(self, "deep_model", deep_model)
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0
            or self.temperature > 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        for name in ("max_tokens", "context_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.quick_thinking, bool) or not isinstance(self.deep_thinking, bool):
            raise ValueError("thinking controls must be bools")
        if not isinstance(self.extra, Mapping):
            raise ValueError("extra must be a mapping")
        normalized_extra = _normalize_extra(self.extra)
        object.__setattr__(self, "extra", _freeze_extra(normalized_extra))
        conflicting = sorted(_RUNTIME_OWNED_CONFIG_KEYS.intersection(self.extra))
        if conflicting:
            raise ValueError(
                "extra cannot override runtime-owned configuration: "
                + ", ".join(conflicting)
            )

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
        # ``extra`` is an additive extension point only.  Runtime-owned
        # provider/model/budget/thinking fields are rejected in __post_init__
        # rather than silently weakening the dedicated shadow contract.
        config.update(_thaw_extra(self.extra))
        return config

    def safe_dict(self) -> dict[str, Any]:
        safe = {
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
        if self.extra:
            safe["extra_fingerprint"] = hashlib.sha256(
                _canonical_extra_json(self.extra).encode()
            ).hexdigest()
        return safe

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
