from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tradingagents.default_config as default_config
from tradingagents.forex.runtime_config import (
    ForexShadowRuntimeConfig,
    collect_runtime_provenance,
)


def test_runtime_config_is_explicit_and_does_not_depend_on_shell_environment():
    original = default_config.DEFAULT_CONFIG
    default_config.DEFAULT_CONFIG = dict(original, forex_quick_max_tokens=999, forex_pm_max_tokens=7)
    try:
        runtime = ForexShadowRuntimeConfig()
        config = runtime.to_tradingagents_config()
    finally:
        default_config.DEFAULT_CONFIG = original

    assert config["llm_provider"] == "ollama"
    assert config["backend_url"] == "http://127.0.0.1:11435/v1"
    assert config["quick_think_llm"] == "qwen3.5:2b"
    assert config["deep_think_llm"] == "qwen3.5:4b"
    assert config["temperature"] == 0
    assert config["max_tokens"] == 1024
    assert config["forex_quick_max_tokens"] == 512
    assert config["forex_pm_max_tokens"] == 2048
    assert config["forex_quick_thinking"] is False
    assert config["forex_deep_thinking"] is True
    assert runtime.context_length == 16384
    assert len(runtime.fingerprint) == 64


def test_runtime_owned_fields_cannot_be_overridden_by_extra():
    with pytest.raises(ValueError, match="runtime-owned configuration"):
        ForexShadowRuntimeConfig(
            extra={
                "llm_provider": "openai",
                "backend_url": "http://127.0.0.1:11434/v1",
                "quick_think_llm": "other-quick-model",
                "deep_think_llm": "other-deep-model",
                "forex_quick_max_tokens": 1,
                "forex_pm_max_tokens": 1,
                "forex_quick_thinking": True,
                "forex_deep_thinking": False,
            }
        )


@pytest.mark.parametrize(
    "field,value",
    [("provider", None), ("provider", 1), ("temperature", "0"), ("temperature", True)],
)
def test_runtime_config_rejects_invalid_provider_and_temperature_types(field, value):
    with pytest.raises(ValueError):
        ForexShadowRuntimeConfig(**{field: value})


def test_additive_extra_settings_change_fingerprint_without_persisting_values():
    first = ForexShadowRuntimeConfig(extra={"custom_runtime_setting": "first-secret"})
    second = ForexShadowRuntimeConfig(extra={"custom_runtime_setting": "second-secret"})

    assert first.fingerprint != second.fingerprint
    safe = json.dumps(first.safe_dict(), sort_keys=True)
    assert "first-secret" not in safe
    assert len(first.safe_dict()["extra_fingerprint"]) == 64


def test_extra_is_detached_and_immutable():
    extra = {"nested": {"value": 1}}
    runtime = ForexShadowRuntimeConfig(extra=extra)
    fingerprint = runtime.fingerprint

    extra["nested"]["value"] = 2
    extra["new"] = True

    assert runtime.fingerprint == fingerprint
    assert runtime.extra["nested"]["value"] == 1
    with pytest.raises(TypeError):
        runtime.extra["new"] = True
    with pytest.raises(TypeError):
        runtime.extra["nested"]["value"] = 2


def test_runtime_provenance_is_safe_and_source_attributable(tmp_path):
    calls = []

    def fake_git(args, **kwargs):
        calls.append((args, kwargs))
        if args[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n")
        return SimpleNamespace(returncode=0, stdout="")

    provenance = collect_runtime_provenance(
        ForexShadowRuntimeConfig(), repo_root=tmp_path, git_runner=fake_git
    )

    assert provenance["git_commit"] == "abc123"
    assert provenance["working_tree_dirty"] is False
    assert provenance["config_fingerprint"]
    assert "secret" not in provenance["safe_config_json"]
    assert len(calls) == 2


def test_runtime_provenance_can_use_effective_direct_watch_config(tmp_path):
    effective = {
        "llm_provider": "ollama",
        "backend_url": "http://localhost:11434/v1",
        "quick_think_llm": "qwen3.5:2b",
        "deep_think_llm": "qwen3.5:4b",
        "max_tokens": 1024,
        "prompt_config_version": "forex-shadow.v1",
        "collector_contract_version": "forex-watch.v1",
        "application_version": "dev",
    }

    provenance = collect_runtime_provenance(
        None,
        safe_config=effective,
        repo_root=tmp_path,
        git_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="abc\n"),
    )

    assert provenance["git_commit"] == "abc"
    assert provenance["prompt_config_version"] == "forex-shadow.v1"
    assert provenance["collector_contract_version"] == "forex-watch.v1"
    assert provenance["application_version"] == "dev"
    assert json.loads(provenance["safe_config_json"]) == effective
