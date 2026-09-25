from __future__ import annotations

from types import SimpleNamespace

from tradingagents.forex.ollama_runtime import DedicatedOllamaRuntime
from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig


class _Http:
    def __init__(self):
        self.gets = []
        self.posts = []

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        if url.endswith("/api/tags"):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"models": [{"name": "qwen3.5:2b"}, {"name": "qwen3.5:4b"}]},
            )
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"models": [{"name": "qwen3.5:4b", "context_length": 16384}]},
        )

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if url.endswith("/chat/completions"):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": "OK"}}]},
            )
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"done": True})


def test_dedicated_runtime_health_checks_models_and_context():
    http = _Http()
    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=http,
        version_runner=lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="ollama version 0.12"
        ),
    )

    health = runtime.health()

    assert health.healthy
    assert health.context_length == 16384
    assert health.models == ("qwen3.5:2b", "qwen3.5:4b")
    assert health.version == "ollama version 0.12"


def test_dedicated_runtime_starts_only_owned_server_with_bounded_probe():
    http = _Http()
    launched = []

    def launch(command, **kwargs):
        launched.append((command, kwargs))
        return SimpleNamespace(poll=lambda: None)

    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=http,
        process_launcher=launch,
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
        sleep=lambda _: None,
        probe_attempts=1,
    )
    health = runtime.ensure_healthy()

    assert health.healthy
    assert launched == []  # existing healthy endpoint is never disturbed


def test_context_must_be_observed_after_model_load():
    class UnknownContextHttp(_Http):
        def get(self, url, **kwargs):
            response = super().get(url, **kwargs)
            if url.endswith("/api/ps"):
                response.json = lambda: {"models": []}
            return response

    health = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=UnknownContextHttp(),
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
    ).health()

    assert health.status == "DEGRADED"
    assert health.error_code == "CONTEXT_NOT_VERIFIED"


def test_prewarm_uses_bounded_non_persistent_health_requests():
    http = _Http()
    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=http,
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
    )

    result = runtime.prewarm()

    assert result == {"qwen3.5:2b": "OK", "qwen3.5:4b": "OK"}
    assert len(http.posts) == 2
    assert all(item[1]["json"]["stream"] is False for item in http.posts)
    assert all(item[1]["json"]["options"]["num_predict"] == 1 for item in http.posts)


def test_openai_compatible_probe_uses_documented_bounded_fields():
    http = _Http()
    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=http,
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
    )

    assert runtime.probe_openai_compatible() is None
    url, request = http.posts[-1]
    assert url.endswith("/v1/chat/completions")
    assert request["json"]["max_tokens"] == 1
    assert request["json"]["reasoning_effort"] == "none"
