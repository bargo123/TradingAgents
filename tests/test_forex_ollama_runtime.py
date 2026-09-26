from __future__ import annotations

from types import SimpleNamespace

from tradingagents.forex.ollama_runtime import DedicatedOllamaRuntime
from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig


class _Http:
    def __init__(self):
        self.gets = []
        self.posts = []
        self.loaded_model = "qwen3.5:4b"

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        if url.endswith("/api/version"):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"version": "0.12"},
            )
        if url.endswith("/api/tags"):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"models": [{"name": "qwen3.5:2b"}, {"name": "qwen3.5:4b"}]},
            )
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "models": [{"name": self.loaded_model, "context_length": 16384}]
            },
        )

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if url.endswith("/chat/completions"):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": "OK"}}]},
            )
        self.loaded_model = kwargs["json"]["model"]
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

    assert health.error_code == "CONTEXT_NOT_VERIFIED"
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


def test_health_verifies_server_version_endpoint():
    http = _Http()
    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=http,
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
    )

    runtime.health()

    assert any(url.endswith("/api/version") for url, _ in http.gets)


def test_empty_ps_recovers_and_verifies_each_model_after_swap():
    class SwapHttp(_Http):
        def __init__(self):
            super().__init__()
            self.loaded = None

        def get(self, url, **kwargs):
            response = super().get(url, **kwargs)
            if url.endswith("/api/ps"):
                response.json = lambda: (
                    {"models": []}
                    if self.loaded is None
                    else {"models": [{"name": self.loaded, "context_length": 16384}]}
                )
            return response

        def post(self, url, **kwargs):
            response = super().post(url, **kwargs)
            if url.endswith("/api/chat"):
                self.loaded = kwargs["json"]["model"]
            return response

    http = SwapHttp()
    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=http,
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
        sleep=lambda _: None,
    )

    health = runtime.ensure_healthy()

    assert health.healthy
    assert health.quick_context_verified is True
    assert health.deep_context_verified is True
    assert health.verified_context_length == 16384


def test_owned_wrong_context_restarts_only_owned_process():
    class WrongThenCorrectHttp(_Http):
        def __init__(self):
            super().__init__()
            self.correct = False

        def get(self, url, **kwargs):
            response = super().get(url, **kwargs)
            if url.endswith("/api/ps"):
                context = 16384 if self.correct else 4096
                response.json = lambda: {
                    "models": [{"name": self.loaded_model, "context_length": context}]
                }
            return response

    class Process:
        def __init__(self):
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.killed = True

    http = WrongThenCorrectHttp()
    processes = []

    def launch(command, **kwargs):
        process = Process()
        processes.append(process)
        if len(processes) == 2:
            http.correct = True
        return process

    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=http,
        process_launcher=launch,
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
        sleep=lambda _: None,
        probe_attempts=2,
    )
    runtime._start_owned_server()

    health = runtime.ensure_healthy()

    assert health.healthy
    assert len(processes) == 2
    assert processes[0].terminated is True
    assert processes[0].killed is False


def test_unowned_wrong_context_is_never_terminated():
    class WrongHttp(_Http):
        def get(self, url, **kwargs):
            response = super().get(url, **kwargs)
            if url.endswith("/api/ps"):
                response.json = lambda: {
                    "models": [{"name": "qwen3.5:4b", "context_length": 4096}]
                }
            return response

    http = WrongHttp()
    launches = []
    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=http,
        process_launcher=lambda *args, **kwargs: launches.append(True),
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
    )

    health = runtime.ensure_healthy()

    assert health.error_code == "CONTEXT_TOO_SMALL"
    assert launches == []


def test_unavailable_server_starts_with_explicit_dedicated_environment():
    class StartsHttp(_Http):
        def __init__(self):
            super().__init__()
            self.started = False

        def get(self, url, **kwargs):
            if not self.started:
                raise OSError("offline")
            return super().get(url, **kwargs)

        def post(self, url, **kwargs):
            self.started = True
            return super().post(url, **kwargs)

    http = StartsHttp()
    launches = []

    def launch(command, **kwargs):
        launches.append((command, kwargs))
        http.started = True
        return SimpleNamespace(poll=lambda: None)

    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=http,
        process_launcher=launch,
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
        sleep=lambda _: None,
        probe_attempts=3,
    )

    health = runtime.ensure_healthy()

    assert health.healthy
    assert launches
    env = launches[0][1]["env"]
    assert env["OLLAMA_HOST"] == "127.0.0.1:11435"
    assert env["OLLAMA_CONTEXT_LENGTH"] == "16384"
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert env["OLLAMA_NUM_PARALLEL"] == "1"


def test_unavailable_server_uses_configured_loopback_endpoint():
    class StartsHttp(_Http):
        def __init__(self):
            super().__init__()
            self.started = False

        def get(self, _url, **_kwargs):
            if not self.started:
                raise OSError("offline")
            return super().get(_url, **_kwargs)

        def post(self, _url, **kwargs):
            self.started = True
            return super().post(_url, **kwargs)

    http = StartsHttp()
    launches = []

    def launch(command, **kwargs):
        launches.append((command, kwargs))
        http.started = True
        return SimpleNamespace(poll=lambda: None)

    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(backend_url="http://127.0.0.1:12345/v1"),
        http=http,
        process_launcher=launch,
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
        sleep=lambda _: None,
        probe_attempts=3,
    )

    health = runtime.ensure_healthy()

    assert health.healthy
    assert launches[0][1]["env"]["OLLAMA_HOST"] == "127.0.0.1:12345"


def test_repeated_startup_failure_is_bounded_and_operator_visible():
    class OfflineHttp:
        def get(self, _url, **_kwargs):
            raise OSError("offline")

        def post(self, _url, **_kwargs):
            raise OSError("offline")

    class Process:
        pid = 9001

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    launches = []
    runtime = DedicatedOllamaRuntime(
        ForexShadowRuntimeConfig(),
        http=OfflineHttp(),
        process_launcher=lambda *args, **kwargs: (launches.append(True) or Process()),
        version_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="v"),
        sleep=lambda _: None,
        probe_attempts=1,
        max_recovery_attempts=2,
    )

    health = runtime.ensure_healthy()

    assert health.status == "UNAVAILABLE"
    assert len(launches) == 3
    assert health.recovery_attempts == 2
