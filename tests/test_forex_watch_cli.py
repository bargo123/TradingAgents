from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import cli.forex_watch as forex_watch
from cli.forex_watch import build_parser, main
from cli.stats_handler import StatsCallbackHandler
from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig
from tradingagents.forex.watch_store import LeaseOwner, LeaseStatus, WatcherStore
from tradingagents.forex.watcher import SerializedMt5OperationGate


def _owner():
    return LeaseOwner(
        owner_token="owner",
        pid=123,
        host="host",
        process_started_at=datetime.now(timezone.utc),
    )


class _FalseCallable:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __bool__(self):
        return False

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.value(*args, **kwargs) if callable(self.value) else self.value


def test_forex_watch_defaults_and_subcommands():
    parser = build_parser()
    args = parser.parse_args(["once"])
    assert args.command == "once"
    assert args.symbols == "EURUSD"
    assert args.analysis_profile == "INTRADAY"
    assert args.analysts == "market,news"
    assert args.schedule_timeframe == "M15"


def test_forex_watch_has_no_execution_or_credential_options():
    parser = build_parser()
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }
    assert "--order-send" not in option_strings
    assert "--api-key" not in option_strings
    assert "--llm-provider" not in option_strings
    assert "--deep-model" not in option_strings


def test_once_prints_shadow_collector_banner(capsys, monkeypatch, tmp_path):
    class FakeCoordinator:
        def __init__(self, **kwargs):
            pass

        def start(self, now=None):
            return SimpleNamespace(status=LeaseStatus.ACQUIRED)

        def run_once(self, now=None):
            return SimpleNamespace(
                lifecycle_status="IDLE", active_run_id=None, error_code=None
            )

        def shutdown(self):
            pass

    monkeypatch.setattr("cli.forex_watch.WatcherCoordinator", FakeCoordinator)
    assert main(["once", "--db-path", str(tmp_path / "watch.db")]) == 0
    output = capsys.readouterr().out
    assert "MT5 FOREX — SHADOW COLLECTOR" in output
    assert "NO ORDER WILL BE SENT" in output


def test_demo_handoff_prints_demo_banner_without_shadow_only_claim(
    capsys, monkeypatch, tmp_path
):
    class FakeCoordinator:
        def __init__(self, **kwargs):
            pass

        def start(self, now=None):
            return SimpleNamespace(status=LeaseStatus.ACQUIRED)

        def run_once(self, now=None):
            return SimpleNamespace(lifecycle_status="IDLE", active_run_id=None, error_code=None)

        def shutdown(self):
            pass

    class Context:
        worker = SimpleNamespace(config=SimpleNamespace(execution_mode="DEMO"))

    monkeypatch.setattr("cli.forex_watch.WatcherCoordinator", FakeCoordinator)
    assert main(
        ["once", "--db-path", str(tmp_path / "watch.db")],
        hft_context=Context(),
    ) == 0
    output = capsys.readouterr().out
    assert "MT5 FOREX — DEMO MODE" in output
    assert "DEMO ACCOUNT ONLY" in output
    assert "NO ORDER WILL BE SENT" not in output


def test_once_propagates_active_analysis_failure(capsys, monkeypatch, tmp_path):
    class FakeCoordinator:
        def __init__(self, **kwargs):
            pass

        def start(self):
            return SimpleNamespace(status=LeaseStatus.ACQUIRED)

        def run_once(self, now=None):
            return SimpleNamespace(
                lifecycle_status="ANALYZING",
                active_run_id="run-1",
                error_code=None,
            )

        def wait_for_active(self):
            return "ANALYSIS_FAILED"

        def shutdown(self):
            pass

    monkeypatch.setattr("cli.forex_watch.WatcherCoordinator", FakeCoordinator)

    assert main(["once", "--db-path", str(tmp_path / "watch.db")]) == 1
    assert "ANALYSIS_FAILED" in capsys.readouterr().err


def test_once_preserves_analysis_failure_when_coordinator_shutdown_fails(
    capsys, monkeypatch, tmp_path
):
    class FakeCoordinator:
        def __init__(self, **kwargs):
            del kwargs

        def start(self):
            return SimpleNamespace(status=LeaseStatus.ACQUIRED)

        def run_once(self, now=None):
            del now
            raise RuntimeError("analysis failure")

        def shutdown(self):
            raise RuntimeError("shutdown failure")

    monkeypatch.setattr("cli.forex_watch.WatcherCoordinator", FakeCoordinator)

    assert main(["once", "--db-path", str(tmp_path / "watch.db")]) == 1
    assert "analysis failure" in capsys.readouterr().err


def test_once_shuts_down_coordinator_when_start_raises(capsys, monkeypatch, tmp_path):
    calls = []

    class FakeCoordinator:
        def __init__(self, **kwargs):
            del kwargs

        def start(self):
            raise RuntimeError("startup failure")

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr("cli.forex_watch.WatcherCoordinator", FakeCoordinator)

    assert main(["once", "--db-path", str(tmp_path / "watch.db")]) == 1
    assert calls == ["shutdown"]
    assert "FOREX WATCH ERROR:" in capsys.readouterr().err


def test_status_does_not_construct_mt5_or_llm(capsys, tmp_path):
    db_path = tmp_path / "watch.db"
    assert main(["status", "--db-path", str(db_path)]) == 0
    output = capsys.readouterr().out
    assert "WATCHER STATUS" in output
    assert "LLM CALLS: unknown" in output
    assert not db_path.exists()


def test_status_reports_malformed_database_without_traceback(capsys, tmp_path):
    db_path = tmp_path / "malformed.db"
    db_path.write_bytes(b"not a sqlite database")

    assert main(["status", "--db-path", str(db_path)]) == 1
    captured = capsys.readouterr()
    assert "FOREX WATCH ERROR:" in captured.err
    assert "Traceback" not in captured.err


def test_status_prefers_read_only_summary(capsys, tmp_path):
    class Store:
        def read_only_summary(self):
            return {
                "lifecycle_status": "STOPPED",
                "lease_expires_at": None,
                "current_run_id": None,
                "evaluation_due_pending": False,
                "database_path": str(tmp_path / "watch.db"),
            }

        def summary(self):
            raise AssertionError("status must not initialize the writer store")

    assert main(
        ["status", "--db-path", str(tmp_path / "watch.db")],
        store_factory=lambda _path: Store(),
    ) == 0
    assert "WATCHER STATUS" in capsys.readouterr().out


def test_status_preserves_falsey_store_factory(capsys, tmp_path):
    class Store:
        def read_only_summary(self):
            return {
                "lifecycle_status": "STOPPED",
                "lease_expires_at": None,
                "current_run_id": None,
                "evaluation_due_pending": False,
                "database_path": str(tmp_path / "watch.db"),
            }

    factory = _FalseCallable(lambda _path: Store())
    assert main(
        ["status", "--db-path", str(tmp_path / "watch.db")],
        store_factory=factory,
    ) == 0
    assert factory.calls == 1
    assert not (tmp_path / "watch.db").exists()


def test_status_probe_prefers_read_only_lease(capsys, tmp_path, monkeypatch):
    class Store:
        def read_only_active_lease(self, _now):
            return SimpleNamespace(
                lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5)
            )

        def active_lease(self, _now):
            raise AssertionError("status probe must not initialize the writer store")

    monkeypatch.setattr(
        "cli.forex_watch._provider_factory",
        lambda: (_ for _ in ()).throw(AssertionError("probe must stop at active lease")),
    )

    assert main(
        ["status", "--probe", "--db-path", str(tmp_path / "watch.db")],
        store_factory=lambda _path: Store(),
    ) == 1
    assert "WATCHER_ALREADY_RUNNING" in capsys.readouterr().err


def test_status_probe_refuses_while_watcher_lease_is_valid(tmp_path, capsys):
    store = WatcherStore(tmp_path / "watch.db")
    store.acquire_lease(_owner(), datetime.now(timezone.utc))

    result = main(["status", "--probe", "--db-path", str(tmp_path / "watch.db")])

    assert result == 1
    assert "WATCHER_ALREADY_RUNNING" in capsys.readouterr().err


def test_make_coordinator_wires_existing_numeric_stats_callback(tmp_path, monkeypatch):
    args = build_parser().parse_args(["once", "--db-path", str(tmp_path / "watch.db")])
    config = forex_watch._make_config(args)
    captured = {}

    class _Executor:
        def shutdown(self, wait=True):
            del wait

    class _Coordinator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(forex_watch, "SingleSlotAnalysisExecutor", _Executor)
    monkeypatch.setattr(forex_watch, "WatcherCoordinator", _Coordinator)

    forex_watch._make_coordinator(args, config)

    callbacks = captured["callbacks"]
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], StatsCallbackHandler)


def test_make_coordinator_accepts_explicit_runtime_config_and_provenance(
    tmp_path, monkeypatch
):
    args = build_parser().parse_args(["once", "--db-path", str(tmp_path / "watch.db")])
    config = forex_watch._make_config(args)
    captured = {}

    class _Executor:
        def shutdown(self, wait=True):
            del wait

    class _Coordinator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(forex_watch, "SingleSlotAnalysisExecutor", _Executor)
    monkeypatch.setattr(forex_watch, "WatcherCoordinator", _Coordinator)

    forex_watch._make_coordinator(args, config, runtime_config=ForexShadowRuntimeConfig())

    assert captured["runner"].config["backend_url"] == "http://127.0.0.1:11435/v1"
    assert captured["runner"].config["quick_think_llm"] == "qwen3.5:2b"
    assert captured["store"].provenance["prompt_config_version"] == "forex-shadow.runtime.v1"


def test_make_coordinator_wires_shared_hft_gate_and_plan_callback(tmp_path, monkeypatch):
    args = build_parser().parse_args(
        ["run", "--db-path", str(tmp_path / "watch.db"), "--hft-shadow"]
    )
    config = forex_watch._make_config(args)
    captured = {}
    gate = SerializedMt5OperationGate()

    class _Executor:
        def shutdown(self, wait=True):
            del wait

    class _Context:
        def __init__(self):
            self.gate = gate

        def provider_factory(self, terminal_path=None):
            del terminal_path
            return SimpleNamespace(initialize=lambda: True, shutdown=lambda: None)

        def handle_decision(self, decision):
            del decision
            return True

    class _Coordinator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(forex_watch, "SingleSlotAnalysisExecutor", _Executor)
    monkeypatch.setattr(forex_watch, "WatcherCoordinator", _Coordinator)

    context = _Context()
    forex_watch._make_coordinator(
        args,
        config,
        runtime_config=ForexShadowRuntimeConfig(),
        hft_context=context,
    )

    assert captured["gate"] is gate
    assert captured["mt5_operations_gated"] is True
    assert captured["on_decision"].__self__ is context


def test_once_refuses_active_watcher_before_coordinator_resource_construction(
    tmp_path, capsys, monkeypatch
):
    db_path = tmp_path / "watch.db"
    store = WatcherStore(db_path)
    now = datetime.now(timezone.utc)
    store.acquire_lease(_owner(), now)

    def must_not_construct(*args, **kwargs):
        raise AssertionError("coordinator resources must not be constructed")

    monkeypatch.setattr("cli.forex_watch._make_coordinator", must_not_construct)

    assert main(["once", "--db-path", str(db_path)]) == 1
    assert "WATCHER_ALREADY_RUNNING" in capsys.readouterr().err


def test_main_accepts_injected_coordinator_factory_for_deterministic_smoke(
    tmp_path, monkeypatch
):
    calls = []

    class FakeCoordinator:
        def start(self):
            return SimpleNamespace(status=LeaseStatus.WATCHER_ALREADY_RUNNING)

    def factory(args, config):
        calls.append((args.command, config.db_path))
        return FakeCoordinator()

    # The lease guard remains authoritative; this no-lease path reaches the
    # injected coordinator and never imports/contructs the production runner.
    monkeypatch.setattr("cli.forex_watch._watcher_lease_is_active", lambda *args: False)
    assert main(
        ["once", "--db-path", str(tmp_path / "watch.db")],
        coordinator_factory=factory,
    ) == 1
    assert calls and calls[0][0] == "once"


def test_main_preserves_falsey_coordinator_factory(tmp_path, monkeypatch):
    class FakeCoordinator:
        def start(self):
            return SimpleNamespace(status=LeaseStatus.ACQUIRED)

        def run_once(self):
            return SimpleNamespace(active_run_id=None, error_code=None)

        def shutdown(self):
            return None

    factory = _FalseCallable(lambda _args, _config: FakeCoordinator())
    monkeypatch.setattr("cli.forex_watch._watcher_lease_is_active", lambda *args: False)
    monkeypatch.setattr(
        "cli.forex_watch._make_coordinator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("falsey coordinator factory was ignored")
        ),
    )
    assert main(
        ["once", "--db-path", str(tmp_path / "watch.db")],
        coordinator_factory=factory,
    ) == 0
    assert factory.calls == 1


def test_forex_watch_run_owns_hft_context_lifecycle(tmp_path, monkeypatch):
    lifecycle = []

    class FakeContext:
        def start(self):
            lifecycle.append("start")

        def stop(self):
            lifecycle.append("stop")

    class FakeCoordinator:
        def start(self):
            return SimpleNamespace(status=LeaseStatus.ACQUIRED)

        def run_forever(self):
            lifecycle.append("watch")

        def shutdown(self):
            lifecycle.append("shutdown")

    monkeypatch.setattr(
        forex_watch,
        "_make_coordinator",
        lambda *args, **kwargs: FakeCoordinator(),
    )
    monkeypatch.setattr(forex_watch, "_watcher_lease_is_active", lambda *args: False)

    assert forex_watch.main(
        ["run", "--db-path", str(tmp_path / "watch.db")],
        runtime_config=ForexShadowRuntimeConfig(),
        hft_context=FakeContext(),
    ) == 0
    assert lifecycle == ["start", "watch", "shutdown", "stop"]


def test_pyproject_registers_only_new_forex_watch_script():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'forex-watch = "cli.forex_watch:main"' in text
    assert 'tradingagents = "cli.main:app"' in text


def test_forex_shadow_docs_describe_serialized_collector_and_no_execution():
    text = Path("docs/forex-shadow.md").read_text(encoding="utf-8")
    assert "forex-watch" in text
    assert "WATCHER_ALREADY_RUNNING" in text
    assert "NO ORDER WILL BE SENT" in text
    assert "evaluation_due_pending" in text
