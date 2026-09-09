from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from cli.forex_watch import build_parser, main
from tradingagents.forex.watch_store import LeaseOwner, LeaseStatus, WatcherStore


def _owner():
    return LeaseOwner(
        owner_token="owner",
        pid=123,
        host="host",
        process_started_at=datetime.now(timezone.utc),
    )


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


def test_status_does_not_construct_mt5_or_llm(capsys, tmp_path):
    assert main(["status", "--db-path", str(tmp_path / "watch.db")]) == 0
    output = capsys.readouterr().out
    assert "WATCHER STATUS" in output
    assert "LLM CALLS" in output


def test_status_probe_refuses_while_watcher_lease_is_valid(tmp_path, capsys):
    store = WatcherStore(tmp_path / "watch.db")
    store.acquire_lease(_owner(), datetime.now(timezone.utc))

    result = main(["status", "--probe", "--db-path", str(tmp_path / "watch.db")])

    assert result == 1
    assert "WATCHER_ALREADY_RUNNING" in capsys.readouterr().err


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
