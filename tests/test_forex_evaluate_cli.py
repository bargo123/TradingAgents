from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from cli.forex_evaluate import build_parser, main
from tradingagents.forex.watch_store import LeaseOwner, WatcherStore


def _fake_result() -> SimpleNamespace:
    decision = SimpleNamespace(
        decision_id="decision-cli-001",
        resolved_symbol="EURUSDm",
        requested_symbol="EURUSD",
    )
    return SimpleNamespace(
        decision=decision,
        evaluations=(),
        status_by_basis={
            "ANALYSIS_SNAPSHOT": "COMPLETE",
            "DECISION_REFERENCE": "DATA_UNAVAILABLE",
        },
        errors=(),
        metrics={
            "decisions_scanned": 1,
            "horizons_evaluated": 1,
            "historical_ticks_processed": 4,
            "database_seconds": 0.01,
            "mt5_read_seconds": 0.02,
            "total_runtime_seconds": 0.03,
            "llm_calls": 0,
        },
    )


def test_forex_evaluate_requires_exactly_one_selector() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--decision-id", "one", "--pending"])


def test_forex_evaluate_cli_labels_bases_and_reports_zero_llm_calls(capsys, tmp_path) -> None:
    calls: list[tuple[str, str | None]] = []

    class FakeEvaluator:
        def __init__(self, **kwargs):
            calls.append(("construct", kwargs["config"].observation_tolerance_seconds))

        def evaluate_decision(self, decision_id, *, now=None, terminal_path=None):
            calls.append((decision_id, terminal_path))
            return _fake_result()

    result = main(
        [
            "--decision-id",
            "decision-cli-001",
            "--db-path",
            str(tmp_path / "evaluate.db"),
            "--terminal-path",
            "terminal.exe",
            "--observation-tolerance-seconds",
            "45",
        ],
        evaluator_factory=FakeEvaluator,
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "MT5 FOREX — OUTCOME EVALUATION (READ ONLY)" in output
    assert "NO ORDER WILL BE SENT" in output
    assert "ANALYSIS_SNAPSHOT STATUS: COMPLETE" in output
    assert "DECISION_REFERENCE STATUS: DATA_UNAVAILABLE" in output
    assert "LLM CALLS: 0" in output
    assert calls == [("construct", 45), ("decision-cli-001", "terminal.exe")]


def test_forex_evaluate_cli_returns_nonzero_on_provider_failure(capsys) -> None:
    class FailingEvaluator:
        def __init__(self, **kwargs):
            pass

        def evaluate_decision(self, decision_id, *, now=None, terminal_path=None):
            raise RuntimeError("MT5 unavailable")

    result = main(
        ["--decision-id", "decision-cli-001"],
        evaluator_factory=FailingEvaluator,
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "FOREX EVALUATION ERROR: MT5 unavailable" in captured.err
    assert "NO ORDER WILL BE SENT" in captured.out


def test_forex_evaluate_cli_rejects_nonzero_llm_metric(capsys) -> None:
    class NonZeroLlmEvaluator:
        def __init__(self, **kwargs):
            pass

        def evaluate_decision(self, decision_id, *, now=None, terminal_path=None):
            result = _fake_result()
            result.metrics["llm_calls"] = 1
            return result

    result = main(
        ["--decision-id", "decision-cli-001"],
        evaluator_factory=NonZeroLlmEvaluator,
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "non-zero LLM calls" in captured.err


def test_forex_evaluate_has_no_llm_or_execution_options() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--llm-provider" not in option_strings
    assert "--order-send" not in option_strings
    assert "--symbol" not in option_strings


def test_forex_evaluate_refuses_nonexpired_watcher_lease(capsys, tmp_path):
    db_path = tmp_path / "watch.db"
    store = WatcherStore(db_path)
    now = datetime.now(timezone.utc)
    store.acquire_lease(
        LeaseOwner("owner", 123, "host", now),
        now,
    )

    class MustNotConstruct:
        def __init__(self, **kwargs):
            raise AssertionError("evaluator must not be constructed")

    result = main(
        ["--pending", "--db-path", str(db_path)],
        evaluator_factory=MustNotConstruct,
    )

    assert result == 1
    assert "WATCHER_ALREADY_RUNNING" in capsys.readouterr().err
