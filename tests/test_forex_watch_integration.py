from __future__ import annotations

import ast
import os
from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_forex_watcher import _complete_run_result, _Harness, _utc

FORBIDDEN = {
    "order_send", "buy", "sell", "open_position", "close_position",
    "modify_position", "modify_order", "place_order", "cancel_order",
}


def test_mocked_collector_trace_has_one_mt5_operation_at_a_time(tmp_path):
    harness = _Harness(tmp_path, runner_result=_complete_run_result())
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))
    harness.complete_runner()
    harness.poll(_utc("2026-09-09T12:30:31Z"))

    assert harness.gate.overlaps == []
    assert harness.gate.events[:4] == [
        "probe:start", "probe:end", "runner:start", "runner:end"
    ]
    decision = harness.store.list_decisions()[0]
    assert decision.executed is False


def test_phase6_sources_define_no_forbidden_mutation_methods():
    root = Path(__file__).resolve().parents[1]
    for path in (
        root / "tradingagents" / "forex" / "watcher.py",
        root / "tradingagents" / "forex" / "watch_store.py",
        root / "cli" / "forex_watch.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {
            node.name.casefold()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert FORBIDDEN.isdisjoint(defined), path


def test_run_summary_keeps_temporal_bases_and_staleness_separate(tmp_path):
    result = _complete_run_result()
    completed = _utc("2026-09-09T13:00:00Z")
    result = SimpleNamespace(
        decision=replace(
            result.decision,
            decision_completed_timestamp=completed,
            decision_reference_timestamp=completed + timedelta(seconds=1),
            decision_reference_bid=1.1001,
            decision_reference_ask=1.1003,
            decision_reference_spread=0.0002,
            decision_reference_spread_points=2.0,
            decision_reference_status="AVAILABLE",
        ),
        elapsed_seconds=result.elapsed_seconds,
        metrics=result.metrics,
    )
    harness = _Harness(tmp_path, runner_result=result)
    harness.start()
    harness.poll(_utc("2026-09-09T12:15:31Z"))
    harness.complete_runner()
    harness.poll(completed + timedelta(seconds=1))

    run = harness.store.list_runs()[0]
    assert run.analysis_snapshot_timestamp < run.decision_completed_timestamp
    assert run.decision_reference_timestamp >= run.decision_completed_timestamp
    assert run.analysis_latency_seconds > 0
    assert run.stale_by_completion is True
    assert "API_KEY" not in run.safe_config_json.upper()
    assert "secret" not in run.safe_config_json.lower()
    assert "training_eligible" not in harness.store.summary()


@pytest.mark.integration
def test_optional_local_mt5_probe_is_read_only():
    if os.getenv("RUN_MT5_INTEGRATION") != "1":
        pytest.skip("set RUN_MT5_INTEGRATION=1")
    pytest.importorskip("MetaTrader5")
    from tradingagents.dataflows.mt5.provider import MT5Provider
    from tradingagents.forex.watcher import ReadOnlyMarketProbe

    terminal_path = os.getenv("MT5_TERMINAL_PATH")
    requested = os.getenv("MT5_TEST_SYMBOL", "EURUSD")

    def account_state(symbol):
        provider = MT5Provider(terminal_path=terminal_path)
        try:
            provider.initialize()
            return provider.get_positions(symbol), provider.get_orders(symbol)
        finally:
            provider.shutdown()

    before_positions, before_orders = account_state(requested)
    result = ReadOnlyMarketProbe(
        lambda terminal_path=None: MT5Provider(terminal_path=terminal_path),
        terminal_path=terminal_path,
    ).probe(requested)
    after_positions, after_orders = account_state(result.resolved_symbol)
    assert result.timestamp.tzinfo == timezone.utc
    assert result.ask >= result.bid > 0
    assert before_positions == after_positions
    assert before_orders == after_orders
    assert all(not hasattr(MT5Provider, name) for name in FORBIDDEN)
