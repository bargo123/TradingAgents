"""Opt-in live smoke coverage for the standalone forex shadow path."""

from __future__ import annotations

import ast
import inspect
import os
from datetime import timezone
from pathlib import Path

import pytest

from tradingagents.dataflows.mt5.errors import (
    Mt5AccountDisconnectedError,
    Mt5DependencyError,
    Mt5InitializationError,
)
from tradingagents.dataflows.mt5.provider import MT5Provider
from tradingagents.forex.runner import ForexShadowRunner
from tradingagents.forex.tools import MT5ToolAdapter

FORBIDDEN_MUTATION_METHODS = (
    "order_send",
    "buy",
    "sell",
    "open_position",
    "close_position",
    "modify_position",
    "modify_order",
    "place_order",
    "cancel_order",
)


@pytest.mark.integration
def test_live_mt5_forex_snapshot_is_read_only() -> None:
    if os.getenv("RUN_MT5_INTEGRATION") != "1":
        pytest.skip("set RUN_MT5_INTEGRATION=1 to use the local MT5 terminal")

    pytest.importorskip("MetaTrader5")

    provider = MT5Provider()
    symbol = os.getenv("MT5_TEST_SYMBOL", "EURUSD")

    try:
        try:
            provider.initialize()
        except (Mt5DependencyError, Mt5InitializationError, Mt5AccountDisconnectedError) as exc:
            pytest.skip(f"MT5 unavailable: {exc}")

        before_positions = provider.get_positions(symbol)
        before_orders = provider.get_orders(symbol)
        snapshot = provider.get_market_snapshot(symbol, count=2)
        after_positions = provider.get_positions(symbol)
        after_orders = provider.get_orders(symbol)

        assert snapshot.timestamp.tzinfo == timezone.utc
        assert snapshot.bid > 0
        assert snapshot.ask >= snapshot.bid
        assert snapshot.symbol_info is not None
        assert snapshot.symbol_info.digits is not None
        assert snapshot.symbol_info.point not in (None, 0)
        assert before_positions == after_positions
        assert before_orders == after_orders

        # The adapter consumes the already captured snapshot and must not
        # initiate another provider snapshot request.
        cached = MT5ToolAdapter(provider, snapshot).get_mt5_market_snapshot(
            snapshot.symbol
        )
        assert cached["symbol"] == snapshot.symbol

        for method_name in FORBIDDEN_MUTATION_METHODS:
            assert not hasattr(provider, method_name)
    finally:
        provider.shutdown()


def test_phase4_components_expose_no_mutation_or_execution_api() -> None:
    for component in (MT5Provider, MT5ToolAdapter, ForexShadowRunner):
        public_callables = {
            name
            for name, member in inspect.getmembers(component, predicate=callable)
            if not name.startswith("_")
        }
        for forbidden in FORBIDDEN_MUTATION_METHODS:
            assert forbidden not in public_callables


def test_phase4_sources_define_no_mutation_methods() -> None:
    """Scan definitions, not guard-list string literals, for execution APIs."""
    root = Path(__file__).resolve().parents[1]
    production_sources = (
        root / "cli" / "forex_shadow.py",
        root / "tradingagents" / "forex" / "context.py",
        root / "tradingagents" / "forex" / "runner.py",
        root / "tradingagents" / "forex" / "shadow.py",
        root / "tradingagents" / "forex" / "tools.py",
        root / "tradingagents" / "dataflows" / "mt5" / "provider.py",
    )
    forbidden = set(FORBIDDEN_MUTATION_METHODS)
    for source_path in production_sources:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        defined = {
            node.name.casefold()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert forbidden.isdisjoint(defined), source_path
