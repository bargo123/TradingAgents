from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.dataflows.mt5.models import (
    ForexMarketSnapshot,
    Mt5AccountInfo,
    Mt5Bar,
    Mt5Position,
    Mt5SymbolInfo,
)
from tradingagents.forex.runner import ForexShadowRunner, ForexShadowRunResult
from tradingagents.forex.shadow import ShadowDecisionStore
from tradingagents.graph.propagation import Propagator


def _snapshot() -> ForexMarketSnapshot:
    timestamp = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    bar = Mt5Bar(
        timestamp=timestamp,
        open=1.1000,
        high=1.1006,
        low=1.0997,
        close=1.1002,
        tick_volume=120,
        spread=2,
        real_volume=120,
    )
    return ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSDm",
        bid=1.1000,
        ask=1.1002,
        spread=0.0002,
        spread_points=20.0,
        symbol_info=Mt5SymbolInfo(
            name="EURUSDm",
            description="Euro / US Dollar",
            digits=5,
            point=0.00001,
            visible=True,
            trade_mode=0,
            currency_base="EUR",
            currency_profit="USD",
        ),
        m1_candles=(bar,),
        m5_candles=(bar,),
        m15_candles=(bar,),
        h1_candles=(bar,),
        account=Mt5AccountInfo(
            login=123456,
            server="Fake-Demo",
            currency="USD",
            balance=10_000.0,
            equity=9_980.0,
            profit=-20.0,
            margin=100.0,
            free_margin=9_880.0,
            leverage=100,
        ),
        positions=(
            Mt5Position(
                ticket=1,
                symbol="EURUSDm",
                type=0,
                volume=0.10,
                price_open=1.1000,
                price_current=1.1002,
                profit=2.0,
                time=timestamp,
            ),
        ),
    )


class _FakeProvider:
    def __init__(self, snapshot: ForexMarketSnapshot) -> None:
        self.snapshot = snapshot
        self.initialize_calls = 0
        self.shutdown_calls = 0
        self.market_snapshot_calls = 0
        self.resolved_symbols: list[str] = []

    def initialize(self) -> bool:
        self.initialize_calls += 1
        return True

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def ensure_symbol(self, symbol: str) -> str:
        self.resolved_symbols.append(symbol)
        return self.snapshot.symbol

    def get_market_snapshot(self, symbol: str, count: int = 100) -> ForexMarketSnapshot:
        self.market_snapshot_calls += 1
        self.resolved_symbols.append(symbol)
        return self.snapshot

    def get_tick(self, symbol: str):
        return SimpleNamespace(
            symbol=symbol,
            timestamp=self.snapshot.timestamp,
            bid=self.snapshot.bid,
            ask=self.snapshot.ask,
            last=self.snapshot.ask,
            volume=12,
            volume_real=12.0,
        )

    def get_bars(self, symbol: str, timeframe: str, count: int):
        return getattr(self.snapshot, f"{timeframe.lower()}_candles")

    def get_account_info(self):
        return self.snapshot.account

    def get_positions(self, symbol: str | None = None):
        return self.snapshot.positions

    def get_spread(self, symbol: str):
        return SimpleNamespace(
            symbol=symbol,
            bid=self.snapshot.bid,
            ask=self.snapshot.ask,
            price=self.snapshot.spread,
            points=self.snapshot.spread_points,
            timestamp=self.snapshot.timestamp,
        )


@dataclass
class _FakeGraph:
    final_state: dict

    def __post_init__(self) -> None:
        self.graph = SimpleNamespace(invoke=self._invoke_compiled)
        self.invocations = []
        self.propagator = SimpleNamespace(
            create_initial_state=self._create_initial_state,
            get_graph_args=lambda callbacks=None: {"config": {"callbacks": callbacks or []}},
        )
        self.config = {
            "llm_provider": "local",
            "quick_think_llm": "qwen",
            "deep_think_llm": "qwen",
            "backend_url": None,
            "data_cache_dir": "data_cache",
            "results_dir": "results",
            "max_recur_limit": 16,
        }

    def resolve_instrument_context(self, symbol: str, asset_type: str = "forex") -> str:
        return f"Instrument context for {symbol} ({asset_type})"

    def _create_initial_state(self, company_name, trade_date, **kwargs):
        return Propagator().create_initial_state(company_name, trade_date, **kwargs)

    def _invoke_compiled(self, init_state, **kwargs):
        self.invocations.append((init_state, kwargs))
        return self.final_state


def _make_runner(tmp_path: Path, final_state: dict):
    provider = _FakeProvider(_snapshot())
    graph = _FakeGraph(final_state)
    store = ShadowDecisionStore(tmp_path / "shadow.db")
    runner = ForexShadowRunner(
        provider_factory=lambda terminal_path=None: provider,
        graph_factory=lambda **kwargs: graph,
        store=store,
        config={
            "llm_provider": "local",
            "quick_think_llm": "qwen",
            "deep_think_llm": "qwen",
            "backend_url": None,
            "data_cache_dir": str(tmp_path / "cache"),
            "results_dir": str(tmp_path / "results"),
            "max_recur_limit": 16,
        },
    )
    return runner, provider, graph, store


def test_runner_fetches_one_snapshot_and_persists_normalized_decision(tmp_path):
    runner, provider, graph, store = _make_runner(
        tmp_path,
        {
            "final_trade_decision": PortfolioDecision(
                rating=PortfolioRating.BUY,
                executive_summary="x",
                investment_thesis="y",
            ),
            "portfolio_manager_raw_result": PortfolioDecision(
                rating=PortfolioRating.BUY,
                executive_summary="x",
                investment_thesis="y",
            ),
            "trader_investment_plan": "Trader sees upside",
            "investment_plan": "Research says buy",
            "market_report": "Market is constructive",
            "news_report": "Global news is stable",
            "fundamentals_report": "",
            "risk_debate_state": {
                "history": "debate",
                "aggressive_history": "agg",
                "conservative_history": "con",
                "neutral_history": "neu",
                "judge_decision": "judge",
            },
        },
    )

    result = runner.run(symbol="EURUSD", analysis_date=date(2026, 9, 8))

    assert isinstance(result, ForexShadowRunResult)
    assert provider.initialize_calls == 1
    assert provider.market_snapshot_calls == 1
    assert provider.shutdown_calls == 1
    assert result.decision.action == "BUY"
    assert result.decision.normalization_status == "NORMALIZED"
    assert result.decision.executed is False
    assert store.get(result.decision.decision_id).decision_id == result.decision.decision_id
    assert graph.invocations
    assert graph.invocations[0][0]["market_context"].startswith("SOURCE: LIVE MT5 BROKER DATA")
    assert graph.invocations[0][1]["config"]["callbacks"] == []
    assert graph.invocations[0][0]["asset_type"] == "forex"
    assert graph.invocations[0][0]["market_data_mode"] == "forex_mt5"


def test_runner_prefers_raw_structured_result_over_rendered_prose(tmp_path):
    runner, provider, graph, store = _make_runner(
        tmp_path,
        {
            "final_trade_decision": "BUY now; hold if uncertain",
            "portfolio_manager_raw_result": {
                "rating": "Sell",
                "executive_summary": "x",
                "investment_thesis": "y",
            },
            "trader_investment_plan": "Trader sees upside",
            "investment_plan": "Research says buy",
            "market_report": "Market is constructive",
            "news_report": "Global news is stable",
            "fundamentals_report": "",
            "risk_debate_state": {
                "history": "debate",
                "aggressive_history": "agg",
                "conservative_history": "con",
                "neutral_history": "neu",
                "judge_decision": "judge",
            },
        },
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert provider.shutdown_calls == 1
    assert result.decision.action == "SELL"
    assert result.decision.normalization_status == "NORMALIZED"
    assert "Sell" in result.decision.raw_portfolio_manager_result_json
    assert store.get(result.decision.decision_id).normalization_status == "NORMALIZED"


def test_runner_persists_failed_normalization_without_guessing(tmp_path):
    runner, provider, _, store = _make_runner(
        tmp_path,
        {
            "final_trade_decision": "BUY now; hold if uncertain",
            "portfolio_manager_raw_result": "BUY now; hold if uncertain",
            "trader_investment_plan": "Trader sees upside",
            "investment_plan": "Research says buy",
            "market_report": "Market is constructive",
            "news_report": "Global news is stable",
            "fundamentals_report": "",
            "risk_debate_state": {},
        },
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert provider.shutdown_calls == 1
    assert result.decision.action is None
    assert result.decision.normalization_status == "FAILED"
    assert "structured" in result.decision.normalization_error.lower()
    assert "BUY now" in result.decision.raw_portfolio_manager_result_json
    assert store.get(result.decision.decision_id).normalization_status == "FAILED"


def test_runner_shuts_down_when_graph_fails(tmp_path):
    provider = _FakeProvider(_snapshot())

    class FailingGraph(_FakeGraph):
        def _invoke_compiled(self, init_state, **kwargs):
            raise RuntimeError("boom")

    graph = FailingGraph(
        {
            "final_trade_decision": "Hold",
            "trader_investment_plan": "",
            "investment_plan": "",
            "market_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "risk_debate_state": {
                "history": "",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "judge_decision": "",
            },
        }
    )

    runner = ForexShadowRunner(
        provider_factory=lambda terminal_path=None: provider,
        graph_factory=lambda **kwargs: graph,
        store=ShadowDecisionStore(tmp_path / "shadow.db"),
        config={
            "llm_provider": "local",
            "quick_think_llm": "qwen",
            "deep_think_llm": "qwen",
            "backend_url": None,
            "data_cache_dir": str(tmp_path / "cache"),
            "results_dir": str(tmp_path / "results"),
            "max_recur_limit": 16,
        },
    )

    with suppress(RuntimeError):
        runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert provider.shutdown_calls == 1


def test_runner_rejects_invalid_inputs_and_still_shuts_down(tmp_path):
    provider = _FakeProvider(_snapshot())
    runner, _, _, _ = _make_runner(tmp_path, {"final_trade_decision": None})
    runner.provider_factory = lambda terminal_path=None: provider

    for kwargs in (
        {"count": 0},
        {"analysts": ("market", "fundamentals")},
    ):
        try:
            runner.run(symbol="EURUSD", analysis_date="2026-09-08", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid runner input should raise ValueError")

    assert provider.initialize_calls == 0
    assert provider.shutdown_calls == 2
