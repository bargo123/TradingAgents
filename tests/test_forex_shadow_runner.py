from __future__ import annotations

import hashlib
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.dataflows.mt5.errors import Mt5BrokerClockError
from tradingagents.dataflows.mt5.models import (
    ForexMarketSnapshot,
    Mt5AccountInfo,
    Mt5Bar,
    Mt5Position,
    Mt5Spread,
    Mt5SymbolInfo,
)
from tradingagents.forex import runner as runner_module
from tradingagents.forex.evidence_audit import AuditWriteError
from tradingagents.forex.evidence_context import (
    CanonicalEvidenceItem,
    EvidenceContext,
    EvidenceIntegrationStatus,
    EvidenceSourceKind,
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
    assert result.decision.analysis_profile == "INTRADAY"
    assert result.decision.valid_for_seconds == 3600
    assert result.decision.valid_until == result.decision.snapshot_timestamp + timedelta(seconds=3600)


def test_runner_does_not_construct_graph_when_broker_clock_is_unavailable(tmp_path):
    provider = _FakeProvider(_snapshot())

    def unavailable_snapshot(symbol: str, count: int = 100):
        raise Mt5BrokerClockError("broker clock calibration is unavailable")

    provider.get_market_snapshot = unavailable_snapshot
    graph_constructions = []

    runner = ForexShadowRunner(
        provider_factory=lambda terminal_path=None: provider,
        graph_factory=lambda **kwargs: graph_constructions.append(kwargs),
        store=ShadowDecisionStore(tmp_path / "shadow.db"),
        config={"llm_provider": "local"},
    )

    with pytest.raises(Mt5BrokerClockError, match="broker clock"):
        runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert graph_constructions == []
    assert provider.shutdown_calls == 1


def test_runner_uses_watcher_source_run_id(tmp_path):
    runner, _, _, store = _make_runner(
        tmp_path,
        {
            "final_trade_decision": {"rating": "Hold"},
            "portfolio_manager_raw_result": {"rating": "Hold"},
            "investment_debate_state": {"bull_history": "bull", "bear_history": "bear"},
            "risk_debate_state": {"history": "risk"},
        },
    )

    result = runner.run(
        symbol="EURUSD",
        analysis_date="2026-09-09",
        source_run_id="watch-run-001",
    )

    assert result.decision.source_run_id == "watch-run-001"
    assert store.find_by_source_run_id("watch-run-001")[0].decision_id == result.decision.decision_id


def test_runner_rejects_empty_source_run_id(tmp_path):
    runner, _, _, _ = _make_runner(
        tmp_path,
        {
            "final_trade_decision": {"rating": "Hold"},
            "portfolio_manager_raw_result": {"rating": "Hold"},
            "risk_debate_state": {},
        },
    )

    with pytest.raises(ValueError):
        runner.run(symbol="EURUSD", source_run_id=" ")


def test_runner_fails_closed_for_month_horizon(tmp_path):
    runner, _, _, _ = _make_runner(
        tmp_path,
        {
            "final_trade_decision": "**Rating**: Hold",
            "portfolio_manager_raw_result": {
                "rating": "Hold",
                "time_horizon": "3-6 months",
                "analysis_profile": "INTRADAY",
            },
            "risk_debate_state": {},
        },
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.decision.action is None
    assert result.decision.normalization_status == "FAILED"
    assert result.decision.valid_for_seconds is None
    assert result.decision.valid_until is None


def test_runner_passes_callbacks_and_reports_llm_metrics(tmp_path):
    callback = type(
        "Callback",
        (),
        {
            "get_stats": lambda self: {
                "llm_calls": 17,
                "tool_calls": 4,
                "tokens_in": 100,
                "tokens_out": 50,
            }
        },
    )()
    runner, provider, graph, _ = _make_runner(
        tmp_path,
        {
            "final_trade_decision": {"rating": "Hold"},
            "portfolio_manager_raw_result": {"rating": "Hold"},
            "investment_debate_state": {"bull_history": "bull", "bear_history": "bear"},
            "risk_debate_state": {},
        },
    )

    result = runner.run(
        symbol="EURUSD",
        analysis_date="2026-09-08",
        callbacks=[callback],
    )

    assert graph.invocations[0][1]["config"]["callbacks"] == [callback]
    assert result.metrics["llm_calls"] == 17
    assert result.metrics["tool_calls"] == 4
    assert result.metrics["tokens_in"] == 100
    assert result.metrics["tokens_out"] == 50
    assert result.elapsed_seconds >= 0


def test_runner_resets_opt_in_reusable_callback_before_each_decision(tmp_path):
    class Callback:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def get_stats(self):
            return {"llm_calls": 1, "tool_calls": 0}

    callback = Callback()
    runner, _, _, _ = _make_runner(
        tmp_path,
        {
            "final_trade_decision": {"rating": "Hold"},
            "portfolio_manager_raw_result": {"rating": "Hold"},
            "risk_debate_state": {},
        },
    )

    runner.run(symbol="EURUSD", analysis_date="2026-09-08", callbacks=[callback])

    assert callback.reset_calls == 1


def test_runner_marks_analysis_telemetry_unavailable_without_callback(tmp_path):
    runner, _, _, _ = _make_runner(
        tmp_path,
        {
            "final_trade_decision": {"rating": "Hold"},
            "portfolio_manager_raw_result": {"rating": "Hold"},
            "investment_debate_state": {"bull_history": "bull", "bear_history": "bear"},
            "risk_debate_state": {},
        },
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.metrics["telemetry_status"] == "UNAVAILABLE"
    assert result.metrics["llm_calls"] is None
    assert result.metrics["tool_calls"] is None
    assert result.metrics["tokens_in"] is None
    assert result.metrics["tokens_out"] is None
    assert result.metrics["reasoning_tokens"] is None
    assert result.metrics["agents"] is None


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


def test_runner_uses_configured_analyst_default_and_empty_placeholder_falls_back(
    tmp_path,
):
    runner, provider, graph, _ = _make_runner(
        tmp_path,
        {
            "final_trade_decision": {"rating": "Hold"},
            # Propagator initializes this field to an empty mapping.  A graph
            # wrapper that preserves that placeholder must not hide the
            # structured final result.
            "portfolio_manager_raw_result": {},
            "trader_investment_plan": "",
            "investment_plan": "",
            "market_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "risk_debate_state": {},
        },
    )
    runner.selected_analysts = ("market",)

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.decision.action == "HOLD"
    assert graph.invocations
    assert provider.shutdown_calls == 1


def test_runner_merges_partial_runtime_config_with_graph_defaults():
    runner = ForexShadowRunner(config={"llm_provider": "ollama"})

    assert runner.config["llm_provider"] == "ollama"
    assert runner.config["quick_think_llm"]
    assert runner.config["deep_think_llm"]
    assert runner.config["max_debate_rounds"] >= 0


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


def test_runner_persists_incomplete_context_status_without_relabeling_action(tmp_path):
    runner, provider, _, store = _make_runner(
        tmp_path,
        {
            "final_trade_decision": "**Rating**: Hold",
            "portfolio_manager_raw_result": {"rating": "Hold"},
            "market_report": "MARKET_REPORT",
            "news_report": "NEWS_REPORT",
            "investment_plan": "RM_PLAN",
            "trader_investment_plan": "TRADER_PLAN",
            "investment_debate_state": {
                "history": "\nBull Analyst: \nBear Analyst: ",
                "bull_history": "\nBull Analyst: ",
                "bear_history": "\nBear Analyst: ",
            },
            "risk_debate_state": {
                "history": "\nAggressive Analyst: ",
                "aggressive_history": "\nAggressive Analyst: ",
                "conservative_history": "\nConservative Analyst: ",
                "neutral_history": "\nNeutral Analyst: ",
            },
        },
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert provider.market_snapshot_calls == 1
    assert result.decision.action == "HOLD"
    assert result.decision.normalization_status == "NORMALIZED"
    assert result.decision.decision_context_status == "INCOMPLETE"
    assert result.metrics["decision_context_status"] == "INCOMPLETE"
    assert store.get(result.decision.decision_id).decision_context_status == "INCOMPLETE"


def test_runner_marks_complete_context_when_all_artifacts_are_present(tmp_path):
    runner, _, _, store = _make_runner(
        tmp_path,
        {
            "final_trade_decision": "PM_RESULT",
            "portfolio_manager_raw_result": {"rating": "Hold"},
            "market_report": "MARKET_REPORT",
            "news_report": "NEWS_REPORT",
            "investment_plan": "RM_PLAN",
            "trader_investment_plan": "TRADER_PLAN",
            "investment_debate_state": {
                "history": "\nBull Analyst: BULL\nBear Analyst: BEAR",
                "bull_history": "\nBull Analyst: BULL",
                "bear_history": "\nBear Analyst: BEAR",
            },
            "risk_debate_state": {
                "history": "\nAggressive Analyst: AGG\nConservative Analyst: CON\nNeutral Analyst: NEU",
                "aggressive_history": "\nAggressive Analyst: AGG",
                "conservative_history": "\nConservative Analyst: CON",
                "neutral_history": "\nNeutral Analyst: NEU",
            },
        },
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.decision.decision_context_status == "COMPLETE"
    assert result.metrics["context_integrity"]["missing"] == []
    assert store.get(result.decision.decision_id).decision_context_status == "COMPLETE"


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
    assert provider.shutdown_calls == 0

# Phase 5 temporal/reference tests
ANALYSIS_TIMESTAMP = datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc)
COMPLETION_TIMESTAMP = datetime(2026, 9, 8, 0, 0, 41, tzinfo=timezone.utc)
BROKER_REFERENCE_TIMESTAMP = datetime(2026, 9, 8, 0, 0, 42, 250_000, tzinfo=timezone.utc)


def _temporal_snapshot() -> ForexMarketSnapshot:
    bar = Mt5Bar(
        timestamp=ANALYSIS_TIMESTAMP,
        open=1.1,
        high=1.101,
        low=1.099,
        close=1.1005,
        tick_volume=10,
    )
    return ForexMarketSnapshot(
        timestamp=ANALYSIS_TIMESTAMP,
        symbol="EURUSDm",
        bid=1.1,
        ask=1.1002,
        spread=0.0002,
        spread_points=2.0,
        m1_candles=(bar,),
        m5_candles=(bar,),
        m15_candles=(bar,),
        h1_candles=(bar,),
        account=None,
        positions=(),
        symbol_info=Mt5SymbolInfo(name="EURUSDm", digits=5, point=0.00001),
    )


class _TemporalProvider:
    def __init__(
        self,
        events: list[str],
        *,
        spread_error: Exception | None = None,
        old_timestamp: bool = False,
    ):
        self.events = events
        self.spread_error = spread_error
        self.old_timestamp = old_timestamp
        self.market_snapshot_calls = 0

    def initialize(self) -> bool:
        self.events.append("initialize")
        return True

    def shutdown(self) -> None:
        self.events.append("shutdown")

    def ensure_symbol(self, symbol: str) -> str:
        self.events.append("ensure_symbol")
        return "EURUSDm"

    def get_market_snapshot(self, symbol: str, count: int = 100) -> ForexMarketSnapshot:
        self.events.append("snapshot")
        self.market_snapshot_calls += 1
        return _temporal_snapshot()

    def get_spread(self, symbol: str) -> Mt5Spread:
        self.events.append("spread")
        if self.spread_error is not None:
            raise self.spread_error
        timestamp = ANALYSIS_TIMESTAMP if self.old_timestamp else BROKER_REFERENCE_TIMESTAMP
        return Mt5Spread(
            symbol="EURUSDm",
            bid=1.1004,
            ask=1.1006,
            price=0.0002,
            points=2.0,
            timestamp=timestamp,
        )


class _TemporalPropagator:
    def create_initial_state(self, *args, **kwargs):
        return {}

    def get_graph_args(self, **kwargs):
        return {}


class _TemporalGraph:
    def __init__(self, events: list[str]):
        self.events = events
        self.propagator = _TemporalPropagator()
        self.config = {"llm_provider": "fake", "quick_think_llm": "quick"}

    def invoke(self, initial_state, **kwargs):
        self.events.append("graph")
        return {
            "portfolio_manager_raw_result": {"rating": "Hold"},
            "final_trade_decision": "Hold",
        }


def _run_temporal(
    tmp_path: Path,
    monkeypatch,
    *,
    spread_error: Exception | None = None,
    old_timestamp: bool = False,
):
    events: list[str] = []
    provider = _TemporalProvider(
        events,
        spread_error=spread_error,
        old_timestamp=old_timestamp,
    )
    graph = _TemporalGraph(events)
    monkeypatch.setattr(runner_module, "_utc_now", lambda: COMPLETION_TIMESTAMP)
    result = ForexShadowRunner(
        provider_factory=lambda terminal_path=None: provider,
        graph_factory=lambda **kwargs: graph,
    ).run(
        symbol="EURUSD",
        count=1,
        analysis_date=date(2026, 9, 8),
        db_path=tmp_path / "shadow.db",
    )
    return provider, result, events


def test_runner_persists_broker_timestamp_after_graph_without_extra_llm_call(
    tmp_path: Path, monkeypatch
) -> None:
    provider, result, events = _run_temporal(tmp_path, monkeypatch)

    assert events.index("spread") > events.index("graph")
    assert events.count("spread") == 1
    assert events.count("graph") == 1
    assert provider.market_snapshot_calls == 1
    decision = result.decision
    assert decision.decision_completed_timestamp == COMPLETION_TIMESTAMP
    assert decision.decision_reference_timestamp == BROKER_REFERENCE_TIMESTAMP
    assert decision.decision_reference_timestamp != decision.decision_completed_timestamp
    assert decision.decision_reference_bid == pytest.approx(1.1004)
    assert decision.decision_reference_ask == pytest.approx(1.1006)
    assert decision.decision_reference_spread == pytest.approx(0.0002)
    assert decision.decision_reference_spread_points == pytest.approx(2.0)
    assert decision.analysis_latency_seconds == pytest.approx(41.0)
    assert decision.decision_reference_delay_seconds == pytest.approx(1.25)
    assert decision.decision_reference_status == "AVAILABLE"


def test_runner_persists_analysis_when_fresh_quote_fails(
    tmp_path: Path, monkeypatch
) -> None:
    _, result, events = _run_temporal(
        tmp_path,
        monkeypatch,
        spread_error=RuntimeError("temporary quote failure"),
    )

    decision = result.decision
    assert events.count("spread") == 1
    assert decision.decision_reference_timestamp is None
    assert decision.decision_reference_bid is None
    assert decision.decision_reference_status == "UNAVAILABLE"
    assert "temporary quote failure" in (decision.decision_reference_error or "")
    assert decision.analysis_snapshot_timestamp == ANALYSIS_TIMESTAMP


def test_runner_marks_older_broker_quote_temporally_invalid(
    tmp_path: Path, monkeypatch
) -> None:
    _, result, _ = _run_temporal(tmp_path, monkeypatch, old_timestamp=True)

    decision = result.decision
    assert decision.decision_reference_timestamp == ANALYSIS_TIMESTAMP
    assert decision.decision_reference_status == "INVALID_TEMPORAL"
    assert decision.decision_reference_delay_seconds == pytest.approx(-41.0)
    assert decision.decision_reference_bid == pytest.approx(1.1004)


class _RunnerEvidenceService:
    def __init__(self, events: list[str], context: EvidenceContext | None = None):
        self.events = events
        self.context = context or EvidenceContext(
            integration_status=EvidenceIntegrationStatus.INJECTED,
            as_of=_snapshot().timestamp,
            rendered_context="evidence",
            rendered_context_hash=hashlib.sha256(b"evidence").hexdigest(),
        )
        self.calls = 0
        self.snapshots = []

    def retrieve(self, snapshot, *, resolved_symbol, analysis_profile, analysis_timeframe):
        self.calls += 1
        self.events.append("evidence")
        self.snapshots.append((snapshot, resolved_symbol, analysis_profile, analysis_timeframe))
        return self.context


class _RunnerAuditStore:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.records = []

    def append(self, audit):
        if self.error:
            raise self.error
        self.records.append(audit)


def _phase9_runner(
    tmp_path: Path,
    final_state: dict,
    *,
    enabled: bool = True,
    events: list[str] | None = None,
    evidence_context: EvidenceContext | None = None,
    audit_store=None,
    evidence_service_factory=None,
    evidence_audit_store_factory=None,
):
    event_log = events if events is not None else []
    provider = _FakeProvider(_snapshot())
    original_snapshot = provider.get_market_snapshot

    def get_market_snapshot(*args, **kwargs):
        event_log.append("snapshot")
        return original_snapshot(*args, **kwargs)

    provider.get_market_snapshot = get_market_snapshot
    graph = _FakeGraph(final_state)
    service = _RunnerEvidenceService(event_log, evidence_context)
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
            "forex_evidence_enabled": enabled,
        },
        evidence_service_factory=evidence_service_factory or (lambda **kwargs: service),
        evidence_audit_store_factory=evidence_audit_store_factory or (lambda **kwargs: audit_store),
    )
    return runner, provider, graph, store, service, event_log


def test_enabled_runner_retrieves_once_after_snapshot(tmp_path):
    events = []
    runner, _, _, _, service, events = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        events=events,
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert service.calls == 1
    assert events.index("evidence") > events.index("snapshot")
    assert result.metrics["evidence_integration_status"] == "INJECTED"


def test_enabled_runner_passes_snapshot_as_of(tmp_path):
    runner, _, _, _, service, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.snapshot_json["timestamp"] == "2026-09-08T12:00:00Z"
    assert result.metrics["evidence_as_of"] == _snapshot().timestamp


def test_enabled_runner_injects_one_context_hash(tmp_path):
    context = EvidenceContext(
        integration_status=EvidenceIntegrationStatus.INJECTED,
        as_of=_snapshot().timestamp,
        rendered_context="evidence",
        rendered_context_hash=hashlib.sha256(b"evidence").hexdigest(),
    )
    runner, _, graph, _, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        evidence_context=context,
    )

    runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    initial_state = graph.invocations[0][0]
    assert initial_state["evidence_context"] is context
    assert graph.invocations[0][1]["config"]["forex_evidence_context_hash"]


def test_disabled_runner_constructs_no_evidence_service(tmp_path):
    runner, _, graph, _, service, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        enabled=False,
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert service.calls == 0
    assert "evidence_context" not in graph.invocations[0][0]
    assert result.metrics["evidence_integration_status"] == "DISABLED"


def test_analyze_does_not_write_shadow_store(tmp_path):
    runner, _, _, _, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
    )
    runner.store.record = lambda _decision: (_ for _ in ()).throw(AssertionError("analyze persisted a decision"))

    result = runner.analyze(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.normalized_action == "HOLD"


def test_normal_run_persists_existing_shadow_decision(tmp_path):
    runner, _, _, store, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert store.get(result.decision.decision_id).executed is False


def test_enabled_fallback_audit_records_fallback_status(tmp_path):
    audit_store = _RunnerAuditStore()
    context = EvidenceContext(integration_status=EvidenceIntegrationStatus.FALLBACK)
    runner, _, _, _, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        evidence_context=context,
        audit_store=audit_store,
    )

    runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert len(audit_store.records) == 1
    assert audit_store.records[0].integration_status == "FALLBACK"


def test_shadow_row_excludes_transient_evidence_fields(tmp_path):
    audit_store = _RunnerAuditStore()
    runner, _, _, store, _, _ = _phase9_runner(
        tmp_path,
        {
            "final_trade_decision": {"rating": "Hold"},
            "portfolio_manager_raw_result": {
                "rating": "Hold",
                "evidence_use_status": "USED",
                "evidence_refs_used": ["K1"],
                "evidence_context_hash": "secret-hash",
            },
        },
        audit_store=audit_store,
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")
    raw_json = store.get(result.decision.decision_id).raw_portfolio_manager_result_json

    assert "evidence_use_status" not in raw_json
    assert "evidence_refs_used" not in raw_json
    assert "evidence_context_hash" not in raw_json


def test_audit_failure_preserves_shadow_decision(tmp_path):
    audit_store = _RunnerAuditStore(AuditWriteError("audit unavailable"))
    runner, _, _, store, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        audit_store=audit_store,
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.decision.executed is False
    assert store.get(result.decision.decision_id) is not None
    assert result.metrics["audit_status"] == "AUDIT_WRITE_FAILED"


def test_baseline_runs_without_phase7_or_phase8_artifacts(tmp_path):
    runner, _, graph, _, service, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        enabled=False,
    )

    runner.config["data_cache_dir"] = str(tmp_path / "missing-cache")
    runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert service.calls == 0
    assert "evidence_context" not in graph.invocations[0][0]


def test_evidence_factory_failure_falls_back_and_persists(tmp_path):
    def broken_factory(**_kwargs):
        raise RuntimeError("bad evidence configuration")

    runner, _, _, store, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        evidence_service_factory=broken_factory,
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.metrics["evidence_integration_status"] == "FALLBACK"
    assert result.decision.executed is False
    assert store.get(result.decision.decision_id) is not None


def test_invalid_evidence_context_becomes_fallback(tmp_path):
    invalid = EvidenceContext(
        integration_status=EvidenceIntegrationStatus.INJECTED,
        as_of=datetime(2026, 9, 8, 11, 0, tzinfo=timezone.utc),
        rendered_context="evidence",
        rendered_context_hash="wrong",
    )
    runner, _, graph, _, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        evidence_context=invalid,
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.metrics["evidence_integration_status"] == "FALLBACK"
    assert graph.invocations[0][0]["evidence_context"].integration_status is EvidenceIntegrationStatus.FALLBACK


def test_fallback_audit_never_claims_used(tmp_path):
    fallback = EvidenceContext(
        integration_status=EvidenceIntegrationStatus.FALLBACK,
        as_of=_snapshot().timestamp,
        rendered_context="evidence",
        rendered_context_hash=hashlib.sha256(b"evidence").hexdigest(),
        knowledge_items=(CanonicalEvidenceItem(
            display_id="K1", source_kind=EvidenceSourceKind.KNOWLEDGE,
            authoritative_id="chunk-1", text="evidence", content_type="text", score=1.0, provenance={},
        ),),
    )
    audit_store = _RunnerAuditStore()
    runner, _, _, _, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {
            "rating": "Hold", "evidence_use_status": "USED", "evidence_refs_used": ["K1"],
        }},
        evidence_context=fallback,
        audit_store=audit_store,
    )

    runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert audit_store.records[0].evidence_use_status == "UNAVAILABLE"
    assert audit_store.records[0].evidence_refs_used == ()


def test_fallback_context_identity_is_snapshot_specific():
    first_snapshot = _snapshot()
    second_snapshot = replace(first_snapshot, bid=first_snapshot.bid + 0.001)
    first = ForexShadowRunner._fallback_evidence_context(first_snapshot, "timeout")
    second = ForexShadowRunner._fallback_evidence_context(second_snapshot, "timeout")
    assert first.rendered_context_hash
    assert first.rendered_context_hash != second.rendered_context_hash


def test_validator_failure_preserves_persisted_decision(tmp_path, monkeypatch):
    audit_store = _RunnerAuditStore()
    runner, _, _, store, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        audit_store=audit_store,
    )
    monkeypatch.setattr(runner_module, "validate_evidence_references", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("closed vocabulary")))

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.decision.executed is False
    assert store.get(result.decision.decision_id) is not None
    assert result.metrics["audit_status"] in {"INVALID_REFERENCE", "AUDIT_WRITE_FAILED"}


def test_audit_factory_failure_preserves_persisted_decision(tmp_path):
    def broken_factory(**_kwargs):
        raise RuntimeError("audit unavailable")

    runner, _, _, store, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        evidence_audit_store_factory=broken_factory,
    )

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08")

    assert result.decision.executed is False
    assert store.get(result.decision.decision_id) is not None
    assert result.metrics["audit_status"] == "AUDIT_WRITE_FAILED"


def test_audit_captures_metadata_only_telemetry(tmp_path):
    audit_store = _RunnerAuditStore()
    runner, _, _, _, _, _ = _phase9_runner(
        tmp_path,
        {"final_trade_decision": {"rating": "Hold"}, "portfolio_manager_raw_result": {"rating": "Hold"}},
        audit_store=audit_store,
    )

    runner.run(symbol="EURUSD", analysis_date="2026-09-08")
    audit = audit_store.records[0]

    assert audit.builder_latency_seconds == 0.0
    assert audit.node_context_hashes is not None
    assert audit.missing_nodes is not None
    assert audit.telemetry_references is not None
