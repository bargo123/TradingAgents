from __future__ import annotations

import importlib
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.forex.context import build_forex_market_context, snapshot_to_dict
from tradingagents.forex.context_integrity import evaluate_context_integrity
from tradingagents.forex.profile import MACRO_EVENT_UNAVAILABLE, resolve_forex_profile
from tradingagents.forex.shadow import (
    ShadowDecisionStore,
    ShadowNormalization,
    ShadowTradeDecision,
    normalize_portfolio_manager_result,
)
from tradingagents.forex.telemetry import capture_state_trace
from tradingagents.forex.tools import MT5ToolAdapter


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_date(value: date | str | None) -> date:
    if value is None:
        return _utc_now().date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _identifier(value: Any) -> str:
    """Return an explicit evidence label even when a test graph omits config."""
    text = _text(value).strip()
    return text or "unknown"


def _callback_metrics(callbacks: Sequence[Any]) -> dict[str, Any]:
    """Read optional callback statistics without making telemetry required."""
    metrics: dict[str, Any] = {
        "llm_calls": 0,
        "tool_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "reasoning_tokens": 0,
        "agents": {},
    }
    for callback in callbacks:
        getter = getattr(callback, "get_stats", None)
        if not callable(getter):
            continue
        try:
            reported = getter()
        except Exception:  # noqa: BLE001 — telemetry must not alter analysis
            continue
        if not isinstance(reported, Mapping):
            continue
        for key in metrics:
            value = reported.get(key)
            if key == "agents" and isinstance(value, Mapping):
                metrics[key] = dict(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = value
        break
    return metrics


def _failed_normalization(exc: Exception) -> ShadowNormalization:
    return ShadowNormalization(
        action=None,
        normalization_status="FAILED",
        normalization_error=f"structured Portfolio Manager output could not be normalized: {exc}",
        raw_result={"error": str(exc)},
    )


@dataclass(frozen=True, slots=True)
class ForexShadowRunResult:
    """The persisted shadow decision and immutable run evidence."""

    decision: ShadowTradeDecision
    final_state: Mapping[str, Any]
    snapshot_json: Mapping[str, Any]
    provider_snapshot_calls: int
    elapsed_seconds: float
    metrics: Mapping[str, Any]

    @property
    def raw_state(self) -> Mapping[str, Any]:
        """Backward-compatible alias for callers of the initial candidate API."""
        return self.final_state


class ForexShadowRunner:
    """Run one read-only forex analysis session against one cached snapshot."""

    def __init__(
        self,
        provider_factory: Callable[..., Any] | None = None,
        graph_factory: Callable[..., Any] | None = None,
        store: ShadowDecisionStore | None = None,
        config: Mapping[str, Any] | None = None,
        selected_analysts: Sequence[str] = ("market", "news"),
    ) -> None:
        self.provider_factory = provider_factory or self._default_provider_factory
        self.graph_factory = graph_factory
        # Keep programmatic/CLI overrides partial and non-secret while still
        # supplying every setting required by TradingAgentsGraph.
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(config or {})
        self.store = store or ShadowDecisionStore(
            Path(self.config.get("data_cache_dir", "data_cache"))
            / "shadow_decisions.db"
        )
        # Retained as a compatibility convenience; run() is the public place
        # to choose analysts and defaults to exactly market/news.
        self.selected_analysts = tuple(selected_analysts)

    def _default_provider_factory(self, terminal_path: str | None = None) -> Any:
        module = importlib.import_module("tradingagents.dataflows.mt5.provider")
        return module.MT5Provider(terminal_path=terminal_path)

    def _default_graph_factory(self, **kwargs: Any) -> Any:
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        return TradingAgentsGraph(**kwargs)

    @staticmethod
    def _validate_inputs(
        symbol: str,
        count: int,
        analysis_date: date | str | None,
        analysts: Sequence[str],
    ) -> date:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        try:
            parsed_date = _as_date(analysis_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("analysis_date must be an ISO date") from exc
        if isinstance(analysts, (str, bytes)):
            raise ValueError("analysts must be a sequence of analyst names")
        selected = tuple(analysts)
        if not selected:
            raise ValueError("at least one forex analyst is required")
        if any(not isinstance(name, str) or not name for name in selected):
            raise ValueError("analysts must contain non-empty string names")
        if len(set(selected)) != len(selected):
            raise ValueError("analysts must not contain duplicates")
        allowed = {"market", "news"}
        invalid = [name for name in selected if name not in allowed]
        if invalid:
            raise ValueError(
                "forex shadow mode only supports market/news analysts; "
                f"unsupported: {', '.join(map(str, invalid))}"
            )
        return parsed_date

    @staticmethod
    def _extract_raw_pm_result(final_state: Mapping[str, Any]) -> Any:
        # Task 4 writes this field as the serialized structured output. It is
        # authoritative over the rendered prose in final_trade_decision.
        if "portfolio_manager_raw_result" in final_state:
            raw_result = final_state["portfolio_manager_raw_result"]
            # A state assembled by a graph wrapper may retain the propagator's
            # empty placeholder when the PM node returns a structured result.
            # Empty placeholders are the one case where the structured final
            # field is a safe fallback; a non-empty raw field always wins.
            if raw_result is not None and raw_result != "" and raw_result != {}:
                return raw_result
        final_result = final_state.get("final_trade_decision")
        if isinstance(final_result, (Mapping,)):
            return final_result
        # PortfolioDecision is intentionally checked without importing it here:
        # normalization itself owns the exact structured type contract.
        if hasattr(final_result, "model_dump"):
            return final_result
        return final_result

    @staticmethod
    def _invoke_compiled_graph(graph: Any, initial_state: Mapping[str, Any], graph_args: Mapping[str, Any]) -> Any:
        compiled_graph = getattr(graph, "graph", None) or graph
        invoke = getattr(compiled_graph, "invoke", None)
        if not callable(invoke):
            raise TypeError("forex graph must expose a compiled graph invoke() method")
        return invoke(initial_state, **dict(graph_args))

    def run(
        self,
        symbol: str = "EURUSD",
        count: int = 100,
        analysis_date: date | str | None = None,
        terminal_path: str | None = None,
        analysts: Sequence[str] | None = None,
        *,
        db_path: str | Path | None = None,
        callbacks: Sequence[Any] | None = None,
        analysis_profile: str = "INTRADAY",
    ) -> ForexShadowRunResult:
        started = time.perf_counter()
        provider: Any | None = None
        callback_list = list(callbacks or ())
        if db_path is not None:
            self.store = ShadowDecisionStore(db_path)

        try:
            selected_analysts = (
                self.selected_analysts if analysts is None else tuple(analysts)
            )
            parsed_date = self._validate_inputs(
                symbol, count, analysis_date, selected_analysts
            )
            profile = resolve_forex_profile(analysis_profile)
            provider = self.provider_factory(terminal_path=terminal_path)
            if not provider.initialize():
                raise RuntimeError("MT5 provider initialization failed")
            resolved_symbol = provider.ensure_symbol(symbol)
            if not isinstance(resolved_symbol, str) or not resolved_symbol.strip():
                raise RuntimeError("MT5 provider returned an invalid resolved symbol")
            snapshot = provider.get_market_snapshot(resolved_symbol, count=count)
            snapshot_json = snapshot_to_dict(snapshot)
            market_context = build_forex_market_context(snapshot, profile)
            instrument_context = build_instrument_context(resolved_symbol, "forex", {})
            adapter = MT5ToolAdapter(provider, snapshot)

            graph_factory = self.graph_factory or self._default_graph_factory
            graph = graph_factory(
                selected_analysts=selected_analysts,
                market_data_mode="forex_mt5",
                mt5_tools=adapter,
                config=self.config,
                callbacks=callback_list,
                forex_analysis_profile=profile.name,
            )
            initial_state = graph.propagator.create_initial_state(
                resolved_symbol,
                parsed_date.isoformat(),
                asset_type="forex",
                past_context="",
                instrument_context=instrument_context,
                market_data_mode="forex_mt5",
                market_context=market_context,
                forex_analysis_profile=profile.name,
            )
            graph_args = graph.propagator.get_graph_args(callbacks=callback_list)
            # The runner deliberately calls the already compiled graph. The
            # stock propagate helper performs Yahoo identity/memory-log work.
            with capture_state_trace() as state_trace:
                final_state = self._invoke_compiled_graph(graph, initial_state, graph_args)
            if not isinstance(final_state, Mapping):
                raise TypeError("compiled forex graph must return a mapping state")
            final_state = dict(final_state)
            context_integrity = evaluate_context_integrity(
                final_state,
                trace=state_trace,
            )
            # The trace contains only node/phase names and artifact
            # presence/size metadata; report content and private reasoning
            # are deliberately never retained.
            context_integrity["boundaries"] = list(state_trace)

            raw_pm_result = self._extract_raw_pm_result(final_state)
            try:
                normalized = normalize_portfolio_manager_result(
                    raw_pm_result,
                    forex_profile=profile.name,
                )
            except Exception as exc:  # malformed structured output fails closed
                normalized = _failed_normalization(exc)

            if normalized.normalization_status == "FAILED":
                state_error = final_state.get("normalization_error")
                if isinstance(state_error, str) and state_error.strip():
                    normalized = ShadowNormalization(
                        action=None,
                        normalization_status="FAILED",
                        normalization_error=state_error.strip(),
                        raw_result=normalized.raw_result,
                    )

            valid_for_seconds = None
            valid_until = None
            if normalized.normalization_status == "NORMALIZED":
                raw_validity = (
                    normalized.raw_result.get("valid_for_seconds")
                    if isinstance(normalized.raw_result, Mapping)
                    else None
                )
                valid_for_seconds = (
                    raw_validity
                    if isinstance(raw_validity, int) and not isinstance(raw_validity, bool)
                    else profile.valid_for_seconds
                )
                valid_until = snapshot.timestamp + timedelta(seconds=valid_for_seconds)

            source_run_id = str(uuid.uuid4())
            graph_config = getattr(graph, "config", {})
            effective_config = dict(graph_config) if isinstance(graph_config, Mapping) else {}
            effective_config.update(self.config)
            investment_debate_state = final_state.get("investment_debate_state", {})
            if not isinstance(investment_debate_state, Mapping):
                investment_debate_state = {}
            decision = ShadowTradeDecision(
                decision_id=str(uuid.uuid4()),
                created_at=_utc_now(),
                snapshot_timestamp=snapshot.timestamp,
                analysis_date=parsed_date,
                requested_symbol=symbol,
                resolved_symbol=snapshot.symbol,
                action=normalized.action,
                raw_portfolio_manager_result=normalized.raw_result,
                normalization_status=normalized.normalization_status,
                normalization_error=normalized.normalization_error,
                decision_context_status=context_integrity["status"],
                confidence=None,
                reference_bid=snapshot.bid,
                reference_ask=snapshot.ask,
                reference_mid=(snapshot.bid + snapshot.ask) / 2,
                spread=snapshot.spread,
                spread_points=snapshot.spread_points,
                analysis_timeframe="M1/M5/M15/H1",
                trader_summary=_text(final_state.get("trader_investment_plan")),
                portfolio_manager_summary=_text(final_state.get("final_trade_decision")),
                bull_summary=_text(investment_debate_state.get("bull_history")) or None,
                bear_summary=_text(investment_debate_state.get("bear_history")) or None,
                llm_provider=_identifier(effective_config.get("llm_provider")),
                quick_model=_identifier(effective_config.get("quick_think_llm")),
                deep_model=_identifier(effective_config.get("deep_think_llm")),
                snapshot_json=dict(snapshot_json),
                source_run_id=source_run_id,
                analysis_profile=profile.name,
                valid_for_seconds=valid_for_seconds,
                valid_until=valid_until,
            )
            self.store.record(decision)
            elapsed_seconds = time.perf_counter() - started
            callback_metrics = _callback_metrics(callback_list)
            metrics = {
                "provider_snapshot_calls": getattr(provider, "market_snapshot_calls", 1),
                "elapsed_seconds": elapsed_seconds,
                "market_data_mode": "forex_mt5",
                "selected_analysts": selected_analysts,
                "analysis_profile": profile.name,
                "bars_used": {
                    timeframe: snapshot_json.get("features", {})
                    .get(timeframe, {})
                    .get("candle_count", 0)
                    for timeframe in ("M1", "M5", "M15", "H1")
                },
                "market_features": snapshot_json.get("features", {}),
                "macro_event_status": MACRO_EVENT_UNAVAILABLE,
                "decision_valid_for_seconds": valid_for_seconds,
                "decision_valid_until": valid_until,
                "decision_context_status": context_integrity["status"],
                "context_integrity": context_integrity,
                "state_boundaries": list(state_trace),
                **callback_metrics,
            }
            return ForexShadowRunResult(
                decision=decision,
                final_state=final_state,
                snapshot_json=snapshot_json,
                provider_snapshot_calls=metrics["provider_snapshot_calls"],
                elapsed_seconds=elapsed_seconds,
                metrics=metrics,
            )
        finally:
            shutdown = getattr(provider, "shutdown", None) if provider is not None else None
            if callable(shutdown):
                shutdown()
