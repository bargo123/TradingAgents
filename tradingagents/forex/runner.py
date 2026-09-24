from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.dataflows.mt5.models import ForexMarketSnapshot
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.forex.context import build_forex_market_context, snapshot_to_dict
from tradingagents.forex.context_integrity import evaluate_context_integrity
from tradingagents.forex.evidence_audit import EvidenceAuditStore, EvidenceUsageAudit
from tradingagents.forex.evidence_context import (
    EvidenceAuditStatus,
    EvidenceContext,
    EvidenceIntegrationStatus,
    EvidenceReferenceValidation,
    EvidenceUseStatus,
    strip_transient_evidence_metadata,
    validate_evidence_references,
)
from tradingagents.forex.evidence_runtime import EvidenceIntegrationService
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


def _normalize_utc_timestamp(value: Any) -> datetime:
    """Normalize a provider timestamp while requiring an explicit UTC value."""
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("MT5 reference timestamp must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("MT5 reference timestamp must be UTC")
    return value.astimezone(timezone.utc)


def _fresh_reference_quote(
    provider: Any,
    resolved_symbol: str,
    completed_timestamp: datetime,
) -> dict[str, Any]:
    """Read one fresh quote and retain its broker tick timestamp.

    ``Mt5Spread`` already carries the timestamp from ``symbol_info_tick``.  A
    provider implementation that cannot expose that field may use its
    read-only ``get_tick`` seam as a fallback; the application completion time
    is never used as a market timestamp.
    """
    spread = provider.get_spread(resolved_symbol)
    timestamp = getattr(spread, "timestamp", None)
    tick = None
    if timestamp is None:
        get_tick = getattr(provider, "get_tick", None)
        if not callable(get_tick):
            raise ValueError("fresh MT5 spread did not include a broker timestamp")
        tick = get_tick(resolved_symbol)
        timestamp = getattr(tick, "timestamp", None)
    reference_timestamp = _normalize_utc_timestamp(timestamp)
    bid = getattr(tick, "bid", getattr(spread, "bid", None))
    ask = getattr(tick, "ask", getattr(spread, "ask", None))
    spread_price = getattr(spread, "price", None)
    if spread_price is None and bid is not None and ask is not None:
        spread_price = float(ask) - float(bid)
    points = getattr(spread, "points", None)
    values = (bid, ask, spread_price, points)
    if any(value is None for value in values):
        raise ValueError("fresh MT5 quote did not include a complete bid/ask/spread set")
    try:
        bid = float(bid)
        ask = float(ask)
        spread_price = float(spread_price)
        points = float(points)
    except (TypeError, ValueError) as exc:
        raise ValueError("fresh MT5 quote contains non-numeric values") from exc
    if not all(math.isfinite(value) for value in (bid, ask, spread_price, points)):
        raise ValueError("fresh MT5 quote contains non-finite values")
    if ask < bid or spread_price < 0 or points < 0:
        raise ValueError("fresh MT5 quote contains invalid spread values")
    delay = (reference_timestamp - completed_timestamp).total_seconds()
    return {
        "timestamp": reference_timestamp,
        "bid": bid,
        "ask": ask,
        "spread": spread_price,
        "spread_points": points,
        "delay": delay,
        "status": "AVAILABLE" if delay >= 0 else "INVALID_TEMPORAL",
    }


def _fresh_reference_quote_until_post_completion(
    provider: Any,
    resolved_symbol: str,
    completed_timestamp: datetime,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    max_attempts: int,
    monotonic: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], Exception | None]:
    """Read a broker quote until its timestamp is not before completion.

    The first read is always attempted.  A pre-completion quote is retained as
    temporal evidence, but never promoted to ``AVAILABLE``.  Subsequent reads
    are bounded by both a monotonic timeout and an explicit attempt cap so a
    quiet market or a stalled test/provider cannot create an unbounded loop.
    """
    if timeout_seconds < 0 or not math.isfinite(timeout_seconds):
        raise ValueError("reference quote timeout must be finite and non-negative")
    if poll_interval_seconds < 0 or not math.isfinite(poll_interval_seconds):
        raise ValueError("reference quote poll interval must be finite and non-negative")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("reference quote max attempts must be a positive integer")

    clock = monotonic or time.monotonic
    wait = sleeper or time.sleep
    started = clock()
    attempts = 0
    candidate: dict[str, Any] | None = None
    poll_error: Exception | None = None

    while attempts < max_attempts:
        # Always perform the first immediate read.  After sleeping exactly to
        # the deadline, one final broker read is allowed; no further wait is.
        if attempts and clock() - started > timeout_seconds:
            break
        attempts += 1
        try:
            candidate = _fresh_reference_quote(
                provider, resolved_symbol, completed_timestamp
            )
        except Exception as exc:  # provider/validation failure is fail-closed
            poll_error = exc
            break
        if candidate["delay"] >= 0:
            break
        elapsed = max(0.0, clock() - started)
        if elapsed >= timeout_seconds or attempts >= max_attempts:
            break
        remaining = timeout_seconds - elapsed
        try:
            wait(min(poll_interval_seconds, remaining))
        except Exception as exc:  # a broken wait seam is also fail-closed
            poll_error = exc
            break

    elapsed = max(0.0, clock() - started)
    telemetry = {
        "reference_poll_attempts": attempts,
        "reference_wait_seconds": elapsed,
        "final_reference_delay_seconds": (
            None if candidate is None else candidate["delay"]
        ),
        "reference_status": (
            "UNAVAILABLE" if candidate is None else candidate["status"]
        ),
    }
    return candidate, telemetry, poll_error


def _reference_poll_settings(
    config: Mapping[str, Any],
    snapshot_timestamp: datetime,
    completed_timestamp: datetime,
) -> tuple[float, float, int]:
    """Resolve and freshness-cap the bounded reference polling settings."""
    try:
        timeout_seconds = float(
            config.get("forex_reference_poll_timeout_seconds", 5.0)
        )
        poll_interval_seconds = float(
            config.get("forex_reference_poll_interval_seconds", 0.25)
        )
        max_attempts_value = config.get("forex_reference_poll_max_attempts", 21)
        max_attempts = int(max_attempts_value)
        freshness_budget_value = config.get("freshness_budget_seconds")
        freshness_budget = (
            None
            if freshness_budget_value is None
            else float(freshness_budget_value)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid reference quote polling configuration") from exc
    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds < 0
        or not math.isfinite(poll_interval_seconds)
        or poll_interval_seconds < 0
        or isinstance(max_attempts_value, bool)
        or max_attempts <= 0
        or (
            freshness_budget is not None
            and (not math.isfinite(freshness_budget) or freshness_budget < 0)
        )
    ):
        raise ValueError("invalid reference quote polling configuration")

    # Freshness is measured from the analyzed broker snapshot through decision
    # completion.  Do not spend a polling wait that would cross that budget.
    if freshness_budget is None:
        return timeout_seconds, poll_interval_seconds, max_attempts
    analysis_elapsed = (completed_timestamp - snapshot_timestamp).total_seconds()
    remaining_freshness = max(0.0, freshness_budget - analysis_elapsed)
    return min(timeout_seconds, remaining_freshness), poll_interval_seconds, max_attempts


def _safe_reference_error(exc: Exception) -> str:
    """Keep reference diagnostics bounded and free of model/prompt content."""
    detail = str(exc).strip()
    if len(detail) > 200:
        detail = detail[:200]
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


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


def _summary_without_evidence(value: Any) -> str:
    if isinstance(value, Mapping):
        return _text(strip_transient_evidence_metadata(value))
    return _text(value)


def _callback_metrics(callbacks: Sequence[Any]) -> dict[str, Any]:
    """Read optional callback statistics without making telemetry required."""
    metrics: dict[str, Any] = {
        "telemetry_status": "UNAVAILABLE",
        "llm_calls": None,
        "tool_calls": None,
        "tokens_in": None,
        "tokens_out": None,
        "reasoning_tokens": None,
        "agents": None,
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
        if not any(key in reported for key in metrics if key != "telemetry_status"):
            continue
        metrics["telemetry_status"] = "AVAILABLE"
        for key in metrics:
            if key == "telemetry_status":
                continue
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


@dataclass(frozen=True, slots=True)
class ForexAnalysisResult:
    """Complete in-memory forex analysis, before optional shadow persistence."""

    analysis_date: date
    requested_symbol: str
    resolved_symbol: str
    snapshot: Any
    snapshot_json: Mapping[str, Any]
    final_state: Mapping[str, Any]
    raw_portfolio_manager_result: Mapping[str, Any]
    normalized_action: str | None
    normalization_status: str
    normalization_error: str | None
    context_integrity: Mapping[str, Any]
    evidence_context: EvidenceContext | None
    decision_reference: Mapping[str, Any] | None
    decision_reference_error: str | None
    decision_completed_timestamp: datetime
    state_trace: tuple[Mapping[str, Any], ...]
    analysis_telemetry: Mapping[str, Any]
    valid_for_seconds: int | None
    valid_until: datetime | None
    profile_name: str
    source_run_id: str | None

    @property
    def raw_state(self) -> Mapping[str, Any]:
        return self.final_state

    @property
    def decision_reference_quote(self) -> Mapping[str, Any] | None:
        return self.decision_reference

    @property
    def evidence_context_hash(self) -> str:
        return ForexShadowRunner._evidence_hash(self.evidence_context)


class _SavedSnapshotProvider:
    """Read-only provider facade used by saved-snapshot replay."""

    market_snapshot_calls = 0

    def __init__(self, snapshot: ForexMarketSnapshot) -> None:
        self.snapshot = snapshot

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def ensure_symbol(self, symbol: str) -> str:
        if symbol.casefold() != self.snapshot.symbol.casefold():
            raise ValueError("saved snapshot symbol does not match requested symbol")
        return self.snapshot.symbol

    def get_market_snapshot(self, *_args: Any, **_kwargs: Any) -> ForexMarketSnapshot:
        raise AssertionError("saved replay must not retrieve a second market snapshot")

    def get_tick(self, symbol: str) -> Any:
        return type("SavedTick", (), {"symbol": symbol, "timestamp": self.snapshot.timestamp, "bid": self.snapshot.bid, "ask": self.snapshot.ask})()

    def get_bars(self, symbol: str, timeframe: str, count: int) -> Any:
        return getattr(self.snapshot, f"{timeframe.lower()}_candles")[:count]

    def get_account_info(self) -> Any:
        return self.snapshot.account

    def get_positions(self, symbol: str | None = None) -> Any:
        return self.snapshot.positions

    def get_spread(self, symbol: str) -> Any:
        return type("SavedSpread", (), {"symbol": symbol, "bid": self.snapshot.bid, "ask": self.snapshot.ask, "price": self.snapshot.spread, "points": self.snapshot.spread_points, "timestamp": self.snapshot.timestamp})()


class ForexShadowRunner:
    """Run one read-only forex analysis session against one cached snapshot."""

    def __init__(
        self,
        provider_factory: Callable[..., Any] | None = None,
        graph_factory: Callable[..., Any] | None = None,
        store: ShadowDecisionStore | None = None,
        config: Mapping[str, Any] | None = None,
        selected_analysts: Sequence[str] = ("market", "news"),
        *,
        evidence_service_factory: Callable[..., Any] | None = None,
        evidence_audit_store_factory: Callable[..., Any] | None = None,
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
        self.evidence_service_factory = evidence_service_factory
        self.evidence_audit_store_factory = evidence_audit_store_factory

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

    @staticmethod
    def _call_factory(factory: Callable[..., Any], **kwargs: Any) -> Any:
        """Call injected factories without imposing one test/runtime signature."""
        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            return factory()
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return factory(**kwargs)
        accepted = {name: value for name, value in kwargs.items() if name in parameters}
        required_positional = [
            parameter
            for parameter in parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is inspect.Parameter.empty
        ]
        if required_positional and not accepted:
            return factory(kwargs.get("config", {}))
        return factory(**accepted)

    @staticmethod
    def _evidence_is_enabled(config: Mapping[str, Any]) -> bool:
        return bool(config.get("forex_evidence_enabled", False))

    def _default_evidence_service_factory(self, **kwargs: Any) -> EvidenceIntegrationService:
        config = kwargs.get("config", self.config)
        from tradingagents.forex.evidence_context import EvidenceQueryPolicy

        policy = EvidenceQueryPolicy(
            evidence_timeout_seconds=float(
                config.get(
                    "forex_evidence_timeout_seconds",
                    config.get("evidence_timeout_seconds", 10.0),
                )
            ),
            evaluation_basis=config.get(
                "forex_evidence_evaluation_basis", "ANALYSIS_SNAPSHOT"
            ),
            statistics_horizon_seconds=config.get(
                "forex_evidence_statistics_horizon_seconds"
            ),
            knowledge_top_k=int(config.get("forex_evidence_knowledge_top_k", 10)),
            experience_top_k=int(config.get("forex_evidence_experience_top_k", 50)),
        )
        roots = config.get(
            "forex_evidence_artifact_roots",
            config.get("evidence_artifact_roots", {}),
        )
        if not roots:
            roots = {
                key: config[key]
                for key in (
                    "forex_evidence_knowledge_artifact_root",
                    "forex_evidence_knowledge_embedding_model_path",
                    "forex_evidence_experience_artifact_root",
                )
                if config.get(key) is not None
            }
            roots = {
                "knowledge": roots.get("forex_evidence_knowledge_artifact_root"),
                "knowledge_embedding_model_path": roots.get(
                    "forex_evidence_knowledge_embedding_model_path"
                ),
                "experience": roots.get("forex_evidence_experience_artifact_root"),
            }
            roots = {key: value for key, value in roots.items() if value is not None}
        return EvidenceIntegrationService(
            policy=policy,
            orchestrator_factory=config.get("evidence_orchestrator_factory"),
            generation_provider=config.get("evidence_generation_provider", (None, None)),
            provider_endpoint=config.get("backend_url"),
            artifact_roots=roots if isinstance(roots, Mapping) else {},
        )

    def _default_evidence_audit_store_factory(self, **kwargs: Any) -> EvidenceAuditStore:
        config = kwargs.get("config", self.config)
        root = Path(config.get("data_cache_dir", "data_cache")) / "evidence_runtime"
        return EvidenceAuditStore(root / "evidence_audit.sqlite3")

    @staticmethod
    def _fallback_evidence_context(
        snapshot: Any,
        detail: Any = "",
        *,
        code: str = "ORCHESTRATOR_FAILURE",
    ) -> EvidenceContext:
        snapshot_payload = json.dumps(
            snapshot_to_dict(snapshot, include_candles=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        rendered_context = (
            f"EVIDENCE_FALLBACK snapshot={snapshot_payload} "
            f"code={str(code).replace(chr(10), ' ')[:80]}"
        )
        rendered_context_hash = hashlib.sha256(rendered_context.encode("utf-8")).hexdigest()
        return EvidenceContext(
            as_of=snapshot.timestamp,
            integration_status=EvidenceIntegrationStatus.FALLBACK,
            diagnostics={
                "integration": {
                    "code": code,
                    "message": str(detail).replace("\r", " ").replace("\n", " ")[:500],
                    "error_type": type(detail).__name__
                    if isinstance(detail, BaseException)
                    else None,
                }
            },
            rendered_context=rendered_context,
            rendered_context_hash=rendered_context_hash,
            rendered_character_count=len(rendered_context),
        )

    @staticmethod
    def _validate_evidence_context(snapshot: Any, context: EvidenceContext) -> None:
        if context.as_of != snapshot.timestamp:
            raise ValueError("evidence context as_of does not match snapshot timestamp")
        expected_hash = hashlib.sha256(context.rendered_context.encode("utf-8")).hexdigest()
        if context.rendered_context_hash != expected_hash:
            raise ValueError("evidence context hash does not match canonical payload")

    @staticmethod
    def _evidence_hash(context: EvidenceContext | None) -> str:
        if context is None:
            return ""
        return str(
            context.rendered_context_hash
            or hashlib.sha256(context.rendered_context.encode("utf-8")).hexdigest()
        )

    def analyze(
        self,
        symbol: str = "EURUSD",
        count: int = 100,
        analysis_date: date | str | None = None,
        terminal_path: str | None = None,
        analysts: Sequence[str] | None = None,
        *,
        callbacks: Sequence[Any] | None = None,
        analysis_profile: str = "INTRADAY",
        source_run_id: str | None = None,
        snapshot: ForexMarketSnapshot | None = None,
        snapshot_bytes: bytes | None = None,
        forex_evidence_enabled: bool | None = None,
        evidence_enabled: bool | None = None,
        pinned_phase7_generation_id: str | None = None,
        pinned_phase8_generation_id: str | None = None,
        snapshot_fingerprint: str | None = None,
        provider: str | None = None,
        models: Mapping[str, Any] | None = None,
        model_settings: Mapping[str, Any] | None = None,
        normalization_path: str | None = None,
        persist: bool = False,
    ) -> ForexAnalysisResult:
        """Analyze one snapshot without constructing or writing a shadow store row."""
        if persist:
            raise ValueError("analyze() is non-persisting; use run() for normal shadow persistence")
        if snapshot is not None and not isinstance(snapshot, ForexMarketSnapshot):
            raise ValueError("snapshot must be a ForexMarketSnapshot")
        if snapshot is not None and symbol.casefold() != snapshot.symbol.casefold():
            raise ValueError("saved snapshot symbol does not match requested symbol")
        if snapshot is not None and snapshot_bytes is not None and not isinstance(snapshot_bytes, bytes):
            raise ValueError("snapshot_bytes must be immutable bytes")
        mt5_provider: Any | None = None
        callback_list = list(callbacks or ())
        # The watcher intentionally reuses its callback handler for each
        # scheduled opportunity.  Reset only handlers that explicitly expose
        # this opt-in seam so persisted metrics represent this decision, while
        # arbitrary integration callbacks retain their existing behavior.
        for callback in callback_list:
            reset = getattr(callback, "reset", None)
            if callable(reset):
                with suppress(Exception):
                    reset()
        if source_run_id is not None:
            if not isinstance(source_run_id, str) or not source_run_id.strip():
                raise ValueError("source_run_id must be a non-empty string")
            source_run_id = source_run_id.strip()
        selected_analysts = self.selected_analysts if analysts is None else tuple(analysts)
        parsed_date = self._validate_inputs(symbol, count, analysis_date, selected_analysts)
        profile = resolve_forex_profile(analysis_profile)
        analysis_telemetry: dict[str, Any] = {}
        try:
            mt5_provider = _SavedSnapshotProvider(snapshot) if snapshot is not None else self.provider_factory(terminal_path=terminal_path)
            if not mt5_provider.initialize():
                raise RuntimeError("MT5 provider initialization failed")
            resolved_symbol = snapshot.symbol if snapshot is not None else mt5_provider.ensure_symbol(symbol)
            if not isinstance(resolved_symbol, str) or not resolved_symbol.strip():
                raise RuntimeError("MT5 provider returned an invalid resolved symbol")
            if snapshot is None:
                snapshot = mt5_provider.get_market_snapshot(resolved_symbol, count=count)
            snapshot_json = snapshot_to_dict(snapshot)
            evidence_context: EvidenceContext | None = None
            config_for_graph = dict(self.config)
            if (
                forex_evidence_enabled is not None
                and evidence_enabled is not None
                and bool(forex_evidence_enabled) != bool(evidence_enabled)
            ):
                raise ValueError("conflicting evidence-enabled replay flags")
            effective_evidence_enabled = (
                self._evidence_is_enabled(self.config)
                if forex_evidence_enabled is None and evidence_enabled is None
                else bool(
                    forex_evidence_enabled
                    if forex_evidence_enabled is not None
                    else evidence_enabled
                )
            )
            config_for_graph["forex_evidence_enabled"] = effective_evidence_enabled
            config_for_graph["forex_replay"] = snapshot_bytes is not None
            config_for_graph["forex_replay_persist"] = False
            if snapshot_fingerprint is not None:
                config_for_graph["forex_replay_snapshot_fingerprint"] = snapshot_fingerprint
            if provider is not None:
                config_for_graph["llm_provider"] = provider
            if models is not None:
                model_values = dict(models)
                config_for_graph["replay_models"] = model_values
                quick_model = (
                    model_values.get("quick_think_llm")
                    or model_values.get("quick_model")
                    or model_values.get("quick")
                )
                deep_model = (
                    model_values.get("deep_think_llm")
                    or model_values.get("deep_model")
                    or model_values.get("deep")
                )
                if quick_model is not None:
                    config_for_graph["quick_think_llm"] = quick_model
                if deep_model is not None:
                    config_for_graph["deep_think_llm"] = deep_model
            if model_settings is not None:
                settings = dict(model_settings)
                config_for_graph["replay_model_settings"] = settings
                for key in (
                    "temperature",
                    "llm_max_retries",
                    "max_tokens",
                    "google_thinking_level",
                    "openai_reasoning_effort",
                    "anthropic_effort",
                ):
                    if key in settings:
                        config_for_graph[key] = settings[key]
            if normalization_path is not None:
                config_for_graph["replay_normalization_path"] = normalization_path
            if pinned_phase7_generation_id is not None:
                config_for_graph["pinned_phase7_generation_id"] = pinned_phase7_generation_id
            if pinned_phase8_generation_id is not None:
                config_for_graph["pinned_phase8_generation_id"] = pinned_phase8_generation_id
            if (
                pinned_phase7_generation_id is not None
                or pinned_phase8_generation_id is not None
            ):
                config_for_graph["evidence_generation_provider"] = (
                    pinned_phase7_generation_id,
                    pinned_phase8_generation_id,
                )
            if effective_evidence_enabled:
                factory = self.evidence_service_factory or self._default_evidence_service_factory
                started_retrieval = time.perf_counter()
                service: Any | None = None
                try:
                    service = self._call_factory(factory, config=config_for_graph, runner=self)
                    evidence_context = service.retrieve(
                        snapshot,
                        resolved_symbol=resolved_symbol,
                        analysis_profile=profile.name,
                        analysis_timeframe="M1/M5/M15/H1",
                    )
                    if not isinstance(evidence_context, EvidenceContext):
                        raise TypeError("evidence service must return EvidenceContext")
                    self._validate_evidence_context(snapshot, evidence_context)
                except Exception as exc:  # evidence is best-effort and advisory
                    code = "EVIDENCE_CONTEXT_INVALID" if isinstance(evidence_context, EvidenceContext) else "ORCHESTRATOR_FAILURE"
                    evidence_context = self._fallback_evidence_context(snapshot, exc, code=code)
                analysis_telemetry["evidence_retrieval_latency_seconds"] = time.perf_counter() - started_retrieval
                analysis_telemetry["evidence_retrieval_count"] = getattr(service, "retrieval_count", 1)
                integration_diagnostic = evidence_context.diagnostics.get("integration")
                if isinstance(integration_diagnostic, Mapping):
                    # Only type/code metadata crosses into replay telemetry;
                    # provider exception text remains in the in-memory context
                    # and is never persisted as a prompt/completion artifact.
                    analysis_telemetry["evidence_integration_diagnostic"] = {
                        "code": str(integration_diagnostic.get("code", ""))[:100],
                        "error_type": str(integration_diagnostic.get("error_type", ""))[:100],
                    }
            context_hash = self._evidence_hash(evidence_context)
            config_for_graph["forex_evidence_enabled"] = effective_evidence_enabled
            config_for_graph["forex_evidence_context_hash"] = context_hash

            market_context = build_forex_market_context(snapshot, profile)
            instrument_context = build_instrument_context(resolved_symbol, "forex", {})
            adapter = MT5ToolAdapter(mt5_provider, snapshot)
            graph_factory = self.graph_factory or self._default_graph_factory
            graph = graph_factory(
                selected_analysts=selected_analysts,
                market_data_mode="forex_mt5",
                mt5_tools=adapter,
                config=config_for_graph,
                callbacks=callback_list,
                forex_analysis_profile=profile.name,
            )
            initial_kwargs = {
                "asset_type": "forex",
                "past_context": "",
                "instrument_context": instrument_context,
                "market_data_mode": "forex_mt5",
                "market_context": market_context,
                "forex_analysis_profile": profile.name,
            }
            if evidence_context is not None:
                initial_kwargs["evidence_context"] = evidence_context
            initial_state = graph.propagator.create_initial_state(
                resolved_symbol, parsed_date.isoformat(), **initial_kwargs
            )
            graph_args = graph.propagator.get_graph_args(callbacks=callback_list)
            graph_args = dict(graph_args or {})
            graph_args_config = dict(graph_args.get("config", {}))
            graph_args_config["forex_evidence_enabled"] = effective_evidence_enabled
            graph_args_config["forex_evidence_context_hash"] = context_hash
            graph_args["config"] = graph_args_config
            with capture_state_trace() as state_trace:
                final_state = self._invoke_compiled_graph(graph, initial_state, graph_args)
            if not isinstance(final_state, Mapping):
                raise TypeError("compiled forex graph must return a mapping state")
            final_state = dict(final_state)
            decision_completed_timestamp = _utc_now()
            context_integrity = evaluate_context_integrity(final_state, trace=state_trace)
            context_integrity["boundaries"] = list(state_trace)
            raw_pm_result = self._extract_raw_pm_result(final_state)
            try:
                normalized = normalize_portfolio_manager_result(raw_pm_result, forex_profile=profile.name)
            except Exception as exc:
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
            decision_reference: dict[str, Any] | None = None
            decision_reference_error: str | None = None
            reference_telemetry: dict[str, Any] = {
                "reference_poll_attempts": 0,
                "reference_wait_seconds": 0.0,
                "final_reference_delay_seconds": None,
                "reference_status": "UNAVAILABLE",
            }
            try:
                timeout_seconds, poll_interval_seconds, max_attempts = _reference_poll_settings(
                    config_for_graph,
                    snapshot.timestamp,
                    decision_completed_timestamp,
                )
                decision_reference, reference_telemetry, poll_error = (
                    _fresh_reference_quote_until_post_completion(
                        mt5_provider,
                        resolved_symbol,
                        decision_completed_timestamp,
                        timeout_seconds=timeout_seconds,
                        poll_interval_seconds=poll_interval_seconds,
                        max_attempts=max_attempts,
                    )
                )
                if poll_error is not None:
                    decision_reference_error = _safe_reference_error(poll_error)
            except Exception as exc:
                decision_reference_error = _safe_reference_error(exc)
                reference_telemetry = {
                    "reference_poll_attempts": 0,
                    "reference_wait_seconds": 0.0,
                    "final_reference_delay_seconds": None,
                    "reference_status": "UNAVAILABLE",
                }
            valid_for_seconds = None
            valid_until = None
            if normalized.normalization_status == "NORMALIZED":
                raw_validity = normalized.raw_result.get("valid_for_seconds")
                valid_for_seconds = raw_validity if isinstance(raw_validity, int) and not isinstance(raw_validity, bool) else profile.valid_for_seconds
                valid_until = snapshot.timestamp + timedelta(seconds=valid_for_seconds)
            analysis_telemetry.update(_callback_metrics(callback_list))
            analysis_telemetry.update(reference_telemetry)
            analysis_telemetry.update(
                {
                    "provider_snapshot_calls": getattr(mt5_provider, "market_snapshot_calls", 1),
                    "evidence_integration_status": (
                        "DISABLED" if evidence_context is None else str(evidence_context.integration_status)
                    ),
                    "evidence_context_hash": context_hash,
                    "evidence_as_of": None if evidence_context is None else evidence_context.as_of,
                }
            )
            return ForexAnalysisResult(
                analysis_date=parsed_date,
                requested_symbol=symbol,
                resolved_symbol=resolved_symbol,
                snapshot=snapshot,
                snapshot_json=snapshot_json,
                final_state=final_state,
                raw_portfolio_manager_result=normalized.raw_result,
                normalized_action=normalized.action,
                normalization_status=normalized.normalization_status,
                normalization_error=normalized.normalization_error,
                context_integrity=context_integrity,
                evidence_context=evidence_context,
                decision_reference=decision_reference,
                decision_reference_error=decision_reference_error,
                decision_completed_timestamp=decision_completed_timestamp,
                state_trace=tuple(state_trace),
                analysis_telemetry=analysis_telemetry,
                valid_for_seconds=valid_for_seconds,
                valid_until=valid_until,
                profile_name=profile.name,
                source_run_id=source_run_id,
            )
        finally:
            shutdown = getattr(mt5_provider, "shutdown", None) if mt5_provider is not None else None
            if callable(shutdown):
                shutdown()

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
        source_run_id: str | None = None,
    ) -> ForexShadowRunResult:
        started = time.perf_counter()
        if db_path is not None:
            self.store = ShadowDecisionStore(db_path)
        result = self.analyze(
            symbol=symbol,
            count=count,
            analysis_date=analysis_date,
            terminal_path=terminal_path,
            analysts=analysts,
            callbacks=callbacks,
            analysis_profile=analysis_profile,
            source_run_id=source_run_id,
        )
        completed = result.decision_completed_timestamp
        decision_reference = result.decision_reference
        investment_debate_state = result.final_state.get("investment_debate_state", {})
        if not isinstance(investment_debate_state, Mapping):
            investment_debate_state = {}
        effective_config = dict(self.config)
        raw_result = dict(strip_transient_evidence_metadata(result.raw_portfolio_manager_result))
        decision = ShadowTradeDecision(
            decision_id=str(uuid.uuid4()),
            created_at=completed,
            snapshot_timestamp=result.snapshot.timestamp,
            analysis_date=result.analysis_date,
            requested_symbol=result.requested_symbol,
            resolved_symbol=result.snapshot.symbol,
            action=result.normalized_action,
            raw_portfolio_manager_result=raw_result,
            normalization_status=result.normalization_status,
            normalization_error=result.normalization_error,
            decision_context_status=result.context_integrity["status"],
            confidence=None,
            reference_bid=result.snapshot.bid,
            reference_ask=result.snapshot.ask,
            reference_mid=(result.snapshot.bid + result.snapshot.ask) / 2,
            spread=result.snapshot.spread,
            spread_points=result.snapshot.spread_points,
            analysis_timeframe="M1/M5/M15/H1",
            trader_summary=_summary_without_evidence(result.final_state.get("trader_investment_plan")),
            portfolio_manager_summary=_summary_without_evidence(result.final_state.get("final_trade_decision")),
            bull_summary=_summary_without_evidence(investment_debate_state.get("bull_history")) or None,
            bear_summary=_summary_without_evidence(investment_debate_state.get("bear_history")) or None,
            llm_provider=_identifier(effective_config.get("llm_provider")),
            quick_model=_identifier(effective_config.get("quick_think_llm")),
            deep_model=_identifier(effective_config.get("deep_think_llm")),
            snapshot_json=dict(result.snapshot_json),
            source_run_id=result.source_run_id or str(uuid.uuid4()),
            analysis_profile=result.profile_name,
            valid_for_seconds=result.valid_for_seconds,
            valid_until=result.valid_until,
            analysis_snapshot_timestamp=result.snapshot.timestamp,
            analysis_snapshot_bid=result.snapshot.bid,
            analysis_snapshot_ask=result.snapshot.ask,
            analysis_snapshot_spread=result.snapshot.spread,
            analysis_snapshot_spread_points=result.snapshot.spread_points,
            decision_completed_timestamp=completed,
            analysis_latency_seconds=(completed - result.snapshot.timestamp).total_seconds(),
            decision_reference_timestamp=None if decision_reference is None else decision_reference["timestamp"],
            decision_reference_bid=None if decision_reference is None else decision_reference["bid"],
            decision_reference_ask=None if decision_reference is None else decision_reference["ask"],
            decision_reference_spread=None if decision_reference is None else decision_reference["spread"],
            decision_reference_spread_points=None if decision_reference is None else decision_reference["spread_points"],
            decision_reference_status="UNAVAILABLE" if decision_reference is None else decision_reference["status"],
            decision_reference_delay_seconds=None if decision_reference is None else decision_reference["delay"],
            decision_reference_error=result.decision_reference_error,
            research_manager_recommendation=(
                result.final_state.get("research_manager_recommendation")
                if isinstance(result.final_state.get("research_manager_recommendation"), str)
                else None
            ),
        )
        self.store.record(decision)
        metrics = dict(result.analysis_telemetry)
        metrics.update(
            {
                "provider_snapshot_calls": metrics.get("provider_snapshot_calls", 1),
                "elapsed_seconds": time.perf_counter() - started,
                "market_data_mode": "forex_mt5",
                "selected_analysts": self.selected_analysts if analysts is None else tuple(analysts),
                "analysis_profile": result.profile_name,
                "bars_used": {
                    timeframe: result.snapshot_json.get("features", {}).get(timeframe, {}).get("candle_count", 0)
                    for timeframe in ("M1", "M5", "M15", "H1")
                },
                "market_features": result.snapshot_json.get("features", {}),
                "macro_event_status": MACRO_EVENT_UNAVAILABLE,
                "decision_valid_for_seconds": result.valid_for_seconds,
                "decision_valid_until": result.valid_until,
                "decision_context_status": result.context_integrity["status"],
                "context_integrity": result.context_integrity,
                "state_boundaries": list(result.state_trace),
                "audit_status": "NOT_REQUESTED",
            }
        )
        context = result.evidence_context
        if context is not None and context.integration_status in (
            EvidenceIntegrationStatus.INJECTED,
            EvidenceIntegrationStatus.FALLBACK,
        ):
            try:
                validation = validate_evidence_references(
                    context,
                    result.raw_portfolio_manager_result,
                    runtime_integration_status=context.integration_status,
                )
            except Exception:
                validation = EvidenceReferenceValidation(
                    evidence_use_status=EvidenceUseStatus.UNAVAILABLE,
                    evidence_refs_used=(),
                    evidence_refs_rejected=(),
                    evidence_audit_status=EvidenceAuditStatus.INVALID_REFERENCE,
                )
            if context.integration_status is EvidenceIntegrationStatus.FALLBACK and (
                validation.evidence_use_status is EvidenceUseStatus.USED
            ):
                validation = EvidenceReferenceValidation(
                    evidence_use_status=EvidenceUseStatus.UNAVAILABLE,
                    evidence_refs_used=(),
                    evidence_refs_rejected=(),
                    evidence_audit_status=(
                        EvidenceAuditStatus.INVALID_REFERENCE
                        if validation.evidence_refs_used
                        else validation.evidence_audit_status
                    ),
                )
            try:
                audit_factory = self.evidence_audit_store_factory or self._default_evidence_audit_store_factory
                audit_path = Path(self.config.get("data_cache_dir", "data_cache")) / "evidence_runtime" / "evidence_audit.sqlite3"
                audit_store = self._call_factory(
                    audit_factory,
                    config=self.config,
                    runner=self,
                    path=audit_path,
                )
                if audit_store is None:
                    audit_store = self._default_evidence_audit_store_factory(config=self.config)
                available = {
                    "knowledge": tuple(item.display_id for item in context.knowledge_items),
                    "experience": tuple(item.display_id for item in context.experience_items),
                    "statistics": tuple(item.display_id for item in context.statistics_items),
                }
                node_context_hashes = {
                    str(item.get("node")): str(item.get("artifacts", {}).get("evidence_context_hash"))
                    for item in result.state_trace
                    if isinstance(item, Mapping)
                    and item.get("phase") == "after"
                    and isinstance(item.get("artifacts"), Mapping)
                    and item.get("artifacts", {}).get("evidence_context_hash")
                }
                telemetry_references = result.analysis_telemetry.get("telemetry_references", ())
                if not isinstance(telemetry_references, (tuple, list)):
                    telemetry_references = ()
                audit = EvidenceUsageAudit(
                    decision_id=decision.decision_id,
                    source_run_id=decision.source_run_id or "",
                    integration_status=str(context.integration_status),
                    bundle_status=str(context.bundle_status),
                    as_of=context.as_of,
                    knowledge_generation_id=context.knowledge_generation_id,
                    experience_generation_id=context.experience_generation_id,
                    query_normalization_fingerprint=context.query_normalization_fingerprint,
                    knowledge_query=context.knowledge_query,
                    knowledge_query_fingerprint=context.knowledge_query_fingerprint,
                    query_policy_version=context.knowledge_query_policy_version,
                    rendered_context=context.rendered_context,
                    rendered_context_hash=self._evidence_hash(context),
                    available_knowledge_ids=available["knowledge"],
                    available_experience_ids=available["experience"],
                    available_statistics_ids=available["statistics"],
                    evidence_use_status=str(validation.evidence_use_status),
                    evidence_refs_used=validation.evidence_refs_used,
                    evidence_refs_rejected=validation.evidence_refs_rejected,
                    evidence_audit_status=str(validation.evidence_audit_status),
                    source_status=context.diagnostics.get("source_status", {}) if isinstance(context.diagnostics, Mapping) else {},
                    diagnostics=context.diagnostics,
                    source_errors=context.source_errors,
                    retrieval_count=int(metrics.get("evidence_retrieval_count", 1)),
                    retrieval_latency_seconds=metrics.get("evidence_retrieval_latency_seconds"),
                    builder_latency_seconds=float(metrics.get("evidence_builder_latency_seconds", 0.0)),
                    selected_counts={"knowledge": context.selected_knowledge_count, "experience": context.selected_experience_count, "statistics": context.selected_statistics_count},
                    dropped_counts={"knowledge": context.dropped_knowledge_count, "experience": context.dropped_experience_count, "statistics": context.dropped_statistics_count},
                    telemetry_references=tuple(str(value) for value in telemetry_references),
                    node_context_hashes=node_context_hashes,
                    missing_nodes=tuple(str(value) for value in result.context_integrity.get("missing_nodes", ())),
                    provider=_identifier(effective_config.get("llm_provider")),
                    model=_identifier(effective_config.get("deep_think_llm")),
                )
                audit_store.append(audit)
            except Exception:
                metrics["audit_status"] = "AUDIT_WRITE_FAILED"
            else:
                metrics["audit_status"] = str(validation.evidence_audit_status)
        return ForexShadowRunResult(
            decision=decision,
            final_state=result.final_state,
            snapshot_json=result.snapshot_json,
            provider_snapshot_calls=metrics["provider_snapshot_calls"],
            elapsed_seconds=metrics["elapsed_seconds"],
            metrics=metrics,
        )
