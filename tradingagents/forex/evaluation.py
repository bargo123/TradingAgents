"""Deterministic, read-only outcome evaluation for forex shadow decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from tradingagents.dataflows.mt5.models import Mt5Tick
from tradingagents.forex.shadow import ShadowTradeDecision

EvaluationBasis = Literal["ANALYSIS_SNAPSHOT", "DECISION_REFERENCE"]
EvaluationStatus = Literal["PENDING", "COMPLETE", "DATA_UNAVAILABLE", "INELIGIBLE"]
AllowedAction = Literal["BUY", "SELL", "HOLD"]
BestCounterfactualAction = Literal["BUY", "SELL", "TIE"]

_BASIS_VALUES = ("ANALYSIS_SNAPSHOT", "DECISION_REFERENCE")
_STATUS_VALUES = ("PENDING", "COMPLETE", "DATA_UNAVAILABLE", "INELIGIBLE")
_TRAINING_REASONS = {
    "DEFERRED_TO_CORPUS_BUILDER",
    "SOURCE_CONTEXT_INELIGIBLE",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware UTC values")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("timestamps must be UTC values")
    return value.astimezone(timezone.utc)


def _optional_utc(value: datetime | None) -> datetime | None:
    return None if value is None else _utc(value)


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    horizons_seconds: tuple[int, ...] = (300, 900, 1800, 3600)
    observation_tolerance_seconds: int = 30
    evaluation_version: str = "phase5.v1"
    market_data_source: str = "MT5"

    def __post_init__(self) -> None:
        horizons = tuple(self.horizons_seconds)
        if not horizons:
            raise ValueError("horizons_seconds must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in horizons
        ):
            raise ValueError("horizons_seconds must contain positive integers")
        if tuple(sorted(horizons)) != horizons or len(set(horizons)) != len(horizons):
            raise ValueError("horizons_seconds must be strictly increasing")
        object.__setattr__(self, "horizons_seconds", horizons)
        if (
            isinstance(self.observation_tolerance_seconds, bool)
            or not isinstance(self.observation_tolerance_seconds, int)
            or self.observation_tolerance_seconds < 0
        ):
            raise ValueError("observation_tolerance_seconds must be non-negative")
        if not isinstance(self.evaluation_version, str) or not self.evaluation_version.strip():
            raise ValueError("evaluation_version must be a non-empty string")
        if self.market_data_source != "MT5":
            raise ValueError("market_data_source must be MT5")


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    eligible: bool
    reason: str


@dataclass(frozen=True, slots=True)
class DirectionalOutcome:
    buy_net_price: float
    buy_net_points: float
    sell_net_price: float
    sell_net_points: float
    selected_action: AllowedAction | None
    selected_action_net_price: float
    selected_action_net_points: float
    best_counterfactual_action: BestCounterfactualAction
    best_counterfactual_net_points: float
    hold_opportunity_cost_points: float


@dataclass(frozen=True, slots=True)
class ExcursionMetrics:
    buy_mfe_price: float
    buy_mfe_points: float
    buy_mae_price: float
    buy_mae_points: float
    sell_mfe_price: float
    sell_mfe_points: float
    sell_mae_price: float
    sell_mae_points: float


@dataclass(frozen=True, slots=True)
class ShadowOutcomeEvaluation:
    decision_id: str
    resolved_symbol: str
    evaluation_basis: EvaluationBasis
    horizon_seconds: int
    evaluation_version: str
    market_data_source: str
    source_context_eligible: bool
    training_eligible: bool | None
    training_eligibility_reason: str
    target_timestamp: datetime | None = None
    observation_timestamp: datetime | None = None
    observation_lag_ms: int | None = None
    entry_timestamp: datetime | None = None
    entry_bid: float | None = None
    entry_ask: float | None = None
    entry_spread: float | None = None
    entry_spread_points: float | None = None
    future_bid: float | None = None
    future_ask: float | None = None
    future_spread: float | None = None
    future_spread_points: float | None = None
    point: float | None = None
    digits: int | None = None
    buy_net_price: float | None = None
    buy_net_points: float | None = None
    sell_net_price: float | None = None
    sell_net_points: float | None = None
    selected_action: AllowedAction | None = None
    selected_action_net_price: float | None = None
    selected_action_net_points: float | None = None
    best_counterfactual_action: BestCounterfactualAction | None = None
    best_counterfactual_net_points: float | None = None
    hold_opportunity_cost_points: float | None = None
    buy_mfe_price: float | None = None
    buy_mfe_points: float | None = None
    buy_mae_price: float | None = None
    buy_mae_points: float | None = None
    sell_mfe_price: float | None = None
    sell_mfe_points: float | None = None
    sell_mae_price: float | None = None
    sell_mae_points: float | None = None
    evaluation_status: EvaluationStatus = "PENDING"
    unavailable_reason: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    evaluated_at: datetime | None = None
    recovered_from_unavailable_at: datetime | None = None
    previous_unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.evaluation_basis not in _BASIS_VALUES:
            raise ValueError(f"unsupported evaluation basis: {self.evaluation_basis!r}")
        if self.evaluation_status not in _STATUS_VALUES:
            raise ValueError(f"unsupported evaluation status: {self.evaluation_status!r}")
        if (
            isinstance(self.horizon_seconds, bool)
            or not isinstance(self.horizon_seconds, int)
            or self.horizon_seconds <= 0
        ):
            raise ValueError("horizon_seconds must be a positive integer")
        if self.training_eligible is not None:
            raise ValueError("Phase 5 must defer training eligibility")
        if self.training_eligibility_reason not in _TRAINING_REASONS:
            raise ValueError("unsupported training eligibility reason")
        for name in (
            "target_timestamp",
            "observation_timestamp",
            "entry_timestamp",
            "created_at",
            "evaluated_at",
            "recovered_from_unavailable_at",
        ):
            value = getattr(self, name)
            if value is not None:
                _utc(value)
        for name in (
            "entry_bid",
            "entry_ask",
            "entry_spread",
            "entry_spread_points",
            "future_bid",
            "future_ask",
            "future_spread",
            "future_spread_points",
            "point",
            "buy_net_price",
            "buy_net_points",
            "sell_net_price",
            "sell_net_points",
            "selected_action_net_price",
            "selected_action_net_points",
            "best_counterfactual_net_points",
            "hold_opportunity_cost_points",
            "buy_mfe_price",
            "buy_mfe_points",
            "buy_mae_price",
            "buy_mae_points",
            "sell_mfe_price",
            "sell_mfe_points",
            "sell_mae_price",
            "sell_mae_points",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        if self.evaluation_status == "COMPLETE":
            if self.observation_timestamp is None or self.entry_timestamp is None:
                raise ValueError("complete evaluations require observation and entry timestamps")


def decision_source_eligibility(decision: ShadowTradeDecision) -> EligibilityResult:
    if getattr(decision, "decision_context_status", None) != "COMPLETE":
        return EligibilityResult(False, "DECISION_CONTEXT_INCOMPLETE")
    if getattr(decision, "normalization_status", None) != "NORMALIZED":
        return EligibilityResult(False, "NORMALIZATION_FAILED")
    if getattr(decision, "action", None) not in ("BUY", "SELL", "HOLD"):
        return EligibilityResult(False, "ACTION_UNAVAILABLE")
    timestamp = getattr(decision, "analysis_snapshot_timestamp", None)
    try:
        _utc(timestamp)
    except (TypeError, ValueError):
        return EligibilityResult(False, "ANALYSIS_TIMESTAMP_INVALID")
    return EligibilityResult(True, "ELIGIBLE_SOURCE_CONTEXT")


def evaluation_target(
    anchor_timestamp: datetime,
    horizon_seconds: int,
    *,
    tolerance_seconds: int = 30,
) -> tuple[datetime, datetime]:
    anchor = _utc(anchor_timestamp)
    if isinstance(horizon_seconds, bool) or not isinstance(horizon_seconds, int) or horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be a positive integer")
    if (
        isinstance(tolerance_seconds, bool)
        or not isinstance(tolerance_seconds, int)
        or tolerance_seconds < 0
    ):
        raise ValueError("tolerance_seconds must be non-negative")
    target = anchor + timedelta(seconds=horizon_seconds)
    return target, target + timedelta(seconds=tolerance_seconds)


def _valid_tick(tick: Any) -> bool:
    try:
        timestamp = _utc(tick.timestamp)
        bid = _finite(tick.bid, "tick.bid")
        ask = _finite(tick.ask, "tick.ask")
    except (AttributeError, TypeError, ValueError):
        return False
    return timestamp == tick.timestamp.astimezone(timezone.utc) and ask >= bid


def first_valid_tick(
    ticks: Sequence[Mt5Tick],
    target_timestamp: datetime,
    deadline_timestamp: datetime,
) -> Mt5Tick | None:
    target = _utc(target_timestamp)
    deadline = _utc(deadline_timestamp)
    if deadline < target:
        return None
    valid = [
        tick
        for tick in ticks
        if _valid_tick(tick)
        and target <= _utc(tick.timestamp) <= deadline
    ]
    return min(valid, key=lambda tick: _utc(tick.timestamp)) if valid else None


def evaluate_directional_outcomes(
    *,
    entry_bid: float,
    entry_ask: float,
    future_bid: float,
    future_ask: float,
    point: float,
    selected_action: AllowedAction | None,
) -> DirectionalOutcome:
    entry_bid = _finite(entry_bid, "entry_bid")
    entry_ask = _finite(entry_ask, "entry_ask")
    future_bid = _finite(future_bid, "future_bid")
    future_ask = _finite(future_ask, "future_ask")
    point = _finite(point, "point")
    if point <= 0:
        raise ValueError("point must be positive")
    if entry_ask < entry_bid or future_ask < future_bid:
        raise ValueError("ask must be greater than or equal to bid")
    buy_net_price = future_bid - entry_ask
    sell_net_price = entry_bid - future_ask
    buy_net_points = buy_net_price / point
    sell_net_points = sell_net_price / point
    if buy_net_points > sell_net_points:
        best_action: BestCounterfactualAction = "BUY"
    elif sell_net_points > buy_net_points:
        best_action = "SELL"
    else:
        best_action = "TIE"
    if selected_action == "BUY":
        selected_price, selected_points = buy_net_price, buy_net_points
    elif selected_action == "SELL":
        selected_price, selected_points = sell_net_price, sell_net_points
    else:
        selected_price, selected_points = 0.0, 0.0
    return DirectionalOutcome(
        buy_net_price=buy_net_price,
        buy_net_points=buy_net_points,
        sell_net_price=sell_net_price,
        sell_net_points=sell_net_points,
        selected_action=selected_action,
        selected_action_net_price=selected_price,
        selected_action_net_points=selected_points,
        best_counterfactual_action=best_action,
        best_counterfactual_net_points=max(buy_net_points, sell_net_points),
        hold_opportunity_cost_points=max(0.0, buy_net_points, sell_net_points),
    )


def evaluate_excursions(
    *,
    entry_bid: float,
    entry_ask: float,
    point: float,
    ticks: Sequence[Mt5Tick],
    anchor_timestamp: datetime,
    target_timestamp: datetime,
) -> ExcursionMetrics:
    entry_bid = _finite(entry_bid, "entry_bid")
    entry_ask = _finite(entry_ask, "entry_ask")
    point = _finite(point, "point")
    if point <= 0:
        raise ValueError("point must be positive")
    anchor = _utc(anchor_timestamp)
    target = _utc(target_timestamp)
    if target < anchor:
        raise ValueError("target_timestamp must not precede anchor_timestamp")
    window = [
        tick
        for tick in ticks
        if _valid_tick(tick)
        and anchor <= _utc(tick.timestamp) <= target
    ]
    bids = [entry_bid] + [_finite(tick.bid, "tick.bid") for tick in window]
    asks = [entry_ask] + [_finite(tick.ask, "tick.ask") for tick in window]
    buy_mfe_price = max(bids) - entry_ask
    buy_mae_price = min(bids) - entry_ask
    sell_mfe_price = entry_bid - min(asks)
    sell_mae_price = entry_bid - max(asks)
    return ExcursionMetrics(
        buy_mfe_price=buy_mfe_price,
        buy_mfe_points=buy_mfe_price / point,
        buy_mae_price=buy_mae_price,
        buy_mae_points=buy_mae_price / point,
        sell_mfe_price=sell_mfe_price,
        sell_mfe_points=sell_mfe_price / point,
        sell_mae_price=sell_mae_price,
        sell_mae_points=sell_mae_price / point,
    )


def _entry_quote(
    decision: ShadowTradeDecision,
    basis: EvaluationBasis,
) -> tuple[datetime, float, float, float, float] | None:
    if basis == "ANALYSIS_SNAPSHOT":
        values = (
            getattr(decision, "analysis_snapshot_timestamp", None),
            getattr(decision, "analysis_snapshot_bid", None),
            getattr(decision, "analysis_snapshot_ask", None),
            getattr(decision, "analysis_snapshot_spread", None),
            getattr(decision, "analysis_snapshot_spread_points", None),
        )
    else:
        if getattr(decision, "decision_reference_status", None) != "AVAILABLE":
            return None
        values = (
            getattr(decision, "decision_reference_timestamp", None),
            getattr(decision, "decision_reference_bid", None),
            getattr(decision, "decision_reference_ask", None),
            getattr(decision, "decision_reference_spread", None),
            getattr(decision, "decision_reference_spread_points", None),
        )
    if any(value is None for value in values):
        return None
    timestamp, bid, ask, spread, spread_points = values
    try:
        timestamp = _utc(timestamp)
        bid = _finite(bid, "entry_bid")
        ask = _finite(ask, "entry_ask")
        spread = _finite(spread, "entry_spread")
        spread_points = _finite(spread_points, "entry_spread_points")
    except (TypeError, ValueError):
        return None
    if ask < bid or spread < 0 or spread_points < 0:
        return None
    return timestamp, bid, ask, spread, spread_points


def _market_metadata(decision: ShadowTradeDecision) -> tuple[float, int] | None:
    snapshot_json = getattr(decision, "snapshot_json", None)
    metadata = snapshot_json.get("symbol_metadata") if isinstance(snapshot_json, Mapping) else None
    if not isinstance(metadata, Mapping):
        return None
    try:
        point = _finite(metadata.get("point"), "point")
        digits_value = metadata.get("digits")
        if (
            isinstance(digits_value, bool)
            or not isinstance(digits_value, int)
            or digits_value < 0
            or point <= 0
        ):
            return None
    except (TypeError, ValueError):
        return None
    return point, digits_value


def _base_fields(
    decision: ShadowTradeDecision,
    basis: EvaluationBasis,
    horizon_seconds: int,
    config: EvaluationConfig,
    *,
    target_timestamp: datetime | None,
    entry: tuple[datetime, float, float, float, float] | None,
    status: EvaluationStatus,
    reason: str | None,
    now: datetime,
) -> dict[str, Any]:
    eligible = decision_source_eligibility(decision).eligible
    return {
        "decision_id": decision.decision_id,
        "resolved_symbol": decision.resolved_symbol,
        "evaluation_basis": basis,
        "horizon_seconds": horizon_seconds,
        "evaluation_version": config.evaluation_version,
        "market_data_source": config.market_data_source,
        "source_context_eligible": eligible,
        "training_eligible": None,
        "training_eligibility_reason": (
            "DEFERRED_TO_CORPUS_BUILDER" if eligible else "SOURCE_CONTEXT_INELIGIBLE"
        ),
        "target_timestamp": target_timestamp,
        "entry_timestamp": None if entry is None else entry[0],
        "entry_bid": None if entry is None else entry[1],
        "entry_ask": None if entry is None else entry[2],
        "entry_spread": None if entry is None else entry[3],
        "entry_spread_points": None if entry is None else entry[4],
        "evaluation_status": status,
        "unavailable_reason": reason,
        "created_at": now,
        "evaluated_at": now,
    }


def build_horizon_evaluation(
    decision: ShadowTradeDecision,
    *,
    evaluation_basis: EvaluationBasis,
    horizon_seconds: int,
    now: datetime,
    ticks: Sequence[Mt5Tick],
    config: EvaluationConfig,
) -> ShadowOutcomeEvaluation:
    if evaluation_basis not in _BASIS_VALUES:
        raise ValueError(f"unsupported evaluation basis: {evaluation_basis!r}")
    if horizon_seconds not in config.horizons_seconds:
        raise ValueError("horizon_seconds is not enabled by EvaluationConfig")
    now = _utc(now)
    eligibility = decision_source_eligibility(decision)
    entry = _entry_quote(decision, evaluation_basis)
    if not eligibility.eligible:
        target = None
        if entry is not None:
            target, _ = evaluation_target(
                entry[0],
                horizon_seconds,
                tolerance_seconds=config.observation_tolerance_seconds,
            )
        return ShadowOutcomeEvaluation(
            **_base_fields(
                decision,
                evaluation_basis,
                horizon_seconds,
                config,
                target_timestamp=target,
                entry=entry,
                status="INELIGIBLE",
                reason=eligibility.reason,
                now=now,
            )
        )
    if entry is None:
        reason = (
            "decision reference unavailable"
            if evaluation_basis == "DECISION_REFERENCE"
            else "analysis snapshot quote unavailable"
        )
        return ShadowOutcomeEvaluation(
            **_base_fields(
                decision,
                evaluation_basis,
                horizon_seconds,
                config,
                target_timestamp=None,
                entry=None,
                status="DATA_UNAVAILABLE",
                reason=reason,
                now=now,
            )
        )
    anchor, entry_bid, entry_ask, entry_spread, entry_spread_points = entry
    target, deadline = evaluation_target(
        anchor,
        horizon_seconds,
        tolerance_seconds=config.observation_tolerance_seconds,
    )
    common = _base_fields(
        decision,
        evaluation_basis,
        horizon_seconds,
        config,
        target_timestamp=target,
        entry=entry,
        status="PENDING",
        reason=None,
        now=now,
    )
    if now < deadline:
        return ShadowOutcomeEvaluation(**common)
    future = first_valid_tick(ticks, target, deadline)
    if future is None:
        common["evaluation_status"] = "DATA_UNAVAILABLE"
        common["unavailable_reason"] = "no valid MT5 tick in target tolerance window"
        return ShadowOutcomeEvaluation(**common)
    metadata = _market_metadata(decision)
    if metadata is None:
        common["evaluation_status"] = "DATA_UNAVAILABLE"
        common["unavailable_reason"] = "invalid symbol point/digits metadata"
        return ShadowOutcomeEvaluation(**common)
    point, digits = metadata
    future_bid = _finite(future.bid, "future_bid")
    future_ask = _finite(future.ask, "future_ask")
    future_spread = future_ask - future_bid
    future_spread_points = future_spread / point
    direction = evaluate_directional_outcomes(
        entry_bid=entry_bid,
        entry_ask=entry_ask,
        future_bid=future_bid,
        future_ask=future_ask,
        point=point,
        selected_action=decision.action,
    )
    excursions = evaluate_excursions(
        entry_bid=entry_bid,
        entry_ask=entry_ask,
        point=point,
        ticks=ticks,
        anchor_timestamp=anchor,
        target_timestamp=target,
    )
    common.update(
        {
            "evaluation_status": "COMPLETE",
            "observation_timestamp": _utc(future.timestamp),
            "observation_lag_ms": round((_utc(future.timestamp) - target).total_seconds() * 1000),
            "future_bid": future_bid,
            "future_ask": future_ask,
            "future_spread": future_spread,
            "future_spread_points": future_spread_points,
            "point": point,
            "digits": digits,
            "buy_net_price": direction.buy_net_price,
            "buy_net_points": direction.buy_net_points,
            "sell_net_price": direction.sell_net_price,
            "sell_net_points": direction.sell_net_points,
            "selected_action": direction.selected_action,
            "selected_action_net_price": direction.selected_action_net_price,
            "selected_action_net_points": direction.selected_action_net_points,
            "best_counterfactual_action": direction.best_counterfactual_action,
            "best_counterfactual_net_points": direction.best_counterfactual_net_points,
            "hold_opportunity_cost_points": direction.hold_opportunity_cost_points,
            "buy_mfe_price": excursions.buy_mfe_price,
            "buy_mfe_points": excursions.buy_mfe_points,
            "buy_mae_price": excursions.buy_mae_price,
            "buy_mae_points": excursions.buy_mae_points,
            "sell_mfe_price": excursions.sell_mfe_price,
            "sell_mfe_points": excursions.sell_mfe_points,
            "sell_mae_price": excursions.sell_mae_price,
            "sell_mae_points": excursions.sell_mae_points,
        }
    )
    return ShadowOutcomeEvaluation(**common)


__all__ = [
    "AllowedAction",
    "BestCounterfactualAction",
    "DirectionalOutcome",
    "EligibilityResult",
    "EvaluationBasis",
    "EvaluationConfig",
    "EvaluationStatus",
    "ExcursionMetrics",
    "ShadowOutcomeEvaluation",
    "build_horizon_evaluation",
    "decision_source_eligibility",
    "evaluate_directional_outcomes",
    "evaluate_excursions",
    "evaluation_target",
    "first_valid_tick",
]
