"""Deterministic, read-only outcome evaluation for forex shadow decisions."""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


_EVALUATION_COLUMNS = (
    "decision_id",
    "resolved_symbol",
    "evaluation_basis",
    "horizon_seconds",
    "evaluation_version",
    "market_data_source",
    "source_context_eligible",
    "training_eligible",
    "training_eligibility_reason",
    "target_timestamp",
    "observation_timestamp",
    "observation_lag_ms",
    "entry_timestamp",
    "entry_bid",
    "entry_ask",
    "entry_spread",
    "entry_spread_points",
    "future_bid",
    "future_ask",
    "future_spread",
    "future_spread_points",
    "point",
    "digits",
    "buy_net_price",
    "buy_net_points",
    "sell_net_price",
    "sell_net_points",
    "selected_action",
    "selected_action_net_price",
    "selected_action_net_points",
    "best_counterfactual_action",
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
    "evaluation_status",
    "unavailable_reason",
    "created_at",
    "evaluated_at",
    "recovered_from_unavailable_at",
    "previous_unavailable_reason",
)

_CREATE_EVALUATIONS_SQL = """
CREATE TABLE IF NOT EXISTS shadow_decision_evaluations (
    decision_id TEXT NOT NULL,
    evaluation_basis TEXT NOT NULL CHECK (evaluation_basis IN ('ANALYSIS_SNAPSHOT','DECISION_REFERENCE')),
    horizon_seconds INTEGER NOT NULL CHECK (horizon_seconds > 0),
    resolved_symbol TEXT NOT NULL,
    evaluation_version TEXT NOT NULL,
    market_data_source TEXT NOT NULL CHECK (market_data_source = 'MT5'),
    source_context_eligible INTEGER NOT NULL CHECK (source_context_eligible IN (0, 1)),
    training_eligible INTEGER CHECK (training_eligible IS NULL OR training_eligible IN (0, 1)),
    training_eligibility_reason TEXT NOT NULL,
    target_timestamp TEXT,
    observation_timestamp TEXT,
    observation_lag_ms INTEGER,
    entry_timestamp TEXT,
    entry_bid REAL,
    entry_ask REAL,
    entry_spread REAL,
    entry_spread_points REAL,
    future_bid REAL,
    future_ask REAL,
    future_spread REAL,
    future_spread_points REAL,
    point REAL,
    digits INTEGER,
    buy_net_price REAL,
    buy_net_points REAL,
    sell_net_price REAL,
    sell_net_points REAL,
    selected_action TEXT CHECK (selected_action IS NULL OR selected_action IN ('BUY','SELL','HOLD')),
    selected_action_net_price REAL,
    selected_action_net_points REAL,
    best_counterfactual_action TEXT CHECK (best_counterfactual_action IS NULL OR best_counterfactual_action IN ('BUY','SELL','TIE')),
    best_counterfactual_net_points REAL,
    hold_opportunity_cost_points REAL,
    buy_mfe_price REAL,
    buy_mfe_points REAL,
    buy_mae_price REAL,
    buy_mae_points REAL,
    sell_mfe_price REAL,
    sell_mfe_points REAL,
    sell_mae_price REAL,
    sell_mae_points REAL,
    evaluation_status TEXT NOT NULL CHECK (evaluation_status IN ('PENDING','COMPLETE','DATA_UNAVAILABLE','INELIGIBLE')),
    unavailable_reason TEXT,
    created_at TEXT NOT NULL,
    evaluated_at TEXT,
    recovered_from_unavailable_at TEXT,
    previous_unavailable_reason TEXT,
    PRIMARY KEY (decision_id, evaluation_basis, horizon_seconds)
)
"""


def _iso(value: datetime | None) -> str | None:
    return None if value is None else _utc(value).isoformat().replace("+00:00", "Z")


def _record_values(
    record: ShadowOutcomeEvaluation,
    *,
    created_at: datetime | None = None,
    evaluated_at: datetime | None = None,
    recovered_from_unavailable_at: datetime | None = None,
    previous_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    values = {
        "decision_id": record.decision_id,
        "resolved_symbol": record.resolved_symbol,
        "evaluation_basis": record.evaluation_basis,
        "horizon_seconds": record.horizon_seconds,
        "evaluation_version": record.evaluation_version,
        "market_data_source": record.market_data_source,
        "source_context_eligible": int(record.source_context_eligible),
        "training_eligible": None,
        "training_eligibility_reason": record.training_eligibility_reason,
        "target_timestamp": _iso(record.target_timestamp),
        "observation_timestamp": _iso(record.observation_timestamp),
        "observation_lag_ms": record.observation_lag_ms,
        "entry_timestamp": _iso(record.entry_timestamp),
        "entry_bid": record.entry_bid,
        "entry_ask": record.entry_ask,
        "entry_spread": record.entry_spread,
        "entry_spread_points": record.entry_spread_points,
        "future_bid": record.future_bid,
        "future_ask": record.future_ask,
        "future_spread": record.future_spread,
        "future_spread_points": record.future_spread_points,
        "point": record.point,
        "digits": record.digits,
        "buy_net_price": record.buy_net_price,
        "buy_net_points": record.buy_net_points,
        "sell_net_price": record.sell_net_price,
        "sell_net_points": record.sell_net_points,
        "selected_action": record.selected_action,
        "selected_action_net_price": record.selected_action_net_price,
        "selected_action_net_points": record.selected_action_net_points,
        "best_counterfactual_action": record.best_counterfactual_action,
        "best_counterfactual_net_points": record.best_counterfactual_net_points,
        "hold_opportunity_cost_points": record.hold_opportunity_cost_points,
        "buy_mfe_price": record.buy_mfe_price,
        "buy_mfe_points": record.buy_mfe_points,
        "buy_mae_price": record.buy_mae_price,
        "buy_mae_points": record.buy_mae_points,
        "sell_mfe_price": record.sell_mfe_price,
        "sell_mfe_points": record.sell_mfe_points,
        "sell_mae_price": record.sell_mae_price,
        "sell_mae_points": record.sell_mae_points,
        "evaluation_status": record.evaluation_status,
        "unavailable_reason": record.unavailable_reason,
        "created_at": _iso(created_at or record.created_at),
        "evaluated_at": _iso(evaluated_at or record.evaluated_at),
        "recovered_from_unavailable_at": _iso(
            recovered_from_unavailable_at or record.recovered_from_unavailable_at
        ),
        "previous_unavailable_reason": (
            previous_unavailable_reason
            if previous_unavailable_reason is not None
            else record.previous_unavailable_reason
        ),
    }
    return values


class ShadowEvaluationStore:
    """SQLite persistence for immutable-per-basis shadow outcome evidence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                ("shadow_decision_evaluations",),
            ).fetchone()
            if not exists:
                conn.execute(_CREATE_EVALUATIONS_SQL)
                return
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(shadow_decision_evaluations)")
            }
            if not {"evaluation_basis", "source_context_eligible", "training_eligibility_reason"}.issubset(columns):
                self._migrate_legacy_table(conn)
                return
            for column in (
                "evaluated_at",
                "recovered_from_unavailable_at",
                "previous_unavailable_reason",
            ):
                if column not in columns:
                    conn.execute(f"ALTER TABLE shadow_decision_evaluations ADD COLUMN {column} TEXT")

    def _migrate_legacy_table(self, conn: sqlite3.Connection) -> None:
        legacy_name = "shadow_decision_evaluations_legacy"
        suffix = 1
        while conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (legacy_name,)
        ).fetchone():
            suffix += 1
            legacy_name = f"shadow_decision_evaluations_legacy_{suffix}"
        conn.execute(f"ALTER TABLE shadow_decision_evaluations RENAME TO {legacy_name}")
        conn.execute(_CREATE_EVALUATIONS_SQL)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {legacy_name}").fetchall()
        for row in rows:
            row_keys = set(row.keys())
            status = row["evaluation_status"] if "evaluation_status" in row_keys else "DATA_UNAVAILABLE"
            if status not in _STATUS_VALUES:
                status = "DATA_UNAVAILABLE"
            created_at = (
                row["created_at"]
                if "created_at" in row_keys and row["created_at"]
                else _iso(_utc_now())
            )
            values = {
                "decision_id": row["decision_id"],
                "resolved_symbol": row["resolved_symbol"] if "resolved_symbol" in row_keys else "",
                "evaluation_basis": "ANALYSIS_SNAPSHOT",
                "horizon_seconds": row["horizon_seconds"] if "horizon_seconds" in row_keys else 1,
                "evaluation_version": row["evaluation_version"] if "evaluation_version" in row_keys else "legacy",
                "market_data_source": row["market_data_source"] if "market_data_source" in row_keys else "MT5",
                "source_context_eligible": 0,
                "training_eligible": None,
                "training_eligibility_reason": "SOURCE_CONTEXT_INELIGIBLE",
                "target_timestamp": row["target_timestamp"] if "target_timestamp" in row_keys else None,
                "evaluation_status": status,
                "unavailable_reason": row["unavailable_reason"] if "unavailable_reason" in row_keys else None,
                "created_at": created_at,
                "evaluated_at": None,
                "recovered_from_unavailable_at": None,
                "previous_unavailable_reason": None,
            }
            for column in _EVALUATION_COLUMNS:
                values.setdefault(column, row[column] if column in row_keys else None)
            values["source_context_eligible"] = 0
            values["training_eligible"] = None
            values["training_eligibility_reason"] = "SOURCE_CONTEXT_INELIGIBLE"
            values["evaluation_basis"] = "ANALYSIS_SNAPSHOT"
            values["created_at"] = created_at
            placeholders = ", ".join("?" for _ in _EVALUATION_COLUMNS)
            conn.execute(
                f"INSERT INTO shadow_decision_evaluations ({', '.join(_EVALUATION_COLUMNS)}) VALUES ({placeholders})",
                tuple(values[column] for column in _EVALUATION_COLUMNS),
            )

    def get(
        self,
        decision_id: str,
        evaluation_basis: EvaluationBasis,
        horizon_seconds: int,
    ) -> ShadowOutcomeEvaluation:
        self.initialize()
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM shadow_decision_evaluations WHERE decision_id=? AND evaluation_basis=? AND horizon_seconds=?",
                (decision_id, evaluation_basis, horizon_seconds),
            ).fetchone()
        if row is None:
            raise KeyError((decision_id, evaluation_basis, horizon_seconds))
        return self._row_to_record(row)

    def list_for_decision(self, decision_id: str) -> list[ShadowOutcomeEvaluation]:
        self.initialize()
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM shadow_decision_evaluations WHERE decision_id=? ORDER BY evaluation_basis, horizon_seconds",
                (decision_id,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def upsert(self, records: Sequence[ShadowOutcomeEvaluation]) -> None:
        self.initialize()
        records = tuple(records)
        if not records:
            return
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            for record in records:
                key = (record.decision_id, record.evaluation_basis, record.horizon_seconds)
                existing = conn.execute(
                    "SELECT * FROM shadow_decision_evaluations WHERE decision_id=? AND evaluation_basis=? AND horizon_seconds=?",
                    key,
                ).fetchone()
                evaluated_at = record.evaluated_at or record.created_at or _utc_now()
                if existing is None:
                    values = _record_values(record, evaluated_at=evaluated_at)
                    placeholders = ", ".join("?" for _ in _EVALUATION_COLUMNS)
                    conn.execute(
                        f"INSERT INTO shadow_decision_evaluations ({', '.join(_EVALUATION_COLUMNS)}) VALUES ({placeholders})",
                        tuple(values[column] for column in _EVALUATION_COLUMNS),
                    )
                    continue
                current_status = existing["evaluation_status"]
                incoming_status = record.evaluation_status
                if current_status in ("COMPLETE", "INELIGIBLE"):
                    continue
                if current_status == "DATA_UNAVAILABLE" and incoming_status != "COMPLETE":
                    continue
                if current_status == "PENDING" and incoming_status not in (
                    "PENDING",
                    "COMPLETE",
                    "DATA_UNAVAILABLE",
                ):
                    continue
                recovered_at = None
                previous_reason = None
                if current_status == "DATA_UNAVAILABLE" and incoming_status == "COMPLETE":
                    recovered_at = evaluated_at
                    previous_reason = existing["unavailable_reason"]
                values = _record_values(
                    record,
                    created_at=_parse_db_timestamp(existing["created_at"]),
                    evaluated_at=evaluated_at,
                    recovered_from_unavailable_at=recovered_at,
                    previous_unavailable_reason=previous_reason,
                )
                update_columns = [
                    column
                    for column in _EVALUATION_COLUMNS
                    if column not in ("decision_id", "evaluation_basis", "horizon_seconds", "created_at")
                ]
                conn.execute(
                    f"UPDATE shadow_decision_evaluations SET {', '.join(f'{column}=?' for column in update_columns)} WHERE decision_id=? AND evaluation_basis=? AND horizon_seconds=?",
                    tuple(values[column] for column in update_columns) + key,
                )

    def upsert_pending(self, records: Sequence[ShadowOutcomeEvaluation]) -> None:
        self.upsert(records)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ShadowOutcomeEvaluation:
        keys = set(row.keys())

        def value(name: str, default: Any = None) -> Any:
            return row[name] if name in keys else default

        return ShadowOutcomeEvaluation(
            decision_id=row["decision_id"],
            resolved_symbol=row["resolved_symbol"],
            evaluation_basis=row["evaluation_basis"],
            horizon_seconds=int(row["horizon_seconds"]),
            evaluation_version=row["evaluation_version"],
            market_data_source=row["market_data_source"],
            source_context_eligible=bool(row["source_context_eligible"]),
            training_eligible=None,
            training_eligibility_reason=row["training_eligibility_reason"],
            target_timestamp=_parse_db_timestamp(value("target_timestamp")),
            observation_timestamp=_parse_db_timestamp(value("observation_timestamp")),
            observation_lag_ms=value("observation_lag_ms"),
            entry_timestamp=_parse_db_timestamp(value("entry_timestamp")),
            entry_bid=value("entry_bid"),
            entry_ask=value("entry_ask"),
            entry_spread=value("entry_spread"),
            entry_spread_points=value("entry_spread_points"),
            future_bid=value("future_bid"),
            future_ask=value("future_ask"),
            future_spread=value("future_spread"),
            future_spread_points=value("future_spread_points"),
            point=value("point"),
            digits=value("digits"),
            buy_net_price=value("buy_net_price"),
            buy_net_points=value("buy_net_points"),
            sell_net_price=value("sell_net_price"),
            sell_net_points=value("sell_net_points"),
            selected_action=value("selected_action"),
            selected_action_net_price=value("selected_action_net_price"),
            selected_action_net_points=value("selected_action_net_points"),
            best_counterfactual_action=value("best_counterfactual_action"),
            best_counterfactual_net_points=value("best_counterfactual_net_points"),
            hold_opportunity_cost_points=value("hold_opportunity_cost_points"),
            buy_mfe_price=value("buy_mfe_price"),
            buy_mfe_points=value("buy_mfe_points"),
            buy_mae_price=value("buy_mae_price"),
            buy_mae_points=value("buy_mae_points"),
            sell_mfe_price=value("sell_mfe_price"),
            sell_mfe_points=value("sell_mfe_points"),
            sell_mae_price=value("sell_mae_price"),
            sell_mae_points=value("sell_mae_points"),
            evaluation_status=row["evaluation_status"],
            unavailable_reason=value("unavailable_reason"),
            created_at=_parse_db_timestamp(row["created_at"]) or _utc_now(),
            evaluated_at=_parse_db_timestamp(value("evaluated_at")),
            recovered_from_unavailable_at=_parse_db_timestamp(
                value("recovered_from_unavailable_at")
            ),
            previous_unavailable_reason=value("previous_unavailable_reason"),
        )


def _parse_db_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


@dataclass(frozen=True, slots=True)
class ShadowDecisionEvaluationResult:
    decision: ShadowTradeDecision
    evaluations: tuple[ShadowOutcomeEvaluation, ...]
    status_by_basis: Mapping[EvaluationBasis, str]
    errors: tuple[str, ...]
    metrics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ShadowEvaluationBatchResult:
    decisions: tuple[ShadowTradeDecision, ...]
    evaluations: tuple[ShadowOutcomeEvaluation, ...]
    status_by_basis: Mapping[EvaluationBasis, str]
    status_by_decision: Mapping[str, Mapping[EvaluationBasis, str]]
    errors: tuple[str, ...]
    metrics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _MatureRequest:
    basis: EvaluationBasis
    horizon_seconds: int
    anchor_timestamp: datetime
    target_timestamp: datetime
    deadline_timestamp: datetime
    existing_status: EvaluationStatus | None


def _aggregate_status(records: Sequence[ShadowOutcomeEvaluation]) -> str:
    if not records:
        return "PENDING"
    statuses = [record.evaluation_status for record in records]
    if any(status == "INELIGIBLE" for status in statuses):
        return "INELIGIBLE"
    if any(status == "PENDING" for status in statuses):
        return "PARTIAL" if any(status != "PENDING" for status in statuses) else "PENDING"
    if any(status == "COMPLETE" for status in statuses):
        return "COMPLETE"
    return "DATA_UNAVAILABLE"


class ShadowOutcomeEvaluator:
    """Evaluate persisted decisions with one bounded, read-only MT5 read."""

    def __init__(
        self,
        *,
        decision_store: Any,
        evaluation_store: ShadowEvaluationStore | None = None,
        config: EvaluationConfig | None = None,
        provider_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.decision_store = decision_store
        self.evaluation_store = evaluation_store or ShadowEvaluationStore(
            decision_store.path
        )
        self.config = config or EvaluationConfig()
        self.provider_factory = provider_factory or self._default_provider_factory

    @staticmethod
    def _default_provider_factory(terminal_path: str | None = None) -> Any:
        from tradingagents.dataflows.mt5.provider import MT5Provider

        return MT5Provider(terminal_path=terminal_path)

    def _prepare_decision(
        self,
        decision: ShadowTradeDecision,
        now: datetime,
    ) -> tuple[list[ShadowOutcomeEvaluation], list[_MatureRequest]]:
        existing = {
            (row.evaluation_basis, row.horizon_seconds): row
            for row in self.evaluation_store.list_for_decision(decision.decision_id)
        }
        records: list[ShadowOutcomeEvaluation] = []
        mature: list[_MatureRequest] = []
        eligibility = decision_source_eligibility(decision)
        for basis in _BASIS_VALUES:
            for horizon in self.config.horizons_seconds:
                key = (basis, horizon)
                current = existing.get(key)
                if current is not None and current.evaluation_status in (
                    "COMPLETE",
                    "INELIGIBLE",
                ):
                    records.append(current)
                    continue
                if not eligibility.eligible:
                    records.append(
                        build_horizon_evaluation(
                            decision,
                            evaluation_basis=basis,
                            horizon_seconds=horizon,
                            now=now,
                            ticks=(),
                            config=self.config,
                        )
                    )
                    continue
                entry = _entry_quote(decision, basis)
                if entry is None:
                    records.append(
                        build_horizon_evaluation(
                            decision,
                            evaluation_basis=basis,
                            horizon_seconds=horizon,
                            now=now,
                            ticks=(),
                            config=self.config,
                        )
                    )
                    continue
                anchor, _, _, _, _ = entry
                target, deadline = evaluation_target(
                    anchor,
                    horizon,
                    tolerance_seconds=self.config.observation_tolerance_seconds,
                )
                if now < deadline:
                    records.append(
                        build_horizon_evaluation(
                            decision,
                            evaluation_basis=basis,
                            horizon_seconds=horizon,
                            now=now,
                            ticks=(),
                            config=self.config,
                        )
                    )
                    continue
                mature.append(
                    _MatureRequest(
                        basis=basis,
                        horizon_seconds=horizon,
                        anchor_timestamp=anchor,
                        target_timestamp=target,
                        deadline_timestamp=deadline,
                        existing_status=(None if current is None else current.evaluation_status),
                    )
                )
                if current is not None and current.evaluation_status == "DATA_UNAVAILABLE":
                    records.append(current)
                else:
                    records.append(
                        _pending_evaluation(
                            decision,
                            basis,
                            horizon,
                            self.config,
                            now,
                        )
                    )
        return records, mature

    def _new_provider(self, terminal_path: str | None) -> Any:
        return self.provider_factory(terminal_path=terminal_path)

    @staticmethod
    def _initialize_provider(provider: Any) -> None:
        if not callable(getattr(provider, "initialize", None)):
            raise RuntimeError("MT5 provider does not expose initialize()")
        if not provider.initialize():
            raise RuntimeError("MT5 provider initialization failed")

    def _evaluate_one(
        self,
        decision: ShadowTradeDecision,
        *,
        now: datetime,
        terminal_path: str | None,
        provider: Any | None,
        provider_initialized: bool,
    ) -> tuple[ShadowDecisionEvaluationResult, Any | None, bool]:
        started = time.perf_counter()
        db_seconds = 0.0
        mt5_seconds = 0.0
        errors: list[str] = []
        db_started = time.perf_counter()
        records, mature = self._prepare_decision(decision, now)
        db_seconds += time.perf_counter() - db_started
        db_started = time.perf_counter()
        self.evaluation_store.upsert(records)
        db_seconds += time.perf_counter() - db_started
        if mature:
            if provider is None:
                try:
                    provider = self._new_provider(terminal_path)
                except Exception as exc:
                    errors.append(str(exc))
            if provider is not None and not provider_initialized and not errors:
                try:
                    self._initialize_provider(provider)
                    provider_initialized = True
                except Exception as exc:
                    errors.append(str(exc))
            if provider is not None and provider_initialized and not errors:
                start = min(item.anchor_timestamp for item in mature)
                end = max(item.deadline_timestamp for item in mature)
                try:
                    mt5_started = time.perf_counter()
                    ticks = tuple(
                        provider.get_ticks_range(
                            decision.resolved_symbol,
                            start,
                            end,
                        )
                    )
                    mt5_seconds += time.perf_counter() - mt5_started
                    evaluated = [
                        build_horizon_evaluation(
                            decision,
                            evaluation_basis=item.basis,
                            horizon_seconds=item.horizon_seconds,
                            now=now,
                            ticks=ticks,
                            config=self.config,
                        )
                        for item in mature
                    ]
                    db_started = time.perf_counter()
                    self.evaluation_store.upsert(evaluated)
                    db_seconds += time.perf_counter() - db_started
                except Exception as exc:
                    errors.append(str(exc))
        db_started = time.perf_counter()
        final_records = tuple(self.evaluation_store.list_for_decision(decision.decision_id))
        db_seconds += time.perf_counter() - db_started
        status_by_basis = {
            basis: _aggregate_status(
                [row for row in final_records if row.evaluation_basis == basis]
            )
            for basis in _BASIS_VALUES
        }
        metrics = {
            "decisions_scanned": 1,
            "horizons_evaluated": len(mature),
            "historical_ticks_processed": 0 if not mature or errors else len(ticks),
            "database_seconds": db_seconds,
            "mt5_read_seconds": mt5_seconds,
            "total_runtime_seconds": time.perf_counter() - started,
            "llm_calls": 0,
        }
        return (
            ShadowDecisionEvaluationResult(
                decision=decision,
                evaluations=final_records,
                status_by_basis=status_by_basis,
                errors=tuple(errors),
                metrics=metrics,
            ),
            provider,
            provider_initialized,
        )

    def evaluate_decision(
        self,
        decision_id: str,
        *,
        now: datetime | None = None,
        terminal_path: str | None = None,
    ) -> ShadowDecisionEvaluationResult:
        decision = self.decision_store.get(decision_id)
        provider: Any | None = None
        try:
            result, provider, _ = self._evaluate_one(
                decision,
                now=_utc(now or _utc_now()),
                terminal_path=terminal_path,
                provider=None,
                provider_initialized=False,
            )
            return result
        finally:
            shutdown = getattr(provider, "shutdown", None) if provider is not None else None
            if callable(shutdown):
                shutdown()

    def evaluate_pending(
        self,
        *,
        now: datetime | None = None,
        terminal_path: str | None = None,
    ) -> ShadowEvaluationBatchResult:
        started = time.perf_counter()
        now = _utc(now or _utc_now())
        db_started = time.perf_counter()
        decisions = tuple(self.decision_store.list_pending())
        database_seconds = time.perf_counter() - db_started
        provider: Any | None = None
        provider_initialized = False
        all_evaluations: list[ShadowOutcomeEvaluation] = []
        all_errors: list[str] = []
        statuses: dict[str, Mapping[EvaluationBasis, str]] = {}
        horizons_evaluated = 0
        ticks_processed = 0
        mt5_seconds = 0.0
        try:
            for decision in decisions:
                result, provider, provider_initialized = self._evaluate_one(
                    decision,
                    now=now,
                    terminal_path=terminal_path,
                    provider=provider,
                    provider_initialized=provider_initialized,
                )
                all_evaluations.extend(result.evaluations)
                all_errors.extend(result.errors)
                statuses[decision.decision_id] = result.status_by_basis
                horizons_evaluated += int(result.metrics["horizons_evaluated"])
                ticks_processed += int(result.metrics["historical_ticks_processed"])
                database_seconds += float(result.metrics["database_seconds"])
                mt5_seconds += float(result.metrics["mt5_read_seconds"])
        finally:
            shutdown = getattr(provider, "shutdown", None) if provider is not None else None
            if callable(shutdown):
                shutdown()
        status_by_basis = {
            basis: _aggregate_status(
                [row for row in all_evaluations if row.evaluation_basis == basis]
            )
            for basis in _BASIS_VALUES
        }
        return ShadowEvaluationBatchResult(
            decisions=decisions,
            evaluations=tuple(all_evaluations),
            status_by_basis=status_by_basis,
            status_by_decision=statuses,
            errors=tuple(all_errors),
            metrics={
                "decisions_scanned": len(decisions),
                "horizons_evaluated": horizons_evaluated,
                "historical_ticks_processed": ticks_processed,
                "database_seconds": database_seconds,
                "mt5_read_seconds": mt5_seconds,
                "total_runtime_seconds": time.perf_counter() - started,
                "llm_calls": 0,
            },
        )


def _pending_evaluation(
    decision: ShadowTradeDecision,
    basis: EvaluationBasis,
    horizon_seconds: int,
    config: EvaluationConfig,
    now: datetime,
) -> ShadowOutcomeEvaluation:
    entry = _entry_quote(decision, basis)
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
            basis,
            horizon_seconds,
            config,
            target_timestamp=target,
            entry=entry,
            status="PENDING",
            reason=None,
            now=now,
        )
    )


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
    "ShadowDecisionEvaluationResult",
    "ShadowEvaluationBatchResult",
    "ShadowOutcomeEvaluator",
    "ShadowEvaluationStore",
    "build_horizon_evaluation",
    "decision_source_eligibility",
    "evaluate_directional_outcomes",
    "evaluate_excursions",
    "evaluation_target",
    "first_valid_tick",
]
