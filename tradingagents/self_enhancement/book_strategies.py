"""Finite deterministic HFT primitives built only from validated StrategySpecs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import Any

from tradingagents.forex.hft.features import TickFeatures
from tradingagents.forex.hft.models import FastAction
from tradingagents.forex.hft.strategies import StrategySignal
from tradingagents.self_enhancement.book_rule_grammar import (
    FEATURE_UNITS,
    parse_supported_rule_quote,
)
from tradingagents.self_enhancement.strategy_specs import (
    RuleDirection,
    RuleOperator,
    RuleStage,
    StrategySpec,
    StrategySuitability,
)

_FAMILY_PERMISSION = {
    "MOMENTUM_CONTINUATION": "momentum_continuation",
    "RANGE_REJECTION": "range_rejection",
}
_CONDITIONS = {
    RuleStage.ENTRY: "after confirmation",
    RuleStage.CONFIRMATION: "after three ticks",
    RuleStage.INVALIDATION: "after reversal",
    RuleStage.EXPECTED_MOVE: "after entry",
    RuleStage.EXIT: "after reversal",
    RuleStage.PROFIT_PROTECTION: "after target retracement",
    RuleStage.STOP_BEHAVIOR: "after adverse move",
    RuleStage.HORIZON: "after entry",
}
_PRICE_FEATURES = {"momentum", "rolling_range", "spread", "spread_expansion"}


@dataclass(frozen=True, slots=True)
class _Threshold:
    stage: RuleStage
    feature: str
    operator: RuleOperator
    direction: RuleDirection
    value: float
    unit: str


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _feature_value(features: TickFeatures, rule: _Threshold) -> float | None:
    raw = _number(getattr(features, rule.feature, None))
    if raw is None:
        return None
    if rule.unit == "points" and rule.feature in _PRICE_FEATURES:
        point = _number(features.point)
        if point is None or point <= 0:
            return None
        return raw / point
    if rule.unit != FEATURE_UNITS.get(rule.feature):
        return None
    return raw


def _matches(features: TickFeatures, rule: _Threshold) -> bool:
    value = _feature_value(features, rule)
    if value is None:
        return False
    return {
        RuleOperator.GREATER_THAN: value > rule.value,
        RuleOperator.GREATER_OR_EQUAL: value >= rule.value,
        RuleOperator.LESS_THAN: value < rule.value,
        RuleOperator.LESS_OR_EQUAL: value <= rule.value,
    }.get(rule.operator, False)


@dataclass(frozen=True, slots=True)
class DeterministicBookStrategy:
    """A stateless causal signal primitive; it has no broker/order authority."""

    candidate_id: str
    strategy_id: str
    spec_id: str
    entry: _Threshold
    confirmation: _Threshold
    invalidation: tuple[_Threshold, ...]
    exits: tuple[_Threshold, ...]
    profit_protection: tuple[_Threshold, ...]
    stop_behavior: tuple[_Threshold, ...]
    expected_move_points: float
    expected_move_horizon_seconds: int
    max_horizon_seconds: int

    def reset(self) -> None:
        """Stateless primitive: reset is an explicit segment-boundary no-op."""

    def evaluate(self, features: TickFeatures) -> StrategySignal | None:
        if not isinstance(features, TickFeatures):
            raise TypeError("features must be TickFeatures")
        if features.tick_count < 3 or not _matches(features, self.confirmation):
            return None
        if not _matches(features, self.entry):
            return None
        action = FastAction.ENTER_LONG if self.entry.direction is RuleDirection.LONG else FastAction.ENTER_SHORT
        return StrategySignal(
            self.strategy_id,
            action,
            1.0,
            self.expected_move_points,
            f"BOOK_CANDIDATE:{self.candidate_id}",
        )

    def should_exit(
        self,
        features: TickFeatures,
        *,
        elapsed_seconds: float,
        target_reached: bool = False,
        adverse_move: bool = False,
    ) -> bool:
        if not isinstance(features, TickFeatures):
            raise TypeError("features must be TickFeatures")
        if type(target_reached) is not bool or type(adverse_move) is not bool:
            raise TypeError("target_reached and adverse_move must be booleans")
        elapsed = _number(elapsed_seconds)
        if elapsed is None or elapsed < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if elapsed >= self.max_horizon_seconds:
            return True
        if any(_matches(features, rule) for rule in (*self.invalidation, *self.exits)):
            return True
        if target_reached and any(_matches(features, rule) for rule in self.profit_protection):
            return True
        return adverse_move and any(_matches(features, rule) for rule in self.stop_behavior)


class BookStrategyRegistry:
    """Create reviewed threshold primitives; unknown concepts fail closed."""

    def create(self, spec: StrategySpec) -> DeterministicBookStrategy:
        if not isinstance(spec, StrategySpec) or not spec.is_executable:
            raise ValueError("only a complete validated HFT StrategySpec can be instantiated")
        if spec.suitability is not StrategySuitability.HFT_SUITABLE:
            raise ValueError("book candidate is not HFT suitable")
        strategy_id = _FAMILY_PERMISSION.get(spec.family)
        if strategy_id is None:
            raise ValueError("strategy family has no reviewed HFT permission mapping")

        stage_rules: dict[RuleStage, list[_Threshold]] = {}
        move_points: float | None = None
        move_direction: RuleDirection | None = None
        move_horizon: int | None = None
        max_horizon: int | None = None
        for claim in spec.source_supported_rules:
            quote = claim.evidence.quote if claim.evidence is not None else ""
            parsed = parse_supported_rule_quote(quote, claim.stage)
            if parsed is None:
                raise ValueError(f"unsupported deterministic rule stage: {claim.stage.value}")
            value = _number(claim.value)
            parsed_value = _number(parsed["value"])
            if (
                parsed["stage"] is not claim.stage
                or parsed["direction"] is not claim.direction
                or parsed["operator"] is not claim.operator
                or value is None
                or value != parsed_value
                or parsed["unit"] != claim.unit
                or parsed["condition"] != claim.condition
                or parsed["horizon_seconds"] != claim.horizon_seconds
                or claim.condition != _CONDITIONS[claim.stage]
            ):
                raise ValueError(f"source quote and structured rule disagree: {claim.stage.value}")
            feature = parsed["feature"]
            if claim.stage is RuleStage.EXPECTED_MOVE:
                if claim.unit != "points" or value <= 0:
                    raise ValueError("expected move must be positive points")
                if move_points is not None and (
                    move_points != value
                    or move_direction is not claim.direction
                    or move_horizon != claim.horizon_seconds
                ):
                    raise ValueError("conflicting source expected-move rules")
                move_points = value
                move_direction = claim.direction
                assert claim.horizon_seconds is not None
                move_horizon = claim.horizon_seconds
                continue
            if claim.stage is RuleStage.HORIZON:
                if claim.unit != "seconds" or value <= 0 or claim.horizon_seconds is None:
                    raise ValueError("horizon rule must provide positive bounded seconds")
                if max_horizon is not None and max_horizon != claim.horizon_seconds:
                    raise ValueError("conflicting source horizons")
                max_horizon = claim.horizon_seconds
                continue
            if not isinstance(feature, str) or feature not in FEATURE_UNITS:
                raise ValueError(f"feature is not implemented: {feature}")
            if FEATURE_UNITS[feature] != claim.unit:
                raise ValueError(f"feature unit is not implemented: {feature}")
            if feature not in spec.required_data:
                raise ValueError(f"feature is not declared as required data: {feature}")
            if claim.operator not in {
                RuleOperator.GREATER_THAN,
                RuleOperator.GREATER_OR_EQUAL,
                RuleOperator.LESS_THAN,
                RuleOperator.LESS_OR_EQUAL,
            }:
                raise ValueError(f"operator is not implemented: {claim.operator.value}")
            stage_rules.setdefault(claim.stage, []).append(
                _Threshold(claim.stage, feature, claim.operator, claim.direction, value, claim.unit)
            )

        for stage, rules in stage_rules.items():
            if len({_threshold_identity(rule) for rule in rules}) > 1:
                raise ValueError(f"conflicting independently sourced rules for stage: {stage.value}")
        if move_points is None or move_horizon is None or max_horizon is None or max_horizon > 60:
            raise ValueError("HFT candidate requires a positive expected move and horizon <= 60 seconds")
        if move_horizon > max_horizon:
            raise ValueError("expected-move horizon exceeds the source maximum holding horizon")

        entry = _one(stage_rules, RuleStage.ENTRY)
        confirmation = _one(stage_rules, RuleStage.CONFIRMATION)
        if entry.direction not in {RuleDirection.LONG, RuleDirection.SHORT}:
            raise ValueError("entry direction must be LONG or SHORT")
        if confirmation.direction is not entry.direction:
            raise ValueError("confirmation direction must match entry direction")
        if move_direction is not entry.direction:
            raise ValueError("expected-move direction must match entry direction")
        for stage in (RuleStage.INVALIDATION, RuleStage.EXIT, RuleStage.PROFIT_PROTECTION, RuleStage.STOP_BEHAVIOR):
            rules = stage_rules.get(stage, [])
            if any(rule.direction is not entry.direction for rule in rules):
                raise ValueError(f"{stage.value} direction must match entry direction")

        return DeterministicBookStrategy(
            candidate_id=f"book-spec-{spec.content_hash[:24]}",
            strategy_id=strategy_id,
            spec_id=spec.spec_id,
            entry=entry,
            confirmation=confirmation,
            invalidation=tuple(stage_rules.get(RuleStage.INVALIDATION, ())),
            exits=tuple(stage_rules.get(RuleStage.EXIT, ())),
            profit_protection=tuple(stage_rules.get(RuleStage.PROFIT_PROTECTION, ())),
            stop_behavior=tuple(stage_rules.get(RuleStage.STOP_BEHAVIOR, ())),
            expected_move_points=move_points,
            expected_move_horizon_seconds=move_horizon,
            max_horizon_seconds=max_horizon,
        )


def _threshold_identity(rule: _Threshold) -> tuple[object, ...]:
    return rule.feature, rule.operator, rule.direction, rule.value, rule.unit


def _one(rules: dict[RuleStage, list[_Threshold]], stage: RuleStage) -> _Threshold:
    values = rules.get(stage, [])
    if not values:
        raise ValueError(f"required executable stage is absent: {stage.value}")
    return values[0]


__all__ = ["BookStrategyRegistry", "DeterministicBookStrategy"]
