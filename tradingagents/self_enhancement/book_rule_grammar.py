"""Small reviewed grammar shared by offline validation and deterministic primitives."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from .strategy_specs import RuleDirection, RuleOperator, RuleStage

_NUMERIC_RULE_RE = re.compile(
    r"^(?:(?P<stage>ENTRY|CONFIRMATION|INVALIDATION|EXIT|PROFIT_PROTECTION|STOP_BEHAVIOR): )?"
    r"(?P<direction>LONG|SHORT) when "
    r"(?P<feature>[a-z][a-z0-9_]*) "
    r"(?P<operator>>=|>|<=|<) "
    r"(?P<value>-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?) "
    r"(?P<unit>points|pips|ticks|bps|fraction|hz|ratio) "
    r"(?P<condition>after [a-z][a-z0-9_-]*(?: [a-z][a-z0-9_-]*)*)\.?$"
)
_EXPECTED_MOVE_RE = re.compile(
    r"^EXPECTED_MOVE: (?P<direction>LONG|SHORT) target "
    r"(?P<value>-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?) points "
    r"within (?P<horizon>[1-9][0-9]*) seconds "
    r"(?P<condition>after [a-z][a-z0-9_-]*(?: [a-z][a-z0-9_-]*)*)$"
)
_HORIZON_RE = re.compile(
    r"^HORIZON: hold no longer than (?P<horizon>[1-9][0-9]*) seconds "
    r"(?P<condition>after [a-z][a-z0-9_-]*(?: [a-z][a-z0-9_-]*)*)$"
)
_OPERATOR_MAP = {
    ">": RuleOperator.GREATER_THAN,
    ">=": RuleOperator.GREATER_OR_EQUAL,
    "<": RuleOperator.LESS_THAN,
    "<=": RuleOperator.LESS_OR_EQUAL,
}
FEATURE_UNITS = {
    "momentum": "points",
    "rolling_range": "points",
    "spread": "points",
    "spread_points": "points",
    "spread_expansion": "points",
    "return_1": "fraction",
    "m1_return": "fraction",
    "m5_return": "fraction",
    "volatility": "fraction",
    "direction_persistence": "fraction",
    "range_position": "fraction",
    "tick_count": "ticks",
    "tick_frequency": "hz",
    "burst": "ratio",
}
_NUMERIC_RULE_STAGES = {
    RuleStage.ENTRY,
    RuleStage.CONFIRMATION,
    RuleStage.INVALIDATION,
    RuleStage.EXIT,
    RuleStage.PROFIT_PROTECTION,
    RuleStage.STOP_BEHAVIOR,
}


def parse_supported_rule_quote(quote: str, stage: RuleStage) -> Mapping[str, Any] | None:
    """Parse only exact, stage-bound, bounded V1 rule sentences."""

    if not isinstance(quote, str) or not isinstance(stage, RuleStage):
        return None
    if stage in _NUMERIC_RULE_STAGES:
        match = _NUMERIC_RULE_RE.fullmatch(quote)
        if match is None:
            return None
        groups = match.groupdict()
        explicit_stage = groups["stage"]
        parsed_stage = RuleStage(explicit_stage) if explicit_stage else RuleStage.ENTRY
        if parsed_stage is not stage:
            return None
        try:
            value = Decimal(groups["value"])
        except InvalidOperation:
            return None
        if not value.is_finite():
            return None
        return {
            "stage": parsed_stage,
            "direction": RuleDirection(groups["direction"]),
            "feature": groups["feature"],
            "operator": _OPERATOR_MAP[groups["operator"]],
            "value": value,
            "unit": groups["unit"],
            "condition": groups["condition"],
            "horizon_seconds": None,
        }
    if stage is RuleStage.EXPECTED_MOVE:
        match = _EXPECTED_MOVE_RE.fullmatch(quote)
        if match is None:
            return None
        groups = match.groupdict()
        try:
            value = Decimal(groups["value"])
            horizon = int(groups["horizon"])
        except (InvalidOperation, ValueError):
            return None
        if not value.is_finite():
            return None
        return {
            "stage": stage,
            "direction": RuleDirection(groups["direction"]),
            "feature": None,
            "operator": RuleOperator.GREATER_OR_EQUAL,
            "value": value,
            "unit": "points",
            "condition": groups["condition"],
            "horizon_seconds": horizon,
        }
    if stage is RuleStage.HORIZON:
        match = _HORIZON_RE.fullmatch(quote)
        if match is None:
            return None
        groups = match.groupdict()
        horizon = int(groups["horizon"])
        return {
            "stage": stage,
            "direction": RuleDirection.BOTH,
            "feature": None,
            "operator": RuleOperator.LESS_OR_EQUAL,
            "value": Decimal(horizon),
            "unit": "seconds",
            "condition": groups["condition"],
            "horizon_seconds": horizon,
        }
    return None


__all__ = ["FEATURE_UNITS", "parse_supported_rule_quote"]
