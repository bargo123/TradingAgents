"""Versioned offline audit of book claims against the implemented HFT contracts."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Sequence
from dataclasses import asdict
from decimal import Decimal, InvalidOperation

from tradingagents.forex.hft.engines import (
    MOMENTUM_EXIT_PROFILE,
    RANGE_EXIT_PROFILE,
    HftExecutionEngine,
)
from tradingagents.forex.hft.risk import RiskConfig, RiskEngine
from tradingagents.forex.hft.strategies import MomentumContinuationStrategy, RangeRejectionStrategy
from tradingagents.self_enhancement.book_rule_grammar import parse_supported_rule_quote
from tradingagents.self_enhancement.phase14c_models import (
    CURRENT_STRATEGY_IDS,
    CurrentStrategyComponentContract,
    CurrentStrategyComponentMapping,
    CurrentStrategyContract,
    CurrentStrategyContractSnapshot,
    CurrentStrategyMapping,
    CurrentStrategyMappingStatus,
    CurrentStrategyRuleSignature,
)
from tradingagents.self_enhancement.strategy_specs import (
    RuleDirection,
    RuleOperator,
    RuleOrigin,
    RuleStage,
    RuleValidationStatus,
    StrategyRuleClaim,
    StrategySpec,
)

PHASE14C_MAPPING_SCHEMA_VERSION = "phase14c-current-strategy-mapping-v1"
PHASE14C_MAPPING_VERSION = "phase14c-code-contract-capture-v1"
_NO_DIRECT_MATCH = "NO_EXACT_COMPONENT_MATCH"
_EXACT_MATCH = "EXACT_CANONICAL_RULE_MATCH"
_PARTIAL_MATCH = "PARTIAL_CANONICAL_RULE_MATCH"
_FAMILY_TO_STRATEGY = {
    "MOMENTUM_CONTINUATION": "momentum_continuation",
    "RANGE_REJECTION": "range_rejection",
}


class StrategyContractDriftError(ValueError):
    """A supplied code snapshot is stale relative to current strategy defaults."""


def _number_text(value: object) -> str:
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("strategy contract number is invalid") from exc
    if not number.is_finite():
        raise ValueError("strategy contract number must be finite")
    text = format(number.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _implementation_fingerprint(function: object) -> str:
    try:
        source = inspect.getsource(function).replace("\r\n", "\n")
    except (OSError, TypeError) as exc:
        raise RuntimeError("current strategy source is unavailable for drift-safe mapping") from exc
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _risk_contract() -> dict[str, object]:
    return {
        **asdict(RiskConfig()),
        "owner": "global deterministic runtime risk engine",
        "strategy_specific_book_rule": False,
        "implementation_sha256": _implementation_fingerprint(RiskEngine.evaluate),
    }


def _exit_profile_contract(profile: object) -> dict[str, object]:
    values = asdict(profile)
    values["implementation_sha256"] = _implementation_fingerprint(HftExecutionEngine._exit_reason)
    return values


def _signature(
    stage: RuleStage,
    direction: RuleDirection,
    operator: RuleOperator,
    feature: str | None,
    value: object,
    unit: str,
    condition: str,
    horizon_seconds: int | None = None,
) -> CurrentStrategyRuleSignature:
    return CurrentStrategyRuleSignature(
        stage=stage,
        direction=direction,
        operator=operator,
        feature=feature,
        value=_number_text(value),
        unit=unit,
        condition=condition,
        horizon_seconds=horizon_seconds,
    )


def _horizon_signature(seconds: float) -> CurrentStrategyRuleSignature:
    if not float(seconds).is_integer():
        raise ValueError("profile max duration must be a whole number of seconds")
    duration = int(seconds)
    return _signature(
        RuleStage.HORIZON,
        RuleDirection.BOTH,
        RuleOperator.LESS_OR_EQUAL,
        None,
        duration,
        "seconds",
        "after entry",
        duration,
    )


def _momentum_contract() -> CurrentStrategyContract:
    strategy = MomentumContinuationStrategy()
    engine = HftExecutionEngine()
    profile = MOMENTUM_EXIT_PROFILE
    implementation_fingerprint = _implementation_fingerprint(strategy.evaluate)
    threshold = strategy.minimum_momentum_points
    persistence = strategy.minimum_persistence
    minimum_ticks = strategy.minimum_ticks
    entry_rules = (
        _signature(
            RuleStage.ENTRY,
            RuleDirection.LONG,
            RuleOperator.GREATER_THAN,
            "momentum",
            threshold,
            "points",
            "after confirmation",
        ),
        _signature(
            RuleStage.ENTRY,
            RuleDirection.SHORT,
            RuleOperator.LESS_THAN,
            "momentum",
            -threshold,
            "points",
            "after confirmation",
        ),
    )
    confirmation_rules = tuple(
        _signature(
            RuleStage.CONFIRMATION,
            direction,
            RuleOperator.GREATER_OR_EQUAL,
            "direction_persistence",
            persistence,
            "fraction",
            "after three ticks",
        )
        for direction in (RuleDirection.LONG, RuleDirection.SHORT)
    )
    components = (
        CurrentStrategyComponentContract(
            "entry",
            {
                "minimum_momentum_points": threshold,
                "direction_rule": "direction follows signed momentum",
                "long_operator": RuleOperator.GREATER_THAN.value,
                "short_operator": RuleOperator.LESS_THAN.value,
                "short_threshold_points": -threshold,
                "implementation_sha256": implementation_fingerprint,
            },
            entry_rules,
        ),
        CurrentStrategyComponentContract(
            "confirmation",
            {
                "minimum_persistence": persistence,
                "minimum_ticks": minimum_ticks,
                "implementation_sha256": implementation_fingerprint,
            },
            confirmation_rules,
        ),
        CurrentStrategyComponentContract(
            "expected_move",
            {
                "mode": "observed_absolute_momentum_points",
                "fixed_target": False,
                "emitted_by": "signal magnitude",
                "implementation_sha256": implementation_fingerprint,
            },
        ),
        CurrentStrategyComponentContract("exit", _exit_profile_contract(profile)),
        CurrentStrategyComponentContract("risk", _risk_contract()),
        CurrentStrategyComponentContract(
            "holding_horizon",
            {
                "profile_name": profile.name,
                "profile_max_duration_seconds": profile.max_duration_seconds,
                "runtime_hard_max_duration_seconds": engine.hard_max_duration_seconds,
                "max_duration_seconds": min(
                    engine.hard_max_duration_seconds,
                    profile.max_duration_seconds,
                ),
            },
            (_horizon_signature(min(engine.hard_max_duration_seconds, profile.max_duration_seconds)),),
        ),
    )
    return CurrentStrategyContract(strategy.strategy_id, components)


def _range_contract() -> CurrentStrategyContract:
    strategy = RangeRejectionStrategy()
    engine = HftExecutionEngine()
    profile = RANGE_EXIT_PROFILE
    implementation_fingerprint = _implementation_fingerprint(strategy.evaluate)
    components = (
        CurrentStrategyComponentContract(
            "entry",
            {
                "edge_fraction": strategy.edge_fraction,
                "minimum_range_points": strategy.minimum_range_points,
                "minimum_ticks": strategy.minimum_ticks,
                "long_requires": "range_position <= edge_fraction and return_1 > 0",
                "short_requires": "range_position >= 1 - edge_fraction and return_1 < 0",
                "implementation_sha256": implementation_fingerprint,
            },
        ),
        CurrentStrategyComponentContract(
            "confirmation",
            {
                "minimum_ticks": strategy.minimum_ticks,
                "separate_confirmation_stage": False,
                "implementation_sha256": implementation_fingerprint,
            },
        ),
        CurrentStrategyComponentContract(
            "expected_move",
            {
                "mode": "observed_rolling_range_points",
                "fixed_target": False,
                "emitted_by": "rolling range divided by point",
                "implementation_sha256": implementation_fingerprint,
            },
        ),
        CurrentStrategyComponentContract("exit", _exit_profile_contract(profile)),
        CurrentStrategyComponentContract("risk", _risk_contract()),
        CurrentStrategyComponentContract(
            "holding_horizon",
            {
                "profile_name": profile.name,
                "profile_max_duration_seconds": profile.max_duration_seconds,
                "runtime_hard_max_duration_seconds": engine.hard_max_duration_seconds,
                "max_duration_seconds": min(
                    engine.hard_max_duration_seconds,
                    profile.max_duration_seconds,
                ),
            },
            (_horizon_signature(min(engine.hard_max_duration_seconds, profile.max_duration_seconds)),),
        ),
    )
    return CurrentStrategyContract(strategy.strategy_id, components)


def capture_current_strategy_contracts() -> CurrentStrategyContractSnapshot:
    """Snapshot current offline HFT defaults and profiles without touching runtime state."""

    strategies = (_momentum_contract(), _range_contract())
    if tuple(sorted(item.strategy_id for item in strategies)) != CURRENT_STRATEGY_IDS:
        raise ValueError("current strategy IDs differ from the reviewed Phase 14C contract")
    return CurrentStrategyContractSnapshot(
        schema_version=PHASE14C_MAPPING_SCHEMA_VERSION,
        contract_version=PHASE14C_MAPPING_VERSION,
        strategies=strategies,
    )


def _claim_signature(claim: StrategyRuleClaim) -> CurrentStrategyRuleSignature | None:
    if claim.origin is not RuleOrigin.SOURCE_SUPPORTED_CONCEPT or claim.evidence is None or not claim.is_complete:
        return None
    parsed = parse_supported_rule_quote(claim.evidence.quote, claim.stage)
    if parsed is None:
        return None
    if (
        parsed["stage"] is not claim.stage
        or parsed["direction"] is not claim.direction
        or parsed["operator"] is not claim.operator
        or _number_text(parsed["value"]) != _number_text(claim.value)
        or parsed["unit"] != claim.unit
        or parsed["condition"] != claim.condition
        or parsed["horizon_seconds"] != claim.horizon_seconds
    ):
        return None
    return _signature(
        parsed["stage"],
        parsed["direction"],
        parsed["operator"],
        parsed["feature"],
        parsed["value"],
        parsed["unit"],
        parsed["condition"],
        parsed["horizon_seconds"],
    )


def _validated_source_signatures(spec: StrategySpec) -> dict[CurrentStrategyRuleSignature, set[str]]:
    validation = {item.rule_fingerprint: item.status for item in spec.validation_results}
    result: dict[CurrentStrategyRuleSignature, set[str]] = {}
    for claim in spec.source_supported_rules:
        if validation.get(claim.fingerprint) is not RuleValidationStatus.SUPPORTED:
            continue
        signature = _claim_signature(claim)
        if signature is None:
            continue
        if signature.feature is not None and signature.feature not in spec.required_data:
            continue
        result.setdefault(signature, set()).add(claim.fingerprint)
    return result


def map_current_strategies(
    specs: Sequence[StrategySpec],
    *,
    snapshot: CurrentStrategyContractSnapshot,
) -> tuple[CurrentStrategyMapping, ...]:
    """Compare exact validated source rules to current code; names never establish a match."""

    if not isinstance(snapshot, CurrentStrategyContractSnapshot):
        raise TypeError("snapshot must be CurrentStrategyContractSnapshot")
    values = tuple(specs)
    if any(not isinstance(spec, StrategySpec) for spec in values):
        raise TypeError("specs must contain StrategySpec values")
    current = capture_current_strategy_contracts()
    if current.fingerprint != snapshot.fingerprint:
        raise StrategyContractDriftError("current strategy implementation changed after snapshot capture")

    by_strategy: dict[str, list[dict[CurrentStrategyRuleSignature, set[str]]]] = {
        strategy_id: [] for strategy_id in CURRENT_STRATEGY_IDS
    }
    for spec in values:
        strategy_id = _FAMILY_TO_STRATEGY.get(spec.family)
        if strategy_id is not None:
            by_strategy[strategy_id].append(_validated_source_signatures(spec))

    mappings = []
    for strategy in snapshot.strategies:
        source_rules = by_strategy[strategy.strategy_id]
        component_results = []
        for component in strategy.components:
            expected = set(component.matchable_rules)
            fingerprints = tuple(
                sorted(
                    fingerprint
                    for source in source_rules
                    for signature, source_fingerprints in source.items()
                    if signature in expected
                    for fingerprint in source_fingerprints
                )
            )
            matched_signatures = {
                signature
                for source in source_rules
                for signature in source
                if signature in expected
            }
            if not matched_signatures:
                status = CurrentStrategyMappingStatus.NO_DIRECT_MATCH
                reason = _NO_DIRECT_MATCH
            elif matched_signatures == expected:
                status = CurrentStrategyMappingStatus.MATCHED
                reason = _EXACT_MATCH
            else:
                status = CurrentStrategyMappingStatus.PARTIALLY_MATCHED
                reason = _PARTIAL_MATCH
            component_results.append(
                CurrentStrategyComponentMapping(
                    component=component.component,
                    status=status,
                    matched_source_rule_fingerprints=fingerprints,
                    reason_code=reason,
                )
            )
        mappings.append(CurrentStrategyMapping(strategy.strategy_id, tuple(component_results)))
    return tuple(mappings)


__all__ = [
    "PHASE14C_MAPPING_SCHEMA_VERSION",
    "PHASE14C_MAPPING_VERSION",
    "StrategyContractDriftError",
    "capture_current_strategy_contracts",
    "map_current_strategies",
]
