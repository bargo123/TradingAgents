from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.forex.hft.features import TickFeatures
from tradingagents.forex.hft.models import FastAction
from tradingagents.forex.hft.regime import DirectionPolicy, Regime, StrategicRegimeState
from tradingagents.forex.hft.strategies import MomentumContinuationStrategy, SignalArbiter
from tradingagents.self_enhancement.book_pipeline import classify_suitability
from tradingagents.self_enhancement.book_strategies import BookStrategyRegistry
from tradingagents.self_enhancement.strategy_specs import (
    EXECUTABLE_REQUIRED_STAGES,
    EvidenceSpan,
    RuleDirection,
    RuleOperator,
    RuleOrigin,
    RuleStage,
    RuleValidationResult,
    RuleValidationStatus,
    StrategyRuleClaim,
    StrategySpec,
    StrategySuitability,
)

_POINT = 0.00001


def _spec(*, direction: RuleDirection = RuleDirection.LONG) -> StrategySpec:
    entry_operator = RuleOperator.GREATER_THAN if direction is RuleDirection.LONG else RuleOperator.LESS_THAN
    reverse_operator = RuleOperator.LESS_OR_EQUAL if direction is RuleDirection.LONG else RuleOperator.GREATER_OR_EQUAL
    reverse_threshold = 0.0
    stop_threshold = -3.0 if direction is RuleDirection.LONG else 3.0
    stage_values = {
        RuleStage.ENTRY: (entry_operator, direction, 2.0 if direction is RuleDirection.LONG else -2.0, "points", "after confirmation", None),
        RuleStage.CONFIRMATION: (
            RuleOperator.GREATER_OR_EQUAL,
            direction,
            0.6,
            "fraction",
            "after three ticks",
            None,
        ),
        RuleStage.INVALIDATION: (reverse_operator, direction, reverse_threshold, "points", "after reversal", None),
        RuleStage.EXPECTED_MOVE: (
            RuleOperator.GREATER_OR_EQUAL,
            direction,
            5.0,
            "points",
            "after entry",
            30,
        ),
        RuleStage.EXIT: (reverse_operator, direction, reverse_threshold, "points", "after reversal", None),
        RuleStage.PROFIT_PROTECTION: (
            reverse_operator,
            direction,
            reverse_threshold,
            "points",
            "after target retracement",
            None,
        ),
        RuleStage.STOP_BEHAVIOR: (
            reverse_operator,
            direction,
            stop_threshold,
            "points",
            "after adverse move",
            None,
        ),
        RuleStage.HORIZON: (
            RuleOperator.LESS_OR_EQUAL,
            RuleDirection.BOTH,
            30,
            "seconds",
            "after entry",
            30,
        ),
    }
    claims = []
    for stage in EXECUTABLE_REQUIRED_STAGES:
        operator, rule_direction, value, unit, condition, horizon = stage_values[stage]
        if stage is RuleStage.ENTRY:
            quote = f"{rule_direction.value} when momentum {operator.value.replace('_', ' ').lower()} {value:g} {unit} {condition}"
            quote = quote.replace("greater than", ">").replace("greater or equal", ">=")
            quote = quote.replace("less than", "<").replace("less or equal", "<=")
        elif stage is RuleStage.EXPECTED_MOVE:
            quote = f"EXPECTED_MOVE: {rule_direction.value} target {value:g} points within {horizon} seconds {condition}"
        elif stage is RuleStage.HORIZON:
            quote = f"HORIZON: hold no longer than {horizon} seconds {condition}"
        else:
            feature = "direction_persistence" if stage is RuleStage.CONFIRMATION else "momentum"
            quote = (
                f"{stage.value}: {rule_direction.value} when {feature} "
                f"{operator.value.replace('_', ' ').lower()} {value:g} {unit} {condition}"
            )
            quote = quote.replace("greater than", ">").replace("greater or equal", ">=")
            quote = quote.replace("less than", "<").replace("less or equal", "<=")
        span = EvidenceSpan("gen-1", "doc-1", f"chunk-{stage.value}", "a" * 64, 0, len(quote), quote)
        claims.append(
            StrategyRuleClaim(
                stage,
                operator,
                rule_direction,
                value,
                unit,
                condition,
                horizon,
                RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
                span,
            )
        )
    validations = tuple(
        RuleValidationResult(claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE")
        for claim in claims
    )
    return StrategySpec(
        "spec-1",
        "Momentum threshold challenger",
        "MOMENTUM_CONTINUATION",
        ("momentum", "direction_persistence"),
        tuple(claims),
        "gen-1",
        "b" * 64,
        "ollama-local",
        "qwen3.5:2b",
        "phase14b-draft-v1",
        "strategy-spec-v1",
        datetime(2026, 10, 3, tzinfo=UTC),
        0.8,
        StrategySuitability.HFT_SUITABLE,
        validations,
    )


def _features(
    *, momentum_points: float = 3.0, persistence: float = 0.8, tick_count: int = 5
) -> TickFeatures:
    now = datetime(2026, 10, 3, 12, tzinfo=UTC)
    return TickFeatures(
        "EURUSD",
        now,
        1.1,
        1.10002,
        1.10001,
        0.00002,
        2.0,
        _POINT,
        0.00001,
        momentum_points * _POINT,
        0.000001,
        0.0,
        0.000001,
        0.0001,
        0.0,
        3.0,
        0.0,
        persistence,
        tick_count,
        "LONDON",
    )


def _regime(*, momentum_enabled: bool = True, direction: DirectionPolicy = DirectionPolicy.BOTH):
    now = datetime(2026, 10, 3, 12, tzinfo=UTC)
    return StrategicRegimeState(
        "state-1",
        "EURUSD",
        Regime.NEUTRAL,
        direction,
        0.25,
        0.0,
        now - timedelta(seconds=1),
        now + timedelta(seconds=5),
        momentum_enabled=momentum_enabled,
        range_enabled=False,
        breakout_enabled=False,
        mean_reversion_enabled=False,
    )


def test_registry_creates_only_complete_allowlisted_deterministic_strategy() -> None:
    spec = _spec()

    strategy = BookStrategyRegistry().create(spec)

    assert strategy.candidate_id.startswith("book-spec-")
    assert strategy.strategy_id == "momentum_continuation"
    assert strategy.expected_move_horizon_seconds == 30
    assert strategy.max_horizon_seconds == 30
    assert classify_suitability(_spec(), {"momentum", "direction_persistence"}).suitability is StrategySuitability.HFT_SUITABLE


def test_book_strategy_emits_existing_signal_shape_from_causal_features() -> None:
    signal = BookStrategyRegistry().create(_spec()).evaluate(_features())

    assert signal is not None
    assert signal.strategy_id == "momentum_continuation"
    assert signal.action is FastAction.ENTER_LONG
    assert signal.expected_move_points == 5.0
    assert signal.reason.startswith("BOOK_CANDIDATE:")


def test_short_entry_uses_the_validated_short_direction_and_threshold() -> None:
    signal = BookStrategyRegistry().create(_spec(direction=RuleDirection.SHORT)).evaluate(
        _features(momentum_points=-3.0)
    )

    assert signal is not None
    assert signal.action is FastAction.ENTER_SHORT


def test_entry_and_confirmation_thresholds_are_both_required() -> None:
    strategy = BookStrategyRegistry().create(_spec())

    assert strategy.evaluate(_features(momentum_points=1.9)) is None
    assert strategy.evaluate(_features(momentum_points=3.0, persistence=0.59)) is None
    assert strategy.evaluate(_features(momentum_points=3.0, persistence=0.6)) is not None


def test_strategy_obeys_existing_regime_direction_and_cost_gates() -> None:
    signal = BookStrategyRegistry().create(_spec()).evaluate(_features())
    assert signal is not None
    arbiter = SignalArbiter(cost_safety_margin_points=1.0)

    selected, reason = arbiter.select((signal,), _regime(momentum_enabled=False), spread_points=2.0)
    assert selected is None
    assert reason == "NO_VALID_SIGNAL"

    selected, reason = arbiter.select(
        (signal,), _regime(direction=DirectionPolicy.SHORT_ONLY), spread_points=2.0
    )
    assert selected is None
    assert reason == "DIRECTION_POLICY"

    selected, reason = arbiter.select((signal,), _regime(), spread_points=5.0)
    assert selected is None
    assert reason == "TRANSACTION_COST"


def test_candidate_exit_and_horizon_rules_are_exposed_without_order_authority() -> None:
    strategy = BookStrategyRegistry().create(_spec())

    assert strategy.should_exit(_features(momentum_points=-0.1), elapsed_seconds=1.0) is True
    assert strategy.should_exit(_features(momentum_points=3.0), elapsed_seconds=30.0) is True
    assert strategy.should_exit(_features(momentum_points=3.0), elapsed_seconds=1.0) is False
    with pytest.raises(TypeError, match="booleans"):
        strategy.should_exit(_features(), elapsed_seconds=1.0, target_reached=1)
    assert not hasattr(strategy, "order_send")
    assert not hasattr(strategy, "gateway")


def test_incomplete_unsupported_or_unavailable_specs_cannot_be_instantiated() -> None:
    spec = _spec()
    missing_exit = replace(
        spec,
        rule_claims=tuple(claim for claim in spec.rule_claims if claim.stage is not RuleStage.EXIT),
        validation_results=tuple(
            result
            for result in spec.validation_results
            if result.rule_fingerprint
            in {
                claim.fingerprint
                for claim in spec.rule_claims
                if claim.stage is not RuleStage.EXIT
            }
        ),
    )
    with pytest.raises(ValueError, match="complete validated HFT"):
        BookStrategyRegistry().create(missing_exit)

    unsupported = replace(
        spec,
        suitability=StrategySuitability.UNIMPLEMENTABLE_AUTONOMOUSLY,
    )
    with pytest.raises(ValueError, match="complete validated HFT"):
        BookStrategyRegistry().create(unsupported)


def test_registry_rejects_unreviewed_family_unit_or_operator() -> None:
    with pytest.raises(ValueError, match="strategy family"):
        BookStrategyRegistry().create(replace(_spec(), family="UNREVIEWED"))

    base = _spec()
    entry_claim = next(claim for claim in base.rule_claims if claim.stage is RuleStage.ENTRY)
    wrong_entry = replace(entry_claim, unit="pips")
    spec_with_wrong_unit = replace(
        base,
        rule_claims=tuple(wrong_entry if claim.stage is RuleStage.ENTRY else claim for claim in base.rule_claims),
        validation_results=tuple(
            RuleValidationResult(
                (wrong_entry if claim.stage is RuleStage.ENTRY else claim).fingerprint,
                RuleValidationStatus.SUPPORTED,
                "EXACT_SOURCE_RULE",
            )
            for claim in base.rule_claims
        ),
    )
    with pytest.raises(ValueError, match="source quote"):
        BookStrategyRegistry().create(spec_with_wrong_unit)


def test_registry_rejects_claim_operator_that_differs_from_source_quote() -> None:
    base = _spec()
    entry_claim = next(claim for claim in base.rule_claims if claim.stage is RuleStage.ENTRY)
    wrong_entry = replace(entry_claim, operator=RuleOperator.LESS_THAN)
    claims = tuple(wrong_entry if claim.stage is RuleStage.ENTRY else claim for claim in base.rule_claims)
    validations = tuple(
        RuleValidationResult(claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE")
        for claim in claims
    )
    inconsistent = replace(base, rule_claims=claims, validation_results=validations)

    with pytest.raises(ValueError, match="source quote"):
        BookStrategyRegistry().create(inconsistent)


def test_registry_rejects_conflicting_independent_maximum_horizons() -> None:
    base = _spec()
    source_horizon = next(claim for claim in base.rule_claims if claim.stage is RuleStage.HORIZON)
    quote = "HORIZON: hold no longer than 45 seconds after entry"
    span = EvidenceSpan("gen-1", "doc-2", "chunk-horizon-2", "d" * 64, 0, len(quote), quote)
    conflict = replace(source_horizon, value=45, horizon_seconds=45, evidence=span)
    claims = (*base.rule_claims, conflict)
    validations = (
        *base.validation_results,
        RuleValidationResult(conflict.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE"),
    )
    conflicting = replace(base, rule_claims=claims, validation_results=validations)

    assert conflicting.is_executable is True
    with pytest.raises(ValueError, match="conflicting source horizons"):
        BookStrategyRegistry().create(conflicting)


def test_registry_rejects_conflicting_independent_expected_move_rules() -> None:
    base = _spec()
    source_move = next(claim for claim in base.rule_claims if claim.stage is RuleStage.EXPECTED_MOVE)
    quote = "EXPECTED_MOVE: LONG target 7 points within 30 seconds after entry"
    span = EvidenceSpan("gen-1", "doc-2", "chunk-move-2", "d" * 64, 0, len(quote), quote)
    conflict = replace(source_move, value=7, evidence=span)
    claims = (*base.rule_claims, conflict)
    validations = (
        *base.validation_results,
        RuleValidationResult(conflict.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE"),
    )
    conflicting = replace(base, rule_claims=claims, validation_results=validations)

    assert conflicting.is_executable is True
    with pytest.raises(ValueError, match="conflicting source expected-move"):
        BookStrategyRegistry().create(conflicting)


def test_expected_move_window_must_fit_inside_source_maximum_horizon() -> None:
    base = _spec()
    source_horizon = next(claim for claim in base.rule_claims if claim.stage is RuleStage.HORIZON)
    quote = "HORIZON: hold no longer than 20 seconds after entry"
    span = EvidenceSpan("gen-1", "doc-1", "chunk-horizon-short", "a" * 64, 0, len(quote), quote)
    short_horizon = replace(source_horizon, value=20, horizon_seconds=20, evidence=span)
    claims = tuple(short_horizon if claim.stage is RuleStage.HORIZON else claim for claim in base.rule_claims)
    validations = tuple(
        RuleValidationResult(claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE")
        for claim in claims
    )
    inconsistent = replace(base, rule_claims=claims, validation_results=validations)

    with pytest.raises(ValueError, match="exceeds the source maximum"):
        BookStrategyRegistry().create(inconsistent)


def test_candidate_instances_have_no_cross_segment_state_and_incumbent_is_unchanged() -> None:
    features = _features()
    incumbent_before = MomentumContinuationStrategy(minimum_momentum_points=2.0).evaluate(features)
    candidate_a = BookStrategyRegistry().create(_spec())
    candidate_b = BookStrategyRegistry().create(_spec())

    assert candidate_a.evaluate(features) == candidate_b.evaluate(features)
    assert MomentumContinuationStrategy(minimum_momentum_points=2.0).evaluate(features) == incumbent_before
