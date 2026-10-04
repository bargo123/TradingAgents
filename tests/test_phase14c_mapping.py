from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from tradingagents.self_enhancement import phase14c_mapping
from tradingagents.self_enhancement.book_rule_grammar import parse_supported_rule_quote
from tradingagents.self_enhancement.phase14c_models import (
    CurrentStrategyMappingStatus as MappingStatus,
)
from tradingagents.self_enhancement.strategy_specs import (
    EvidenceSpan,
    RuleOrigin,
    RuleStage,
    RuleValidationResult,
    RuleValidationStatus,
    StrategyRuleClaim,
    StrategySpec,
    StrategySuitability,
)

GENERATION = "gen-phase14c-contract-fixture"
GENERATION_FINGERPRINT = "d" * 64


def _spec(
    *,
    family: str = "MOMENTUM_CONTINUATION",
    name: str = "Book momentum concept",
    entry_quote: str = "LONG when momentum > 2 points after confirmation",
    confirmation_quote: str = (
        "CONFIRMATION: LONG when direction_persistence >= 0.6 fraction after three ticks"
    ),
    horizon_seconds: int = 45,
) -> StrategySpec:
    quote_pairs = [
        (RuleStage.ENTRY, entry_quote),
        (RuleStage.CONFIRMATION, confirmation_quote),
        (RuleStage.HORIZON, f"HORIZON: hold no longer than {horizon_seconds} seconds after entry"),
    ]
    text = "\n".join(quote for _, quote in quote_pairs)
    source_hash = hashlib.sha256(b"mapping-fixture-source").hexdigest()
    claims = []
    for stage, quote in quote_pairs:
        start = text.index(quote)
        span = EvidenceSpan(
            GENERATION,
            "doc-mapping-fixture",
            "chunk-mapping-fixture",
            source_hash,
            start,
            start + len(quote),
            quote,
        )
        parsed = parse_supported_rule_quote(quote, stage)
        assert parsed is not None
        claims.append(
            StrategyRuleClaim(
                stage=stage,
                operator=parsed["operator"],
                direction=parsed["direction"],
                value=(
                    int(parsed["value"])
                    if parsed["value"] == parsed["value"].to_integral_value()
                    else float(parsed["value"])
                ),
                unit=parsed["unit"],
                condition=parsed["condition"],
                horizon_seconds=parsed["horizon_seconds"],
                origin=RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
                evidence=span,
            )
        )
    validations = tuple(
        RuleValidationResult(claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE")
        for claim in claims
    )
    return StrategySpec(
        spec_id="spec-mapping-fixture",
        name=name,
        family=family,
        required_data=("momentum", "direction_persistence"),
        rule_claims=tuple(claims),
        generation_id=GENERATION,
        knowledge_fingerprint=GENERATION_FINGERPRINT,
        model_provider="ollama-local",
        model_id="qwen3.5:2b",
        prompt_version="fixture-v1",
        schema_version="fixture-v1",
        created_at=datetime.now(UTC),
        implementation_confidence=0.0,
        suitability=StrategySuitability.HFT_SUITABLE,
        validation_results=validations,
    )


def test_snapshot_captures_versioned_current_defaults_and_exit_profiles() -> None:
    snapshot = phase14c_mapping.capture_current_strategy_contracts()
    repeated = phase14c_mapping.capture_current_strategy_contracts()

    assert snapshot.fingerprint == repeated.fingerprint
    assert len(snapshot.fingerprint) == 64
    assert snapshot.contract_version
    assert {strategy.strategy_id for strategy in snapshot.strategies} == {
        "momentum_continuation",
        "range_rejection",
    }
    for strategy in snapshot.strategies:
        assert tuple(component.component for component in strategy.components) == (
            "entry",
            "confirmation",
            "expected_move",
            "exit",
            "risk",
            "holding_horizon",
        )

    momentum = snapshot.strategy("momentum_continuation")
    assert momentum.component("entry").canonical_values["minimum_momentum_points"] == 2.0
    assert momentum.component("confirmation").canonical_values["minimum_persistence"] == 0.6
    assert momentum.component("confirmation").canonical_values["minimum_ticks"] == 3
    assert len(momentum.component("confirmation").canonical_values["implementation_sha256"]) == 64
    assert momentum.component("exit").canonical_values["name"] == "momentum_continuation"
    assert momentum.component("holding_horizon").canonical_values["max_duration_seconds"] == 45.0

    range_contract = snapshot.strategy("range_rejection")
    assert range_contract.component("entry").canonical_values["edge_fraction"] == 0.2
    assert range_contract.component("entry").canonical_values["minimum_range_points"] == 2.0
    assert range_contract.component("entry").canonical_values["minimum_ticks"] == 4
    assert range_contract.component("entry").canonical_values["long_requires"] == (
        "range_position <= edge_fraction and return_1 > 0"
    )
    assert range_contract.component("entry").canonical_values["short_requires"] == (
        "range_position >= 1 - edge_fraction and return_1 < 0"
    )
    assert range_contract.component("exit").canonical_values["name"] == "range_rejection"
    assert range_contract.component("holding_horizon").canonical_values["max_duration_seconds"] == 30.0
    assert range_contract.component("risk").canonical_values["max_risk_fraction"] == 0.01


def test_changed_runtime_default_invalidates_old_snapshot_before_mapping(monkeypatch) -> None:
    snapshot = phase14c_mapping.capture_current_strategy_contracts()

    class ChangedMomentumStrategy(phase14c_mapping.MomentumContinuationStrategy):
        def __init__(self):
            super().__init__(minimum_momentum_points=3.0)

    monkeypatch.setattr(phase14c_mapping, "MomentumContinuationStrategy", ChangedMomentumStrategy)

    with pytest.raises(phase14c_mapping.StrategyContractDriftError):
        phase14c_mapping.map_current_strategies((_spec(),), snapshot=snapshot)


def test_changed_strategy_logic_invalidates_old_snapshot_before_mapping(monkeypatch) -> None:
    snapshot = phase14c_mapping.capture_current_strategy_contracts()

    def changed_evaluate(_self, _features):
        return None

    monkeypatch.setattr(phase14c_mapping.MomentumContinuationStrategy, "evaluate", changed_evaluate)

    with pytest.raises(phase14c_mapping.StrategyContractDriftError):
        phase14c_mapping.map_current_strategies((_spec(),), snapshot=snapshot)


def test_component_mapping_cites_only_exact_source_rule_fingerprints() -> None:
    snapshot = phase14c_mapping.capture_current_strategy_contracts()
    spec = _spec()

    mappings = phase14c_mapping.map_current_strategies((spec,), snapshot=snapshot)

    momentum = next(item for item in mappings if item.strategy_id == "momentum_continuation")
    assert len(momentum.components) == 6
    components = {item.component: item for item in momentum.components}
    entry_claim = spec.rule_claims[0]
    assert components["entry"].status is MappingStatus.PARTIALLY_MATCHED
    assert components["entry"].matched_source_rule_fingerprints == (entry_claim.fingerprint,)
    assert components["entry"].reason_code == "PARTIAL_CANONICAL_RULE_MATCH"
    assert components["holding_horizon"].status is MappingStatus.MATCHED
    assert components["holding_horizon"].matched_source_rule_fingerprints == (
        spec.rule_claims[-1].fingerprint,
    )
    assert components["risk"].status is MappingStatus.NO_DIRECT_MATCH
    assert components["risk"].matched_source_rule_fingerprints == ()


def test_strategy_name_mention_without_matching_source_rules_yields_no_match() -> None:
    snapshot = phase14c_mapping.capture_current_strategy_contracts()
    spec = _spec(
        family="MOMENTUM_CONTINUATION",
        name="momentum_continuation range_rejection review",
        entry_quote="LONG when momentum > 99 points after confirmation",
        confirmation_quote=(
            "CONFIRMATION: LONG when direction_persistence >= 0.9 fraction after three ticks"
        ),
        horizon_seconds=90,
    )

    mappings = phase14c_mapping.map_current_strategies((spec,), snapshot=snapshot)

    momentum = next(item for item in mappings if item.strategy_id == "momentum_continuation")
    assert len(momentum.components) == 6
    assert all(item.status is MappingStatus.NO_DIRECT_MATCH for item in momentum.components)
    assert all(item.matched_source_rule_fingerprints == () for item in momentum.components)
