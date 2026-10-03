from __future__ import annotations

import inspect
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.forex.hft.features import TickFeatures
from tradingagents.forex.hft.regime import (
    DirectionPolicy,
    Regime,
    StrategicRegimeState,
)
from tradingagents.forex.hft.risk import RiskContext
from tradingagents.self_enhancement.book_factory import BookStrategyFactory
from tradingagents.self_enhancement.models import StrategyVersion
from tradingagents.self_enhancement.store import SelfEnhancementStore
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

_RULES = {
    RuleStage.ENTRY: (
        RuleOperator.GREATER_THAN,
        RuleDirection.LONG,
        2,
        "points",
        "after confirmation",
        None,
        "LONG when momentum > 2 points after confirmation",
    ),
    RuleStage.CONFIRMATION: (
        RuleOperator.GREATER_OR_EQUAL,
        RuleDirection.LONG,
        0.6,
        "fraction",
        "after three ticks",
        None,
        "CONFIRMATION: LONG when direction_persistence >= 0.6 fraction after three ticks",
    ),
    RuleStage.INVALIDATION: (
        RuleOperator.LESS_OR_EQUAL,
        RuleDirection.LONG,
        0,
        "points",
        "after reversal",
        None,
        "INVALIDATION: LONG when momentum <= 0 points after reversal",
    ),
    RuleStage.EXPECTED_MOVE: (
        RuleOperator.GREATER_OR_EQUAL,
        RuleDirection.LONG,
        30,
        "points",
        "after entry",
        30,
        "EXPECTED_MOVE: LONG target 30 points within 30 seconds after entry",
    ),
    RuleStage.EXIT: (
        RuleOperator.LESS_OR_EQUAL,
        RuleDirection.LONG,
        0,
        "points",
        "after reversal",
        None,
        "EXIT: LONG when momentum <= 0 points after reversal",
    ),
    RuleStage.PROFIT_PROTECTION: (
        RuleOperator.LESS_OR_EQUAL,
        RuleDirection.LONG,
        0,
        "points",
        "after target retracement",
        None,
        "PROFIT_PROTECTION: LONG when momentum <= 0 points after target retracement",
    ),
    RuleStage.STOP_BEHAVIOR: (
        RuleOperator.LESS_OR_EQUAL,
        RuleDirection.LONG,
        -3,
        "points",
        "after adverse move",
        None,
        "STOP_BEHAVIOR: LONG when momentum <= -3 points after adverse move",
    ),
    RuleStage.HORIZON: (
        RuleOperator.LESS_OR_EQUAL,
        RuleDirection.BOTH,
        30,
        "seconds",
        "after entry",
        30,
        "HORIZON: hold no longer than 30 seconds after entry",
    ),
}


def _registered_challenger(
    tmp_path, *, promote: bool = True, rolled_back: bool = False
) -> tuple[str, str]:
    claims = []
    for stage in EXECUTABLE_REQUIRED_STAGES:
        operator, direction, value, unit, condition, horizon, quote = _RULES[stage]
        claims.append(
            StrategyRuleClaim(
                stage,
                operator,
                direction,
                value,
                unit,
                condition,
                horizon,
                RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
                EvidenceSpan(
                    "gen-pinned",
                    "doc-source",
                    f"chunk-{stage.value}",
                    "a" * 64,
                    0,
                    len(quote),
                    quote,
                ),
            )
        )
    spec = StrategySpec(
        "spec-challenger",
        "Source grounded momentum",
        "MOMENTUM_CONTINUATION",
        ("momentum", "direction_persistence"),
        tuple(claims),
        "gen-pinned",
        "b" * 64,
        "ollama-local",
        "qwen3.5:2b",
        "strategy-spec-draft-v1",
        "strategy-spec-v1",
        datetime(2026, 10, 3, tzinfo=UTC),
        0.8,
        StrategySuitability.HFT_SUITABLE,
        tuple(
            RuleValidationResult(
                claim.fingerprint,
                RuleValidationStatus.SUPPORTED,
                "EXACT_SOURCE_RULE",
            )
            for claim in claims
        ),
    )
    parent = StrategyVersion(
        "range_rejection",
        "incumbent-v1",
        "cfg-v1",
        {"config": "fixture"},
        "fixture-commit",
    )
    candidate = BookStrategyFactory().from_validated_spec(spec, parent=parent)
    db_path = tmp_path / "phase14.sqlite3"
    store = SelfEnhancementStore(db_path)
    store.initialize()
    store.create_experiment(
        "experiment-1",
        parent_version="incumbent-v1",
        dataset_fingerprint="c" * 64,
    )
    store.record_candidate("experiment-1", candidate)
    if promote:
        deployment_id = store.record_promotion(
            "experiment-1",
            candidate.candidate_id,
            decision="SHADOW_CHALLENGER",
            reason="all unchanged promotion gates passed",
            rollback_package={"previous_version": "incumbent-v1"},
        )
        if rolled_back:
            assert deployment_id is not None
            store.rollback(deployment_id, "fixture rollback")
    return str(db_path), candidate.candidate_id


def _features(at: datetime, mid: float, *, momentum: float = 0.00006) -> TickFeatures:
    point = 0.00001
    return TickFeatures(
        symbol="EURUSD",
        timestamp=at,
        bid=mid - 0.00005,
        ask=mid + 0.00005,
        mid=mid,
        spread=0.0001,
        spread_points=10,
        point=point,
        return_1=0.00001,
        momentum=momentum,
        velocity=0.00001,
        acceleration=0.0,
        volatility=0.00001,
        rolling_range=0.0002,
        spread_expansion=0.0,
        tick_frequency=1.0,
        burst=0.0,
        direction_persistence=1.0,
        tick_count=3,
        session="LONDON",
    )


def _regime(at: datetime, *, expires_at: datetime | None = None, validity: str = "VALID") -> StrategicRegimeState:
    return StrategicRegimeState(
        state_id="state-1",
        symbol="EURUSD",
        regime=Regime.BULLISH,
        direction_policy=DirectionPolicy.LONG_ONLY,
        risk_multiplier=0.25,
        confidence=0.0,
        created_at=at - timedelta(seconds=3),
        expires_at=expires_at or at + timedelta(minutes=2),
        momentum_enabled=True,
        validity=validity,
    )


def _risk_context(at: datetime) -> RiskContext:
    return RiskContext(
        equity=1000.0,
        open_exposure=0.0,
        daily_loss=0.0,
        drawdown=0.0,
        consecutive_losses=0,
        tick_timestamp=at,
        observed_at=at + timedelta(milliseconds=100),
        session="LONDON",
    )


def test_shadow_challenger_captures_causal_shadow_fill_exit_cost_and_excursions(tmp_path):
    from tradingagents.self_enhancement.challenger import ShadowChallengerObserver

    catalog, candidate_id = _registered_challenger(tmp_path)
    observer = ShadowChallengerObserver(
        catalog_path=catalog,
        observer_path=tmp_path / "observer.sqlite3",
        slippage_points=0.5,
    )
    start = datetime(2026, 10, 3, 12, tzinfo=UTC)
    regime = _regime(start)

    entry = observer.on_tick(_features(start, 1.10006), regime, _risk_context(start))
    excursion = observer.on_tick(
        _features(start + timedelta(seconds=5), 1.09995),
        regime,
        _risk_context(start + timedelta(seconds=5)),
    )
    exit_events = observer.on_tick(
        _features(start + timedelta(seconds=31), 1.10030),
        regime,
        _risk_context(start + timedelta(seconds=31)),
    )

    assert len(entry) == 1
    assert entry[0].candidate_id == candidate_id
    assert entry[0].event == "ENTRY"
    assert entry[0].entry_price > _features(start, 1.10006).ask
    assert entry[0].executed is False
    assert entry[0].order_intent_emitted is False
    assert excursion == ()
    assert len(exit_events) == 1
    closed = exit_events[0]
    assert closed.event == "EXIT"
    assert closed.reason_code == "MAX_HORIZON"
    assert closed.entry_price == entry[0].entry_price
    assert closed.exit_price is not None
    assert closed.cost_points == pytest.approx(21.0)
    assert closed.mfe_points > 0
    assert closed.mae_points < 0
    assert closed.holding_seconds == pytest.approx(31.0)
    assert closed.commission_status == "UNKNOWN"
    assert closed.gross_pnl is not None
    assert closed.executed is False
    assert closed.order_intent_emitted is False

    with sqlite3.connect(tmp_path / "observer.sqlite3") as db:
        rows = db.execute(
            "SELECT event, executed, order_intent_emitted FROM challenger_observations ORDER BY observation_id"
        ).fetchall()
    assert rows == [("ENTRY", 0, 0), ("EXIT", 0, 0)]


@pytest.mark.parametrize(
    ("validity", "expires_delta"),
    (("INVALID", 120), ("VALID", -1)),
)
def test_invalid_or_stale_regime_suppresses_challenger_observation(
    tmp_path, validity: str, expires_delta: int
):
    from tradingagents.self_enhancement.challenger import ShadowChallengerObserver

    catalog, _ = _registered_challenger(tmp_path)
    observer = ShadowChallengerObserver(
        catalog_path=catalog,
        observer_path=tmp_path / "observer.sqlite3",
    )
    at = datetime(2026, 10, 3, 12, tzinfo=UTC)
    state = _regime(at, expires_at=at + timedelta(seconds=expires_delta), validity=validity)

    assert observer.on_tick(_features(at, 1.10006), state, _risk_context(at)) == ()
    with sqlite3.connect(tmp_path / "observer.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM challenger_observations").fetchone()[0] == 0


def test_observer_loads_only_registry_rows_with_shadow_challenger_promotion(tmp_path):
    from tradingagents.self_enhancement.challenger import ShadowChallengerObserver

    catalog, _ = _registered_challenger(tmp_path, promote=False)
    observer = ShadowChallengerObserver(
        catalog_path=catalog,
        observer_path=tmp_path / "observer.sqlite3",
    )

    assert observer.candidate_ids == ()


def test_observer_excludes_rolled_back_shadow_challenger(tmp_path):
    from tradingagents.self_enhancement.challenger import ShadowChallengerObserver

    catalog, _ = _registered_challenger(tmp_path, rolled_back=True)
    observer = ShadowChallengerObserver(
        catalog_path=catalog,
        observer_path=tmp_path / "observer.sqlite3",
    )

    assert observer.candidate_ids == ()


def test_observer_does_not_write_to_phase14_catalog(tmp_path):
    from tradingagents.self_enhancement.challenger import ShadowChallengerObserver

    catalog, _ = _registered_challenger(tmp_path, promote=False)
    catalog_file = tmp_path / "phase14.sqlite3"
    before = catalog_file.read_bytes()

    ShadowChallengerObserver(
        catalog_path=catalog,
        observer_path=tmp_path / "observer.sqlite3",
    )

    assert catalog_file.read_bytes() == before


def test_observer_suppresses_inconsistent_quote_features(tmp_path):
    from tradingagents.self_enhancement.challenger import ShadowChallengerObserver

    catalog, _ = _registered_challenger(tmp_path)
    observer = ShadowChallengerObserver(
        catalog_path=catalog,
        observer_path=tmp_path / "observer.sqlite3",
    )
    at = datetime(2026, 10, 3, 12, tzinfo=UTC)
    inconsistent = replace(_features(at, 1.10006), spread_points=1.0)

    assert observer.on_tick(inconsistent, _regime(at), _risk_context(at)) == ()
    with sqlite3.connect(tmp_path / "observer.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM challenger_observations").fetchone()[0] == 0


def test_observer_refuses_to_share_catalog_and_observation_database(tmp_path):
    from tradingagents.self_enhancement.challenger import ShadowChallengerObserver

    catalog, _ = _registered_challenger(tmp_path)

    with pytest.raises(ValueError, match="separate"):
        ShadowChallengerObserver(catalog_path=catalog, observer_path=catalog)


def test_observer_rejects_tampered_registered_strategy_spec(tmp_path):
    from tradingagents.self_enhancement.challenger import (
        ChallengerRegistryError,
        ShadowChallengerObserver,
    )

    catalog, candidate_id = _registered_challenger(tmp_path)
    with sqlite3.connect(catalog) as db:
        row = db.execute(
            "SELECT payload_json FROM candidates WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        payload = json.loads(row[0])
        payload["strategy_spec"]["name"] = "tampered"
        db.execute(
            "UPDATE candidates SET payload_json=? WHERE candidate_id=?",
            (json.dumps(payload), candidate_id),
        )

    with pytest.raises(ChallengerRegistryError, match="StrategySpec"):
        ShadowChallengerObserver(
            catalog_path=catalog,
            observer_path=tmp_path / "observer.sqlite3",
        )


def test_observer_has_no_provider_or_gateway_constructor_or_reference(tmp_path):
    from tradingagents.self_enhancement.challenger import ShadowChallengerObserver

    catalog, _ = _registered_challenger(tmp_path)
    observer = ShadowChallengerObserver(
        catalog_path=catalog,
        observer_path=tmp_path / "observer.sqlite3",
    )

    assert "gateway" not in inspect.signature(ShadowChallengerObserver).parameters
    assert "provider" not in inspect.signature(ShadowChallengerObserver).parameters
    assert not hasattr(observer, "gateway")
    assert not hasattr(observer, "provider")
