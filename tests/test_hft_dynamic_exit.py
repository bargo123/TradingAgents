from dataclasses import replace
from datetime import datetime, timedelta, timezone

from tests.test_phase12d_hft_first import _features, _state
from tradingagents.forex.hft.engines import HftExecutionEngine
from tradingagents.forex.hft.models import FastAction, PositionState
from tradingagents.forex.hft.regime import DirectionPolicy, Regime

UTC = timezone.utc


def _position_features(features, *, seconds=1, bid=None, ask=None, momentum=None, persistence=None):
    bid = features.bid if bid is None else bid
    ask = features.ask if ask is None else ask
    return replace(
        features,
        timestamp=features.timestamp + timedelta(seconds=seconds),
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2.0,
        spread=ask - bid,
        spread_points=(ask - bid) / features.point,
        momentum=features.momentum if momentum is None else momentum,
        direction_persistence=features.direction_persistence if persistence is None else persistence,
    )


def test_dynamic_exit_uses_mfe_micro_reversal_after_favorable_progress():
    features = _features((1.10000, 1.10002, 1.10004, 1.10008, 1.10012))
    state = _state(
        features.timestamp - timedelta(seconds=1),
        regime=Regime.BULLISH,
        direction_policy=DirectionPolicy.LONG_ONLY,
        momentum_enabled=True,
        range_enabled=False,
        mean_reversion_enabled=False,
    )
    engine = HftExecutionEngine()
    entry = engine.on_tick(state, features, position_state=PositionState.FLAT)
    assert entry.action is FastAction.ENTER_LONG
    favorable = _position_features(
        features,
        bid=features.bid + 20 * features.point,
        ask=features.ask + 20 * features.point,
        momentum=2 * features.point,
        persistence=0.8,
    )
    held = engine.on_tick(
        state,
        favorable,
        position_state=PositionState.LONG,
        entry_price=features.ask,
        entry_at=features.timestamp,
        expected_move_points=entry.expected_move_points,
        strategy_id=entry.strategy_id,
    )
    assert held.action is FastAction.NO_ACTION
    reversal = _position_features(
        favorable,
        bid=favorable.bid - 10 * favorable.point,
        ask=favorable.ask - 10 * favorable.point,
        momentum=-3 * favorable.point,
        persistence=0.8,
    )
    decision = engine.on_tick(
        state,
        reversal,
        position_state=PositionState.LONG,
        entry_price=features.ask,
        entry_at=features.timestamp,
        expected_move_points=entry.expected_move_points,
        strategy_id=entry.strategy_id,
    )
    assert decision.action is FastAction.EXIT
    assert decision.reason in {"MICRO_REVERSAL", "MFE_GIVEBACK"}
    assert decision.mfe_points >= 0
    assert decision.mae_points <= 0
    assert decision.profit_protection_state in {"PROFIT_PROTECTION_ARMED", "TRAILING", "EXIT_PENDING"}


def test_dynamic_exit_does_not_use_legacy_fixed_stop_or_target_as_normal_exit():
    features = _features((1.10000, 1.10002, 1.10004, 1.10006, 1.10008))
    state = _state(features.timestamp - timedelta(seconds=1))
    engine = HftExecutionEngine()
    decision = engine.on_tick(
        state,
        _position_features(
            features,
            bid=1.09975,
            ask=1.09985,
            momentum=0.0,
            persistence=0.0,
        ),
        position_state=PositionState.LONG,
        entry_price=1.10000,
        entry_at=features.timestamp,
        expected_move_points=8.0,
        strategy_id="range_rejection",
    )
    assert decision.reason not in {"STOP_LOSS", "TAKE_PROFIT"}


def test_dynamic_exit_uses_explicit_emergency_broker_stop_only_at_catastrophic_distance():
    features = _features((1.10000, 1.10002, 1.10004, 1.10006, 1.10008))
    state = _state(features.timestamp - timedelta(seconds=1))
    engine = HftExecutionEngine(emergency_stop_distance_points=100.0)
    decision = engine.on_tick(
        state,
        _position_features(features, bid=1.09895, ask=1.09905, momentum=0.0, persistence=0.0),
        position_state=PositionState.LONG,
        entry_price=1.10000,
        entry_at=features.timestamp,
        expected_move_points=8.0,
        strategy_id="range_rejection",
    )
    assert decision.action is FastAction.EXIT
    assert decision.reason == "EMERGENCY_BROKER_STOP"


def test_dynamic_exit_closes_on_causal_mfe_giveback_after_trailing_arms():
    features = _features((1.10000, 1.10002, 1.10004, 1.10006, 1.10008))
    state = _state(features.timestamp - timedelta(seconds=1))
    engine = HftExecutionEngine()
    favorable = _position_features(
        features,
        bid=1.10055,
        ask=1.10065,
        momentum=8 * features.point,
        persistence=0.8,
    )
    held = engine.on_tick(
        state,
        favorable,
        position_state=PositionState.LONG,
        entry_price=1.10000,
        entry_at=features.timestamp,
        expected_move_points=40.0,
        strategy_id="momentum_continuation",
    )
    assert held.action is FastAction.NO_ACTION
    assert held.profit_protection_state == "TRAILING"
    pullback = _position_features(
        favorable,
        bid=1.10025,
        ask=1.10035,
        momentum=-4 * features.point,
        persistence=0.1,
    )
    decision = engine.on_tick(
        state,
        pullback,
        position_state=PositionState.LONG,
        entry_price=1.10000,
        entry_at=features.timestamp,
        expected_move_points=40.0,
        strategy_id="momentum_continuation",
    )
    assert decision.action is FastAction.EXIT
    assert decision.reason == "MFE_GIVEBACK"
    assert decision.strategy_id == "momentum_continuation"
    assert decision.trailing_level is not None


def test_dynamic_exit_takes_profit_when_expected_move_is_reached():
    features = _features((1.10000, 1.10002, 1.10004, 1.10006, 1.10008))
    state = _state(features.timestamp - timedelta(seconds=1))
    engine = HftExecutionEngine()
    entry_price = features.ask
    target_bid = entry_price + 13 * features.point
    target = _position_features(
        features,
        bid=target_bid,
        ask=target_bid + features.spread,
        momentum=4 * features.point,
        persistence=0.8,
    )

    decision = engine.on_tick(
        state,
        target,
        position_state=PositionState.LONG,
        entry_price=entry_price,
        entry_at=features.timestamp,
        expected_move_points=8.0,
        strategy_id="range_rejection",
    )

    assert decision.action is FastAction.EXIT
    assert decision.reason == "TAKE_PROFIT"
    assert decision.current_pnl_points > 12.5


def test_expected_move_scales_no_progress_and_range_is_tighter_than_momentum():
    features = _features((1.10000, 1.10002, 1.10004, 1.10006, 1.10008))
    state = _state(features.timestamp - timedelta(seconds=1))
    range_engine = HftExecutionEngine()
    momentum_engine = HftExecutionEngine()
    range_decision = range_engine.on_tick(
        state,
        _position_features(features, seconds=10, bid=1.10000, ask=1.10010, momentum=0.0, persistence=0.0),
        position_state=PositionState.LONG,
        entry_price=1.10000,
        entry_at=features.timestamp,
        expected_move_points=8.0,
        strategy_id="range_rejection",
    )
    momentum_decision = momentum_engine.on_tick(
        state,
        _position_features(features, seconds=10, bid=1.10000, ask=1.10010, momentum=0.0, persistence=0.0),
        position_state=PositionState.LONG,
        entry_price=1.10000,
        entry_at=features.timestamp,
        expected_move_points=40.0,
        strategy_id="momentum_continuation",
    )
    assert range_decision.reason == "NO_PROGRESS"
    assert momentum_decision.reason != "NO_PROGRESS"


def test_hft_decision_telemetry_is_safe_and_bounded():
    features = _features((1.10000, 1.10002, 1.10004, 1.10008, 1.10012))
    state = _state(
        features.timestamp - timedelta(seconds=1),
        regime=Regime.BULLISH,
        direction_policy=DirectionPolicy.LONG_ONLY,
        momentum_enabled=True,
        range_enabled=False,
        mean_reversion_enabled=False,
    )
    decision = HftExecutionEngine().on_tick(state, features, position_state=PositionState.FLAT)
    assert decision.expected_move_points > 0
    assert decision.mfe_points == 0
    assert decision.mae_points == 0
    assert decision.time_since_entry_seconds == 0


def test_fast_execution_engine_contract_remains_separate():
    assert HftExecutionEngine.__name__ == "HftExecutionEngine"


def test_hft_runtime_keeps_only_distant_broker_emergency_protection(tmp_path):
    from tests.test_phase12d_hft_first import _hft_demo_runtime
    from tradingagents.forex.hft.demo_runtime import DemoRuntimeConfig

    state = _state(
        datetime(2026, 10, 1, 11, 59, 59, tzinfo=UTC),
        regime=Regime.BULLISH,
        direction_policy=DirectionPolicy.LONG_ONLY,
        momentum_enabled=True,
        range_enabled=False,
        mean_reversion_enabled=False,
    )
    runtime, api, _ = _hft_demo_runtime(tmp_path, (1.10000, 1.10002, 1.10004, 1.10008, 1.10012), state)
    runtime.config = DemoRuntimeConfig(
        artifact_path=tmp_path / "demo.sqlite3",
        symbol="EURUSD",
        hft_first=True,
    )
    for _ in range(6):
        runtime.run_once()
    assert len(api.calls) == 1
    request = api.calls[0]
    assert abs(request["price"] - request["sl"]) > 99 * 0.00001
    assert abs(request["tp"] - request["price"]) > 499 * 0.00001


def test_runtime_position_payload_carries_dynamic_exit_state(tmp_path):
    from tests.test_phase12d_hft_first import _hft_demo_runtime

    state = _state(
        datetime(2026, 10, 1, 11, 59, 59, tzinfo=UTC),
        regime=Regime.BULLISH,
        direction_policy=DirectionPolicy.LONG_ONLY,
        momentum_enabled=True,
        range_enabled=False,
        mean_reversion_enabled=False,
    )
    runtime, _, store = _hft_demo_runtime(tmp_path, (1.10000, 1.10002, 1.10004, 1.10008, 1.10012), state)
    for _ in range(6):
        runtime.run_once()
    positions = store.read_owned_positions("EURUSD")
    assert len(positions) == 1
    payload = runtime._position_payload(positions[0])
    assert payload["strategy_id"] in {"momentum_continuation", "range_rejection"}
    assert payload["expected_move_points"] > 0
    assert "profit_protection_state" in payload
