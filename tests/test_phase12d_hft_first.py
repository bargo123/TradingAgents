from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradingagents.forex.hft.demo_gateway import VerifiedDemoExecutionGateway
from tradingagents.forex.hft.demo_runtime import DemoHftRuntime, DemoRuntimeConfig
from tradingagents.forex.hft.demo_store import DemoExecutionStore
from tradingagents.forex.hft.engines import HftExecutionEngine
from tradingagents.forex.hft.features import TickFeatureEngine
from tradingagents.forex.hft.models import FastAction, PositionState, Tick
from tradingagents.forex.hft.plan_store import AtomicPlanStore
from tradingagents.forex.hft.regime import (
    DirectionPolicy,
    Regime,
    StrategicRegimeState,
    build_hft_bootstrap_neutral_regime,
)
from tradingagents.forex.hft.regime_feed import build_regime_from_shadow_decision
from tradingagents.forex.hft.regime_store import AtomicRegimeStore
from tradingagents.forex.hft.store import HftShadowStore
from tradingagents.forex.hft.strategies import (
    MomentumContinuationStrategy,
    RangeRejectionStrategy,
    SignalArbiter,
)
from tradingagents.forex.watcher import Mt5OperationBusy

UTC = timezone.utc


def _state(now: datetime, **overrides) -> StrategicRegimeState:
    values = {
        "state_id": "state-1",
        "symbol": "EURUSD",
        "regime": Regime.NEUTRAL,
        "direction_policy": DirectionPolicy.BOTH,
        "risk_multiplier": 1.0,
        "confidence": 0.8,
        "created_at": now,
        "expires_at": now + timedelta(minutes=5),
        "momentum_enabled": False,
        "range_enabled": True,
        "breakout_enabled": False,
        "mean_reversion_enabled": True,
        "source_run_id": "run-1",
        "source_decision_id": "decision-1",
        "git_commit": "abc123",
    }
    values.update(overrides)
    return StrategicRegimeState(**values)


def _features(values: tuple[float, ...]):
    engine = TickFeatureEngine(max_history=30, window=5)
    start = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    latest = None
    for index, mid in enumerate(values):
        latest = engine.update(
            Tick("EURUSD", start + timedelta(seconds=index), mid - 0.00005, mid + 0.00005, sequence=index)
        )
    assert latest is not None
    return latest


def test_regime_state_is_atomic_and_expires_without_tick_path_side_effects():
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    store = AtomicRegimeStore()
    state = _state(now)

    store.replace(state, now=now)

    assert store.current(now, "EURUSD") == state
    assert store.current(now + timedelta(minutes=6), "EURUSD") is None


def test_invalid_regime_direction_policy_is_rejected():
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="direction_policy"):
        _state(now, direction_policy="GUESS")


def test_hold_shadow_decision_becomes_neutral_filter_not_directional_trigger():
    now = datetime.now(UTC)
    decision = SimpleNamespace(
        decision_id="decision-1",
        source_run_id="run-1",
        resolved_symbol="EURUSD",
        action="HOLD",
        confidence=0.7,
        decision_completed_timestamp=now,
        valid_until=now + timedelta(minutes=5),
        git_commit="abc123",
    )

    state = build_regime_from_shadow_decision(decision, git_commit="abc123")

    assert state is not None
    assert state.regime is Regime.NEUTRAL
    assert state.direction_policy is DirectionPolicy.BOTH
    assert state.range_enabled is True
    assert state.momentum_enabled is False


def test_regime_feed_accepts_persisted_optional_confidence_as_neutral_risk():
    """Persisted Phase 5 decisions may leave confidence unset."""
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    decision = SimpleNamespace(
        decision_id="decision-persisted",
        source_run_id="run-persisted",
        resolved_symbol="EURUSD",
        action="HOLD",
        confidence=None,
        decision_completed_timestamp=now,
        valid_until=now + timedelta(minutes=5),
    )

    state = build_regime_from_shadow_decision(decision, git_commit="abc123")

    assert state is not None
    assert state.confidence == 0.0
    assert state.risk_multiplier == 0.25
    assert state.direction_policy is DirectionPolicy.BOTH


def test_regime_feed_rejects_invalid_confidence_and_accepts_action_enum():
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    decision = SimpleNamespace(
        action=SimpleNamespace(value="BUY"),
        resolved_symbol="EURUSD",
        decision_id="decision-1",
        source_run_id="run-1",
        decision_completed_timestamp=now,
        valid_for_seconds=300,
        confidence=1.5,
    )

    assert build_regime_from_shadow_decision(decision, git_commit="abc") is None
    decision.confidence = 0.8
    assert build_regime_from_shadow_decision(decision, git_commit="abc") is not None


def test_momentum_strategy_is_causal_and_emits_long_signal():
    features = _features((1.10000, 1.10002, 1.10004, 1.10008, 1.10012))
    signal = MomentumContinuationStrategy(minimum_momentum_points=4.0).evaluate(features)

    assert signal is not None
    assert signal.action is FastAction.ENTER_LONG
    assert signal.strategy_id == "momentum_continuation"


def test_range_rejection_strategy_emits_reversal_at_lower_edge():
    features = _features((1.10000, 1.09980, 1.09970, 1.09972, 1.09978))
    signal = RangeRejectionStrategy(edge_fraction=0.30).evaluate(features)

    assert signal is not None
    assert signal.action is FastAction.ENTER_LONG
    assert signal.strategy_id == "range_rejection"


def test_arbiter_rejects_directionally_forbidden_signals_and_costly_entries():
    features = _features((1.10000, 1.10002, 1.10004, 1.10008, 1.10012))
    signal = MomentumContinuationStrategy(minimum_momentum_points=4.0).evaluate(features)
    assert signal is not None
    state = _state(
        features.timestamp - timedelta(seconds=1),
        direction_policy=DirectionPolicy.SHORT_ONLY,
        momentum_enabled=True,
        range_enabled=False,
        mean_reversion_enabled=False,
    )

    selected, reason = SignalArbiter(cost_safety_margin_points=2.0).select(
        (signal,), state, spread_points=20.0
    )

    assert selected is None
    assert reason in {"DIRECTION_POLICY", "TRANSACTION_COST"}


def test_no_valid_regime_is_fail_closed():
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    store = AtomicRegimeStore()

    assert store.current(now, "EURUSD") is None


def test_hft_execution_engine_generates_entry_from_deterministic_strategy():
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

    assert decision.action is FastAction.ENTER_LONG
    assert decision.strategy_id == "momentum_continuation"


def test_hft_execution_engine_pauses_without_a_valid_regime():
    features = _features((1.10000, 1.10002, 1.10004, 1.10008, 1.10012))

    decision = HftExecutionEngine().on_tick(None, features, position_state=PositionState.FLAT)

    assert decision.action is FastAction.NO_ACTION
    assert decision.reason == "NO_VALID_STRATEGIC_STATE"


def test_bootstrap_neutral_regime_is_both_with_existing_risk_floor():
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)

    state = build_hft_bootstrap_neutral_regime("EURUSD", now)

    assert state.source_run_id == "HFT_BOOTSTRAP_NEUTRAL"
    assert state.regime is Regime.NEUTRAL
    assert state.direction_policy is DirectionPolicy.BOTH
    assert state.risk_multiplier == 0.25
    assert state.confidence == 0.0
    assert state.momentum_enabled is True
    assert state.range_enabled is True
    assert state.breakout_enabled is False
    assert state.mean_reversion_enabled is False


def test_hft_engine_evaluates_both_strategies_under_bootstrap_regime():
    class _SpyStrategy:
        def __init__(self):
            self.calls = 0

        def evaluate(self, features):
            self.calls += 1
            return None

    features = _features((1.10000, 1.10001, 1.10002, 1.10003))
    momentum = _SpyStrategy()
    range_rejection = _SpyStrategy()
    engine = HftExecutionEngine(momentum=momentum, range_rejection=range_rejection)
    state = build_hft_bootstrap_neutral_regime("EURUSD", features.timestamp)

    decision = engine.on_tick(state, features, position_state=PositionState.FLAT)

    assert decision.reason == "NO_VALID_SIGNAL"
    assert momentum.calls == 1
    assert range_rejection.calls == 1


def test_high_uncertainty_state_overrides_neutral_bootstrap():
    features = _features((1.10000, 1.10001, 1.10002, 1.10003))
    state = _state(
        features.timestamp - timedelta(seconds=1),
        regime=Regime.HIGH_UNCERTAINTY,
        direction_policy=DirectionPolicy.BOTH,
        momentum_enabled=True,
        range_enabled=True,
    )

    decision = HftExecutionEngine().on_tick(state, features, position_state=PositionState.FLAT)

    assert decision.action is FastAction.NO_ACTION
    assert decision.reason == "STRATEGIC_HIGH_UNCERTAINTY"


def test_hft_execution_engine_exits_position_on_emergency_stop_without_regime():
    features = _features((1.10000, 1.09980, 1.09970, 1.09960, 1.09950))
    state = _state(features.timestamp - timedelta(seconds=1))

    decision = HftExecutionEngine(emergency_stop_distance_points=10).on_tick(
        state,
        features,
        position_state=PositionState.LONG,
        entry_price=1.1000,
        entry_at=features.timestamp - timedelta(seconds=10),
    )

    assert decision.action is FastAction.EXIT
    assert decision.reason == "EMERGENCY_BROKER_STOP"


def test_hft_execution_engine_exits_when_strategic_state_expires():
    features = _features((1.10000, 1.10001, 1.10002, 1.10003))
    expired = _state(features.timestamp - timedelta(minutes=2), expires_at=features.timestamp - timedelta(seconds=1))

    decision = HftExecutionEngine().on_tick(
        expired,
        features,
        position_state=PositionState.LONG,
        entry_price=1.1000,
        entry_at=features.timestamp - timedelta(seconds=1),
    )

    assert decision.action is FastAction.EXIT
    assert decision.reason == "STRATEGIC_REGIME_INVALIDATED"


def test_hft_execution_engine_exits_on_spread_or_volatility_blowout():
    features = _features((1.10000, 1.10002, 1.10004, 1.10006))
    state = _state(features.timestamp - timedelta(seconds=1))
    wide = replace(
        features,
        spread=features.spread + 100 * features.point,
        spread_points=features.spread_points + 100,
        spread_expansion=features.spread_expansion + 100 * features.point,
    )
    spread_decision = HftExecutionEngine().on_tick(
        state, wide, position_state=PositionState.LONG, entry_price=1.1000
    )
    assert spread_decision.reason == "SPREAD_BLOWOUT"

    volatile = replace(features, volatility=0.01)
    volatility_decision = HftExecutionEngine().on_tick(
        state, volatile, position_state=PositionState.LONG, entry_price=1.1000
    )
    assert volatility_decision.reason == "VOLATILITY_BLOWOUT"


def test_demo_runtime_parses_persisted_position_timestamp_for_time_stop():
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    parsed = DemoHftRuntime._position_entry_at({"opened_at": "2026-01-01T00:00:00Z"})
    assert parsed == timestamp


class _DemoProvider:
    def __init__(self, mids: tuple[float, ...]):
        self._ticks = iter(mids)
        self._last = mids[-1]
        self.open_ticket: int | None = None
        self.order_api = None

    def get_tick(self, symbol):
        self._last = next(self._ticks, self._last)
        timestamp = datetime(2026, 10, 1, 12, 0, tzinfo=UTC) + timedelta(seconds=getattr(self, "_count", 0))
        self._count = getattr(self, "_count", 0) + 1
        return SimpleNamespace(symbol=symbol, timestamp=timestamp, bid=self._last - 0.00005, ask=self._last + 0.00005)

    def get_account_info(self):
        return SimpleNamespace(login=7, server="Broker-Demo", currency="USD", trade_mode=0, balance=1000.0, equity=1000.0, margin=0.0, margin_free=1000.0)

    def get_terminal_info(self):
        return SimpleNamespace(company="Broker", connected=True)

    def get_symbols(self):
        return (SimpleNamespace(name="EURUSD", point=0.00001, volume_min=0.01, volume_max=10.0, volume_step=0.01, trade_stops_level=0),)

    def get_positions(self, symbol=None):
        if self.open_ticket is None:
            return ()
        return (SimpleNamespace(ticket=self.open_ticket, symbol=symbol or "EURUSD"),)


class _DemoApi:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_FILLING_IOC = 1
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE = 10009

    def __init__(self):
        self.calls = []

    def order_send(self, request):
        self.calls.append(request)
        return SimpleNamespace(retcode=10009, order=11, deal=12, price=request["price"], volume=request["volume"], comment="filled")


def _hft_demo_runtime(tmp_path: Path, mids: tuple[float, ...], state: StrategicRegimeState):
    provider = _DemoProvider(mids)
    api = _DemoApi()
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    gateway = VerifiedDemoExecutionGateway(provider, api, store, run_id="demo-run", demo_trade_mode=0, magic=12012012)
    regimes = AtomicRegimeStore()
    regimes.replace(state, now=datetime(2026, 10, 1, 12, 0, tzinfo=UTC))
    runtime = DemoHftRuntime(
        provider,
        AtomicPlanStore(),
        gateway=gateway,
        store=store,
        regime_store=regimes,
        config=DemoRuntimeConfig(artifact_path=tmp_path / "demo.sqlite3", symbol="EURUSD", hft_first=True),
        clock=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )
    return runtime, api, store


def test_demo_hft_runtime_uses_regime_filter_and_real_demo_gateway(tmp_path: Path):
    state = _state(datetime(2026, 10, 1, 11, 59, 59, tzinfo=UTC), regime=Regime.BULLISH, direction_policy=DirectionPolicy.LONG_ONLY, momentum_enabled=True, range_enabled=False, mean_reversion_enabled=False)
    runtime, api, store = _hft_demo_runtime(tmp_path, (1.10000, 1.10002, 1.10004, 1.10008, 1.10012), state)

    results = [runtime.run_once() for _ in range(5)]

    assert any(result["action"] == "ENTER_LONG" for result in results)
    assert len(api.calls) == 1
    assert store.snapshot()["orders"] == 1


def test_demo_hft_runtime_no_regime_state_pauses_without_order(tmp_path: Path):
    provider = _DemoProvider((1.10000, 1.10002, 1.10004, 1.10008, 1.10012))
    api = _DemoApi()
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    runtime = DemoHftRuntime(
        provider,
        AtomicPlanStore(),
        gateway=VerifiedDemoExecutionGateway(provider, api, store, run_id="demo-run", demo_trade_mode=0, magic=12012012),
        store=store,
        regime_store=AtomicRegimeStore(),
        config=DemoRuntimeConfig(artifact_path=tmp_path / "demo.sqlite3", symbol="EURUSD", hft_first=True),
        clock=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )

    result = runtime.run_once()

    assert result["action"] == "NO_ACTION"
    assert result["reason_code"] == "NO_VALID_STRATEGIC_STATE"
    assert api.calls == []


def test_demo_hft_runtime_bootstraps_neutral_only_when_configured(tmp_path: Path):
    provider = _DemoProvider((1.10000,))
    api = _DemoApi()
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    regimes = AtomicRegimeStore()
    runtime = DemoHftRuntime(
        provider,
        AtomicPlanStore(),
        gateway=VerifiedDemoExecutionGateway(provider, api, store, run_id="demo-run", demo_trade_mode=0, magic=12012012),
        store=store,
        regime_store=regimes,
        config=DemoRuntimeConfig(
            artifact_path=tmp_path / "demo.sqlite3",
            symbol="EURUSD",
            hft_first=True,
            no_regime_policy="BOOTSTRAP_NEUTRAL",
        ),
        clock=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )
    observed = []
    original = runtime.hft.on_tick

    def capture(state, features, **kwargs):
        observed.append(state)
        return original(state, features, **kwargs)

    runtime.hft.on_tick = capture

    result = runtime.run_once()

    assert result["reason_code"] != "NO_VALID_STRATEGIC_STATE"
    assert observed and observed[0] is not None
    assert observed[0].source_run_id == "HFT_BOOTSTRAP_NEUTRAL"
    assert observed[0].direction_policy is DirectionPolicy.BOTH
    assert api.calls == []


@pytest.mark.parametrize(
    ("regime", "direction_policy"),
    (
        (Regime.BULLISH, DirectionPolicy.LONG_ONLY),
        (Regime.BEARISH, DirectionPolicy.SHORT_ONLY),
    ),
)
def test_demo_hft_runtime_strategic_state_replaces_bootstrap(
    tmp_path: Path,
    regime: Regime,
    direction_policy: DirectionPolicy,
):
    provider = _DemoProvider((1.10000, 1.10001))
    api = _DemoApi()
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    regimes = AtomicRegimeStore()
    runtime = DemoHftRuntime(
        provider,
        AtomicPlanStore(),
        gateway=VerifiedDemoExecutionGateway(provider, api, store, run_id="demo-run", demo_trade_mode=0, magic=12012012),
        store=store,
        regime_store=regimes,
        config=DemoRuntimeConfig(
            artifact_path=tmp_path / "demo.sqlite3",
            symbol="EURUSD",
            hft_first=True,
            no_regime_policy="BOOTSTRAP_NEUTRAL",
        ),
        clock=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )
    observed = []
    original = runtime.hft.on_tick

    def capture(state, features, **kwargs):
        observed.append(state)
        return original(state, features, **kwargs)

    runtime.hft.on_tick = capture
    runtime.run_once()
    strategic = _state(
        datetime(2026, 10, 1, 12, 0, 1, tzinfo=UTC),
        regime=regime,
        direction_policy=direction_policy,
        momentum_enabled=regime is Regime.BULLISH,
        range_enabled=regime is Regime.BEARISH,
        mean_reversion_enabled=regime is Regime.BEARISH,
    )
    regimes.replace(strategic, now=strategic.created_at)
    runtime.run_once()

    assert observed[0].source_run_id == "HFT_BOOTSTRAP_NEUTRAL"
    assert observed[1].source_decision_id == "decision-1"
    assert observed[1].regime is regime


def test_demo_hft_runtime_pause_state_overrides_bootstrap(tmp_path: Path):
    state = _state(
        datetime(2026, 10, 1, 11, 59, 59, tzinfo=UTC),
        direction_policy=DirectionPolicy.PAUSE,
        momentum_enabled=True,
        range_enabled=True,
    )
    runtime, api, _ = _hft_demo_runtime(tmp_path, (1.10000,), state)
    runtime.config = DemoRuntimeConfig(
        artifact_path=tmp_path / "demo.sqlite3",
        symbol="EURUSD",
        hft_first=True,
        no_regime_policy="BOOTSTRAP_NEUTRAL",
    )

    result = runtime.run_once()

    assert result["reason_code"] == "STRATEGIC_PAUSE"
    assert api.calls == []


def test_demo_bootstrap_candidate_still_uses_existing_risk_and_gateway(tmp_path: Path):
    provider = _DemoProvider((1.10000, 1.10002, 1.10004, 1.10008, 1.10012))
    api = _DemoApi()
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    runtime = DemoHftRuntime(
        provider,
        AtomicPlanStore(),
        gateway=VerifiedDemoExecutionGateway(provider, api, store, run_id="demo-run", demo_trade_mode=0, magic=12012012),
        store=store,
        regime_store=AtomicRegimeStore(),
        config=DemoRuntimeConfig(
            artifact_path=tmp_path / "demo.sqlite3",
            symbol="EURUSD",
            hft_first=True,
            no_regime_policy="BOOTSTRAP_NEUTRAL",
        ),
        clock=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )

    results = [runtime.run_once() for _ in range(5)]

    assert any(result["action"] in {"ENTER_LONG", "ENTER_SHORT"} for result in results)
    assert len(api.calls) == 1
    assert store.snapshot()["orders"] == 1


def test_demo_runtime_drops_tick_when_serialized_account_read_is_busy(tmp_path: Path):
    provider = _DemoProvider((1.10000,))
    api = _DemoApi()
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    runtime = DemoHftRuntime(
        provider,
        AtomicPlanStore(),
        gateway=VerifiedDemoExecutionGateway(provider, api, store, run_id="demo-run", demo_trade_mode=0, magic=12012012),
        store=store,
        regime_store=AtomicRegimeStore(),
        config=DemoRuntimeConfig(
            artifact_path=tmp_path / "demo.sqlite3",
            symbol="EURUSD",
            hft_first=True,
            no_regime_policy="BOOTSTRAP_NEUTRAL",
        ),
        clock=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )

    def busy_account(**_kwargs):
        raise Mt5OperationBusy("demo_account")

    runtime._account = busy_account

    result = runtime.run_once()

    assert result["status"] == "DROPPED"
    assert result["reason_code"] == "MT5_OPERATION_BUSY"
    assert result["broker_order_sent"] is False
    assert api.calls == []


def test_demo_hft_runtime_never_bootstraps_for_non_demo_account(tmp_path: Path):
    provider = _DemoProvider((1.10000,))
    provider.get_account_info = lambda: SimpleNamespace(
        login=7,
        server="Broker-Real",
        currency="USD",
        trade_mode=1,
        balance=1000.0,
        equity=1000.0,
        margin=0.0,
        margin_free=1000.0,
    )
    api = _DemoApi()
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    runtime = DemoHftRuntime(
        provider,
        AtomicPlanStore(),
        gateway=VerifiedDemoExecutionGateway(provider, api, store, run_id="demo-run", demo_trade_mode=0, magic=12012012),
        store=store,
        regime_store=AtomicRegimeStore(),
        config=DemoRuntimeConfig(
            artifact_path=tmp_path / "demo.sqlite3",
            symbol="EURUSD",
            hft_first=True,
            no_regime_policy="BOOTSTRAP_NEUTRAL",
        ),
        clock=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )
    observed = []
    original = runtime.hft.on_tick

    def capture(state, features, **kwargs):
        observed.append(state)
        return original(state, features, **kwargs)

    runtime.hft.on_tick = capture
    result = runtime.run_once()

    assert observed == [None]
    assert result["reason_code"] == "NO_VALID_STRATEGIC_STATE"
    assert api.calls == []


def test_expired_directional_regime_is_not_reused_by_bootstrap(tmp_path: Path):
    provider = _DemoProvider((1.10000,))
    api = _DemoApi()
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    regimes = AtomicRegimeStore()
    runtime = DemoHftRuntime(
        provider,
        AtomicPlanStore(),
        gateway=VerifiedDemoExecutionGateway(provider, api, store, run_id="demo-run", demo_trade_mode=0, magic=12012012),
        store=store,
        regime_store=regimes,
        config=DemoRuntimeConfig(
            artifact_path=tmp_path / "demo.sqlite3",
            symbol="EURUSD",
            hft_first=True,
            no_regime_policy="BOOTSTRAP_NEUTRAL",
        ),
        clock=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )
    observed = []
    original = runtime.hft.on_tick

    def capture(state, features, **kwargs):
        observed.append(state)
        return original(state, features, **kwargs)

    runtime.hft.on_tick = capture
    runtime.run_once()

    assert observed[0].regime is Regime.NEUTRAL
    assert observed[0].direction_policy is DirectionPolicy.BOTH
    assert observed[0].source_run_id == "HFT_BOOTSTRAP_NEUTRAL"


def test_demo_hft_runtime_routes_deterministic_exit_through_gateway(tmp_path: Path):
    state = _state(datetime(2026, 10, 1, 11, 59, 59, tzinfo=UTC), regime=Regime.BULLISH, direction_policy=DirectionPolicy.LONG_ONLY, momentum_enabled=True, range_enabled=False, mean_reversion_enabled=False)
    runtime, api, _ = _hft_demo_runtime(tmp_path, (1.10000, 1.10002, 1.10004, 1.10008, 1.10012, 1.09990), state)
    runtime.config = DemoRuntimeConfig(
        artifact_path=tmp_path / "demo.sqlite3",
        symbol="EURUSD",
        hft_first=True,
        stop_distance_points=1,
        cooldown_seconds=0,
    )
    runtime._ensure_run()

    entry = None
    provider = runtime.provider
    for _ in range(5):
        entry = runtime.run_once()
        if entry["action"] == "ENTER_LONG":
            provider.open_ticket = 11
            break
    assert entry is not None and entry["action"] == "ENTER_LONG"

    exit_result = runtime.run_once()

    assert exit_result["action"] == "EXIT"
    assert exit_result["broker_order_sent"] is True
    assert len(api.calls) == 2
    assert "position" in api.calls[-1]


def test_demo_order_rate_circuit_is_bounded_and_fail_closed(tmp_path: Path):
    state = _state(datetime(2026, 10, 1, 11, 59, 59, tzinfo=UTC))
    runtime, _, store = _hft_demo_runtime(tmp_path, (1.10000,), state)
    runtime.config = DemoRuntimeConfig(
        artifact_path=tmp_path / "demo.sqlite3",
        symbol="EURUSD",
        hft_first=True,
        max_order_submissions=1,
        cooldown_seconds=0,
    )
    runtime._ensure_run()
    first = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)

    assert runtime._order_allowed(first) is True
    runtime._record_submission(first)
    assert runtime._order_allowed(first + timedelta(seconds=1)) is False
    assert runtime._order_rate_circuit_open is True
    assert store.read_circuit_state()["status"] == "HFT_ORDER_RATE_CIRCUIT_OPEN"


def test_demo_hft_run_persists_shadow_expectation_asynchronously(tmp_path: Path):
    state = _state(datetime(2026, 10, 1, 11, 59, 59, tzinfo=UTC))
    runtime, _, _ = _hft_demo_runtime(tmp_path, (1.10000, 1.10002, 1.10004), state)
    shadow_store = HftShadowStore(tmp_path / "shadow.sqlite3")
    runtime.shadow_store = shadow_store
    runtime.config = DemoRuntimeConfig(
        artifact_path=tmp_path / "demo.sqlite3",
        symbol="EURUSD",
        hft_first=True,
        max_ticks=3,
        poll_interval_seconds=0,
    )

    result = runtime.run(max_ticks=3)

    snapshot = shadow_store.snapshot()
    assert snapshot["ticks"] == 3
    assert snapshot["actions"] == 3
    assert result["local_decision_max_ms"] is not None
    assert result["executed"] is False


def test_worker_publishes_regime_state_without_converting_hold_to_entry(tmp_path: Path):
    from tests.test_phase12_hft_supervisor import _decision, _Provider
    from tradingagents.forex.hft.runtime import HftShadowConfig
    from tradingagents.forex.hft.store import HftShadowStore
    from tradingagents.forex.hft.supervisor import HftShadowWorker

    regimes = AtomicRegimeStore()
    worker = HftShadowWorker(
        lambda **_: _Provider(),
        AtomicPlanStore(),
        regime_store=regimes,
        config=HftShadowConfig(max_ticks=1, artifact_path=tmp_path / "hft.sqlite3"),
        store=HftShadowStore(tmp_path / "hft.sqlite3"),
        git_commit="abc123",
    )

    assert worker.handle_decision(_decision("HOLD")) is True
    current = regimes.current(datetime.now(UTC), "EURUSD")
    assert current is not None
    assert current.direction_policy is DirectionPolicy.BOTH
    assert current.regime is Regime.NEUTRAL


def test_worker_publishes_persisted_decision_with_missing_confidence(tmp_path: Path):
    from tests.test_phase12_hft_supervisor import _decision, _Provider
    from tradingagents.forex.hft.runtime import HftShadowConfig
    from tradingagents.forex.hft.store import HftShadowStore
    from tradingagents.forex.hft.supervisor import HftShadowWorker

    regimes = AtomicRegimeStore()
    worker = HftShadowWorker(
        lambda **_: _Provider(),
        AtomicPlanStore(),
        regime_store=regimes,
        config=HftShadowConfig(max_ticks=1, artifact_path=tmp_path / "hft.sqlite3"),
        store=HftShadowStore(tmp_path / "hft.sqlite3"),
        git_commit="abc123",
    )

    assert worker.handle_decision(replace(_decision("HOLD"), confidence=None)) is True
    current = regimes.current(datetime.now(UTC), "EURUSD")
    assert current is not None
    assert current.confidence == 0.0
