from datetime import datetime, timedelta, timezone

from tradingagents.forex.hft.models import FastAction, RiskDecision, Tick
from tradingagents.forex.hft.risk import RiskConfig, RiskContext, RiskEngine

UTC = timezone.utc


def _tick(seconds=0, spread=0.0001):
    return Tick("EURUSD", datetime(2026, 1, 1, 12, 0, seconds, tzinfo=UTC), 1.1, 1.1 + spread)


def _context(**overrides):
    values = {
        "equity": 1000.0,
        "open_exposure": 0.0,
        "daily_loss": 0.0,
        "drawdown": 0.0,
        "consecutive_losses": 0,
        "tick_timestamp": _tick().timestamp,
        "observed_at": _tick().timestamp,
        "session": "LONDON",
        "slippage_points": 0.0,
    }
    values.update(overrides)
    return RiskContext(**values)


def test_risk_engine_accepts_safe_entry_and_rejects_spread_or_daily_loss():
    engine = RiskEngine(RiskConfig(max_spread_points=20, max_daily_loss=10))
    accepted = engine.evaluate(FastAction.ENTER_LONG, _tick(), _context())
    assert isinstance(accepted, RiskDecision)
    assert accepted.accepted is True
    rejected = engine.evaluate(FastAction.ENTER_LONG, _tick(spread=0.001), _context())
    assert rejected.reason_code == "SPREAD_LIMIT"
    rejected_loss = engine.evaluate(FastAction.ENTER_LONG, _tick(), _context(daily_loss=10.1))
    assert rejected_loss.reason_code == "DAILY_LOSS_LIMIT"


def test_risk_engine_rejects_stale_ticks_and_expired_exposure():
    engine = RiskEngine(RiskConfig(stale_after_seconds=5, max_drawdown=0.2))
    old = _tick()
    context = _context(observed_at=old.timestamp + timedelta(seconds=6))
    assert engine.evaluate(FastAction.ENTER_LONG, old, context).reason_code == "STALE_TICK"
    assert engine.evaluate(FastAction.ENTER_LONG, _tick(), _context(drawdown=0.21)).reason_code == "DRAWDOWN_LIMIT"


def test_risk_engine_enforces_explicit_session_constraints():
    engine = RiskEngine()
    decision = engine.evaluate(FastAction.ENTER_LONG, _tick(), _context(session="ASIA"), allowed_sessions=("LONDON",))
    assert decision.accepted is False
    assert decision.reason_code == "SESSION_LIMIT"


def test_manual_pause_policy_ignores_loss_stops_but_keeps_entry_validation():
    engine = RiskEngine(RiskConfig(enforce_loss_limits=False, max_daily_loss=0.02, max_drawdown=0.1, max_consecutive_losses=3))
    losing_context = _context(daily_loss=0.25, drawdown=0.25, consecutive_losses=20)

    decision = engine.evaluate(FastAction.ENTER_LONG, _tick(), losing_context)

    assert decision.accepted is True
    assert decision.risk_fraction == 0.01


def test_manual_pause_policy_does_not_disable_spread_or_exposure_gates():
    engine = RiskEngine(RiskConfig(enforce_loss_limits=False, max_spread_points=20, max_open_exposure=1.0))

    spread_decision = engine.evaluate(FastAction.ENTER_LONG, _tick(spread=0.001), _context(daily_loss=0.5))
    exposure_decision = engine.evaluate(
        FastAction.ENTER_LONG,
        _tick(),
        _context(open_exposure=1.0, daily_loss=0.5),
    )

    assert spread_decision.reason_code == "SPREAD_LIMIT"
    assert exposure_decision.reason_code == "EXPOSURE_LIMIT"
