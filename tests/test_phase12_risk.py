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
