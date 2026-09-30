from datetime import datetime, timezone

import pytest

from tradingagents.forex.hft.demo_risk import DemoCircuitBreaker, normalize_volume, validate_stop_levels
from tradingagents.dataflows.mt5.models import Mt5SymbolInfo


UTC = timezone.utc


def test_volume_normalization_respects_min_max_step_and_cap():
    assert normalize_volume(0.013, minimum=0.01, maximum=1.0, step=0.01, cap=0.02) == pytest.approx(0.01)
    assert normalize_volume(0.019, minimum=0.01, maximum=1.0, step=0.01, cap=0.02) == pytest.approx(0.01)
    assert normalize_volume(0.029, minimum=0.01, maximum=1.0, step=0.01, cap=0.02) == pytest.approx(0.02)


def test_invalid_stop_levels_fail_closed():
    with pytest.raises(ValueError, match="stop_loss"):
        validate_stop_levels("LONG", entry=1.1000, stop_loss=1.1001, take_profit=1.1020, point=0.00001, minimum_distance_points=10)
    with pytest.raises(ValueError, match="take_profit"):
        validate_stop_levels("SHORT", entry=1.1000, stop_loss=1.1010, take_profit=1.1001, point=0.00001, minimum_distance_points=10)


def test_daily_loss_and_consecutive_loss_circuits_pause_new_entries():
    breaker = DemoCircuitBreaker(daily_loss_limit=0.02, max_consecutive_losses=3)
    breaker.start_day(1000.0, datetime(2026, 9, 30, tzinfo=UTC))

    assert breaker.entry_allowed(995.0) is True
    assert breaker.entry_allowed(979.0) is False
    assert breaker.status == "DAILY_LOSS_LIMIT"

    breaker.reset_day(1000.0, datetime(2026, 10, 1, tzinfo=UTC))
    breaker.record_closed_trade(-1.0)
    breaker.record_closed_trade(-1.0)
    breaker.record_closed_trade(-1.0)

    assert breaker.entry_allowed(1000.0) is False
    assert breaker.status == "DEMO_RISK_COOLDOWN"


def test_symbol_metadata_retains_demo_volume_and_stop_constraints():
    info = Mt5SymbolInfo(
        name="EURUSD",
        point=0.00001,
        volume_min=0.01,
        volume_max=10.0,
        volume_step=0.01,
        trade_stops_level=10,
        trade_freeze_level=0,
    )

    assert info.volume_min == 0.01
    assert info.volume_step == 0.01
    assert info.trade_stops_level == 10
