from datetime import datetime, timezone

import pytest

from tradingagents.forex.hft.account import AccountSimulator, CompoundingMode

UTC = timezone.utc


def test_account_simulator_tracks_equity_drawdown_and_benchmark():
    account = AccountSimulator(initial_balance=100.0, risk_fraction=0.01, mode=CompoundingMode.COMPOUNDING_RISK)
    assert account.position_size(stop_distance_points=20, point_value=0.0001) > 0
    account.record_trade(datetime(2026, 1, 1, tzinfo=UTC), 10.0)
    account.record_trade(datetime(2026, 1, 2, tzinfo=UTC), -5.0)
    metrics = account.report()
    assert metrics["balance"] == pytest.approx(105.0)
    assert metrics["max_drawdown"] >= 0
    assert metrics["benchmark_10pct_days"] == 1
    assert metrics["trades"] == 2


def test_max_drawdown_retains_the_worst_loss_after_equity_recovers():
    account = AccountSimulator(initial_balance=100.0, risk_fraction=0.005)
    account.record_trade(datetime(2026, 1, 1, tzinfo=UTC), -20.0)
    account.record_trade(datetime(2026, 1, 2, tzinfo=UTC), 30.0)
    assert account.report()["balance"] == pytest.approx(110.0)
    assert account.report()["max_drawdown"] == pytest.approx(0.20)
