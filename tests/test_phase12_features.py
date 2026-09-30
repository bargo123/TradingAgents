from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.forex.hft.features import TickFeatureEngine
from tradingagents.forex.hft.models import Tick


UTC = timezone.utc


def _tick(seconds: int, bid: float, ask: float | None = None) -> Tick:
    return Tick(
        "EURUSD",
        datetime(2026, 1, 1, 12, 0, seconds, tzinfo=UTC),
        bid,
        bid + 0.0001 if ask is None else ask,
        sequence=seconds,
    )


def test_features_are_causal_and_track_momentum_and_persistence():
    engine = TickFeatureEngine(max_history=10, window=3)
    first = engine.update(_tick(0, 1.1000))
    second = engine.update(_tick(1, 1.1002))
    third = engine.update(_tick(2, 1.1004))
    assert first.return_1 == 0
    assert second.return_1 > 0
    assert third.momentum > 0
    assert third.direction_persistence == pytest.approx(1.0)
    assert third.tick_count == 3


def test_features_reject_out_of_order_or_symbol_switch():
    engine = TickFeatureEngine()
    engine.update(_tick(2, 1.1))
    with pytest.raises(ValueError, match="monotonic"):
        engine.update(_tick(1, 1.1))
    with pytest.raises(ValueError, match="symbol"):
        engine.update(Tick("USDJPY", datetime(2026, 1, 1, 12, 0, 3, tzinfo=UTC), 150, 150.01))

