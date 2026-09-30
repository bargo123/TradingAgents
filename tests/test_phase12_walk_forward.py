from datetime import datetime, timedelta, timezone

from tradingagents.forex.hft.models import Tick
from tradingagents.forex.hft.replay import walk_forward_splits

UTC = timezone.utc


def test_walk_forward_splits_are_chronological_and_non_overlapping():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ticks = tuple(Tick("EURUSD", start + timedelta(minutes=i), 1.1, 1.1001, sequence=i) for i in range(20))
    splits = walk_forward_splits(ticks, train_fraction=0.5, dev_fraction=0.2, validation_fraction=0.15)
    assert splits.train[-1].timestamp < splits.development[0].timestamp
    assert splits.development[-1].timestamp < splits.validation[0].timestamp
    assert splits.validation[-1].timestamp < splits.unseen_test[0].timestamp

