from threading import Event

from tradingagents.forex.hft.persistence import AsyncPersistenceQueue


class _Target:
    def __init__(self):
        self.values = []
        self.done = Event()

    def append(self, value):
        self.values.append(value)
        self.done.set()


def test_async_persistence_queue_flushes_bounded_events_without_unbounded_growth():
    target = _Target()
    queue = AsyncPersistenceQueue(capacity=2)
    queue.start()

    assert queue.submit(target, "append", 1) is True
    assert queue.submit(target, "append", 2) is True
    assert queue.submit(target, "append", 3) is False
    queue.flush()
    queue.stop()

    assert target.values == [1, 2]
    assert queue.dropped == 1
