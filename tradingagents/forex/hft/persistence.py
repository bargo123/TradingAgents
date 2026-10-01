"""Bounded asynchronous persistence for non-critical HFT telemetry."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _Event:
    target: Any
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class AsyncPersistenceQueue:
    """A bounded single-writer queue; broker-critical writes stay synchronous."""

    def __init__(self, *, capacity: int = 512) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._queue: queue.Queue[_Event | None] = queue.Queue(maxsize=capacity)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._dropped = 0
        self._error: str | None = None

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._error = None
            self._thread = threading.Thread(target=self._run, name="phase12-telemetry-writer", daemon=True)
            self._thread.start()

    def submit(self, target: Any, method: str, *args: Any, **kwargs: Any) -> bool:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("persistence queue is not running")
        event = _Event(target, str(method), tuple(args), dict(kwargs))
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped += 1
            return False
        return True

    def flush(self) -> None:
        self._queue.join()

    def stop(self, timeout: float = 10.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self.flush()
        self._stop.set()
        self._queue.put(None)
        thread.join(timeout)
        if thread.is_alive():
            raise RuntimeError("persistence writer did not stop")
        self._thread = None

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            try:
                if event is None:
                    return
                getattr(event.target, event.method)(*event.args, **event.kwargs)
            except Exception as exc:  # telemetry never mutates execution behavior
                self._error = type(exc).__name__.upper()[:80]
            finally:
                self._queue.task_done()


__all__ = ["AsyncPersistenceQueue"]
