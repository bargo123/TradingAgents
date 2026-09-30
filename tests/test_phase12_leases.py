import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradingagents.forex.hft.store import (
    HftLeaseOwner,
    HftLeaseStatus,
    HftShadowStore,
)
from tradingagents.forex.supervisor import _SharedReadOnlyMt5Provider
from tradingagents.forex.watcher import ReadOnlyMt5ProviderProxy, SerializedMt5OperationGate

UTC = timezone.utc


def _owner(token: str, pid: int = 123) -> HftLeaseOwner:
    return HftLeaseOwner(
        owner_token=token,
        pid=pid,
        host="test-host",
        process_started_at=datetime.now(UTC),
    )


def test_hft_lease_refuses_active_owner_and_recovers_after_expiry(tmp_path: Path):
    store = HftShadowStore(tmp_path / "hft.sqlite3", lease_ttl_seconds=5)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    first = store.acquire_lease(_owner("one"), now)
    assert first.status is HftLeaseStatus.ACQUIRED
    active = store.acquire_lease(_owner("two"), now + timedelta(seconds=1))
    assert active.status is HftLeaseStatus.HFT_ALREADY_RUNNING
    recovered = store.acquire_lease(_owner("two"), now + timedelta(seconds=6))
    assert recovered.status is HftLeaseStatus.ACQUIRED
    assert recovered.recovered_expired is True


def test_hft_lease_takeover_reconciles_only_predecessor_running_rows(tmp_path: Path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path, lease_ttl_seconds=5)
    first_time = datetime(2026, 1, 1, tzinfo=UTC)
    assert store.acquire_lease(_owner("one"), first_time).status is HftLeaseStatus.ACQUIRED
    store.start_run("old-run", mode="SHADOW", source_fingerprint="abc")
    with sqlite3.connect(path) as db:
        db.execute("UPDATE hft_runs SET started_at='2025-12-31T23:00:00Z' WHERE run_id='old-run'")
        db.commit()
    takeover = store.acquire_lease(_owner("two"), first_time + timedelta(seconds=6))

    assert takeover.status is HftLeaseStatus.ACQUIRED
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status,ended_at FROM hft_runs WHERE run_id='old-run'").fetchone()[0] == "STOPPED"


def test_hft_read_only_lease_probe_does_not_create_or_write(tmp_path: Path):
    path = tmp_path / "missing.sqlite3"
    store = HftShadowStore(path)
    assert store.read_only_active_lease(datetime.now(UTC)) is None
    assert not path.exists()


def test_read_only_provider_proxy_exposes_only_gated_read_methods():
    events = []

    class Provider:
        def initialize(self):
            events.append("initialize")
            return True

        def shutdown(self):
            events.append("shutdown")

        def get_tick(self, symbol):
            events.append(("tick", symbol))
            return "tick"

    proxy = ReadOnlyMt5ProviderProxy(Provider(), SerializedMt5OperationGate())
    assert proxy.initialize() is True
    assert proxy.get_tick("EURUSD") == "tick"
    proxy.shutdown()
    assert events == ["initialize", ("tick", "EURUSD"), "shutdown"]
    names = {name.casefold() for name in dir(proxy)}
    assert not any(
        token in name
        for name in names
        for token in ("order_send", "buy", "sell", "close_position", "modify_position")
    )


def test_read_only_provider_proxy_serializes_historical_ticks():
    calls = []

    class Provider:
        def get_ticks_range(self, symbol, start, end):
            calls.append((symbol, start, end))
            return ()

    proxy = ReadOnlyMt5ProviderProxy(Provider(), SerializedMt5OperationGate())
    assert proxy.get_ticks_range("EURUSD", "start", "end") == ()
    assert calls == [("EURUSD", "start", "end")]


def test_shared_consumer_shutdown_does_not_contend_with_hft_operation():
    class Provider:
        def initialize(self):
            return True

        def shutdown(self):
            raise AssertionError("consumer shutdown must not close shared session")

    shared = _SharedReadOnlyMt5Provider(Provider())
    proxy = ReadOnlyMt5ProviderProxy(shared, SerializedMt5OperationGate())
    gate = proxy._gate

    with gate.acquire("hft_tick"):
        proxy.shutdown()
