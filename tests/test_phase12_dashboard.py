from datetime import timezone

from tradingagents.forex.hft.dashboard import read_hft_dashboard
from tradingagents.forex.hft.store import HftShadowStore

UTC = timezone.utc


def test_hft_dashboard_reads_scalar_shadow_metrics_read_only(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run-1", mode="REPLAY", source_fingerprint="abc")
    snapshot = read_hft_dashboard(path)
    assert snapshot.executed is False
    assert snapshot.runs == 1
    assert snapshot.ticks == 0


def test_hft_dashboard_missing_root_is_explicit(tmp_path):
    snapshot = read_hft_dashboard(tmp_path / "missing.sqlite3")
    assert snapshot.status == "NOT_INITIALIZED"
    assert snapshot.executed is False

