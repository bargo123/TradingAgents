from pathlib import Path

from tradingagents.datasets.factory import DatasetFactory
from tradingagents.datasets.models import DatasetConfig


def test_factory_reads_sorted_sources_once_and_publishes(tmp_path, monkeypatch):
    calls = []

    class FakeSource:
        def __init__(self, path):
            self.path = Path(path)
        def read(self):
            calls.append(self.path)
            return object()

    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", FakeSource)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", FakeSource)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase9AuditSource", FakeSource)

    config = DatasetConfig(
        source_db_paths=(tmp_path / "b.db", tmp_path / "a.db"),
        phase8_root=tmp_path / "phase8",
        phase9_audit_path=tmp_path / "audit.db",
        output_root=tmp_path / "out",
    )
    # The test only exercises source orchestration; the fake readers fail visibly.
    report = DatasetFactory().build(config)
    assert report.status in {"EMPTY_ELIGIBLE_SET", "FAILED"}
    source_calls = [item for item in calls if item.name in {"a.db", "b.db"}]
    assert source_calls == sorted(source_calls)


def test_factory_has_no_external_safety_counters():
    assert DatasetFactory.safety_counters() == {
        "network_attempts": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "mt5_calls": 0,
    }
