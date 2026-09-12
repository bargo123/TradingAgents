from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradingagents.dataflows.mt5.models import (
    ForexMarketSnapshot,
    Mt5AccountInfo,
    Mt5Bar,
    Mt5Position,
    Mt5SymbolInfo,
)
from tradingagents.forex.context import snapshot_to_dict
from tradingagents.forex.evidence_context import (
    EvidenceBundleStatus,
    EvidenceContext,
    EvidenceIntegrationStatus,
)
from tradingagents.forex.evidence_replay import (
    EvidenceReplayConfig,
    SavedSnapshotCodec,
    SavedSnapshotReplay,
    SnapshotReplayError,
)


def _snapshot() -> ForexMarketSnapshot:
    timestamp = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    bar = Mt5Bar(timestamp, 1.1, 1.101, 1.099, 1.1005, 12, 2, 12)
    return ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSDm",
        bid=1.1,
        ask=1.1002,
        spread=0.0002,
        spread_points=2.0,
        m1_candles=(bar,),
        m5_candles=(bar,),
        m15_candles=(bar,),
        h1_candles=(bar,),
        account=Mt5AccountInfo(login=7, server="demo", currency="USD", balance=1000.0),
        positions=(Mt5Position(ticket=1, symbol="EURUSDm", volume=0.1, time=timestamp),),
        symbol_info=Mt5SymbolInfo(name="EURUSDm", digits=5, point=0.00001, currency_base="EUR", currency_profit="USD"),
    )


def _row(snapshot: ForexMarketSnapshot | None = None) -> dict:
    snapshot = snapshot or _snapshot()
    return {
        "decision_id": "decision-1",
        "resolved_symbol": snapshot.symbol,
        "analysis_profile": "INTRADAY",
        "snapshot_json": json.dumps(snapshot_to_dict(snapshot), sort_keys=True),
    }


class _FakeRunner:
    def __init__(self, calls: list[dict], *, generations: tuple[str, str]):
        self.calls = calls
        self.generations = generations
        self.run_called = False

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        context = None
        if kwargs.get("forex_evidence_enabled"):
            context = EvidenceContext(
                integration_status=EvidenceIntegrationStatus.INJECTED,
                bundle_status=EvidenceBundleStatus.COMPLETE,
                as_of=kwargs["snapshot"].timestamp,
                knowledge_generation_id=self.generations[0],
                experience_generation_id=self.generations[1],
                rendered_context="bounded evidence",
                rendered_context_hash=hashlib.sha256(b"bounded evidence").hexdigest(),
            )
        return {
            "normalized_action": "BUY" if kwargs.get("forex_evidence_enabled") else "HOLD",
            "evidence_context": context,
            "analysis_telemetry": {"telemetry_status": "AVAILABLE"},
        }

    def run(self, **_kwargs):
        self.run_called = True
        raise AssertionError("replay must not call run")


def _config(tmp_path: Path) -> EvidenceReplayConfig:
    db = tmp_path / "source.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE shadow_decisions (decision_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)")
        conn.execute("INSERT INTO shadow_decisions VALUES (?, ?)", ("decision-1", _row()["snapshot_json"]))
    return EvidenceReplayConfig(
        source_decision_id="decision-1",
        profile="INTRADAY",
        analysts=("market", "news"),
        provider="ollama",
        models={"quick": "qwen", "deep": "qwen"},
        model_settings={"temperature": 0},
        pinned_phase7_generation_id="p7",
        pinned_phase8_generation_id="p8",
        source_database_path=db,
    )


def test_saved_snapshot_codec_validates_row():
    snapshot = SavedSnapshotCodec.from_source_row(_row())
    assert snapshot.symbol == "EURUSDm"
    assert snapshot.timestamp.tzinfo == timezone.utc
    assert isinstance(snapshot.account, Mt5AccountInfo)
    assert isinstance(snapshot.symbol_info, Mt5SymbolInfo)
    assert isinstance(snapshot.m1_candles[0], Mt5Bar)
    assert isinstance(snapshot.positions[0], Mt5Position)

    with pytest.raises(SnapshotReplayError):
        SavedSnapshotCodec.from_source_row({"snapshot_json": "{}"})


def test_replay_uses_identical_snapshot_fingerprint(tmp_path: Path):
    calls: list[dict] = []
    runner = _FakeRunner(calls, generations=("p7", "p8"))
    replay = SavedSnapshotReplay(runner_factory=lambda **_: runner, generation_provider=lambda: ("p7", "p8"))
    snapshot = SavedSnapshotCodec.from_source_row(_row())
    result = replay.run(snapshot, snapshot_bytes=b"snapshot-bytes", config=_config(tmp_path))
    assert result.snapshot_fingerprint == hashlib.sha256(b"snapshot-bytes").hexdigest()
    assert calls[0]["snapshot_bytes"] is calls[1]["snapshot_bytes"]
    assert calls[0]["snapshot_fingerprint"] == calls[1]["snapshot_fingerprint"] == result.snapshot_fingerprint


def test_replay_runs_baseline_then_evidence_sequentially(tmp_path: Path):
    calls: list[dict] = []
    runner = _FakeRunner(calls, generations=("p7", "p8"))
    replay = SavedSnapshotReplay(runner_factory=lambda **_: runner, generation_provider=lambda: ("p7", "p8"))
    report = replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=b"bytes", config=_config(tmp_path))
    assert [call["forex_evidence_enabled"] for call in calls] == [False, True]
    assert report.baseline_action == "HOLD"
    assert report.evidence_action == "BUY"
    assert report.did_action_change is True
    assert runner.run_called is False


def test_replay_does_not_change_source_schema_or_rows(tmp_path: Path):
    config = _config(tmp_path)
    before = config.source_database_path.read_bytes()
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")), generation_provider=lambda: ("p7", "p8"))
    replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=b"bytes", config=config)
    assert config.source_database_path.read_bytes() == before
    with sqlite3.connect(config.source_database_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM shadow_decisions").fetchone()[0] == 1


def test_replay_is_not_a_normal_opportunity(tmp_path: Path):
    runner = _FakeRunner([], generations=("p7", "p8"))
    replay = SavedSnapshotReplay(runner_factory=lambda **_: runner, generation_provider=lambda: ("p7", "p8"))
    report = replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=b"bytes", config=_config(tmp_path))
    assert not hasattr(report, "decision")
    assert runner.run_called is False


def test_replay_pins_phase7_and_phase8_generations(tmp_path: Path):
    calls: list[dict] = []
    runner = _FakeRunner(calls, generations=("p7", "p8"))
    replay = SavedSnapshotReplay(runner_factory=lambda **_: runner, generation_provider=lambda: ("p7", "p8"))
    replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=b"bytes", config=_config(tmp_path))
    assert all(call["pinned_phase7_generation_id"] == "p7" for call in calls)
    assert all(call["pinned_phase8_generation_id"] == "p8" for call in calls)


def test_knowledge_generation_change_invalidates_replay(tmp_path: Path):
    generations = iter((("p7", "p8"), ("changed", "p8")))
    calls: list[dict] = []
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner(calls, generations=("p7", "p8")), generation_provider=lambda: next(generations))
    report = replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=b"bytes", config=_config(tmp_path))
    assert report.comparison_status == "INVALID_GENERATION_CHANGED"
    assert report.did_action_change is None


def test_experience_generation_change_invalidates_replay(tmp_path: Path):
    generations = iter((("p7", "p8"), ("p7", "changed")))
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")), generation_provider=lambda: next(generations))
    report = replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=b"bytes", config=_config(tmp_path))
    assert report.comparison_status == "INVALID_GENERATION_CHANGED"
    assert report.did_action_change is None


def test_replay_report_excludes_prompts_and_reasoning(tmp_path: Path):
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")), generation_provider=lambda: ("p7", "p8"))
    report = replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=b"bytes", config=_config(tmp_path))
    payload = report.to_dict()
    forbidden = {"prompt", "completion", "reasoning", "private_reasoning"}
    assert not forbidden.intersection(payload)
    assert all(not any(word in str(key).lower() for word in forbidden) for key in payload)
