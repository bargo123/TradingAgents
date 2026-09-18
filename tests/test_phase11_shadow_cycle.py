from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.finetuning.shadow_cycle import (
    ShadowCycleConfig,
    ShadowCycleError,
    run_shadow_cycle,
)

UTC = timezone.utc


def _config(tmp_path: Path, *, mode: str = "collect") -> ShadowCycleConfig:
    return ShadowCycleConfig(
        db_path=tmp_path / "shadow.db",
        phase8_root=tmp_path / "phase8",
        phase10_output_root=tmp_path / "phase10",
        phase9_audit_path=tmp_path / "phase9-audit.sqlite3",
        phase7_root=tmp_path / "phase7",
        embedding_model_path=tmp_path / "embedding",
        mode=mode,
    )


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_collect_orders_phase8_initialization_watch_and_phase8_refresh(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    config = _config(tmp_path)

    def fake_sync(current: ShadowCycleConfig):
        calls.append("phase8")
        return {"experience_count": len(calls)}

    def fake_runner(command, env):
        calls.append("watch" if "forex_watch" in " ".join(command) else "other")
        return _Result()

    monkeypatch.setattr("tradingagents.finetuning.shadow_cycle._sync_phase8", fake_sync)

    result = run_shadow_cycle(config, command_runner=fake_runner)

    assert calls == ["phase8", "watch", "phase8"]
    assert result["status"] == "COLLECTED"
    assert result["executed"] is False


def test_evaluate_remains_pending_before_shortest_approved_horizon(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path, mode="evaluate")
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    with sqlite3.connect(config.db_path) as db:
        db.execute(
            "CREATE TABLE shadow_decisions (decision_id TEXT, analysis_snapshot_timestamp TEXT, decision_reference_timestamp TEXT)"
        )
        db.execute(
            "INSERT INTO shadow_decisions VALUES (?, ?, ?)",
            ("d1", (now - timedelta(seconds=30)).isoformat(), None),
        )
        db.commit()

    calls: list[str] = []

    def fake_sync(current: ShadowCycleConfig):
        calls.append("phase8")
        return {}

    def fake_runner(command, env):
        calls.append("evaluate")
        return _Result()

    monkeypatch.setattr("tradingagents.finetuning.shadow_cycle._sync_phase8", fake_sync)

    result = run_shadow_cycle(
        config,
        command_runner=fake_runner,
        now=now,
    )

    assert result["status"] == "PENDING"
    assert result["evaluation_status"] == "PENDING"
    assert "evaluate" not in calls


def test_nonempty_phase10_root_is_rejected_before_any_command(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.phase10_output_root.mkdir(parents=True)
    (config.phase10_output_root / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ShadowCycleError, match="fresh"):
        run_shadow_cycle(config, command_runner=lambda *_: _Result())


def test_collector_failure_stops_without_refresh(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_sync(current: ShadowCycleConfig):
        calls.append("phase8")
        return {}

    def fake_runner(command, env):
        calls.append("watch")
        return _Result(returncode=1, stderr="graph failed")

    monkeypatch.setattr("tradingagents.finetuning.shadow_cycle._sync_phase8", fake_sync)

    with pytest.raises(ShadowCycleError, match="forex-watch"):
        run_shadow_cycle(config, command_runner=fake_runner)
    assert calls == ["phase8", "watch"]
