from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.phase11a_knowledge import fixture_source


def test_real_acceptance_plans_read_only_and_reports_unconfigured_teacher(
    monkeypatch, tmp_path: Path
):
    from scripts import phase11a_real_acceptance as acceptance

    source = fixture_source()
    monkeypatch.setattr(acceptance.Phase7KnowledgeSource, "open", lambda *args, **kwargs: source)
    report_path = tmp_path / "report.json"
    result = acceptance.run_real_acceptance(
        tmp_path / "phase7",
        topics=("order flow imbalance",),
        report_path=report_path,
    )

    assert result["status"] == "PLANNED"
    assert result["documents"] == 4
    assert result["chunks"] == len(source.blocks())
    assert result["packets"] >= 1
    assert result["generation_status"] == "DISTILLATION_TEACHER_NOT_CONFIGURED"
    assert result["network_attempts"] == result["mt5_calls"] == result["llm_calls"] == 0
    assert result["source_unchanged"] is True
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "PLANNED"


def test_real_acceptance_missing_source_is_visible(tmp_path: Path):
    from scripts.phase11a_real_acceptance import run_real_acceptance

    result = run_real_acceptance(tmp_path / "missing")
    assert result["status"] == "SourceUnavailableError"
    assert result["source_unchanged"] is True


def test_real_acceptance_rejects_report_inside_source_root(monkeypatch, tmp_path: Path):
    from scripts import phase11a_real_acceptance as acceptance

    source = fixture_source()
    root = tmp_path / "phase7"
    monkeypatch.setattr(acceptance.Phase7KnowledgeSource, "open", lambda *args, **kwargs: source)
    try:
        acceptance.run_real_acceptance(root, report_path=root / "report.json")
    except ValueError as exc:
        assert "outside the Phase 7 root" in str(exc)
    else:
        raise AssertionError("expected source-root report rejection")
