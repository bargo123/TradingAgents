from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from scripts.phase10_dataset_smoke import main
from tests.fixtures.dataset_factory import make_real_fixtures


def _args(artifacts: dict[str, Path], output: Path) -> list[str]:
    return [
        "--source-db", str(artifacts["source"]),
        "--phase8-root", str(artifacts["phase8"]),
        "--phase9-audit", str(artifacts["audit"]),
        "--output-root", str(output),
    ]


def test_smoke_harness_reports_bounded_offline_build(capsys, tmp_path: Path) -> None:
    artifacts = make_real_fixtures(tmp_path)
    assert main(_args(artifacts, tmp_path / "fresh-output")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["offline"] is True
    assert report["safety"] == {
        "network_attempts": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "mt5_calls": 0,
    }


def test_smoke_harness_reports_actual_source_fingerprints_for_empty_build(
    capsys, tmp_path: Path
) -> None:
    """An empty eligible set still proves which immutable sources were read."""
    artifacts = make_real_fixtures(tmp_path)
    with sqlite3.connect(artifacts["source"]) as db:
        db.execute(
            "UPDATE shadow_decisions SET action=NULL, normalization_status='FAILED', "
            "decision_context_status='INCOMPLETE'"
        )
        db.commit()

    assert main(_args(artifacts, tmp_path / "empty-output")) == 0
    report = json.loads(capsys.readouterr().out)
    fingerprints = report["source_fingerprints"]
    assert report["status"] == "EMPTY_ELIGIBLE_SET"
    assert set(fingerprints) == {"phase56", "phase8", "phase9"}
    assert fingerprints["phase56"]["canonical_path"] == str(artifacts["source"].resolve())
    assert fingerprints["phase8"]["canonical_path"] == str(
        (artifacts["phase8"] / "catalog.sqlite3").resolve()
    )
    assert fingerprints["phase9"]["canonical_path"] == str(artifacts["audit"].resolve())


def test_smoke_harness_refuses_existing_artifact_root(capsys, tmp_path: Path) -> None:
    artifacts = make_real_fixtures(tmp_path)
    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "catalog.sqlite3").write_bytes(b"user-data")
    assert main(_args(artifacts, output)) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "OUTPUT_ROOT_NOT_FRESH"
    assert (output / "catalog.sqlite3").read_bytes() == b"user-data"


def test_smoke_script_supports_direct_invocation() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/phase10_dataset_smoke.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Phase 10 dataset smoke" in result.stdout
