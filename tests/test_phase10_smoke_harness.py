from __future__ import annotations

import json
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


def test_smoke_harness_refuses_existing_artifact_root(capsys, tmp_path: Path) -> None:
    artifacts = make_real_fixtures(tmp_path)
    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "catalog.sqlite3").write_bytes(b"user-data")
    assert main(_args(artifacts, output)) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "OUTPUT_ROOT_NOT_FRESH"
    assert (output / "catalog.sqlite3").read_bytes() == b"user-data"
