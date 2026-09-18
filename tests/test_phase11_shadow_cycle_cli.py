from __future__ import annotations

import json
from pathlib import Path

from scripts import phase11_shadow_cycle


def test_parser_defaults_to_eurusd_and_shortest_horizon(tmp_path: Path) -> None:
    args = phase11_shadow_cycle.build_parser().parse_args(
        [
            "collect",
            "--db-path",
            str(tmp_path / "shadow.db"),
            "--phase8-root",
            str(tmp_path / "phase8"),
            "--phase10-output-root",
            str(tmp_path / "phase10"),
        ]
    )
    assert args.symbol == "EURUSD"
    assert args.horizon_seconds == 300
    assert args.analysts == "market,news"


def test_main_prints_shadow_banner_and_bounded_json(monkeypatch, capsys, tmp_path: Path) -> None:
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {"status": "COLLECTED", "executed": False, "symbol": config.symbol}

    monkeypatch.setattr(phase11_shadow_cycle, "run_shadow_cycle", fake_run)
    exit_code = phase11_shadow_cycle.main(
        [
            "collect",
            "--db-path",
            str(tmp_path / "shadow.db"),
            "--phase8-root",
            str(tmp_path / "phase8"),
            "--phase10-output-root",
            str(tmp_path / "phase10"),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "SHADOW ONLY — NO ORDER WILL BE SENT" in output
    payload = json.loads(output.splitlines()[-1])
    assert payload == {"executed": False, "status": "COLLECTED", "symbol": "EURUSD"}
    assert captured["config"].phase8_root == (tmp_path / "phase8").resolve()


def test_main_rejects_missing_frozen_phase7_root(monkeypatch, tmp_path: Path) -> None:
    called = False

    def fail_if_called(config):
        nonlocal called
        called = True
        raise AssertionError("cycle must not start")

    monkeypatch.setattr(phase11_shadow_cycle, "run_shadow_cycle", fail_if_called)
    exit_code = phase11_shadow_cycle.main(
        [
            "collect",
            "--db-path",
            str(tmp_path / "shadow.db"),
            "--phase8-root",
            str(tmp_path / "phase8"),
            "--phase10-output-root",
            str(tmp_path / "phase10"),
            "--phase7-root",
            str(tmp_path / "missing-phase7"),
        ]
    )
    assert exit_code == 2
    assert called is False
