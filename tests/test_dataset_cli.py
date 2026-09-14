import json
from pathlib import Path

import pytest

from tradingagents.datasets import cli


def test_build_emits_bounded_json_and_calls_factory(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source.db"
    phase8 = tmp_path / "phase8"
    output = tmp_path / "output"
    source.touch()
    phase8.mkdir()

    class FakeFactory:
        def build(self, config):
            from tradingagents.datasets.models import BuildReport

            assert config.output_root == output.resolve()
            return BuildReport("EMPTY_ELIGIBLE_SET")

    monkeypatch.setattr(cli, "DatasetFactory", FakeFactory)
    assert cli.main([
        "build", "--source-db", str(source), "--phase8-root", str(phase8),
        "--output-root", str(output), "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "EMPTY_ELIGIBLE_SET"
    assert len(capsys.readouterr().err) == 0


def test_build_requires_explicit_paths(capsys):
    assert cli.main(["build"]) != 0
    assert "usage" in capsys.readouterr().err.lower()


def test_build_rejects_output_inside_input_before_factory(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source.db"
    phase8 = tmp_path / "phase8"
    output = phase8 / "generated"
    source.touch()
    phase8.mkdir()
    monkeypatch.setattr(cli, "DatasetFactory", lambda: pytest.fail("factory must not run"))
    assert cli.main([
        "build", "--source-db", str(source), "--phase8-root", str(phase8),
        "--output-root", str(output),
    ]) != 0
    assert "overlap" in capsys.readouterr().err.lower()


def test_status_empty_is_explicit_and_does_not_construct_runtime(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DatasetFactory", lambda: pytest.fail("factory must not run"))
    assert cli.main(["status", "--output-root", str(tmp_path / "missing"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "EMPTY",
        "output_root": str((tmp_path / "missing").resolve()),
        "generations": [],
    }


def test_status_lists_published_generations_without_writers(monkeypatch, tmp_path, capsys):
    output = tmp_path / "output"
    generation = output / "generation-a"
    generation.mkdir(parents=True)
    (generation / "manifest.json").write_text('{"status":"PUBLISHED","examples":2}\n', encoding="utf-8")
    monkeypatch.setattr(cli, "write_generation", lambda *a, **k: pytest.fail("writer must not run"), raising=False)
    assert cli.main(["status", "--output-root", str(output), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "OK"
    assert payload["generations"][0]["generation_id"] == "generation-a"


def test_validate_reports_generation_failure(tmp_path, capsys):
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / "manifest.json").write_text("{}\n", encoding="utf-8")
    assert cli.main(["validate", "--generation", str(generation), "--json"]) != 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["errors"]


def test_stock_entrypoint_remains_unchanged():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'tradingagents = "cli.main:app"' in text
