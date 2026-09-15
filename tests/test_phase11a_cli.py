import json

from tradingagents.distillation import cli


def test_distill_without_teacher_fails_closed(capsys, tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"generation_id": "g", "packets": [], "request_fingerprint": "f"}),
        encoding="utf-8",
    )
    assert cli.main(["distill", "--plan", str(plan), "--output-root", str(tmp_path / "out")]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "DISTILLATION_TEACHER_NOT_CONFIGURED"


def test_inspect_does_not_require_teacher(capsys, tmp_path):
    generation = tmp_path / "generation-x"
    generation.mkdir()
    (generation / "manifest.json").write_text(
        json.dumps({"generation_id": "generation-x", "status": "EMPTY", "counts": {}}),
        encoding="utf-8",
    )
    assert cli.main(["inspect", "--generation", str(generation)]) == 0
    assert json.loads(capsys.readouterr().out)["generation_id"] == "generation-x"
