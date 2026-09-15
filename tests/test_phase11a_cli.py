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


def test_inspect_reports_lesson_distributions_and_source_coverage(capsys, tmp_path):
    generation = tmp_path / "generation-x"
    generation.mkdir()
    (generation / "manifest.json").write_text(
        json.dumps(
            {
                "generation_id": "generation-x",
                "status": "PUBLISHED",
                "counts": {"examples": 2, "exclusions": 1},
                "split_counts": {"train": 1, "validation": 1, "test": 0},
                "exclusion_reasons": {"SCHEMA_INVALID": 1},
                "metadata": {
                    "lesson_type_distribution": {"DEFINITION": 2},
                    "topic_distribution": {"OFI": 2},
                    "difficulty_distribution": {"FOUNDATIONAL": 2},
                    "source_coverage": {"documents": 1, "sections": 2, "chunks": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    assert cli.main(["inspect", "--generation", str(generation)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["lesson_type_distribution"] == {"DEFINITION": 2}
    assert result["source_coverage"]["sections"] == 2


def test_plan_rejects_output_inside_phase7_root(monkeypatch, capsys, tmp_path):
    from tests.fixtures.phase11a_knowledge import fixture_source

    source_root = tmp_path / "phase7"
    source_root.mkdir()
    monkeypatch.setattr(cli.Phase7KnowledgeSource, "open", lambda *args, **kwargs: fixture_source())
    assert (
        cli.main(
            [
                "plan",
                "--phase7-root",
                str(source_root),
                "--output",
                str(source_root / "plan.json"),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["status"] == "ValueError"
