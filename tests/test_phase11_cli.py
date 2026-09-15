import json
import sys

from tradingagents.finetuning import cli


def _run(capsys, *args):
    code = cli.main(list(args))
    output = capsys.readouterr()
    return code, json.loads(output.out)


def test_inspect_is_metadata_only_and_machine_readable(capsys, tmp_path, monkeypatch):
    generation = tmp_path / "generation-1"
    generation.mkdir()
    (generation / "manifest.json").write_text(
        json.dumps({"dataset_id": generation.name, "status": "EMPTY_ELIGIBLE_SET", "split_counts": {"train": 0, "validation": 0}}),
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    code, payload = _run(capsys, "inspect", "--generation", str(generation))
    assert code == 0
    assert payload["status"] == "EMPTY_ELIGIBLE_SET"
    assert payload["generation_id"] == generation.name


def test_train_requires_explicit_base_model_and_config(capsys, tmp_path):
    code, payload = _run(capsys, "train", "--prepared", str(tmp_path), "--output-root", str(tmp_path / "out"))
    assert code != 0
    assert payload["status"] == "TRAINING_FAILED"
    assert "base model" in payload["message"].lower()


def test_invalid_generation_is_fail_closed(capsys, tmp_path):
    code, payload = _run(capsys, "inspect", "--generation", str(tmp_path / "missing"))
    assert code != 0
    assert payload["status"] == "PHASE10_INVALID"
    assert "traceback" not in json.dumps(payload).lower()


def test_stock_cli_import_remains_available():
    import cli.main  # noqa: F401
