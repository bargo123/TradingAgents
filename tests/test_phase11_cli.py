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


def test_training_config_accepts_nested_adapter_settings_and_revision(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "mode": "lora",
                "epochs": 1,
                "lora": {"r": 2, "alpha": 4, "target_modules": ["q_proj"]},
                "qlora": {"compute_dtype": "float16"},
            }
        ),
        encoding="utf-8",
    )
    resolved = cli._config(str(config), str(tmp_path / "base"), "local-snapshot")
    assert resolved.base_model_revision == "local-snapshot"
    assert resolved.lora.r == 2
    assert resolved.qlora.compute_dtype == "float16"


def test_train_validates_prepared_manifest_before_loading_tokenizer(monkeypatch, capsys, tmp_path):
    from tradingagents.finetuning import cli

    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("tokenizer must not load for invalid preparation")

    monkeypatch.setattr(cli, "_config", lambda *args: __import__("tradingagents.finetuning").finetuning.TrainingConfig(base_model="x"))
    monkeypatch.setitem(__import__("sys").modules, "transformers", type("T", (), {"AutoTokenizer": type("AT", (), {"from_pretrained": staticmethod(forbidden)})}))
    code, payload = _run(
        capsys,
        "train",
        "--prepared",
        str(tmp_path / "missing-prepared"),
        "--base-model",
        str(tmp_path / "base"),
        "--config",
        str(tmp_path / "config.json"),
        "--output-root",
        str(tmp_path / "runs"),
    )
    assert code != 0
    assert payload["status"] == "TRAINING_FAILED"
    assert not called
