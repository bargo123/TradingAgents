from __future__ import annotations

import json
from pathlib import Path

from tradingagents.finetuning import validation
from tradingagents.finetuning.artifacts import publish_run


def _run(tmp_path: Path, *, leaked: bool = False) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    row = {"example_id": "x", "split": "test" if leaked else "train", "messages": [], "target": {}}
    for name in ("train.sft.jsonl", "validation.sft.jsonl"):
        (prepared / name).write_text(json.dumps(row) + "\n", encoding="utf-8")
    pm = {"status": "PREPARED", "generation_id": "g", "source_fingerprints": {"phase56": 1},
          "formatter_version": "f", "tokenization_policy_version": "t",
          "files": {name: {"sha256": validation._digest(prepared / name)} for name in ("train.sft.jsonl", "validation.sft.jsonl")}}
    (prepared / "manifest.json").write_text(json.dumps(pm), encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2,
        "target_modules": ["q_proj"], "base_model_name_or_path": str(tmp_path / "base")}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    run = publish_run(tmp_path / "runs", run_id="run-x", config={},
                      dataset={"generation_id": "g", "generation_path": str(tmp_path / "generation"),
                               "source_fingerprints": {"phase56": 1}}, metrics={},
                      prepared={name: prepared / name for name in ("train.sft.jsonl", "validation.sft.jsonl")},
                      adapter=adapter, environment={"variables": {}})
    # publish_run copies only the data files; install the prepared manifest as
    # part of the immutable test fixture and refresh its artifact hash entry.
    (run / "prepared" / "manifest.json").write_text(json.dumps(pm), encoding="utf-8")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    from tradingagents.finetuning.fingerprints import directory_hash
    manifest["prepared_hashes"] = directory_hash(run / "prepared")
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_valid_tiny_adapter_passes_without_optional_dependencies(tmp_path, monkeypatch):
    class Generation:
        generation_id = "g"
        manifest_hash = ""
        source_fingerprints = {"phase56": 1}
        train = (1,)
        validation = ()

    monkeypatch.setattr(validation.Phase10Generation, "open", classmethod(lambda cls, path: Generation()))
    monkeypatch.setattr(validation, "_reload_smoke", lambda *args: None)
    report = validation.validate_run(_run(tmp_path))
    assert report.valid


def test_tampered_adapter_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "_check_generation", lambda *args: None)
    run = _run(tmp_path)
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
    assert validation.validate_run(run).status.value == "ADAPTER_INVALID"


def test_prepared_test_leakage_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "_check_generation", lambda *args: None)
    assert not validation.validate_run(_run(tmp_path, leaked=True)).valid


def test_missing_adapter_config_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "_check_generation", lambda *args: None)
    run = _run(tmp_path)
    (run / "adapter" / "adapter_config.json").unlink()
    assert not validation.validate_run(run).valid


def test_reload_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "_check_generation", lambda *args: None)
    monkeypatch.setattr(validation, "_reload_smoke", lambda *args: args[-1].append("PEFT adapter reload/forward failed"))
    assert not validation.validate_run(_run(tmp_path)).valid
