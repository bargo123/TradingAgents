from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.finetuning.artifacts import ArtifactError, publish_run, validate_run_hashes
from tradingagents.finetuning.fingerprints import redact_environment, request_fingerprint


def test_request_hash_and_environment_redaction_are_deterministic() -> None:
    assert request_fingerprint({"b": 2, "a": 1}) == request_fingerprint({"a": 1, "b": 2})
    env = redact_environment({"OPENAI_API_KEY": "secret", "LANG": "en_US"})
    assert "secret" not in json.dumps(env)
    assert env["redacted_variable_count"] == 1


def test_environment_secret_values_are_not_persisted_even_under_safe_names() -> None:
    env = redact_environment({"NORMAL_SETTING": "api_key=secret-value", "LANG": "en_US"})
    assert "secret-value" not in json.dumps(env)
    assert env["redacted_variable_count"] == 1


def test_environment_versions_records_available_cuda_identity_without_importing_it(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            return f"Fake GPU {index}"

    fake_torch = SimpleNamespace(
        __version__="fake-torch",
        version=SimpleNamespace(cuda="12.4"),
        cuda=FakeCuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from tradingagents.finetuning.fingerprints import environment_versions

    result = environment_versions()
    assert result["cuda_available"] is True
    assert result["cuda_version"] == "12.4"
    assert result["gpu_count"] == 2
    assert result["gpu_models"] == ["Fake GPU 0", "Fake GPU 1"]


def test_publish_layout_hashes_and_create_only(tmp_path: Path) -> None:
    prepared = tmp_path / "train.sft.jsonl"
    prepared.write_text('{"example_id":"x"}\n', encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    run = publish_run(tmp_path / "runs", run_id="run-fixed", config={"seed": 1},
                      dataset={"generation_id": "g"}, metrics={"steps": 1},
                      prepared={"train.sft.jsonl": prepared}, adapter=adapter,
                      environment={"python": "3.10", "variables": {}})
    assert all((run / name).exists() for name in ("run_manifest.json", "resolved_config.json",
                                                   "dataset_manifest.json", "metrics.json", "environment.json"))
    assert (run / "prepared" / "train.sft.jsonl").exists()
    assert validate_run_hashes(run)
    with pytest.raises(ArtifactError, match="immutable"):
        publish_run(tmp_path / "runs", run_id="run-fixed", config={}, dataset={}, metrics={})


def test_tampering_is_detected(tmp_path: Path) -> None:
    source = tmp_path / "x"
    source.write_text("safe", encoding="utf-8")
    run = publish_run(tmp_path / "runs", config={}, dataset={}, metrics={},
                      prepared={"x": source}, environment={"variables": {}})
    (run / "prepared" / "x").write_text("tampered", encoding="utf-8")
    assert not validate_run_hashes(run)


def test_hash_validation_rejects_path_traversal_in_tampered_manifest(tmp_path: Path) -> None:
    source = tmp_path / "x"
    source.write_text("safe", encoding="utf-8")
    run = publish_run(
        tmp_path / "runs",
        config={},
        dataset={},
        metrics={},
        prepared={"x": source},
        environment={"variables": {}},
    )
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepared_hashes"] = {"../../x": "anything"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not validate_run_hashes(run)


def test_hash_validation_rejects_unlisted_artifact_files(tmp_path: Path) -> None:
    source = tmp_path / "x"
    source.write_text("safe", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "weights").write_text("safe", encoding="utf-8")
    run = publish_run(
        tmp_path / "runs",
        config={},
        dataset={},
        metrics={},
        prepared={"x": source},
        adapter=adapter,
        environment={"variables": {}},
    )
    (run / "prepared" / "extra").write_text("unexpected", encoding="utf-8")
    assert not validate_run_hashes(run)
