from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_provenance_is_stable_for_local_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"architectures": ["TinyLM"], "torch_dtype": "float32"}))
    (model / "tokenizer.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    from tradingagents.finetuning.provenance import ModelProvenance

    first = ModelProvenance.inspect(model, revision="local-snapshot", tokenizer_path=model)
    second = ModelProvenance.inspect(model, revision="local-snapshot", tokenizer_path=model)
    assert first.fingerprint == second.fingerprint
    assert first.parameter_count >= 0


def test_provenance_rejects_mutable_revision(tmp_path: Path) -> None:
    from tradingagents.finetuning.provenance import ModelProvenance

    with pytest.raises(Exception, match="immutable|unpinned"):
        ModelProvenance.inspect(tmp_path, revision="latest")


def test_qlora_cpu_fails_without_full_precision_fallback(monkeypatch) -> None:
    from tradingagents.finetuning.qlora import QloraCapabilityError, check_qlora_capability

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(QloraCapabilityError, match="GPU_REQUIRED"):
        check_qlora_capability()


def test_request_fingerprint_and_file_hash_are_deterministic(tmp_path: Path) -> None:
    from tradingagents.finetuning.fingerprints import file_sha256, request_fingerprint

    path = tmp_path / "x"
    path.write_bytes(b"abc")
    assert file_sha256(path) == file_sha256(path)
    assert request_fingerprint({"b": 2, "a": 1}) == request_fingerprint({"a": 1, "b": 2})


def test_package_import_does_not_load_optional_training_modules() -> None:
    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tradingagents.finetuning; print('peft' in sys.modules, 'bitsandbytes' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "False False"
