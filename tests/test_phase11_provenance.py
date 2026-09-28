from __future__ import annotations

import json
from pathlib import Path

import pytest


def _fixture(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"architectures": ["TinyLM"], "torch_dtype": "float32", "num_parameters": 7}),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def test_local_provenance_fingerprint_is_stable_and_descriptive(tmp_path: Path) -> None:
    from tradingagents.finetuning.provenance import ModelProvenance

    model = _fixture(tmp_path / "model")
    first = ModelProvenance.inspect(model, revision="local-snapshot", tokenizer_path=model)
    second = ModelProvenance.inspect(model, revision="local-snapshot", tokenizer_path=model)

    assert first.fingerprint == second.fingerprint
    assert first.architecture == ("TinyLM",)
    assert first.dtype == "float32"
    assert first.parameter_count == 7
    assert first.config_fingerprint
    assert first.tokenizer_fingerprint
    assert first.weight_fingerprint


def test_provenance_fingerprints_common_tokenizer_assets(tmp_path: Path) -> None:
    from tradingagents.finetuning.provenance import ModelProvenance

    model = _fixture(tmp_path / "model")
    (model / "vocab.json").write_text('{"EURUSD": 1}', encoding="utf-8")
    (model / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    first = ModelProvenance.inspect(model, revision="local-snapshot", tokenizer_path=model)
    (model / "vocab.json").write_text('{"EURUSD": 2}', encoding="utf-8")
    second = ModelProvenance.inspect(model, revision="local-snapshot", tokenizer_path=model)

    assert first.tokenizer_fingerprint != second.tokenizer_fingerprint


@pytest.mark.parametrize("revision", ["latest", "main", "refs/heads/main"])
def test_provenance_rejects_mutable_revision(tmp_path: Path, revision: str) -> None:
    from tradingagents.finetuning.provenance import (
        BaseModelRevisionUnpinnedError,
        ModelProvenance,
        ProvenanceError,
    )

    model = _fixture(tmp_path / "model")
    with pytest.raises(ProvenanceError, match="immutable|unpinned"):
        ModelProvenance.inspect(model, revision=revision)
    if revision == "latest":
        with pytest.raises(BaseModelRevisionUnpinnedError):
            ModelProvenance.inspect(model, revision=revision)


def test_remote_model_requires_an_immutable_revision() -> None:
    from tradingagents.finetuning.provenance import ModelProvenance, ProvenanceError

    with pytest.raises(ProvenanceError, match="immutable|unpinned|local"):
        ModelProvenance.inspect("org/model", revision=None)


def test_remote_model_is_never_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradingagents.finetuning.provenance import ModelProvenance, ProvenanceError

    class Forbidden:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("network/model loader must not be called")

    monkeypatch.setitem(__import__("sys").modules, "transformers", type("T", (), {"AutoConfig": Forbidden}))
    with pytest.raises(ProvenanceError, match="local"):
        ModelProvenance.inspect("org/model", revision="0123456789abcdef0123456789abcdef01234567")


def test_missing_local_snapshot_is_rejected(tmp_path: Path) -> None:
    from tradingagents.finetuning.provenance import ModelProvenance, ProvenanceError

    with pytest.raises(ProvenanceError, match="local|snapshot"):
        ModelProvenance.inspect(tmp_path / "missing", revision="local-snapshot")


def test_non_object_model_config_is_rejected(tmp_path: Path) -> None:
    from tradingagents.finetuning.provenance import ModelProvenance, ProvenanceError

    model = _fixture(tmp_path / "model")
    (model / "config.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="config"):
        ModelProvenance.inspect(model, revision="local-snapshot", tokenizer_path=model)


@pytest.mark.parametrize("architectures", [42, False, {"name": "TinyLM"}, ["TinyLM", 1], [""]])
def test_malformed_architectures_are_rejected(tmp_path: Path, architectures) -> None:
    from tradingagents.finetuning.provenance import ModelProvenance, ProvenanceError

    model = _fixture(tmp_path / "model")
    (model / "config.json").write_text(
        json.dumps({"architectures": architectures}), encoding="utf-8"
    )

    with pytest.raises(ProvenanceError, match="architectures"):
        ModelProvenance.inspect(model, revision="local-snapshot", tokenizer_path=model)


@pytest.mark.parametrize("dtype", [1, False, {}, []])
def test_malformed_dtype_is_rejected(tmp_path: Path, dtype) -> None:
    from tradingagents.finetuning.provenance import ModelProvenance, ProvenanceError

    model = _fixture(tmp_path / "model")
    (model / "config.json").write_text(
        json.dumps({"architectures": ["TinyLM"], "torch_dtype": dtype}),
        encoding="utf-8",
    )

    with pytest.raises(ProvenanceError, match="dtype"):
        ModelProvenance.inspect(model, revision="local-snapshot", tokenizer_path=model)
