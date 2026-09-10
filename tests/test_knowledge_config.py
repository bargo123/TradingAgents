"""Validation tests for the local/offline Phase 7 configuration."""

from dataclasses import FrozenInstanceError

import pytest

from tradingagents.knowledge.config import KnowledgeConfig


def test_config_resolves_paths_and_enforces_offline_ocr_disabled_policy(tmp_path):
    source = tmp_path / "new books"
    source.mkdir()
    artifacts = tmp_path / "artifacts"
    config = KnowledgeConfig(source_root=source, artifact_root=artifacts)

    assert config.source_root == source.resolve()
    assert config.artifact_root == artifacts.resolve()
    assert config.docling_artifacts_path == (artifacts / "docling").resolve()
    assert config.docling_offline is True
    assert config.docling_do_ocr is False
    assert config.offline is True
    assert config.worker_count == 1
    assert config.embedding_batch_size > 0

    with pytest.raises(FrozenInstanceError):
        config.offline = False


def test_config_rejects_source_or_artifact_policy_violations(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="source_root"):
        KnowledgeConfig(source_root=tmp_path / "missing", artifact_root=tmp_path / "out")

    with pytest.raises(ValueError, match="outside"):
        KnowledgeConfig(source_root=source, artifact_root=source / "artifacts")

    with pytest.raises(ValueError, match="outside"):
        KnowledgeConfig(
            source_root=source,
            artifact_root=tmp_path / "out",
            docling_artifacts_path=source / "docling",
        )

    with pytest.raises(ValueError, match="docling_offline"):
        KnowledgeConfig(source_root=source, artifact_root=tmp_path / "out", docling_offline=False)

    with pytest.raises(ValueError, match="OCR"):
        KnowledgeConfig(source_root=source, artifact_root=tmp_path / "out", docling_do_ocr=True)


@pytest.mark.parametrize("worker_count", (0, 7))
def test_config_rejects_worker_count_outside_cpu_bound(tmp_path, worker_count):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="worker_count"):
        KnowledgeConfig(
            source_root=source,
            artifact_root=tmp_path / "out",
            worker_count=worker_count,
        )


def test_config_rejects_nonpositive_batch_and_invalid_formula_policy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="embedding_batch_size"):
        KnowledgeConfig(source_root=source, artifact_root=tmp_path / "out", embedding_batch_size=0)

    with pytest.raises(ValueError, match="formula"):
        KnowledgeConfig(
            source_root=source,
            artifact_root=tmp_path / "out",
            formula_enrichment_enabled=True,
        )

    formula_path = tmp_path / "formula-artifacts"
    formula_path.mkdir()
    config = KnowledgeConfig(
        source_root=source,
        artifact_root=tmp_path / "out-2",
        formula_enrichment_enabled=True,
        formula_artifacts_path=formula_path,
    )
    assert config.formula_artifacts_path == formula_path.resolve()


def test_config_serialization_is_json_safe_and_fingerprint_changes_on_semantic_change(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = KnowledgeConfig(source_root=source, artifact_root=tmp_path / "out")
    payload = config.to_dict()
    assert isinstance(payload["source_root"], str)
    assert isinstance(payload["artifact_root"], str)
    assert config.config_fingerprint
    assert KnowledgeConfig.from_dict(payload) == config

    changed = KnowledgeConfig(
        source_root=source,
        artifact_root=tmp_path / "out",
        embedding_model_id="different-local-model",
    )
    assert changed.config_fingerprint != config.config_fingerprint
