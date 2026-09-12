import pytest

from scripts.experience_phase8_smoke import (
    ExperienceArtifactNotEmptyError,
    NetworkAttempt,
    OfflineNetworkGuard,
    run_smoke,
)
from tradingagents.experience.errors import Phase7KnowledgeUnavailableError


def test_smoke_requires_explicit_existing_phase7_configuration(tmp_path):
    with pytest.raises((SystemExit, ValueError)):
        run_smoke(
            tmp_path / "source.sqlite3",
            experience_artifact_root=tmp_path / "experience",
            knowledge_artifact_root=None,
            knowledge_embedding_model_path=None,
            offline=True,
        )


def test_smoke_rejects_existing_phase8_state(tmp_path):
    root = tmp_path / "experience"
    root.mkdir()
    (root / "catalog.sqlite3").touch()
    with pytest.raises(ExperienceArtifactNotEmptyError):
        run_smoke(
            tmp_path / "source.sqlite3",
            experience_artifact_root=root,
            knowledge_artifact_root=tmp_path / "knowledge",
            knowledge_embedding_model_path=tmp_path / "model",
            offline=True,
        )


def test_smoke_rejects_existing_empty_phase8_directory(tmp_path):
    root = tmp_path / "experience"
    root.mkdir()
    with pytest.raises(ExperienceArtifactNotEmptyError):
        run_smoke(
            tmp_path / "source.sqlite3",
            experience_artifact_root=root,
            knowledge_artifact_root=tmp_path / "knowledge",
            knowledge_embedding_model_path=tmp_path / "model",
            offline=True,
        )


def test_offline_guard_records_and_blocks_socket():
    import socket

    with OfflineNetworkGuard() as guard:
        with pytest.raises(NetworkAttempt):
            socket.create_connection(("203.0.113.1", 9), timeout=0.01)
    assert guard.attempt_count == 1


def test_smoke_requires_existing_phase7_artifact_root_before_catalog(tmp_path):
    source = tmp_path / "source.sqlite3"
    source.write_bytes(b"not-a-database")
    with pytest.raises(Phase7KnowledgeUnavailableError):
        run_smoke(
            source,
            experience_artifact_root=tmp_path / "experience",
            knowledge_artifact_root=tmp_path / "missing-knowledge",
            knowledge_embedding_model_path=tmp_path / "model",
            offline=True,
        )


def test_report_contract_names_experience_artifact_root():
    import scripts.experience_phase8_smoke as smoke

    assert "experience_artifact_root" in smoke.REPORT_KEYS
