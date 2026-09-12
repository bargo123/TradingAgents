from pathlib import Path

from tradingagents.experience.config import (
    EXPERIENCE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    ExperienceConfig,
)


def test_config_has_versioned_defaults_and_artifact_root() -> None:
    config = ExperienceConfig()
    assert config.artifact_root == Path("data_cache/experience")
    assert EXPERIENCE_SCHEMA_VERSION == "phase8.experience.v1"
    assert config.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert config.source_databases == ()


def test_config_serializes_deterministically() -> None:
    a = ExperienceConfig(source_databases=(Path("b.sqlite"), Path("a.sqlite")))
    b = ExperienceConfig(source_databases=(Path("b.sqlite"), Path("a.sqlite")))
    assert a.to_json() == b.to_json()
