from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    "field",
    [
        "experience_schema_version",
        "feature_schema_version",
        "feature_extractor_version",
        "similarity_profile_version",
        "trust_policy_version",
        "statistics_policy_version",
        "projection_version",
    ],
)
@pytest.mark.parametrize("value", [None, False, 0, [], {}, "", "   ", "x" * 257])
def test_config_rejects_invalid_version_metadata(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        ExperienceConfig(**{field: value})


@pytest.mark.parametrize("value", ["db.sqlite", b"db.sqlite", {}, None])
def test_config_rejects_scalar_source_databases(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="source_databases"):
        ExperienceConfig(source_databases=value)
