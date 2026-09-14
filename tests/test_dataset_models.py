import json
from datetime import datetime, timezone

import pytest

from tradingagents.datasets.errors import DatasetConfigError
from tradingagents.datasets.models import (
    DatasetConfig,
    DatasetExclusionReason,
    SourceFingerprint,
    SourceObservation,
)


def test_closed_enum_and_utc_validation():
    assert {x.value for x in DatasetExclusionReason} == {
        "UNTRUSTED_TIER",
        "DECISION_CONTEXT_INCOMPLETE",
        "TEMPORAL_INVALID",
        "OUTCOME_INELIGIBLE",
        "OUTCOME_UNAVAILABLE",
        "NORMALIZATION_FAILED",
        "CITATION_INVALID",
        "EVIDENCE_INCOMPLETE",
        "SCHEMA_UNSUPPORTED",
        "PROVENANCE_INCOMPLETE",
        "DUPLICATE",
        "SOURCE_INTEGRITY_FAILED",
    }
    with pytest.raises(ValueError):
        SourceObservation(decision_id="d", analysis_snapshot_timestamp=datetime.now())
    with pytest.raises(ValueError):
        SourceObservation(
            decision_id="d", analysis_snapshot_timestamp=datetime.now(timezone.utc), action="NOPE"
        )


def test_immutable_mappings_and_json_determinism(tmp_path):
    c = DatasetConfig(
        (tmp_path / "src.sqlite",),
        tmp_path / "p8",
        None,
        tmp_path / "out",
        filters={"z": 1, "a": 2},
    )
    assert c.allow_empty is True
    with pytest.raises(TypeError):
        c.filters["x"] = 1
    SourceFingerprint("src", "/tmp/src", "schema", "file", "snapshot", "v1")
    o = SourceObservation(
        "d", datetime(2025, 1, 1, tzinfo=timezone.utc), action="BUY", fields={"b": 2, "a": 1}
    )
    assert o.to_json() == o.to_json()
    json.loads(o.to_json())
    assert (
        "prompt" not in o.to_dict()
        and "completion" not in o.to_dict()
        and "reasoning" not in o.to_dict()
    )


def test_config_rejects_output_inside_source(tmp_path):
    with pytest.raises(DatasetConfigError):
        DatasetConfig((tmp_path / "src",), tmp_path / "p8", None, tmp_path / "src" / "out")
