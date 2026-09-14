import json
from datetime import datetime, timezone

import pytest

from tradingagents.datasets.errors import DatasetConfigError
from tradingagents.datasets.models import (
    BuildReport,
    CanonicalExampleV1,
    DatasetConfig,
    DatasetExclusion,
    DatasetExclusionReason,
    DatasetManifest,
    EvaluationObservation,
    EvidenceObservation,
    JoinedObservation,
    SourceFingerprint,
    SourceObservation,
    SplitAssignment,
    ValidationReport,
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


def test_forbidden_keys_and_sensitive_values_are_rejected_recursively():
    for payload in ({"nested": {"password": "x"}}, {"nested": ["private token: x"]}):
        with pytest.raises(ValueError):
            SourceObservation("d", datetime(2025, 1, 1, tzinfo=timezone.utc), fields=payload)


def test_closed_statuses_and_reason_types():
    with pytest.raises(ValueError):
        EvaluationObservation("d", evaluation_status="BOGUS")
    with pytest.raises(ValueError):
        EvidenceObservation(context_integrity="BOGUS")
    with pytest.raises(ValueError):
        EvidenceObservation(evidence_use_status="BOGUS")
    with pytest.raises(ValueError):
        DatasetManifest("m", split_status="BOGUS")
    with pytest.raises(ValueError):
        BuildReport("BOGUS")
    with pytest.raises(ValueError):
        DatasetExclusion("d", reasons=("BOGUS",))


def test_nested_collections_are_immutable_and_report_collections_frozen():
    evidence = EvidenceObservation(refs_used=["a"], refs_rejected=["b"], fields={"x": {"y": 1}})
    with pytest.raises(TypeError):
        evidence.fields["x"]["y"] = 2
    with pytest.raises(TypeError):
        evidence.refs_used[0] = "c"
    report = ValidationReport(True, errors=["e"], warnings=["w"])
    with pytest.raises(TypeError):
        report.errors[0] = "x"


def test_insertion_order_does_not_change_serialization():
    a = SourceObservation("d", datetime(2025, 1, 1, tzinfo=timezone.utc), fields={"a": 1, "b": 2})
    b = SourceObservation("d", datetime(2025, 1, 1, tzinfo=timezone.utc), fields={"b": 2, "a": 1})
    assert a.to_json() == b.to_json()


def test_round2_sensitive_variants_and_deterministic_type_errors():
    for value in ("chain-of-thought", "chain_of_thought", "cot", "secret-value"):
        with pytest.raises(ValueError):
            SourceObservation("d", datetime(2025, 1, 1, tzinfo=timezone.utc), fields={"x": value})
    for kwargs in ({"evaluation_status": None}, {"source_context_eligible": "yes"}):
        with pytest.raises(ValueError):
            EvaluationObservation("d", **kwargs)
    with pytest.raises(ValueError):
        DatasetManifest("m", split_status=None)


def test_round2_vocabulary_deep_freeze_and_types():
    for status in ("PUBLISHED", "EMPTY_ELIGIBLE_SET", "FAILED"):
        DatasetManifest("m", split_status="INSUFFICIENT_DATA")
        BuildReport(status)
    report = BuildReport("PUBLISHED", errors=["e"], exclusions=[DatasetExclusion("d")])
    with pytest.raises(TypeError):
        report.exclusions[0] = DatasetExclusion("x")
    with pytest.raises(TypeError):
        report.errors[0] = "x"
    with pytest.raises(ValueError):
        DatasetConfig(("src",), "p8", None, "out", allow_empty="yes")
    with pytest.raises(ValueError):
        DatasetManifest("m", examples="1")
    with pytest.raises(ValueError):
        JoinedObservation(decision="bad")
    with pytest.raises(ValueError):
        CanonicalExampleV1("x", decision=[])


def test_round3_mapping_fields_reject_sequences():
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    cases = [
        lambda: SourceObservation("d", ts, fields=[]),
        lambda: EvaluationObservation("d", fields=[]),
        lambda: EvidenceObservation(fields=[]),
        lambda: JoinedObservation(SourceObservation("d", ts), fields=[]),
        lambda: DatasetExclusion("d", details=[]),
    ]
    for make in cases:
        with pytest.raises(ValueError):
            make()


def test_round3_invalid_collection_types_are_value_errors():
    with pytest.raises(ValueError):
        BuildReport("FAILED", errors=None)
    with pytest.raises(ValueError):
        BuildReport("FAILED", exclusions=None)
    with pytest.raises(ValueError):
        ValidationReport(True, errors=None)
    with pytest.raises(ValueError):
        ValidationReport(True, warnings=None)
    with pytest.raises(ValueError):
        SplitAssignment("e", "train", None)


@pytest.mark.parametrize("invalid_split", [[], {}])
def test_round4_unhashable_split_values_raise_value_error(invalid_split):
    with pytest.raises(ValueError):
        SplitAssignment("e", invalid_split, "g")


class _UnhashableString(str):
    __hash__ = None


def test_round5_unhashable_string_split_value_raise_value_error():
    with pytest.raises(ValueError):
        SplitAssignment("e", _UnhashableString("other"), "g")
