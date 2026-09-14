import json
import sqlite3
from datetime import datetime, timezone

import pytest

from tests.fixtures.experience_source_db import create_source_db
from tradingagents.datasets.eligibility import join_observations
from tradingagents.datasets.errors import SourceIntegrityError, SourceReadError
from tradingagents.datasets.sources import (
    _AUDIT_FIELDS,
    ReadonlyExperienceSource,
    ReadonlyPhase9AuditSource,
    ReadonlyPhase56Source,
    SourceReadResult,
    SourceSchemaIncompatibleError,
    _decode_array,
    _decode_mapping,
    _decode_phase8_object,
    _stamp,
)
from tradingagents.experience.identity import (
    source_decision_fingerprint,
    source_evaluation_fingerprint,
)
from tradingagents.experience.source_reader import ReadonlySourceReader


def test_phase56_read_is_query_only_and_converts_utc(tmp_path):
    path = create_source_db(tmp_path / "source.db")
    result = ReadonlyPhase56Source(path).read()
    assert result.decisions[0].decision_id == "d1"
    assert result.decisions[0].analysis_snapshot_timestamp == datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )
    assert result.query_only is True
    with pytest.raises(sqlite3.OperationalError):
        ReadonlyPhase56Source(path).connection_for_test().execute("CREATE TABLE x(a)")


def test_phase56_retains_null_action_and_large_snapshot_for_eligibility(tmp_path):
    path = create_source_db(tmp_path / "source.db")
    snapshot = {
        "symbol": "EURUSD",
        "point": 0.00001,
        "digits": 5,
        "quote": {"bid": 1.1, "ask": 1.1001, "spread_points": 10},
        "features": {"M5": {"return_over_bars": 0.1}},
        "diagnostic_padding": "x" * 5000,
    }
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE shadow_decisions SET action=NULL, snapshot_json=?, trader_summary=?",
            (json.dumps(snapshot), "private reasoning output"),
        )
        db.commit()

    result = ReadonlyPhase56Source(path).read()
    decision = result.decisions[0]
    assert decision.action is None
    assert decision.fields["snapshot_json"]["quote"]["bid"] == 1.1
    assert decision.fields["snapshot_json"]["features"]["M5"]["return_over_bars"] == 0.1
    assert len(decision.fields["snapshot_json"]["diagnostic_padding"]) <= 2048
    assert decision.fields["source_decision_fingerprint"]
    assert "trader_summary" not in decision.fields
    assert decision.fields["source_prose_diagnostics"]["trader_summary"]["present"] is True


def test_phase56_public_adapter_carries_authoritative_source_fingerprints(tmp_path):
    path = create_source_db(tmp_path / "source.db")
    result = ReadonlyPhase56Source(path).read()
    snapshot = ReadonlySourceReader(path).read_snapshot()

    assert result.decisions[0].fields["source_decision_fingerprint"] == source_decision_fingerprint(
        snapshot.decisions[0]
    )
    assert result.evaluations[0].fields["source_evaluation_fingerprint"] == source_evaluation_fingerprint(
        snapshot.evaluations[0]
    )


def test_phase56_recomputes_fingerprints_when_input_identity_columns_are_bad(tmp_path):
    path = create_source_db(tmp_path / "source.db")
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE shadow_decisions ADD COLUMN source_decision_fingerprint TEXT")
        db.execute("ALTER TABLE shadow_decision_evaluations ADD COLUMN source_evaluation_fingerprint TEXT")
        db.execute("UPDATE shadow_decisions SET source_decision_fingerprint='BAD'")
        db.execute("UPDATE shadow_decision_evaluations SET source_evaluation_fingerprint='BAD'")
        db.commit()

    result = ReadonlyPhase56Source(path).read()
    snapshot = ReadonlySourceReader(path).read_snapshot()
    expected_decision_row = {
        key: value
        for key, value in snapshot.decisions[0].items()
        if key != "source_decision_fingerprint"
    }
    expected_evaluation_row = {
        key: value
        for key, value in snapshot.evaluations[0].items()
        if key != "source_evaluation_fingerprint"
    }

    assert result.decisions[0].fields["source_decision_fingerprint"] == source_decision_fingerprint(
        expected_decision_row
    )
    assert result.evaluations[0].fields["source_evaluation_fingerprint"] == source_evaluation_fingerprint(
        expected_evaluation_row
    )
    assert result.decisions[0].fields["source_decision_fingerprint"] != "BAD"
    assert result.evaluations[0].fields["source_evaluation_fingerprint"] != "BAD"


def test_phase56_authoritative_identity_keeps_phase8_join_matching(tmp_path):
    path = create_source_db(tmp_path / "source.db")
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE shadow_decisions ADD COLUMN source_decision_fingerprint TEXT")
        db.execute("ALTER TABLE shadow_decision_evaluations ADD COLUMN source_evaluation_fingerprint TEXT")
        db.execute("UPDATE shadow_decisions SET source_decision_fingerprint='BAD'")
        db.execute("UPDATE shadow_decision_evaluations SET source_evaluation_fingerprint='BAD'")
        db.commit()

    result = ReadonlyPhase56Source(path).read()
    snapshot = ReadonlySourceReader(path).read_snapshot()
    expected_decision_fp = source_decision_fingerprint(
        {key: value for key, value in snapshot.decisions[0].items() if key != "source_decision_fingerprint"}
    )
    expected_evaluation_fp = source_evaluation_fingerprint(
        {
            key: value
            for key, value in snapshot.evaluations[0].items()
            if key != "source_evaluation_fingerprint"
        }
    )
    record = {
        "source_decision_id": "d1",
        "source_run_id": "run1",
        "source_decision_fingerprint": expected_decision_fp,
        "source_evaluation_fingerprints": {"ANALYSIS_SNAPSHOT:300": expected_evaluation_fp},
    }

    joined = join_observations(result, SourceReadResult(records=(record,)), SourceReadResult())

    assert joined[0].fields["experience"] == record
    assert joined[0].evaluation.fields["source_evaluation_fingerprint"] == expected_evaluation_fp


def test_phase56_detects_file_change(tmp_path):
    path = create_source_db(tmp_path / "source.db")
    source = ReadonlyPhase56Source(path)
    source._before_read_hook = lambda: path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(SourceIntegrityError):
        source.read()


def test_missing_experience_catalog_is_typed_read_error(tmp_path):
    with pytest.raises(SourceReadError):
        ReadonlyExperienceSource(tmp_path / "missing").read()


def test_optional_missing_audit_is_unavailable(tmp_path):
    result = ReadonlyPhase9AuditSource(tmp_path / "missing.sqlite3").read()
    assert result.available is False


def test_phase56_requires_complete_contract_columns(tmp_path):
    path = tmp_path / "drift.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE shadow_decisions (decision_id TEXT)")
        db.execute("CREATE TABLE shadow_decision_evaluations (decision_id TEXT)")
    with pytest.raises(SourceSchemaIncompatibleError):
        ReadonlyPhase56Source(path).read()


def test_phase8_requires_columns_on_each_required_table(tmp_path):
    path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(path) as db:
        for table in (
            "experience_records",
            "experience_source_aliases",
            "experience_outcome_snapshots",
            "experience_feature_projections",
        ):
            db.execute(f"CREATE TABLE {table} (experience_id TEXT)")
    with pytest.raises(SourceSchemaIncompatibleError):
        ReadonlyExperienceSource(path).read()


def test_phase8_adapter_derives_version_provenance_from_catalog_contract(tmp_path):
    from tradingagents.datasets.sources import _PHASE8_REQUIRED

    path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(path) as db:
        for table, columns in _PHASE8_REQUIRED.items():
            db.execute(f'CREATE TABLE "{table}" (' + ", ".join(f'"{column}" TEXT' for column in columns) + ")")
        record_values = dict.fromkeys(_PHASE8_REQUIRED["experience_records"], "")
        record_values.update({
            "experience_id": "e1", "source_decision_id": "d1", "source_database_id": "s1",
            "source_decision_fingerprint": "dfp", "symbol": "EURUSD", "tombstoned": "0",
            "market_state_json": "{}", "decision_evidence_json": "{}", "provenance_json": "{}",
            "source_evaluation_fingerprints_json": "{}", "trust": "TIER_A_HIGH_TRUST",
        })
        cols = sorted(record_values)
        db.execute('INSERT INTO experience_records (' + ','.join('"' + c + '"' for c in cols) + ') VALUES (' + ','.join('?' for _ in cols) + ')', [record_values[c] for c in cols])
        projection_values = dict.fromkeys(_PHASE8_REQUIRED["experience_feature_projections"], "")
        projection_values.update({"experience_id": "e1", "feature_schema_version": "experience-features.v1", "projection_json": '{"version":"phase8-feature-extractor.v1"}'})
        cols = sorted(projection_values)
        db.execute('INSERT INTO experience_feature_projections (' + ','.join('"' + c + '"' for c in cols) + ') VALUES (' + ','.join('?' for _ in cols) + ')', [projection_values[c] for c in cols])
        db.commit()
    result = ReadonlyExperienceSource(path).read()
    row = result.records[0]
    assert row["experience_schema_version"] == "phase8.experience.v1"
    assert row["feature_schema_version"] == "experience-features.v1"
    assert row["feature_extractor_version"] == "phase8-feature-extractor.v1"
    assert row["trust_policy_version"] == "trust-policy.v1"


def test_audit_retains_policy_fingerprints_and_rejects_incomplete_schema(tmp_path):
    path = tmp_path / "audit.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE evidence_usage_audit (decision_id TEXT, source_run_id TEXT)")
    with pytest.raises(SourceSchemaIncompatibleError):
        ReadonlyPhase9AuditSource(path).read()


def test_phase9_as_of_is_exact_utc_datetime():
    assert _stamp("2026-01-01T00:00:00+00:00") == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_phase9_array_fields_decode_to_bounded_arrays():
    assert _decode_array('["telemetry-a", "node-a"]') == ["telemetry-a", "node-a"]


@pytest.mark.parametrize("raw", [0, False, "", "   "])
def test_phase9_array_fields_reject_falsey_or_empty_non_json_values(raw):
    with pytest.raises(SourceReadError, match="array metadata"):
        _decode_array(raw)


@pytest.mark.parametrize("raw", [0, False, "", "   "])
def test_phase9_mapping_fields_reject_falsey_or_empty_non_json_values(raw):
    with pytest.raises(SourceReadError, match="object metadata"):
        _decode_mapping(raw)


def test_phase9_nullable_metadata_values_have_explicit_empty_shapes():
    assert _decode_array(None) == []
    assert _decode_mapping(None) == {}


def test_phase9_array_fields_reject_oversized_arrays_before_bounding():
    with pytest.raises(SourceReadError, match="array metadata exceeds bound"):
        _decode_array(json.dumps(list(range(101))))


def test_phase9_public_adapter_rejects_oversized_array_strings_before_bounding(tmp_path):
    path = tmp_path / "audit.sqlite3"
    columns = tuple(_AUDIT_FIELDS)
    values = dict.fromkeys(columns)
    values["decision_id"] = "decision-1"
    values["source_run_id"] = "run-1"
    values["available_knowledge_ids"] = json.dumps(["x" * 501])
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE evidence_usage_audit ("
            + ", ".join(f'"{column}" TEXT' for column in columns)
            + ")"
        )
        db.execute(
            'INSERT INTO evidence_usage_audit ("'
            + '", "'.join(columns)
            + '") VALUES ('
            + ", ".join("?" for _ in columns)
            + ")",
            [values[column] for column in columns],
        )
    with pytest.raises(SourceReadError, match="string metadata exceeds bound"):
        ReadonlyPhase9AuditSource(path).read()


def test_phase9_mapping_fields_reject_oversized_mappings_before_bounding():
    with pytest.raises(SourceReadError, match="object metadata exceeds bound"):
        _decode_mapping(json.dumps({f"key-{i}": i for i in range(101)}))


def test_phase9_object_fields_reject_non_object_json():
    with pytest.raises(SourceReadError, match="JSON object"):
        _decode_mapping("[1, 2]")


@pytest.mark.parametrize("raw", ["[1, 2]", '"scalar"', '1', 'null'])
def test_phase8_json_object_fields_reject_non_object_json(raw):
    with pytest.raises(SourceReadError, match="Phase 8 JSON object"):
        _decode_phase8_object(raw)


@pytest.mark.parametrize(
    "raw",
    [json.dumps({"x" * 129: "value"}), json.dumps({"value": "x" * 2500})],
)
def test_phase8_json_object_rejects_oversized_members_before_bounding(raw):
    with pytest.raises(SourceReadError, match="Phase 8 JSON object"):
        _decode_phase8_object(raw)


def test_phase8_fingerprint_decoder_rejects_oversized_fingerprint_before_bounding():
    from tradingagents.datasets.sources import _decode_phase8_fingerprints

    with pytest.raises(SourceReadError, match="evaluation fingerprints"):
        _decode_phase8_fingerprints(json.dumps({"ANALYSIS_SNAPSHOT:300": "x" * 300}))


def test_phase9_read_normalizes_audit_timestamp_and_array_fields(tmp_path):
    path = tmp_path / "audit.sqlite3"
    columns = (
        "decision_id",
        "source_run_id",
        "as_of",
        "rendered_context",
        "rendered_context_hash",
        "knowledge_generation_id",
        "experience_generation_id",
        "knowledge_query",
        "integration_status",
        "bundle_status",
        "evidence_use_status",
        "available_knowledge_ids",
        "available_experience_ids",
        "available_statistics_ids",
        "evidence_refs_used",
        "evidence_refs_rejected",
        "query_policy_version",
        "source_status",
        "diagnostics",
        "source_errors",
        "retrieval_count",
        "retrieval_latency_seconds",
        "builder_latency_seconds",
        "selected_counts",
        "dropped_counts",
        "telemetry_references",
        "node_context_hashes",
        "missing_nodes",
        "provider",
        "model",
        "audit_schema_version",
        "evidence_audit_status",
        "query_normalization_fingerprint",
        "knowledge_query_fingerprint",
    )
    values = [None] * len(columns)
    values[columns.index("decision_id")] = "decision-1"
    values[columns.index("source_run_id")] = "run-1"
    values[columns.index("as_of")] = "2026-01-01T01:02:03Z"
    values[columns.index("telemetry_references")] = json.dumps(["telemetry-1"])
    values[columns.index("missing_nodes")] = json.dumps(["node-a", "node-b"])
    values[columns.index("source_errors")] = json.dumps({"phase7": "unavailable"})
    values[columns.index("selected_counts")] = json.dumps({"knowledge": 2, "experience": 0, "statistics": 1})
    values[columns.index("dropped_counts")] = json.dumps({"knowledge": 0, "experience": 1, "statistics": 0})
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE evidence_usage_audit ("
            + ", ".join(f'"{column}" TEXT' for column in columns)
            + ")"
        )
        placeholders = ", ".join("?" for _ in columns)
        db.execute(
            'INSERT INTO evidence_usage_audit ("' + '", "'.join(columns) + f'") VALUES ({placeholders})',
            values,
        )

    result = ReadonlyPhase9AuditSource(path).read()
    assert result.available is True
    assert len(result.audits) == 1
    audit = result.audits[0]
    assert audit["as_of"] == datetime(2026, 1, 1, 1, 2, 3, tzinfo=timezone.utc)
    assert isinstance(audit["as_of"], datetime)
    assert audit["telemetry_references"] == ["telemetry-1"]
    assert isinstance(audit["telemetry_references"], list)
    assert audit["missing_nodes"] == ["node-a", "node-b"]
    assert isinstance(audit["missing_nodes"], list)
    assert audit["source_errors"] == {"phase7": "unavailable"}
    assert isinstance(audit["source_errors"], dict)
    assert audit["selected_counts"] == {"knowledge": 2, "experience": 0, "statistics": 1}
    assert audit["dropped_counts"] == {"knowledge": 0, "experience": 1, "statistics": 0}
