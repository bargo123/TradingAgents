import json
import sqlite3
from datetime import datetime, timezone

import pytest

from tests.fixtures.experience_source_db import create_source_db
from tradingagents.datasets.errors import SourceIntegrityError, SourceReadError
from tradingagents.datasets.sources import (
    ReadonlyExperienceSource,
    ReadonlyPhase9AuditSource,
    ReadonlyPhase56Source,
    SourceSchemaIncompatibleError,
    _decode_array,
    _stamp,
)


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
