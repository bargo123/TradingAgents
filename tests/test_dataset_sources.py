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
