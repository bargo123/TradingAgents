import sqlite3
from datetime import datetime, timezone

import pytest

from tests.fixtures.experience_source_db import create_source_db
from tradingagents.datasets.errors import SourceIntegrityError, SourceReadError
from tradingagents.datasets.sources import (
    ReadonlyExperienceSource,
    ReadonlyPhase9AuditSource,
    ReadonlyPhase56Source,
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
