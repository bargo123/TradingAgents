from __future__ import annotations

import sqlite3

import pytest

from tradingagents.experience.catalog import ExperienceCatalog
from tradingagents.experience.errors import SourceDecisionConflictError


@pytest.fixture
def catalog(tmp_path):
    return ExperienceCatalog(tmp_path / "experience")


def test_duplicate_aliases_share_one_logical_record(catalog: ExperienceCatalog) -> None:
    first = catalog.upsert_source_alias("db-a", decision_id="d1", fingerprint="fp1")
    second = catalog.upsert_source_alias("db-b", decision_id="d1", fingerprint="fp1")
    assert first.experience_id == second.experience_id
    assert len(catalog.active_records()) == 1
    assert catalog.current_alias_count(first.experience_id) == 2


def test_alias_identity_preserves_multiple_decisions_from_one_source(catalog: ExperienceCatalog) -> None:
    first = catalog.upsert_source_alias("db-a", decision_id="d1", fingerprint="fp1")
    second = catalog.upsert_source_alias("db-a", decision_id="d2", fingerprint="fp2")
    assert first.experience_id != second.experience_id
    records = {record.experience_id: record for record in catalog.active_records()}
    assert set(records[first.experience_id].source_aliases) == {"db-a:d1"}
    assert set(records[second.experience_id].source_aliases) == {"db-a:d2"}


def test_last_alias_removal_tombstones_but_failed_scan_does_not(catalog: ExperienceCatalog) -> None:
    catalog.upsert_source_alias("db-a", decision_id="d1", fingerprint="fp1")
    catalog.upsert_source_alias("db-b", decision_id="d1", fingerprint="fp1")
    catalog.mark_alias_removed("db-a", "d1", "fp1")
    assert catalog.active_records()[0].source_aliases["db-b:d1"] == "CURRENT"
    catalog.record_failed_scan("db-b", "SOURCE_DATABASE_UNAVAILABLE")
    assert catalog.active_records()[0].source_aliases["db-b:d1"] == "CURRENT"
    catalog.mark_alias_removed("db-b", "d1", "fp1")
    record = catalog.historical_records()[0]
    assert catalog.is_tombstoned(record.experience_id) is True


def test_conflicting_fingerprint_is_quarantined(catalog: ExperienceCatalog) -> None:
    catalog.upsert_source_alias("db-a", "d1", "fp1")
    with pytest.raises(SourceDecisionConflictError):
        catalog.upsert_source_alias("db-a", "d1", "fp2")
    assert catalog.quarantine_count() == 1
    assert catalog.active_records()[0].source_decision_fingerprint == "fp1"


def test_recovery_snapshot_is_append_only_only_when_observed(catalog: ExperienceCatalog) -> None:
    catalog.append_evaluation_snapshot("exp1", {"evaluation_status": "DATA_UNAVAILABLE"}, "old-fp")
    catalog.append_evaluation_snapshot("exp1", {"evaluation_status": "COMPLETE"}, "new-fp")
    catalog.append_evaluation_snapshot("exp1", {"evaluation_status": "COMPLETE"}, "new-fp")
    assert catalog.evaluation_snapshot_fingerprints("exp1") == ("old-fp", "new-fp")


def test_catalog_has_phase8_tables_and_diagnostics_are_bounded(catalog: ExperienceCatalog) -> None:
    catalog.record_failed_scan("db-a", "SOURCE_DATABASE_UNAVAILABLE", detail="secret report\nshould not be indexed")
    with sqlite3.connect(catalog.database_path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"experience_sources", "experience_records", "experience_source_aliases",
                "experience_decision_evidence", "experience_market_provenance",
                "experience_outcome_snapshots", "experience_feature_projections",
                "experience_generations", "experience_import_events", "experience_quarantine",
                "experience_diagnostic_fts"}.issubset(tables)
        indexed = db.execute("SELECT label FROM experience_diagnostic_fts").fetchone()[0]
        assert "secret report" not in indexed
        assert "SOURCE_DATABASE_UNAVAILABLE" in indexed


def test_publish_generation_is_atomic_and_readable(catalog: ExperienceCatalog) -> None:
    generation = catalog.publish_generation("gen-1", population_fingerprint="pop-1", metadata={"rows": 1})
    assert generation["generation_id"] == "gen-1"
    assert catalog.active_generation()["population_fingerprint"] == "pop-1"
