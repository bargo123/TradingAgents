from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

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


@pytest.mark.parametrize(
    "timestamp",
    [
        False,
        0,
        "",
        [],
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=2))),
    ],
)
def test_explicit_analysis_timestamp_must_be_utc_and_typed(catalog, timestamp) -> None:
    with pytest.raises((TypeError, ValueError), match="analysis_snapshot_timestamp"):
        catalog.upsert_source_alias(
            "db-a", decision_id="timestamp-invalid", fingerprint="fp1",
            analysis_snapshot_timestamp=timestamp,
        )


@pytest.mark.parametrize("field", ["decision_completed_timestamp", "decision_reference_timestamp"])
@pytest.mark.parametrize("timestamp", [False, 0, "", [], datetime(2026, 1, 1)])
def test_explicit_optional_timestamps_must_be_utc_and_typed(catalog, field, timestamp) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        catalog.upsert_source_alias(
            "db-a", decision_id=f"{field}-invalid", fingerprint="fp1",
            **{field: timestamp},
        )


@pytest.mark.parametrize("value", [False, 0, [], ""])
def test_source_decision_fingerprint_must_be_non_empty_text(catalog, value) -> None:
    with pytest.raises((TypeError, ValueError), match="fingerprint"):
        catalog.upsert_source_alias("db-a", decision_id="bad-fingerprint", fingerprint=value)


@pytest.mark.parametrize("field", ["symbol", "source_run_id", "requested_symbol", "analysis_profile"])
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_alias_metadata_rejects_non_text_values(catalog, field, value) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        catalog.upsert_source_alias(
            "db-a",
            decision_id=f"bad-{field}",
            fingerprint="fp1",
            **{field: value},
        )


@pytest.mark.parametrize("value", [False, 0, [], ""])
def test_evaluation_fingerprint_must_be_non_empty_text(catalog, value) -> None:
    with pytest.raises((TypeError, ValueError), match="fingerprint"):
        catalog.append_evaluation_snapshot(
            "exp1", {"evaluation_status": "COMPLETE"}, value
        )


def test_alias_identity_preserves_multiple_decisions_from_one_source(
    catalog: ExperienceCatalog,
) -> None:
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


def test_malformed_tombstone_flag_fails_closed(catalog: ExperienceCatalog) -> None:
    record = catalog.upsert_source_alias("db-a", decision_id="d1", fingerprint="fp1")
    with closing(sqlite3.connect(catalog.database_path)) as db, db:
        db.execute(
            "UPDATE experience_records SET tombstoned=? WHERE experience_id=?",
            ("false", record.experience_id),
        )

    with pytest.raises(ValueError, match="tombstoned"):
        catalog.is_tombstoned(record.experience_id)


def test_conflicting_fingerprint_is_quarantined(catalog: ExperienceCatalog) -> None:
    catalog.upsert_source_alias("db-a", "d1", "fp1")
    with pytest.raises(SourceDecisionConflictError):
        catalog.upsert_source_alias("db-a", "d1", "fp2")
    assert catalog.quarantine_count() == 1
    assert catalog.active_records()[0].source_decision_fingerprint == "fp1"


def test_conflict_does_not_commit_unrelated_writes_in_outer_transaction(
    catalog: ExperienceCatalog,
) -> None:
    catalog.upsert_source_alias("db-a", "d1", "fp1")
    with pytest.raises(SourceDecisionConflictError), catalog.transaction():
        catalog.upsert_source_alias("db-a", "d2", "fp2")
        catalog.upsert_source_alias("db-a", "d1", "conflicting-fp")

    assert [record.source_decision_id for record in catalog.active_records()] == ["d1"]
    assert catalog.quarantine_count() == 0


def test_recovery_snapshot_is_append_only_only_when_observed(catalog: ExperienceCatalog) -> None:
    catalog.append_evaluation_snapshot("exp1", {"evaluation_status": "DATA_UNAVAILABLE"}, "old-fp")
    catalog.append_evaluation_snapshot("exp1", {"evaluation_status": "COMPLETE"}, "new-fp")
    catalog.append_evaluation_snapshot("exp1", {"evaluation_status": "COMPLETE"}, "new-fp")
    assert catalog.evaluation_snapshot_fingerprints("exp1") == ("old-fp", "new-fp")


@pytest.mark.parametrize("provenance", [[], "", 0, False])
def test_append_evaluation_snapshot_rejects_non_object_provenance(
    catalog: ExperienceCatalog, provenance
) -> None:
    with pytest.raises(ValueError, match="provenance"):
        catalog.append_evaluation_snapshot(
            "exp1", {"evaluation_status": "COMPLETE"}, "fp", provenance=provenance
        )


@pytest.mark.parametrize("detail", [[], "", 0, False])
def test_record_quarantine_rejects_non_object_detail(catalog: ExperienceCatalog, detail) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        catalog.record_quarantine("db-a", "d1", "fp", "BAD_SOURCE", detail=detail)


@pytest.mark.parametrize("field", ["source_database_id", "decision_id", "fingerprint", "reason"])
@pytest.mark.parametrize("value", [False, 0, [], {}, ""])
def test_record_quarantine_rejects_malformed_identity(field, value, catalog) -> None:
    kwargs = {
        "source_database_id": "db-a",
        "decision_id": "d1",
        "fingerprint": "fp",
        "reason": "BAD_SOURCE",
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        catalog.record_quarantine(**kwargs)


def test_catalog_has_phase8_tables_and_diagnostics_are_bounded(catalog: ExperienceCatalog) -> None:
    catalog.record_failed_scan(
        "db-a", "SOURCE_DATABASE_UNAVAILABLE", detail="secret report\nshould not be indexed"
    )
    with closing(sqlite3.connect(catalog.database_path)) as db, db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "experience_sources",
            "experience_records",
            "experience_source_aliases",
            "experience_decision_evidence",
            "experience_market_provenance",
            "experience_outcome_snapshots",
            "experience_feature_projections",
            "experience_generations",
            "experience_import_events",
            "experience_quarantine",
            "experience_diagnostic_fts",
        }.issubset(tables)
        indexed = db.execute("SELECT label FROM experience_diagnostic_fts").fetchone()[0]
        assert "secret report" not in indexed
        assert "SOURCE_DATABASE_UNAVAILABLE" in indexed


@pytest.mark.parametrize("detail", [[], "not-an-object", 0, False])
def test_catalog_rejects_non_object_json_details(catalog: ExperienceCatalog, detail) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        catalog.record_import_event("RUNNING", detail=detail)


@pytest.mark.parametrize("value", [False, 0, [], {}, ""])
def test_record_import_event_rejects_malformed_event_type(catalog, value) -> None:
    with pytest.raises((TypeError, ValueError), match="event_type"):
        catalog.record_import_event(value)


@pytest.mark.parametrize("value", [False, 0, [], {}, ""])
def test_store_feature_projection_rejects_malformed_experience_id(catalog, value) -> None:
    with pytest.raises((TypeError, ValueError), match="experience_id"):
        catalog.store_feature_projection(value, {})


@pytest.mark.parametrize("projection", [False, 0, [], ""])
def test_store_feature_projection_requires_mapping(catalog, projection) -> None:
    record = catalog.upsert_source_alias("db-a", decision_id="projection", fingerprint="fp1")
    with pytest.raises((TypeError, ValueError), match="projection"):
        catalog.store_feature_projection(record.experience_id, projection)


def test_orphaned_import_run_is_marked_interrupted_once(catalog: ExperienceCatalog) -> None:
    catalog.record_import_event("RUNNING", detail={"run_id": "stale-run"})
    catalog.record_import_event("RUNNING", detail={"run_id": "finished-run"})
    catalog.record_import_event("COMPLETED", detail={"run_id": "finished-run"})

    assert catalog.reconcile_interrupted_imports() == ("stale-run",)
    assert catalog.reconcile_interrupted_imports() == ()

    with closing(sqlite3.connect(catalog.database_path)) as db, db:
        events = db.execute(
            "SELECT event_type, detail_json FROM experience_import_events "
            "WHERE detail_json LIKE '%stale-run%' ORDER BY event_id"
        ).fetchall()
    assert [event[0] for event in events] == ["RUNNING", "INTERRUPTED"]


def test_publish_generation_is_atomic_and_readable(catalog: ExperienceCatalog) -> None:
    generation = catalog.publish_generation(
        "gen-1", population_fingerprint="pop-1", metadata={"rows": 1}
    )
    assert generation["generation_id"] == "gen-1"
    assert catalog.active_generation()["population_fingerprint"] == "pop-1"


@pytest.mark.parametrize("value", [False, 0, [], {}, ""])
def test_publish_generation_rejects_malformed_generation_id(catalog, value) -> None:
    with pytest.raises((TypeError, ValueError), match="generation_id"):
        catalog.publish_generation(value, population_fingerprint="pop-1")


@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_publish_generation_rejects_malformed_population_fingerprint(catalog, value) -> None:
    with pytest.raises((TypeError, ValueError), match="population_fingerprint"):
        catalog.publish_generation("gen-1", population_fingerprint=value)


@pytest.mark.parametrize("metadata", [False, 0, [], ""])
def test_publish_generation_rejects_malformed_metadata(catalog, metadata) -> None:
    with pytest.raises((TypeError, ValueError), match="JSON object"):
        catalog.publish_generation("gen-1", population_fingerprint="pop-1", metadata=metadata)


@pytest.mark.parametrize("field", ["source_database_id", "canonical_path", "source_fingerprint", "source_schema_fingerprint"])
@pytest.mark.parametrize("value", [False, 0, [], {}, ""])
def test_register_source_rejects_malformed_identity(field, value, catalog) -> None:
    kwargs = {
        "source_database_id": "db-1",
        "canonical_path": "C:/source.db",
        "source_fingerprint": "source-fp",
        "source_schema_fingerprint": "schema-fp",
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        catalog.register_source(**kwargs)
