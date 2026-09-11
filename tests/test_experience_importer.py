from __future__ import annotations

import sqlite3
from unittest.mock import patch
from tradingagents.experience.errors import ExperienceImportLockedError

from tradingagents.experience.catalog import ExperienceCatalog
from tradingagents.experience.importer import ExperienceImporter
from tests.fixtures.experience_source_db import create_source_db


def test_first_import_is_idempotent(tmp_path):
    source = create_source_db(tmp_path / "source.sqlite3")
    catalog = ExperienceCatalog(tmp_path / "artifact")
    importer = ExperienceImporter(catalog)

    first = importer.import_sources((source,))
    second = importer.import_sources((source,))

    assert first.indexed_count > 0
    assert second.unchanged_count == first.indexed_count
    assert second.experience_count == first.experience_count


def test_failed_scan_does_not_remove_alias(tmp_path):
    source = create_source_db(tmp_path / "source.sqlite3")
    catalog = ExperienceCatalog(tmp_path / "artifact")
    importer = ExperienceImporter(catalog)
    importer.import_sources((source,))
    importer.import_sources((tmp_path / "missing.sqlite3",))

    record = catalog.active_records()[0]
    assert catalog.current_alias_count(record.experience_id) == 1


def test_later_import_appends_newly_observed_evaluation_for_unchanged_decision(tmp_path):
    source = create_source_db(tmp_path / "source.sqlite3")
    catalog = ExperienceCatalog(tmp_path / "artifact")
    importer = ExperienceImporter(catalog)

    importer.import_sources((source,))
    with sqlite3.connect(source) as db:
        db.execute("UPDATE shadow_decision_evaluations SET evaluation_status='COMPLETE'")
        db.commit()

    report = importer.import_sources((source,))
    record = catalog.active_records()[0]
    snapshots = catalog.evaluation_snapshots(record.experience_id)
    assert report.evaluation_snapshot_count == 1
    assert [snapshot["evaluation_status"] for snapshot in snapshots] == ["PENDING", "COMPLETE"]
    assert importer.import_sources((source,)).evaluation_snapshot_count == 0


def test_source_snapshot_writes_are_atomic(tmp_path):
    source = create_source_db(tmp_path / "source.sqlite3")
    catalog = ExperienceCatalog(tmp_path / "artifact")
    importer = ExperienceImporter(catalog)
    with patch.object(catalog, "append_evaluation_snapshot", side_effect=RuntimeError("write failed")):
        report = importer.import_sources((source,))
    assert catalog.active_records() == ()
    assert (report.indexed_count, report.alias_count, report.evaluation_snapshot_count) == (0, 0, 0)


def test_lock_collision_does_not_create_staging(tmp_path):
    source = create_source_db(tmp_path / "source.sqlite3")
    catalog = ExperienceCatalog(tmp_path / "artifact")
    importer = ExperienceImporter(catalog)
    importer.lock_path.parent.mkdir(parents=True, exist_ok=True)
    importer.lock_path.touch()
    with patch.object(importer, "_lock", side_effect=ExperienceImportLockedError("busy")):
        try:
            importer.import_sources((source,))
        except ExperienceImportLockedError:
            pass
    assert not (catalog.artifact_root / ".staging").exists()
