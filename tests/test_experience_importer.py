from __future__ import annotations

import json
import sqlite3
from contextlib import suppress
from unittest.mock import patch

from tests.fixtures.experience_source_db import create_source_db
from tradingagents.experience.catalog import ExperienceCatalog
from tradingagents.experience.config import (
    EXPERIENCE_SCHEMA_VERSION,
    FEATURE_EXTRACTOR_VERSION,
    FEATURE_SCHEMA_VERSION,
    PROJECTION_VERSION,
    SIMILARITY_PROFILE_VERSION,
    STATISTICS_POLICY_VERSION,
    TRUST_POLICY_VERSION,
)
from tradingagents.experience.errors import ExperienceImportLockedError
from tradingagents.experience.importer import ExperienceImporter, ExperienceRebuilder


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
    with patch.object(
        catalog, "append_evaluation_snapshot", side_effect=RuntimeError("write failed")
    ):
        report = importer.import_sources((source,))
    assert catalog.active_records() == ()
    assert (report.indexed_count, report.alias_count, report.evaluation_snapshot_count) == (0, 0, 0)


def test_lock_collision_does_not_create_staging(tmp_path):
    source = create_source_db(tmp_path / "source.sqlite3")
    catalog = ExperienceCatalog(tmp_path / "artifact")
    importer = ExperienceImporter(catalog)
    importer.lock_path.parent.mkdir(parents=True, exist_ok=True)
    importer.lock_path.touch()
    with patch.object(importer, "_lock", side_effect=ExperienceImportLockedError("busy")), suppress(
        ExperienceImportLockedError
    ):
        importer.import_sources((source,))
    assert not (catalog.artifact_root / ".staging").exists()


def test_rebuild_publishes_canonical_compatibility_metadata(tmp_path):
    source = create_source_db(tmp_path / "source.sqlite3")
    catalog = ExperienceCatalog(tmp_path / "artifact")
    ExperienceImporter(catalog).import_sources((source,))

    generation = ExperienceRebuilder(catalog).rebuild()
    metadata = generation.metadata

    assert metadata["rows"] == len(catalog.active_records())
    assert metadata["retains_historical_features"] is True
    assert metadata["experience_schema_version"] == EXPERIENCE_SCHEMA_VERSION
    assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert metadata["feature_extractor_version"] == FEATURE_EXTRACTOR_VERSION
    assert metadata["similarity_profile_version"] == SIMILARITY_PROFILE_VERSION
    assert metadata["trust_policy_version"] == TRUST_POLICY_VERSION
    assert metadata["statistics_policy_version"] == STATISTICS_POLICY_VERSION
    assert metadata["projection_version"] == PROJECTION_VERSION


def test_import_publishes_canonical_feature_projection_metadata(tmp_path):
    source = create_source_db(tmp_path / "source.sqlite3")
    snapshot = {
        "point": 0.00001,
        "digits": 5,
        "quote": {"bid": 1.1, "ask": 1.10001, "spread_points": 1.0},
        "features": {
            timeframe: {
                "return_over_bars": 0.01,
                "range_pct": 0.02,
                "close_position": 0.5,
                "average_true_range": 0.001,
                "direction": "UP",
            }
            for timeframe in ("M1", "M5", "M15", "H1")
        },
    }
    with sqlite3.connect(source) as connection:
        connection.execute(
            "UPDATE shadow_decisions SET snapshot_json=?", (json.dumps(snapshot),)
        )
    catalog = ExperienceCatalog(tmp_path / "artifact")
    ExperienceImporter(catalog).import_sources((source,))
    experience_id = catalog.active_records()[0].experience_id

    with sqlite3.connect(catalog.database_path) as connection:
        schema_version, projection_json = connection.execute(
            "SELECT feature_schema_version, projection_json "
            "FROM experience_feature_projections WHERE experience_id=?",
            (experience_id,),
        ).fetchone()

    assert schema_version == FEATURE_SCHEMA_VERSION
    assert json.loads(projection_json)["version"] == FEATURE_EXTRACTOR_VERSION
