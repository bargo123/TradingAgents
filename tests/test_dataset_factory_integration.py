from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tests.fixtures.dataset_factory import make_real_fixtures
from tradingagents.datasets.factory import DatasetFactory
from tradingagents.datasets.models import DatasetConfig
from tradingagents.datasets.writer import validate_generation


def _build(root: Path, paths: tuple[Path, ...], output: Path):
    return DatasetFactory().build(
        DatasetConfig(
            paths, root / "phase8", root / "phase9.sqlite3", output,
            filters={"as_of": datetime(2026, 1, 20, tzinfo=timezone.utc)},
        )
    )


def _published(root: Path) -> Path:
    return next(path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").is_file())


def test_public_factory_uses_real_adapters_end_to_end(tmp_path: Path) -> None:
    artifacts = make_real_fixtures(tmp_path)
    duplicate_source = tmp_path / "phase56-copy.sqlite3"
    shutil.copyfile(artifacts["source"], duplicate_source)

    first_output = tmp_path / "out-first"
    first = _build(tmp_path, (duplicate_source, artifacts["source"]), first_output)
    assert first.status == "PUBLISHED"
    generation = _published(first_output)
    assert validate_generation(generation).valid

    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["safety"] == {
        "network_attempts": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "mt5_calls": 0,
    }
    assert manifest["examples"] == 8
    assert manifest["split_status"] == "COMPLETE"
    assert manifest["group_count"] == 8
    reasons = manifest["counts"]["reason_counts"]
    assert reasons["UNTRUSTED_TIER"] >= 2
    assert reasons["DECISION_CONTEXT_INCOMPLETE"] >= 1
    assert reasons["OUTCOME_INELIGIBLE"] >= 1
    assert reasons["TEMPORAL_INVALID"] >= 1
    assert reasons["DUPLICATE"] >= 1

    rows = [json.loads(line) for line in (generation / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    assert all("prompt" not in json.dumps(row).lower() for row in rows)
    assert all("reasoning" not in json.dumps(row).lower() for row in rows)
    assert {row["provenance"]["phase56"]["source_id"] for row in rows}
    assert {row["provenance"]["phase8"]["source_fingerprint"]["source_id"] for row in rows}
    assert {row["provenance"]["phase9"]["source_fingerprint"]["source_id"] for row in rows}

    # The factory sorts paths before reading, so reversing caller input cannot
    # alter generation bytes (the duplicate copy is deterministically ignored).
    second_output = tmp_path / "out-second"
    second = _build(tmp_path, (artifacts["source"], duplicate_source), second_output)
    assert second.status == "PUBLISHED"
    generation2 = _published(second_output)
    assert {
        path.name: path.read_bytes() for path in generation.iterdir() if path.is_file()
    } == {
        path.name: path.read_bytes() for path in generation2.iterdir() if path.is_file()
    }


def test_fixture_covers_future_and_unavailable_without_fabricating_labels(tmp_path: Path) -> None:
    artifacts = make_real_fixtures(tmp_path)
    report = _build(tmp_path, (artifacts["source"],), tmp_path / "out")
    excluded = {item.decision_id: {reason.value for reason in item.reasons} for item in report.exclusions}
    assert "unavailable-1" in excluded and "OUTCOME_UNAVAILABLE" not in excluded["unavailable-1"]
    assert "future-1" in excluded and "TEMPORAL_INVALID" in excluded["future-1"]
    assert report.manifest is not None
    assert report.manifest.safety["llm_calls"] == 0


def test_null_action_large_snapshot_is_normalization_exclusion_not_source_failure(
    tmp_path: Path,
) -> None:
    artifacts = make_real_fixtures(tmp_path)
    snapshot = {
        "symbol": "EURUSD",
        "point": 0.00001,
        "digits": 5,
        "quote": {"bid": 1.1, "ask": 1.1001, "spread_points": 10},
        "features": {"M5": {"return_over_bars": 0.1}},
        "diagnostic_padding": "x" * 5000,
    }
    with sqlite3.connect(artifacts["source"]) as db:
        db.execute(
            "UPDATE shadow_decisions SET action=NULL, snapshot_json=?",
            (json.dumps(snapshot),),
        )
        db.commit()

    report = _build(tmp_path, (artifacts["source"],), tmp_path / "out")
    reasons = {reason.value for item in report.exclusions for reason in item.reasons}
    assert "NORMALIZATION_FAILED" in reasons
    assert "SOURCE_INTEGRITY_FAILED" not in reasons
