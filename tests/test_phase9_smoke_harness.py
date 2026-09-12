from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.phase9_evidence_smoke import (
    NetworkAttempt,
    OfflineNetworkGuard,
    Phase8PreflightError,
    build_report,
    resolve_verified_phase8_root,
    run_smoke,
)


def _catalog(root: Path, *, tier_c_only: bool = False, published: bool = True) -> Path:
    root.mkdir()
    db = root / "catalog.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.executescript(
            """
            CREATE TABLE experience_catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE experience_generations (
                generation_id TEXT PRIMARY KEY, population_fingerprint TEXT NOT NULL,
                metadata_json TEXT NOT NULL, published_at TEXT NOT NULL, active INTEGER NOT NULL
            );
            CREATE TABLE experience_records (
                experience_id TEXT PRIMARY KEY, source_decision_id TEXT NOT NULL UNIQUE,
                source_database_id TEXT NOT NULL, source_decision_fingerprint TEXT NOT NULL,
                source_run_id TEXT, symbol TEXT NOT NULL, requested_symbol TEXT,
                analysis_profile TEXT, analysis_timeframe TEXT, analysis_snapshot_timestamp TEXT NOT NULL,
                decision_completed_timestamp TEXT, decision_reference_timestamp TEXT,
                market_state_json TEXT NOT NULL, decision_evidence_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL, trust TEXT NOT NULL, tombstoned INTEGER NOT NULL,
                source_evaluation_fingerprints_json TEXT NOT NULL
            );
            CREATE TABLE experience_source_aliases (
                source_database_id TEXT NOT NULL, source_decision_id TEXT NOT NULL,
                accepted_fingerprint TEXT NOT NULL, experience_id TEXT NOT NULL,
                state TEXT NOT NULL
            );
            CREATE TABLE experience_outcome_snapshots (
                experience_id TEXT NOT NULL, evaluation_basis TEXT, horizon_seconds INTEGER,
                fingerprint TEXT NOT NULL, evaluation_json TEXT NOT NULL, provenance_json TEXT NOT NULL,
                observed_at TEXT NOT NULL
            );
            CREATE TABLE experience_feature_projections (
                experience_id TEXT, feature_schema_version TEXT, projection_json TEXT NOT NULL
            );
            """
        )
        metadata = {
            "trust_policy_version": "trust-policy.v1",
            "feature_schema_version": "experience-features.v1",
            "feature_extractor_version": "phase8-feature-extractor.v1",
            "trust_tiers": ["TIER_A_HIGH_TRUST", "TIER_B_LIMITED"],
        }
        connection.execute("INSERT INTO experience_catalog_meta VALUES (?, ?)", ("schema_version", "phase8.catalog.v1"))
        connection.execute(
            "INSERT INTO experience_generations VALUES (?, ?, ?, ?, ?)",
            ("g8", "population", json.dumps(metadata), "2026-09-12T00:00:00+00:00", int(published)),
        )
        if tier_c_only:
            connection.execute(
                "INSERT INTO experience_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("e1", "d1", "s1", "f1", None, "EURUSD", None, "INTRADAY", "M15", "2026-09-12T00:00:00+00:00", None, None, "{}", "{}", "{}", "TIER_C_DIAGNOSTIC_ONLY", 0, "{}"),
            )
            connection.execute("INSERT INTO experience_source_aliases VALUES (?,?,?,?,?)", ("s1", "d1", "f1", "e1", "CURRENT"))
            connection.execute("INSERT INTO experience_feature_projections VALUES (?,?,?)", ("e1", "experience-features.v1", json.dumps({"cohort": ["EURUSD", "INTRADAY", "M15", "experience-features.v1", "phase8-feature-extractor.v1"], "values": [0], "mask": [1]})))
        connection.commit()
    return root


def test_smoke_requires_offline_flag(tmp_path: Path):
    with pytest.raises(ValueError, match="offline"):
        run_smoke(tmp_path / "source.db", experience_artifact_root=tmp_path / "p8", knowledge_artifact_root=tmp_path / "p7", knowledge_embedding_model_path=tmp_path / "model", offline=False)


def test_smoke_rejects_missing_source_or_artifact(tmp_path: Path):
    with pytest.raises((ValueError, Phase8PreflightError, FileNotFoundError)):
        run_smoke(tmp_path / "missing.db", experience_artifact_root=tmp_path / "missing-p8", knowledge_artifact_root=tmp_path / "missing-p7", knowledge_embedding_model_path=tmp_path / "missing-model", offline=True)


def test_phase8_preflight_validates_generation_policy_schema_and_queries(tmp_path: Path):
    root = _catalog(tmp_path / "p8")
    result = resolve_verified_phase8_root(root)
    assert result.generation_id == "g8"
    assert result.trust_policy_version == "trust-policy.v1"


def test_phase8_preflight_accepts_tier_c_only_generation(tmp_path: Path):
    result = resolve_verified_phase8_root(_catalog(tmp_path / "p8", tier_c_only=True))
    assert result.tier_counts["TIER_C_DIAGNOSTIC_ONLY"] == 1
    assert result.numeric_trust_tiers == ("TIER_A_HIGH_TRUST", "TIER_B_LIMITED")


def test_smoke_rejects_external_artifact_mutation_target(tmp_path: Path):
    root = _catalog(tmp_path / "p8")
    with pytest.raises(Phase8PreflightError, match="artifact"):
        resolve_verified_phase8_root(root / "nested")


def test_smoke_reports_source_fingerprints_and_network_counts():
    payload = build_report(source_fingerprint={"sha256": "x"}, artifact_fingerprints={}, loopback_connection_attempts=1, external_network_attempts=0)
    assert payload["source_fingerprint"]["sha256"] == "x"
    assert payload["loopback_connection_attempts"] == 1
    assert payload["external_network_attempts"] == 0


def test_smoke_uses_saved_snapshot_without_mt5():
    from scripts import phase9_evidence_smoke as smoke

    assert smoke._uses_saved_snapshot() is True


def test_smoke_loopback_guard_rejects_external_connection():
    import socket

    with OfflineNetworkGuard() as guard, pytest.raises(NetworkAttempt):
        socket.create_connection(("203.0.113.1", 9), timeout=0.01)
    assert guard.external_network_attempts == 1


def test_smoke_report_forbids_prompt_completion_reasoning():
    payload = build_report(warnings=["safe"], telemetry={"prompt": "secret", "ok": 1})
    encoded = json.dumps(payload).lower()
    assert all(token not in encoded for token in ("prompt", "completion", "reasoning"))
