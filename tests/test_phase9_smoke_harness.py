from __future__ import annotations

import hashlib
import json
import sqlite3
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.phase9_evidence_smoke import (
    NetworkAttempt,
    OfflineNetworkGuard,
    Phase8PreflightError,
    SmokeAcceptanceError,
    _append_audit,
    _configured_ollama_models,
    _local_runner_config,
    _run_fake_ab,
    _validate_smoke_result,
    build_report,
    resolve_verified_phase8_root,
    run_smoke,
)
from tests.fixtures.experience_source_db import create_source_db
from tradingagents.experience.catalog import ExperienceCatalog
from tradingagents.experience.importer import ExperienceImporter, ExperienceRebuilder
from tradingagents.forex.evidence_audit import EvidenceAuditStore
from tradingagents.forex.evidence_replay import EvidenceReplayReport, _json_value


def test_smoke_uses_local_ollama_model_defaults_and_env_overrides(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_QUICK_THINK_LLM", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_DEEP_THINK_LLM", raising=False)
    assert _configured_ollama_models() == ("qwen3.5:2b", "qwen3.5:4b")

    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "local-quick")
    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_LLM", "local-deep")
    assert _configured_ollama_models() == ("local-quick", "local-deep")


def test_smoke_runner_config_pins_local_evidence_deadline_and_roots(tmp_path: Path):
    config = _local_runner_config(
        knowledge_root=tmp_path / "p7",
        experience_root=tmp_path / "p8",
        model_path=tmp_path / "model",
        quick_model="qwen3.5:2b",
        deep_model="qwen3.5:4b",
    )

    assert config["forex_evidence_timeout_seconds"] == pytest.approx(30.0)
    assert config["forex_evidence_artifact_roots"] == {
        "knowledge": str(tmp_path / "p7"),
        "experience": str(tmp_path / "p8"),
        "knowledge_embedding_model_path": str(tmp_path / "model"),
    }


def test_smoke_rejects_bare_qwen_provider_label(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "qwen")
    with pytest.raises(ValueError, match="bare.*qwen"):
        _configured_ollama_models()


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
            connection.execute("INSERT INTO experience_feature_projections VALUES (?,?,?)", ("e1", "experience-features.v1", json.dumps({"version": "phase8-feature-extractor.v1", "cohort": ["EURUSD", "INTRADAY", "M15", "experience-features.v1", "phase8-feature-extractor.v1"], "values": [0], "mask": [1]})))
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
    assert result.numeric_similarity_count == 0
    assert result.numeric_statistics_eligible_count == 0


def test_phase8_numeric_preflight_uses_full_catalog_interfaces(monkeypatch, tmp_path: Path):
    import tradingagents.experience.outcomes as outcomes_module
    import tradingagents.experience.query as query_module

    observed: list[tuple[str, object]] = []
    original_query = query_module.ExperienceQueryService
    original_stats = outcomes_module.OutcomeStatsCalculator

    class QuerySpy(original_query):
        def __init__(self, records, *args, **kwargs):
            observed.append(("query", records))
            super().__init__(records, *args, **kwargs)

    class StatsSpy(original_stats):
        def __init__(self, records, *args, **kwargs):
            observed.append(("stats", records))
            super().__init__(records, *args, **kwargs)

    monkeypatch.setattr(query_module, "ExperienceQueryService", QuerySpy)
    monkeypatch.setattr(outcomes_module, "OutcomeStatsCalculator", StatsSpy)
    resolve_verified_phase8_root(_catalog(tmp_path / "p8"))
    assert [kind for kind, _ in observed] == ["query", "stats"]
    assert all(hasattr(records, "active_records") and hasattr(records, "historical_records") for _, records in observed)


def test_phase8_preflight_rejects_projection_without_persisted_versions(tmp_path: Path):
    root = _catalog(tmp_path / "p8", tier_c_only=True)
    with sqlite3.connect(root / "catalog.sqlite3") as connection:
        connection.execute("UPDATE experience_feature_projections SET projection_json=?", (json.dumps({"cohort": ["EURUSD", "INTRADAY", "M15", "experience-features.v1", "phase8-feature-extractor.v1"], "values": [0], "mask": [1]}),))
        connection.commit()
    with pytest.raises(Phase8PreflightError, match="projection"):
        resolve_verified_phase8_root(root)


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


def test_smoke_pass_requires_valid_non_fallback_replay_and_references():
    with pytest.raises(SmokeAcceptanceError):
        _validate_smoke_result({"comparison_status": "VALID", "bundle_status": "COMPLETE", "evidence_action": "BUY", "evidence_context_hash": "x", "citation_status": "VALID", "normalization_status": "NORMALIZED", "integration_status": "FALLBACK"})
    _validate_smoke_result({"comparison_status": "VALID", "bundle_status": "COMPLETE", "evidence_action": "BUY", "baseline_action": "HOLD", "rendered_context": "x", "evidence_context_hash": hashlib.sha256(b"x").hexdigest(), "context_hash_valid": True, "context_integrity_status": "COMPLETE", "citation_status": "VALID", "reference_validation_status": "VALID", "normalization_status": "NORMALIZED", "baseline_normalization_status": "NORMALIZED", "integration_status": "INJECTED", "evidence_audit_status": "VALID", "evidence_use_status": "NONE_RELEVANT", "available_references": (), "used_references": (), "rejected_references": (), "source_status": {"knowledge": "COMPLETE", "experience": "COMPLETE", "statistics": "COMPLETE"}, "pinned_phase7_generation_id": "p7", "pinned_phase8_generation_id": "p8", "errors": ()})
    with pytest.raises(SmokeAcceptanceError, match="normalization"):
        _validate_smoke_result({"comparison_status": "VALID", "bundle_status": "COMPLETE", "evidence_action": "BUY", "baseline_action": "HOLD", "rendered_context": "x", "evidence_context_hash": hashlib.sha256(b"x").hexdigest(), "context_hash_valid": True, "context_integrity_status": "COMPLETE", "citation_status": "VALID", "reference_validation_status": "VALID", "integration_status": "INJECTED", "evidence_use_status": "NONE_RELEVANT", "available_references": (), "source_status": {"knowledge": "COMPLETE", "experience": "COMPLETE", "statistics": "COMPLETE"}, "pinned_phase7_generation_id": "p7", "pinned_phase8_generation_id": "p8", "errors": ()})


def test_smoke_pass_rejects_partial_bundle():
    with pytest.raises(SmokeAcceptanceError, match="bundle"):
        _validate_smoke_result(
            {
                "comparison_status": "VALID",
                "bundle_status": "PARTIAL",
                "evidence_action": "BUY",
                "baseline_action": "HOLD",
                "rendered_context": "x",
                "evidence_context_hash": hashlib.sha256(b"x").hexdigest(),
                "context_hash_valid": True,
                "context_integrity_status": "COMPLETE",
                "citation_status": "VALID",
                "reference_validation_status": "VALID",
                "normalization_status": "NORMALIZED",
                "baseline_normalization_status": "NORMALIZED",
                "integration_status": "INJECTED",
                "available_references": (),
                "source_status": {"knowledge": "COMPLETE", "experience": "COMPLETE", "statistics": "COMPLETE"},
                "pinned_phase7_generation_id": "p7",
                "pinned_phase8_generation_id": "p8",
                "errors": (),
            }
        )


def test_fake_ab_executes_two_sequential_analyses_and_measures_them():
    calls: list[bool] = []

    class Fake:
        def analyze(self, **kwargs):
            calls.append(bool(kwargs["evidence_enabled"]))
            return {"normalized_action": "BUY" if kwargs["evidence_enabled"] else "HOLD"}

    report = _run_fake_ab(Fake(), snapshot=object(), snapshot_bytes=b"snapshot", config=object())
    assert calls == [False, True]
    assert report["analysis_count"] == 2
    assert report["baseline_action"] == "HOLD"
    assert report["evidence_action"] == "BUY"
    assert report["latency_seconds"] >= 0


def test_loopback_guard_patches_connect_ex_and_rejects_unc():
    import socket

    with OfflineNetworkGuard() as guard:
        with pytest.raises(NetworkAttempt):
            socket.socket().connect_ex(("203.0.113.1", 9))
        with pytest.raises(NetworkAttempt):
            socket.socket().connect((r"\\server\share", 9))
        with pytest.raises(NetworkAttempt):
            urllib.request.urlopen("file:///secret")
        with pytest.raises(NetworkAttempt):
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("203.0.113.1", 9))
    assert guard.external_network_attempts == 4


def test_loopback_guard_blocks_dns_and_request_object_and_allows_documented_pipe():
    import socket

    with OfflineNetworkGuard() as guard:
        with pytest.raises(NetworkAttempt):
            socket.getaddrinfo("198.51.100.4", 443)
        with pytest.raises(NetworkAttempt):
            urllib.request.urlopen(urllib.request.Request("https://198.51.100.4/"))
        with pytest.raises((OSError, TypeError)):
            socket.socket().connect(r"\\.\pipe\phase9-evidence")
    assert guard.external_network_attempts == 2


def test_phase8_preflight_rejects_missing_generation_metadata(tmp_path: Path):
    root = _catalog(tmp_path / "p8")
    with sqlite3.connect(root / "catalog.sqlite3") as connection:
        connection.execute("UPDATE experience_generations SET metadata_json='{}'")
        connection.commit()
    with pytest.raises(Phase8PreflightError, match="trust_policy_version"):
        resolve_verified_phase8_root(root)


def test_real_phase8_generation_passes_preflight_with_tier_c_only(tmp_path: Path):
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
        connection.commit()

    root = tmp_path / "artifact"
    catalog = ExperienceCatalog(root)
    ExperienceImporter(catalog).import_sources((source,))
    ExperienceRebuilder(catalog).rebuild()

    result = resolve_verified_phase8_root(root)

    assert result.tier_counts["TIER_C_DIAGNOSTIC_ONLY"] == 1
    assert result.numeric_trust_tiers == ("TIER_A_HIGH_TRUST", "TIER_B_LIMITED")
    assert result.numeric_similarity_count == 0
    assert result.numeric_statistics_eligible_count == 0


def test_report_redacts_sensitive_values_without_default_string_conversion():
    payload = build_report(error=ValueError("completion secret"), opaque=object())
    assert "completion secret" not in json.dumps(payload).lower()
    assert "object at" not in json.dumps(payload).lower()


def test_spawned_child_boundary_returns_network_telemetry(monkeypatch):
    import tradingagents.forex.evidence_runtime as runtime

    class Conn:
        values = []

        def send(self, value):
            self.values.append(value)

        def close(self):
            return None

    class Guard:
        loopback_connection_attempts = 3
        external_network_attempts = 1

    class Orchestrator:
        _phase9_network_guard = Guard()

        def query(self, _request):
            return {"status": "COMPLETE"}

    monkeypatch.setattr(runtime, "_resolve_factory", lambda _descriptor: lambda _config: Orchestrator())
    monkeypatch.setattr(runtime, "_validate_child_orchestrator", lambda *_args: None)
    monkeypatch.setattr(runtime, "_close_orchestrator", lambda *_args: None)
    connection = Conn()
    runtime._child_query(connection, {}, {"request": {}, "artifact_roots": {}})
    assert connection.values[-1][-1] == {"loopback_connection_attempts": 3, "external_network_attempts": 1}


def test_local_orchestrator_marks_all_readers_and_embedder_read_only(monkeypatch, tmp_path: Path):
    """The spawned evidence validator must accept every local read-only edge."""

    import types

    import scripts.phase9_evidence_smoke as smoke
    import tradingagents.experience.outcomes as outcomes_module
    import tradingagents.experience.query as experience_query_module
    import tradingagents.forex.evidence_runtime as runtime_module
    import tradingagents.knowledge.embeddings as embeddings_module
    import tradingagents.knowledge.lexical_index as lexical_module
    import tradingagents.knowledge.query as knowledge_query_module
    import tradingagents.knowledge.vector_index as vector_module

    class _KnowledgeCatalog:
        def __init__(self, _root):
            self.path = tmp_path / "knowledge"

        def active_generation(self):
            return types.SimpleNamespace(generation_id="p7", vector_location="vector", lexical_location="lexical")

    class _ExperienceCatalog:
        feature_vectors = {}
        profiles = {}

        def __init__(self, _root):
            self.path = tmp_path / "experience"

        def active_generation(self):
            return {"generation_id": "p8"}

    class _Embedder:
        @classmethod
        def from_config(cls, _config):
            return cls()

    class _VectorReader:
        def __init__(self, _location):
            pass

    class _LexicalReader:
        def __init__(self, _location):
            pass

    class _KnowledgeService:
        def __init__(self, *_args):
            pass

    class _ExperienceService:
        def __init__(self, *_args, **_kwargs):
            pass

    class _Stats:
        def __init__(self, *_args):
            pass

    monkeypatch.setattr(runtime_module, "ReadonlyKnowledgeCatalog", _KnowledgeCatalog)
    monkeypatch.setattr(runtime_module, "ReadonlyExperienceCatalog", _ExperienceCatalog)
    monkeypatch.setattr(smoke, "ReadonlyKnowledgeCatalog", _KnowledgeCatalog)
    monkeypatch.setattr(smoke, "ReadonlyExperienceCatalog", _ExperienceCatalog)
    monkeypatch.setattr(embeddings_module, "FastEmbedProvider", _Embedder)
    monkeypatch.setattr(vector_module, "VectorIndexReader", _VectorReader)
    monkeypatch.setattr(lexical_module, "LexicalIndexReader", _LexicalReader)
    monkeypatch.setattr(knowledge_query_module, "KnowledgeQueryService", _KnowledgeService)
    monkeypatch.setattr(experience_query_module, "ExperienceQueryService", _ExperienceService)
    monkeypatch.setattr(outcomes_module, "OutcomeStatsCalculator", _Stats)

    marked: list[object] = []
    monkeypatch.setattr(smoke, "approved_readonly_component", lambda component: marked.append(component) or component)
    model = tmp_path / "model"
    model.mkdir()
    orchestrator = smoke.build_local_orchestrator(
        {
            "knowledge": str(tmp_path / "p7"),
            "experience": str(tmp_path / "p8"),
            "knowledge_embedding_model_path": str(model),
        }
    )
    try:
        assert orchestrator is not None
        assert len(marked) >= 4
    finally:
        # The production child owns this guard until its query finishes.  The
        # unit test builds the factory in-process, so explicitly close that
        # process-scoped patch to avoid leaking network hooks into later tests.
        orchestrator._phase9_network_guard.__exit__(None, None, None)

    # The real smoke path passes the graph/runner configuration mapping, whose
    # artifact roots are nested under ``forex_evidence_artifact_roots``.  The
    # child factory must accept that shape just as it accepts the serialized
    # runtime envelope used by the deterministic evidence gate.
    nested_orchestrator = smoke.build_local_orchestrator(
        {
            "forex_evidence_artifact_roots": {
                "knowledge": str(tmp_path / "p7"),
                "experience": str(tmp_path / "p8"),
                "knowledge_embedding_model_path": str(model),
            },
            "forex_evidence_timeout_seconds": 30.0,
            "pinned_phase7_generation_id": "p7",
            "pinned_phase8_generation_id": "p8",
        }
    )
    try:
        assert nested_orchestrator is not None
    finally:
        nested_orchestrator._phase9_network_guard.__exit__(None, None, None)


def test_smoke_appends_one_metadata_only_audit_outside_source(tmp_path: Path):
    class Snapshot:
        timestamp = datetime(2026, 9, 12, tzinfo=timezone.utc)

    audit_path = tmp_path / "evidence_runtime" / "evidence_audit.sqlite3"
    result = {
        "integration_status": "INJECTED",
        "bundle_status": "COMPLETE",
        "evidence_action": "BUY",
        "rendered_context": "context",
        "evidence_context_hash": hashlib.sha256(b"context").hexdigest(),
        "knowledge_query": {"text": "forex liquidity", "fingerprint": "q-fp", "policy_version": "v2"},
        "knowledge_query_fingerprint": "q-fp",
        "query_policy_version": "v2",
        "available_references": ("K1", "E1", "S1"),
        "used_references": ("K1",),
        "evidence_use_status": "USED",
        "reference_validation_status": "VALID",
    }
    _append_audit(audit_path, decision_id="d1", source_run_id="r1", snapshot=Snapshot(), result=result)
    with sqlite3.connect(audit_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evidence_usage_audit").fetchone()[0] == 1
        row = connection.execute(
            "SELECT knowledge_query, knowledge_query_fingerprint, query_policy_version, available_knowledge_ids, evidence_refs_used FROM evidence_usage_audit"
        ).fetchone()
        assert row == ('{"fingerprint":"q-fp","policy_version":"v2","text":"forex liquidity"}', "q-fp", "v2", '["K1"]', '["K1"]')
        columns = {row[1] for row in connection.execute("PRAGMA table_info(evidence_usage_audit)")}
        assert not {"prompt", "completion", "reasoning"} & columns


def test_replay_smoke_can_write_phase9_audit_without_writing_phase56_decision(
    monkeypatch, tmp_path: Path
):
    """A failed evidence replay still records its Phase 9 audit separately."""

    source = tmp_path / "source.db"
    source.write_bytes(b"phase56-source")
    knowledge = tmp_path / "p7"
    experience = tmp_path / "p8"
    model = tmp_path / "model"
    knowledge.mkdir()
    experience.mkdir()
    (knowledge / "catalog.sqlite3").write_bytes(b"catalog")
    model.write_bytes(b"local-model")
    audit_path = tmp_path / "phase9-audit.sqlite3"
    for name in ("KNOWLEDGE_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "1")

    class _Snapshot:
        timestamp = datetime(2026, 9, 12, tzinfo=timezone.utc)

    snapshot = _Snapshot()
    row = {
        "decision_id": "decision-1",
        "source_run_id": "run-1",
        "snapshot_json": b"snapshot-bytes",
    }

    class _Reader:
        def read_snapshot(self):
            return type("SourceSnapshot", (), {"decisions": (row,)})()

    class _Generation:
        generation_id = "p7"

    class _Phase8:
        generation_id = "p8"

        def to_dict(self):
            return {"generation_id": self.generation_id}

    monkeypatch.setattr("scripts.phase9_evidence_smoke.ReadonlySourceReader", lambda _path: _Reader())
    monkeypatch.setattr(
        "scripts.phase9_evidence_smoke.SavedSnapshotCodec",
        type("Codec", (), {"from_source_row": staticmethod(lambda _row: snapshot)}),
    )
    monkeypatch.setattr("scripts.phase9_evidence_smoke._resolve_verified_phase7_generation", lambda _root: _Generation())
    monkeypatch.setattr("scripts.phase9_evidence_smoke.resolve_verified_phase8_root", lambda _root: _Phase8())
    monkeypatch.setattr("scripts.phase9_evidence_smoke._fingerprint", lambda _path: {"sha256": "same"})
    monkeypatch.setattr("scripts.phase9_evidence_smoke._source_fingerprint", lambda _path: {"db": {"sha256": "same"}})

    rendered = "fallback evidence context"
    replay_result = {
        "comparison_status": "VALID",
        "integration_status": "FALLBACK",
        "bundle_status": "EMPTY",
        "rendered_context": rendered,
        "evidence_context_hash": hashlib.sha256(rendered.encode()).hexdigest(),
        "available_references": (),
        "used_references": (),
        "rejected_references": (),
        "evidence_use_status": "UNAVAILABLE",
        "reference_validation_status": "NOT_RECORDED",
        "source_status": {},
        "knowledge_count": 0,
        "experience_count": 0,
        "statistics_count": 0,
        "pinned_phase7_generation_id": "p7",
        "pinned_phase8_generation_id": "p8",
        "provider": "fake",
        "models": {"quick": "fake", "deep": "fake"},
        "evidence_telemetry": {},
    }

    def replay_factory(**kwargs):
        assert kwargs["config"].source_database_path == source.resolve()
        return replay_result

    source_before = source.read_bytes()
    with pytest.raises(Exception, match="Phase 9 smoke"):
        run_smoke(
            source,
            experience_artifact_root=experience,
            knowledge_artifact_root=knowledge,
            knowledge_embedding_model_path=model,
            offline=True,
            replay_factory=replay_factory,
            audit_store=EvidenceAuditStore(audit_path),
        )

    assert source.read_bytes() == source_before
    with sqlite3.connect(audit_path) as connection:
        row = connection.execute(
            "SELECT decision_id, source_run_id, integration_status, bundle_status FROM evidence_usage_audit"
        ).fetchone()
    assert row == ("decision-1", "run-1", "FALLBACK", "EMPTY")


def test_smoke_rejects_unreferenced_evidence_items():
    with pytest.raises(SmokeAcceptanceError, match="reference"):
        _validate_smoke_result({"comparison_status": "VALID", "bundle_status": "COMPLETE", "baseline_action": "HOLD", "evidence_action": "BUY", "rendered_context": "x", "evidence_context_hash": hashlib.sha256(b"x").hexdigest(), "context_hash_valid": True, "context_integrity_status": "COMPLETE", "citation_status": "VALID", "reference_validation_status": "VALID", "normalization_status": "NORMALIZED", "baseline_normalization_status": "NORMALIZED", "integration_status": "INJECTED", "evidence_use_status": "NONE_RELEVANT", "knowledge_count": 1, "available_references": (), "source_status": {"knowledge": "COMPLETE", "experience": "COMPLETE", "statistics": "COMPLETE"}, "pinned_phase7_generation_id": "p7", "pinned_phase8_generation_id": "p8", "errors": ()})


def test_smoke_rejects_errors_inconsistent_use_and_forged_context_hash():
    base = {"comparison_status": "VALID", "bundle_status": "COMPLETE", "baseline_action": "HOLD", "evidence_action": "BUY", "rendered_context": "x", "evidence_context_hash": hashlib.sha256(b"x").hexdigest(), "context_hash_valid": True, "context_integrity_status": "COMPLETE", "citation_status": "VALID", "reference_validation_status": "VALID", "normalization_status": "NORMALIZED", "baseline_normalization_status": "NORMALIZED", "integration_status": "INJECTED", "evidence_use_status": "NONE_RELEVANT", "available_references": (), "used_references": (), "rejected_references": (), "source_status": {"knowledge": "COMPLETE"}, "pinned_phase7_generation_id": "p7", "pinned_phase8_generation_id": "p8", "errors": ()}
    for update, message in (({"errors": ("bad",)}, "errors"), ({"evidence_use_status": "USED"}, "use"), ({"rendered_context": "tampered"}, "hash"), ({"available_references": None}, "references")):
        with pytest.raises(SmokeAcceptanceError, match=message):
            _validate_smoke_result({**base, **update})


def test_standalone_replay_fails_on_child_external_counter(monkeypatch, tmp_path: Path):
    import scripts.phase9_evidence_replay as replay_module
    import scripts.phase9_evidence_smoke as smoke_module

    source = tmp_path / "source.db"
    p7 = tmp_path / "p7"
    p8 = tmp_path / "p8"
    model = tmp_path / "model"
    source.write_bytes(b"source")
    p7.mkdir()
    p8.mkdir()
    model.write_bytes(b"model")
    for name in ("KNOWLEDGE_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    row = {"decision_id": "d1", "snapshot_json": b"snapshot", "source_run_id": "r1"}
    snapshot = type("Snapshot", (), {"timestamp": datetime(2026, 9, 12, tzinfo=timezone.utc)})()
    monkeypatch.setattr(replay_module, "ReadonlySourceReader", lambda _path: type("Reader", (), {"read_snapshot": lambda self: type("Store", (), {"decisions": (row,)})()})())
    monkeypatch.setattr(replay_module.SavedSnapshotCodec, "from_source_row", staticmethod(lambda _row: snapshot))
    preflight = type("P8", (), {"generation_id": "p8"})()
    monkeypatch.setattr(smoke_module, "resolve_verified_phase8_root", lambda _root: preflight)
    monkeypatch.setattr(smoke_module, "_resolve_verified_phase7_generation", lambda _root: type("P7", (), {"generation_id": "p7"})())
    monkeypatch.setattr(smoke_module, "_fingerprint", lambda _path: {"sha256": "same"})
    monkeypatch.setattr(smoke_module, "_source_fingerprint", lambda _path: {"db": {"sha256": "same"}})

    class Replay:
        def run(self, *_args, **_kwargs):
            return type("Result", (), {"to_dict": lambda self: {"comparison_status": "VALID", "external_network_attempts": 2}})()

    with pytest.raises(replay_module.SnapshotReplayError, match="external network"):
        replay_module.run_replay(source, "d1", experience_artifact_root=p8, knowledge_artifact_root=p7, knowledge_embedding_model_path=model, offline=True, replay=Replay())


def test_child_network_telemetry_survives_fallback():
    from tradingagents.forex.evidence_context import EvidenceQueryPolicy
    from tradingagents.forex.evidence_runtime import EvidenceIntegrationService

    service = EvidenceIntegrationService(policy=EvidenceQueryPolicy(), orchestrator_factory=None, generation_provider=lambda: ("p7", "p8"), provider_endpoint="http://127.0.0.1:11434")
    service.last_child_network_telemetry = {"loopback_connection_attempts": 2, "external_network_attempts": 1}
    context = service._fallback(datetime(2026, 9, 12, tzinfo=timezone.utc), code="ORCHESTRATOR_FAILURE")
    assert context.diagnostics["network"] == {"loopback_connection_attempts": 2, "external_network_attempts": 1}


def test_replay_report_exposes_typed_context_normalization_and_reference_status():
    names = set(EvidenceReplayReport.__dataclass_fields__)
    assert {"integration_status", "normalization_status", "normalization_error", "context_integrity_status", "reference_validation_status", "available_references", "rendered_character_count", "source_status"} <= names


def test_standalone_replay_privacy_is_recursive_and_separator_aware():
    @dataclass
    class Nested:
        password: str
        safe: str

    payload = _json_value(
        {
            "nested": Nested(password="password=secret", safe="chain-of-thought details"),
            "items": [{"api-key": "credential-value", "chain_of_thought": "reasoning"}],
            "safe": object(),
        }
    )
    encoded = json.dumps(payload).lower()
    assert all(token not in encoded for token in ("password", "api-key", "chain_of_thought", "chain-of-thought", "credential", "reasoning"))
    assert "object at" not in encoded
