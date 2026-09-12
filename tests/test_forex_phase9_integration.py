from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5AccountInfo, Mt5SymbolInfo
from tradingagents.experience.catalog import ExperienceCatalog
from tradingagents.experience.models import EvidenceBundle
from tradingagents.forex.evidence_context import (
    EvidenceBundleStatus,
    EvidenceIntegrationStatus,
    EvidenceQueryPolicy,
)
from tradingagents.forex.evidence_runtime import (
    EvidenceIntegrationService,
    ReadonlyExperienceCatalog,
    ReadonlyKnowledgeCatalog,
)
from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.embeddings import EmbeddingSpecMismatch
from tradingagents.knowledge.models import (
    EmbeddingSpec,
    IndexGeneration,
    KnowledgeHit,
    KnowledgeQuery,
)
from tradingagents.knowledge.query import KnowledgeQueryService


def _snapshot() -> ForexMarketSnapshot:
    return ForexMarketSnapshot(
        timestamp=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc),
        symbol="EURUSD",
        bid=1.1,
        ask=1.1002,
        spread=0.0002,
        spread_points=2.0,
        m1_candles=(),
        m5_candles=(),
        m15_candles=(),
        h1_candles=(),
        account=Mt5AccountInfo(currency="USD"),
        positions=(),
        symbol_info=Mt5SymbolInfo(name="EURUSD", point=0.0001, digits=5),
    )


class _FakeOrchestrator:
    def __init__(self, bundle: EvidenceBundle | None = None, error: Exception | None = None):
        self.bundle = bundle or EvidenceBundle(status="EMPTY")
        self.error = error
        self.calls = []

    def query(self, request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return self.bundle


def _empty_factory(_envelope=None):
    return _FakeOrchestrator()


def _partial_factory(_envelope=None):
    return _FakeOrchestrator(
        EvidenceBundle(status="PARTIAL", source_status={"knowledge": "COMPLETE", "experience": "FAILED"})
    )


def _error_factory(_envelope=None):
    return _FakeOrchestrator(error=RuntimeError("boom"))


def _child_evidence_factory(_envelope=None):
    return _FakeOrchestrator(EvidenceBundle(status="COMPLETE", knowledge=(KnowledgeHit(chunk_id="child-chunk", document_id="child-doc", text="child evidence"),)))


class _SlowOrchestrator:
    def query(self, _request):
        time.sleep(1.0)
        return EvidenceBundle(status="EMPTY")


def _slow_factory(_envelope=None):
    return _SlowOrchestrator()


class _ImmediateProcess:
    def __init__(self, *, target, args):
        self.target, self.args, self._alive = target, args, False

    def start(self):
        self.target(*self.args)

    def join(self, _timeout=None):
        return None

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._alive = False


class _HangingProcess(_ImmediateProcess):
    def start(self):
        self._alive = True


def _process_factory(*, target, args):
    return _ImmediateProcess(target=target, args=args)


def _hanging_process_factory(*, target, args):
    return _HangingProcess(target=target, args=args)


def test_real_spawn_query_uses_serializable_envelope_and_reaches_child():
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=5),
        orchestrator_factory=_child_evidence_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.INJECTED
    assert context.knowledge_items[0].authoritative_id == "child-chunk"
    assert service.active_evidence_workers == 0


def test_real_spawn_timeout_leaves_zero_workers():
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=0.05),
        orchestrator_factory=_slow_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert context.diagnostics["integration"]["code"] == "EVIDENCE_TIMEOUT"
    assert service.active_evidence_workers == 0


def test_spawn_process_args_contain_no_callable_closures():
    captured = {}

    def process_factory(*, target, args):
        captured["args"] = args
        return _ImmediateProcess(target=target, args=args)

    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(),
        orchestrator_factory=_empty_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        process_factory=process_factory,
        artifact_roots={"knowledge": "C:/knowledge", "experience": "C:/experience"},
    )
    service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert not any(callable(value) for value in captured["args"])
    assert captured["args"][2]["artifact_roots"] == {
        "knowledge": "C:/knowledge", "experience": "C:/experience"
    }


def test_populated_phase8_readonly_adapter_matches_writer_semantics(tmp_path: Path):
    writer = ExperienceCatalog(tmp_path / "experience")
    first = writer.upsert_source_alias(
        "source", "decision-1", "fp-1", symbol="EURUSD",
        analysis_profile="INTRADAY", analysis_timeframe="M5",
        analysis_snapshot_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        decision_completed_timestamp=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        market_state={"values": [1.0], "mask": [True], "feature_names": ["x"], "cohort": ["EURUSD", "INTRADAY", "M5", "s", "e"]},
        trust="TIER_A_HIGH_TRUST",
    )
    second = writer.upsert_source_alias(
        "source", "decision-2", "fp-2", symbol="EURUSD",
        analysis_profile="INTRADAY", analysis_timeframe="M5",
        analysis_snapshot_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        decision_completed_timestamp=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        market_state={"values": [2.0], "mask": [True], "feature_names": ["x"], "cohort": ["EURUSD", "INTRADAY", "M5", "s", "e"]},
        trust="TIER_B_LIMITED",
    )
    writer.store_feature_projection(first.experience_id, dict(first.market_state))
    writer.append_evaluation_snapshot(first.experience_id, {"evaluation_status": "COMPLETE", "evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 60}, "eval-1")
    writer.mark_last_alias_removed(second.experience_id)
    writer.publish_generation("generation-1", population_fingerprint="population-1", metadata={"trust_policy_version": "trust-policy.v1"})
    readonly = ReadonlyExperienceCatalog(tmp_path / "experience")
    assert tuple(r.experience_id for r in readonly.active_records()) == (first.experience_id,)
    assert tuple(r.experience_id for r in readonly.historical_records()) == (second.experience_id,)
    assert readonly.active_records()[0].source_aliases == writer.active_records()[0].source_aliases
    assert readonly.evaluation_snapshots(first.experience_id) == writer.evaluation_snapshots(first.experience_id)
    assert readonly.feature_projection(first.experience_id)["values"] == [1.0]
    assert readonly.active_generation() == writer.active_generation()
    assert readonly.active_records()[0].trust == writer.active_records()[0].trust


def test_populated_phase7_readonly_adapter_matches_published_generation(tmp_path: Path):
    writer = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    writer.initialize()
    with sqlite3.connect(tmp_path / "catalog.sqlite3") as db:
        spec = EmbeddingSpec().to_dict()
        db.execute("INSERT INTO knowledge_index_generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("g1", "vector/g1", "keyword/g1", json.dumps(spec), "fts5-v1", "{}", "index-v1", "pop", "identity", 1, 1, 1, 1, "VALIDATED", None, None, "{}", 1))
        db.execute("INSERT INTO knowledge_documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("doc", "hash", "{}", "parser", "1", "cfg", "PARSED", 1, 1, 1, "g1", "pop", "{}", None, "", ""))
        db.execute("INSERT INTO knowledge_resources VALUES (?,?,?,?,?,?,?,?,?,?,?)", ("res", None, None, None, "hash", None, None, "PARSED", None, "", ""))
        db.execute("INSERT INTO knowledge_aliases VALUES (?,?,?,?,?,?)", ("res", "doc", "hash", "CURRENT", None, ""))
    readonly = ReadonlyKnowledgeCatalog(tmp_path)
    assert readonly.active_generation().generation_id == writer.active_generation().generation_id
    assert readonly.document_is_retrieval_ready("doc") == writer.document_is_retrieval_ready("doc") is True


class _QueryReader:
    def __init__(self, spec):
        self._spec = spec

    def metadata(self):
        return {"generation_id": "g", "population_hash": "p", "embedding_spec": self._spec.to_dict()}

    def rows(self):
        return ()

    def search(self, *_args, **_kwargs):
        return ()


def test_complete_embedding_spec_compatibility_matrix_fails_closed():
    base = EmbeddingSpec(model_id="model", resolved_model_version="version", runtime="runtime", artifact_hash="hash", dimensions=3, tokenizer_fingerprint="tokenizer", corpus_instruction_policy="corpus", query_instruction_policy="query")
    generation = IndexGeneration(generation_id="g", vector_location="vector", lexical_location="lexical", embedding_spec=base, population_hash="p", population_identity="identity", document_count=0, chunk_count=0, vector_ready=True, lexical_ready=True, status="VALIDATED")
    fields = ("model_id", "resolved_model_version", "runtime", "artifact_hash", "dimensions", "normalization_policy", "tokenizer_fingerprint", "model_max_input_tokens", "special_token_budget", "effective_corpus_content_token_limit", "corpus_instruction_policy", "corpus_instruction_version", "query_instruction_policy", "query_instruction_version")
    for field in fields:
        value = {"dimensions": 4, "model_max_input_tokens": 256, "effective_corpus_content_token_limit": 254, "special_token_budget": 1}.get(field, f"different-{field}")
        if field in {"model_max_input_tokens", "effective_corpus_content_token_limit", "special_token_budget", "dimensions"}:
            value = int(value)
        changes = {field: value}
        if field in {"model_max_input_tokens", "effective_corpus_content_token_limit", "special_token_budget"}:
            changes.update(model_max_input_tokens=256, effective_corpus_content_token_limit=254, special_token_budget=1)
        mismatch = replace(base, **changes)
        catalog = type("Catalog", (), {"active_generation": lambda self: generation, "document_is_retrieval_ready": lambda self, _id: True})()
        reader = _QueryReader(base)
        embedder = type("Embedder", (), {"spec": mismatch, "embed": lambda self, texts, **kwargs: [[0.0, 0.0, 0.0] for _ in texts]})()
        service = KnowledgeQueryService(reader, reader, catalog, embedder)
        with pytest.raises(EmbeddingSpecMismatch):
            service.search(KnowledgeQuery("q"))


def _service(orchestrator=None, *, enabled=True, endpoint="http://127.0.0.1:11434", factory=None):
    return EvidenceIntegrationService(
        policy=EvidenceQueryPolicy() if enabled else None,
        orchestrator_factory=factory or _empty_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint=endpoint,
        clock=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
        process_factory=_process_factory,
    )


def test_disabled_service_constructs_no_dependencies():
    factory_calls = []
    service = EvidenceIntegrationService(
        policy=None,
        orchestrator_factory=lambda: factory_calls.append(True),
        generation_provider=lambda: (_ for _ in ()).throw(AssertionError("must not read generations")),
        provider_endpoint="https://hosted.example",
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.DISABLED
    assert factory_calls == []


def test_enabled_service_calls_orchestrator_once():
    service = _service(_FakeOrchestrator())
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert service.retrieval_count == 1
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK


def test_enabled_request_uses_snapshot_as_of():
    service = _service(_FakeOrchestrator())
    snapshot = _snapshot()
    service.retrieve(snapshot, resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert service.last_request.as_of == snapshot.timestamp


def test_partial_bundle_maps_to_fallback():
    context = _service(factory=_partial_factory).retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert context.bundle_status is EvidenceBundleStatus.PARTIAL


def test_orchestrator_exception_maps_to_fallback():
    context = _service(factory=_error_factory).retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert "ORCHESTRATOR" in str(context.diagnostics).upper()


def test_timeout_returns_evidence_timeout():
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=0),
        orchestrator_factory=_empty_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        process_factory=_hanging_process_factory,
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert "EVIDENCE_TIMEOUT" in str(context.diagnostics)


def test_timeout_terminates_all_evidence_workers():
    pytest.importorskip("multiprocessing")
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=0),
        orchestrator_factory=_empty_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        process_factory=_hanging_process_factory,
    )
    service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert getattr(service, "active_evidence_workers", 0) == 0


def test_timeout_does_not_write_or_start_maintenance(tmp_path: Path):
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=0),
        orchestrator_factory=_empty_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        process_factory=_hanging_process_factory,
    )
    service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert not list(tmp_path.iterdir())


def test_readonly_knowledge_catalog_uses_mode_ro(tmp_path: Path):
    db = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE knowledge_index_generations (is_active INTEGER, generation_id TEXT, vector_location TEXT, lexical_location TEXT, embedding_spec_json TEXT, lexical_index_version TEXT, lexical_tokenizer_settings_json TEXT, index_version TEXT, population_hash TEXT, population_identity TEXT, document_count INTEGER, chunk_count INTEGER, vector_ready INTEGER, lexical_ready INTEGER, status TEXT, created_at TEXT, activated_at TEXT, component_versions_json TEXT)")
    catalog = ReadonlyKnowledgeCatalog(tmp_path)
    assert catalog.active_generation() is None
    with pytest.raises(sqlite3.OperationalError):
        catalog._connection.execute("CREATE TABLE forbidden (x INTEGER)")


def test_readonly_experience_catalog_uses_mode_ro(tmp_path: Path):
    db = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE experience_records (experience_id TEXT PRIMARY KEY, source_decision_id TEXT, source_database_id TEXT, source_decision_fingerprint TEXT, source_run_id TEXT, symbol TEXT, requested_symbol TEXT, analysis_profile TEXT, analysis_timeframe TEXT, analysis_snapshot_timestamp TEXT, decision_completed_timestamp TEXT, decision_reference_timestamp TEXT, market_state_json TEXT, decision_evidence_json TEXT, provenance_json TEXT, trust TEXT, tombstoned INTEGER, source_evaluation_fingerprints_json TEXT)")
        connection.execute("CREATE TABLE experience_source_aliases (source_database_id TEXT, source_decision_id TEXT, accepted_fingerprint TEXT, experience_id TEXT, state TEXT, observed_at TEXT, scan_status TEXT)")
        connection.execute("CREATE TABLE experience_feature_projections (experience_id TEXT, feature_schema_version TEXT, projection_json TEXT NOT NULL)")
    catalog = ReadonlyExperienceCatalog(tmp_path)
    assert catalog.active_records() == ()
    with pytest.raises(sqlite3.OperationalError):
        catalog._connection.execute("CREATE TABLE forbidden (x INTEGER)")


def test_readonly_catalog_context_manager_closes_owned_handle(tmp_path: Path):
    db = tmp_path / "catalog.sqlite3"
    sqlite3.connect(db).close()
    catalog = ReadonlyKnowledgeCatalog(tmp_path)
    with catalog as active:
        assert active.active_generation() is None
    with pytest.raises(sqlite3.ProgrammingError):
        catalog._connection.execute("SELECT 1")


def test_readonly_experience_adapter_matches_phase8_reader_semantics(tmp_path: Path):
    sqlite3.connect(tmp_path / "catalog.sqlite3").close()
    assert ReadonlyExperienceCatalog(tmp_path).historical_records() == ()


def test_readonly_knowledge_adapter_matches_published_generation_semantics(tmp_path: Path):
    sqlite3.connect(tmp_path / "catalog.sqlite3").close()
    assert ReadonlyKnowledgeCatalog(tmp_path).document_is_retrieval_ready("missing") is False


def test_query_embedding_spec_mismatch_fails_closed():
    orchestrator = _FakeOrchestrator(error=ValueError("embedding specification mismatch"))
    context = _service(orchestrator).retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
