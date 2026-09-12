from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5AccountInfo, Mt5SymbolInfo
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


def _service(orchestrator, *, enabled=True, endpoint="http://127.0.0.1:11434"):
    return EvidenceIntegrationService(
        policy=EvidenceQueryPolicy() if enabled else None,
        orchestrator_factory=lambda: orchestrator,
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
    orchestrator = _FakeOrchestrator()
    context = _service(orchestrator).retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert len(orchestrator.calls) == 1
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK


def test_enabled_request_uses_snapshot_as_of():
    orchestrator = _FakeOrchestrator()
    snapshot = _snapshot()
    _service(orchestrator).retrieve(snapshot, resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert orchestrator.calls[0].as_of == snapshot.timestamp


def test_partial_bundle_maps_to_fallback():
    orchestrator = _FakeOrchestrator(EvidenceBundle(status="PARTIAL", source_status={"knowledge": "COMPLETE", "experience": "FAILED"}))
    context = _service(orchestrator).retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert context.bundle_status is EvidenceBundleStatus.PARTIAL


def test_orchestrator_exception_maps_to_fallback():
    orchestrator = _FakeOrchestrator(error=RuntimeError("boom"))
    context = _service(orchestrator).retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert "ORCHESTRATOR" in str(context.diagnostics).upper()


def test_timeout_returns_evidence_timeout():
    orchestrator = _FakeOrchestrator()
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=0),
        orchestrator_factory=lambda: orchestrator,
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
        orchestrator_factory=lambda: _FakeOrchestrator(),
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        process_factory=_hanging_process_factory,
    )
    service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert getattr(service, "active_evidence_workers", 0) == 0


def test_timeout_does_not_write_or_start_maintenance(tmp_path: Path):
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=0),
        orchestrator_factory=lambda: _FakeOrchestrator(),
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
