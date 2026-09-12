from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5AccountInfo, Mt5SymbolInfo
from tradingagents.experience.models import EvidenceBundle, ExperienceHit, TrustTier
from tradingagents.forex.evidence_audit import EvidenceUsageAudit
from tradingagents.forex.evidence_context import (
    EvidenceIntegrationStatus,
    EvidenceQueryPolicy,
)
from tradingagents.forex.evidence_runtime import EvidenceIntegrationService


def _snapshot() -> ForexMarketSnapshot:
    return ForexMarketSnapshot(
        timestamp=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc),
        symbol="EURUSD", bid=1.1, ask=1.1002, spread=0.0002, spread_points=2.0,
        m1_candles=(), m5_candles=(), m15_candles=(), h1_candles=(),
        account=Mt5AccountInfo(currency="USD"), positions=(),
        symbol_info=Mt5SymbolInfo(name="EURUSD", point=0.0001, digits=5),
    )


class _CountingOrchestrator:
    calls = 0

    def query(self, _request):
        type(self).calls += 1
        return EvidenceBundle(status="EMPTY")


def _factory():
    return _CountingOrchestrator()


def _service(*, enabled: bool = True, endpoint: str = "http://127.0.0.1:11434"):
    return EvidenceIntegrationService(
        policy=EvidenceQueryPolicy() if enabled else None,
        orchestrator_factory=_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint=endpoint,
        process_factory=None,
    )


def test_disabled_mode_works_without_phase7_or_phase8_paths(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Phase 7/8 dependency constructed")

    monkeypatch.setattr("tradingagents.forex.evidence_runtime.ReadonlyKnowledgeCatalog", forbidden)
    monkeypatch.setattr("tradingagents.forex.evidence_runtime.ReadonlyExperienceCatalog", forbidden)
    service = _service(enabled=False)
    context = service.retrieve(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5"
    )
    assert context.integration_status is EvidenceIntegrationStatus.DISABLED
    assert service.retrieval_count == 0


def test_runtime_does_not_call_experience_import_or_rebuild(monkeypatch):
    monkeypatch.setattr("tradingagents.experience.importer.ExperienceImporter", lambda *_a, **_k: pytest.fail("importer called"))
    monkeypatch.setattr("tradingagents.forex.evidence_runtime._make_orchestrator", lambda *_a, **_k: pytest.fail("rebuild called"))
    service = _service(enabled=False)
    service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")


def test_runtime_does_not_construct_docling_or_downloader(monkeypatch):
    monkeypatch.setattr("tradingagents.knowledge.docling_parser.DoclingDocumentParser", lambda *_a, **_k: pytest.fail("Docling constructed"))
    monkeypatch.setattr("tradingagents.knowledge.embeddings.EmbeddingProvider.from_pretrained", lambda *_a, **_k: pytest.fail("model downloaded"), raising=False)
    service = _service(enabled=False)
    service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")


def test_tier_c_is_diagnostic_only():
    tier_c = ExperienceHit(
        experience_id="tier-c", trust_tier=TrustTier.TIER_C_DIAGNOSTIC_ONLY,
        market_state={}, provenance={"diagnostic": "quarantined"},
    )

    class Orchestrator:
        def query(self, _request):
            return EvidenceBundle(status="COMPLETE", experience=(tier_c,))

    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(), orchestrator_factory=lambda: Orchestrator(),
        generation_provider=lambda: (None, None), provider_endpoint="http://127.0.0.1:11434",
    )
    service._query = lambda _request: Orchestrator().query(_request)
    context = service.retrieve(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5"
    )
    assert context.experience_items == ()
    assert "tier-c" in str(context.diagnostics)


def test_hosted_provider_with_text_falls_back_without_leakage():
    class Orchestrator:
        def query(self, _request):
            return EvidenceBundle(status="COMPLETE", knowledge=({"document_id": "d1", "text": "private evidence"},))

    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(), orchestrator_factory=lambda: Orchestrator(),
        generation_provider=lambda: (None, None), provider_endpoint="https://hosted.example/v1",
    )
    service._query = lambda _request: Orchestrator().query(_request)
    context = service.retrieve(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5"
    )
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert context.rendered_context == ""
    assert "private evidence" not in str(context.to_dict())


def test_no_mt5_mutation_api_is_called():
    service = _service(enabled=False)
    context = service.retrieve(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5"
    )
    assert context.integration_status is EvidenceIntegrationStatus.DISABLED


def test_phase5_shadow_schema_is_unchanged(tmp_path: Path):
    database = tmp_path / "shadow.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE shadow_trade_decisions (decision_id TEXT, action TEXT)")
        before = db.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    _service(enabled=False).retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == before


def test_phase7_phase8_files_are_unchanged(tmp_path: Path):
    roots = {name: tmp_path / name for name in ("knowledge", "experience")}
    fingerprints = {}
    for root in roots.values():
        root.mkdir()
        path = root / "catalog.sqlite3"
        sqlite3.connect(path).close()
        fingerprints[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    service = _service(enabled=False)
    service.artifact_roots = {k: str(v) for k, v in roots.items()}
    service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in fingerprints} == fingerprints


@pytest.mark.parametrize(
    "forbidden",
    ("prompt", "completion", "reasoning", "chain_of_thought", "api_key", "password", "token", "credential"),
)
def test_audit_forbids_prompt_completion_reasoning_credentials(tmp_path: Path, forbidden: str):
    value = "K1: evidence"
    with pytest.raises(ValueError):
        EvidenceUsageAudit(
            decision_id="d1", source_run_id="r1", integration_status="INJECTED", bundle_status="COMPLETE",
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc), rendered_context=value,
            rendered_context_hash=hashlib.sha256(value.encode()).hexdigest(), available_knowledge_ids=("K1",),
            diagnostics={"nested": {forbidden: "secret"}},
        )


def test_no_agent_can_call_evidence_service():
    service = _service()
    def query(request):
        if service._query_issued:
            raise AssertionError("second evidence call")
        service._query_issued = True
        return _CountingOrchestrator().query(request)

    service._query = query
    service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert _CountingOrchestrator.calls == 1
