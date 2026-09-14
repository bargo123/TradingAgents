from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5AccountInfo, Mt5SymbolInfo
from tradingagents.experience.catalog import ExperienceCatalog
from tradingagents.experience.features import FEATURE_NAMES_V1
from tradingagents.experience.models import (
    EvidenceBundle,
    ExperienceQuery,
    ExperienceSearchResult,
    TrustTier,
)
from tradingagents.experience.normalization import NormalizationCohortV1, build_profile
from tradingagents.experience.orchestrator import EvidenceOrchestrator
from tradingagents.experience.outcomes import OutcomeStatsCalculator
from tradingagents.experience.query import ExperienceQueryService
from tradingagents.forex.evidence_context import (
    EvidenceBundleStatus,
    EvidenceIntegrationStatus,
    EvidenceQueryPolicy,
)
from tradingagents.forex.evidence_runtime import (
    EvidenceIntegrationService,
    ReadonlyEvidenceRuntimeConfiguration,
    ReadonlyExperienceCatalog,
    ReadonlyKnowledgeCatalog,
    _bounded_diagnostic,
    _close_orchestrator,
    _validate_child_orchestrator,
    approved_readonly_component,
    approved_readonly_factory,
)
from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.embeddings import EmbeddingProvider, EmbeddingSpecMismatch
from tradingagents.knowledge.lexical_index import LexicalIndexReader
from tradingagents.knowledge.models import (
    EmbeddingSpec,
    IndexGeneration,
    KnowledgeHit,
    KnowledgeQuery,
)
from tradingagents.knowledge.query import KnowledgeQueryService
from tradingagents.knowledge.vector_index import VectorIndexReader


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


def _feature_state(value: float) -> dict[str, object]:
    return {
        "values": [value] * len(FEATURE_NAMES_V1),
        "mask": [True] * len(FEATURE_NAMES_V1),
        "feature_names": FEATURE_NAMES_V1,
        "cohort": [
            "EURUSD", "INTRADAY", "M5", "experience-features.v1",
            "phase8-feature-extractor.v1",
        ],
    }


def _catalog_profiles(catalog):
    rows = []
    for record in catalog.active_records() + catalog.historical_records():
        state = dict(record.market_state or {})
        state.update(
            experience_id=record.experience_id,
            trust_tier=record.trust,
            feature_fingerprint="persisted",
            source_aliases=record.source_aliases,
            analysis_snapshot_timestamp=record.analysis_snapshot_timestamp,
            decision_completed_timestamp=record.decision_completed_timestamp,
        )
        rows.append(state)
    cohorts = {NormalizationCohortV1(*tuple(row["cohort"])) for row in rows if row.get("cohort")}
    return {
        cohort: build_profile([row for row in rows if NormalizationCohortV1(*tuple(row["cohort"])) == cohort], cohort)
        for cohort in cohorts
    }


def _artifact_roots(tmp_path: Path) -> dict[str, str]:
    roots = {name: tmp_path / name for name in ("knowledge", "experience")}
    for root in roots.values():
        root.mkdir()
        sqlite3.connect(root / "catalog.sqlite3").close()
    return {name: str(root) for name, root in roots.items()}


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


def _readonly_catalogs(config: ReadonlyEvidenceRuntimeConfiguration):
    roots = config.roots()
    return ReadonlyKnowledgeCatalog(roots["knowledge"]), ReadonlyExperienceCatalog(roots["experience"])


class _TypedKnowledgeService(KnowledgeQueryService):
    def __init__(self, catalog, hits=(), delay=0.0):
        self.catalog = catalog
        self.vector_reader = _ReadonlyVectorReader()
        self.lexical_reader = _ReadonlyLexicalReader()
        self.embedder = _ReadonlyEmbeddingProvider()
        self.hits = tuple(hits)
        self.delay = delay

    def search(self, _query):
        if self.delay:
            time.sleep(self.delay)
        return self.hits


@approved_readonly_component
class _ReadonlyTokenizer:
    model_max_input_tokens = 512
    special_token_budget = 2

    def encode(self, text, *, add_special_tokens, purpose, truncation):
        return tuple(range(min(len(text), self.model_max_input_tokens)))


@approved_readonly_component
class _ReadonlyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, spec=None):
        super().__init__(spec=spec or EmbeddingSpec(), tokenizer=_ReadonlyTokenizer())

    def _embed_formatted(self, texts):
        return tuple((0.0,) * self.spec.dimensions for _ in texts)


@approved_readonly_component
class _ReadonlyVectorReader(VectorIndexReader):
    def __init__(self):
        self._metadata = {
            "generation_id": "g1",
            "population_hash": "pop",
            "embedding_spec": EmbeddingSpec().to_dict(),
        }

    def metadata(self):
        return dict(self._metadata)

    def search(self, *_args, **_kwargs):
        return ()


@approved_readonly_component
class _ReadonlyLexicalReader(LexicalIndexReader):
    def __init__(self):
        self._metadata = {
            "generation_id": "g1",
            "population_hash": "pop",
            "embedding_spec": EmbeddingSpec().to_dict(),
        }

    def metadata(self):
        return dict(self._metadata)

    def search(self, *_args, **_kwargs):
        return ()


class _TypedExperienceService(ExperienceQueryService):
    def __init__(self, catalog, *, fail=False):
        self.catalog = catalog
        self.fail = fail

    def search(self, _query):
        if self.fail:
            raise RuntimeError("experience unavailable")
        return ExperienceSearchResult()


class _TypedStatsCalculator(OutcomeStatsCalculator):
    def __init__(self, catalog):
        super().__init__(catalog)


class _CountingEvidenceOrchestrator(EvidenceOrchestrator):
    def __init__(self, *services):
        super().__init__(*services)
        self.query_count = 0

    def query(self, request):
        self.query_count += 1
        bundle = super().query(request)
        return replace(bundle, source_status={**dict(bundle.source_status), "query_count": self.query_count})


def _typed_orchestrator(config, *, hits=(), experience_fails=False, delay=0.0, count_query=False):
    knowledge, experience = _readonly_catalogs(config)
    orchestrator_type = _CountingEvidenceOrchestrator if count_query else EvidenceOrchestrator
    return orchestrator_type(
        _TypedKnowledgeService(knowledge, hits, delay),
        _TypedExperienceService(experience, fail=experience_fails),
        _TypedStatsCalculator(experience),
    )


@approved_readonly_factory
def _empty_factory(config):
    return _typed_orchestrator(config)


@approved_readonly_factory
def _partial_factory(config):
    return _typed_orchestrator(config, experience_fails=True)


class _ErrorEvidenceOrchestrator(EvidenceOrchestrator):
    def query(self, _request):
        raise RuntimeError("boom")


@approved_readonly_factory
def _error_factory(config):
    knowledge, experience = _readonly_catalogs(config)
    return _ErrorEvidenceOrchestrator(
        _TypedKnowledgeService(knowledge),
        _TypedExperienceService(experience),
        _TypedStatsCalculator(experience),
    )


@approved_readonly_factory
def _child_evidence_factory(config):
    return _typed_orchestrator(
        config,
        hits=(KnowledgeHit(chunk_id="child-chunk", document_id="child-doc", text="child evidence"),),
        count_query=True,
    )


@approved_readonly_factory
def _large_child_evidence_factory(config):
    hits = tuple(
        KnowledgeHit(
            chunk_id=f"large-child-{index}",
            document_id="large-doc",
            text="bounded evidence " * 100,
        )
        for index in range(80)
    )
    return _typed_orchestrator(config, hits=hits)


@approved_readonly_factory
def _slow_factory(config):
    return _typed_orchestrator(config, delay=1.0)


@approved_readonly_factory
def _profile_child_factory(config):
    orchestrator = _typed_orchestrator(
        config,
        hits=(KnowledgeHit(chunk_id="profile-child", document_id="doc", text="profile evidence"),),
        count_query=True,
    )
    orchestrator.experience_service.profiles = orchestrator.experience_service.catalog.profiles
    return orchestrator


def _unapproved_factory(_config=None):
    return type("WriterOwner", (), {"writer": object(), "query": lambda self, _request: EvidenceBundle(status="COMPLETE")})()


@approved_readonly_factory
def _approved_writer_factory(_config=None):
    return type("WriterOwner", (), {"writer": object(), "query": lambda self, _request: EvidenceBundle(status="COMPLETE")})()


class _WriterKnowledgeService(_TypedKnowledgeService):
    writer = object()


@approved_readonly_factory
def _typed_writer_factory(config):
    knowledge, experience = _readonly_catalogs(config)
    return EvidenceOrchestrator(
        _WriterKnowledgeService(knowledge),
        _TypedExperienceService(experience),
        _TypedStatsCalculator(experience),
    )


@approved_readonly_factory
def _untyped_dependency_factory(config):
    knowledge, experience = _readonly_catalogs(config)
    service = _TypedKnowledgeService(knowledge)
    service.vector_reader = object()
    return EvidenceOrchestrator(
        service,
        _TypedExperienceService(experience),
        _TypedStatsCalculator(experience),
    )


@approved_readonly_factory
def _nested_writer_dependency_factory(config):
    knowledge, experience = _readonly_catalogs(config)
    service = _TypedKnowledgeService(
        knowledge,
        (KnowledgeHit(chunk_id="nested", document_id="doc", text="nested evidence"),),
    )
    service.dependencies = {"providers": [type("NestedWriter", (), {})()]}
    return EvidenceOrchestrator(
        service,
        _TypedExperienceService(experience),
        _TypedStatsCalculator(experience),
    )


@approved_readonly_factory
def _nested_untyped_dependency_factory(config):
    knowledge, experience = _readonly_catalogs(config)
    service = _TypedKnowledgeService(
        knowledge,
        (KnowledgeHit(chunk_id="nested", document_id="doc", text="nested evidence"),),
    )
    service.dependencies = {"readers": [{"reader": object()}]}
    return EvidenceOrchestrator(
        service,
        _TypedExperienceService(experience),
        _TypedStatsCalculator(experience),
    )


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


class _BoundProcess(_ImmediateProcess):
    def __init__(self, *, target, args, orchestrator):
        super().__init__(target=target, args=args)
        self.orchestrator = orchestrator

    def start(self):
        from tradingagents.experience.models import EvidenceRequest

        request = EvidenceRequest(**dict(self.args[2]["request"]))
        bundle = self.orchestrator.query(request)
        self.args[0].send(("ok", bundle.to_dict()))


class _HangingProcess(_ImmediateProcess):
    def start(self):
        self._alive = True


def _process_factory(*, target, args):
    return _ImmediateProcess(target=target, args=args)


def _hanging_process_factory(*, target, args):
    return _HangingProcess(target=target, args=args)


def test_real_spawn_query_uses_serializable_envelope_and_reaches_child(tmp_path: Path):
    roots = _artifact_roots(tmp_path)
    before = {Path(path) / "catalog.sqlite3": (Path(path) / "catalog.sqlite3").read_bytes() for path in roots.values()}
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=5),
        orchestrator_factory=_child_evidence_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        artifact_roots=roots,
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.INJECTED
    assert context.knowledge_items[0].authoritative_id == "child-chunk"
    assert context.diagnostics["source_status"]["query_count"] == 1
    assert service.active_evidence_workers == 0
    assert {path: path.read_bytes() for path in before} == before


def test_real_spawn_timeout_leaves_zero_workers(tmp_path: Path):
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=0.05),
        orchestrator_factory=_slow_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        artifact_roots=_artifact_roots(tmp_path),
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert context.diagnostics["integration"]["code"] == "EVIDENCE_TIMEOUT"
    assert service.active_evidence_workers == 0


def test_real_spawn_drains_large_bundle_without_pipe_deadlock(tmp_path: Path):
    roots = _artifact_roots(tmp_path)
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=10),
        orchestrator_factory=_large_child_evidence_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        artifact_roots=roots,
    )
    context = service.retrieve(
        _snapshot(),
        resolved_symbol="EURUSD",
        analysis_profile="INTRADAY",
        analysis_timeframe="M5",
    )
    assert context.integration_status is EvidenceIntegrationStatus.INJECTED
    assert context.selected_knowledge_count > 0
    assert service.active_evidence_workers == 0


def test_child_rejects_unapproved_or_writer_factory_without_query():
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=5),
        orchestrator_factory=_unapproved_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert context.diagnostics["integration"]["code"] == "ORCHESTRATOR_FAILURE"
    assert service.active_evidence_workers == 0


def test_worker_failure_diagnostic_keeps_code_and_safe_exception_type():
    diagnostic = _bounded_diagnostic(
        "ORCHESTRATOR_FAILURE", RuntimeError("provider response contained prompt text")
    )

    assert diagnostic["code"] == "ORCHESTRATOR_FAILURE"
    assert diagnostic["error_type"] == "RuntimeError"
    assert "provider response" in diagnostic["message"]


def test_child_guard_rejects_approved_factory_that_owns_writer(tmp_path: Path):
    roots = _artifact_roots(tmp_path)
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=5),
        orchestrator_factory=_approved_writer_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        artifact_roots=roots,
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert service.active_evidence_workers == 0


def test_child_guard_rejects_typed_service_with_writer_component(tmp_path: Path):
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=5),
        orchestrator_factory=_typed_writer_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        artifact_roots=_artifact_roots(tmp_path),
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert context.diagnostics["integration"]["code"] == "ORCHESTRATOR_FAILURE"


def test_child_guard_rejects_untyped_reader_dependency(tmp_path: Path):
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=5),
        orchestrator_factory=_untyped_dependency_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        artifact_roots=_artifact_roots(tmp_path),
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert context.diagnostics["integration"]["code"] == "ORCHESTRATOR_FAILURE"


def test_child_guard_treats_approved_embedding_provider_as_read_only_leaf(tmp_path: Path):
    roots = _artifact_roots(tmp_path)

    @approved_readonly_component
    class _OpaqueEmbeddingProvider(_ReadonlyEmbeddingProvider):
        def __init__(self):
            super().__init__()
            # FastEmbed owns tokenizer/runtime implementation objects that are
            # internal to the approved provider, not independent evidence
            # components.  The guard must not mistake those internals for a
            # writer simply because they are reachable via __dict__.
            self._runtime = type("OpaqueRuntime", (), {})()

    orchestrator = _typed_orchestrator(
        ReadonlyEvidenceRuntimeConfiguration(tuple(sorted(roots.items())))
    )
    orchestrator.knowledge_service.embedder = _OpaqueEmbeddingProvider()
    orchestrator.knowledge_service.vector_reader.backend = type("OpaqueVectorBackend", (), {})()

    _validate_child_orchestrator(
        orchestrator,
        ReadonlyEvidenceRuntimeConfiguration(tuple(sorted(roots.items()))),
    )


@pytest.mark.parametrize("factory", [_nested_writer_dependency_factory, _nested_untyped_dependency_factory])
def test_child_guard_recurses_nested_dependency_containers(tmp_path: Path, factory):
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=5),
        orchestrator_factory=factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        artifact_roots=_artifact_roots(tmp_path),
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert context.diagnostics["integration"]["code"] == "ORCHESTRATOR_FAILURE"


def test_populated_profiles_pass_adapter_backed_spawn_guard(tmp_path: Path):
    roots = _artifact_roots(tmp_path)
    writer = ExperienceCatalog(roots["experience"])
    record = writer.upsert_source_alias(
        "source", "profile-decision", "profile-fingerprint", symbol="EURUSD",
        analysis_profile="INTRADAY", analysis_timeframe="M5",
        analysis_snapshot_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        decision_completed_timestamp=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        market_state=_feature_state(1.0), trust="TIER_A_HIGH_TRUST",
    )
    writer.store_feature_projection(record.experience_id, dict(record.market_state))
    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(evidence_timeout_seconds=5),
        orchestrator_factory=_profile_child_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        artifact_roots=roots,
    )
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.INJECTED
    assert context.diagnostics["source_status"]["query_count"] == 1
    assert service.active_evidence_workers == 0


def test_phase8_profile_and_cohort_models_are_immutable(tmp_path: Path):
    roots = _artifact_roots(tmp_path)
    writer = ExperienceCatalog(roots["experience"])
    record = writer.upsert_source_alias(
        "source", "immutable-decision", "immutable-fingerprint", symbol="EURUSD",
        analysis_profile="INTRADAY", analysis_timeframe="M5",
        analysis_snapshot_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        decision_completed_timestamp=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        market_state=_feature_state(1.0), trust="TIER_A_HIGH_TRUST",
    )
    writer.store_feature_projection(record.experience_id, dict(record.market_state))
    catalog = ReadonlyExperienceCatalog(roots["experience"])
    profile = next(iter(catalog.profiles.values()))
    assert isinstance(profile.cohort, NormalizationCohortV1)
    with pytest.raises(FrozenInstanceError):
        profile.population_count = 99


def test_spawn_process_args_contain_no_callable_closures(tmp_path: Path):
    captured = {}
    roots = _artifact_roots(tmp_path)

    def process_factory(*, target, args):
        captured["args"] = args
        return _ImmediateProcess(target=target, args=args)

    service = EvidenceIntegrationService(
        policy=EvidenceQueryPolicy(),
        orchestrator_factory=_child_evidence_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint="http://127.0.0.1:11434",
        process_factory=process_factory,
        artifact_roots=roots,
    )
    service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert not any(callable(value) for value in captured["args"])
    assert captured["args"][2]["artifact_roots"] == roots


def test_populated_phase8_readonly_adapter_matches_writer_semantics(tmp_path: Path):
    writer = ExperienceCatalog(tmp_path / "experience")
    first = writer.upsert_source_alias(
        "source", "decision-1", "fp-1", symbol="EURUSD",
        analysis_profile="INTRADAY", analysis_timeframe="M5",
        analysis_snapshot_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        decision_completed_timestamp=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        market_state=_feature_state(1.0),
        trust="TIER_A_HIGH_TRUST",
    )
    second = writer.upsert_source_alias(
        "source", "decision-2", "fp-2", symbol="EURUSD",
        analysis_profile="INTRADAY", analysis_timeframe="M5",
        analysis_snapshot_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        decision_completed_timestamp=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        market_state=_feature_state(2.0),
        trust="TIER_B_LIMITED",
    )
    future = writer.upsert_source_alias(
        "source", "decision-3", "fp-3", symbol="EURUSD",
        analysis_profile="INTRADAY", analysis_timeframe="M5",
        analysis_snapshot_timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
        decision_completed_timestamp=datetime(2026, 1, 3, 1, tzinfo=timezone.utc),
        market_state=_feature_state(3.0),
        trust="TIER_A_HIGH_TRUST",
    )
    writer.store_feature_projection(first.experience_id, dict(first.market_state))
    writer.store_feature_projection(future.experience_id, dict(future.market_state))
    writer.append_evaluation_snapshot(first.experience_id, {"evaluation_status": "COMPLETE", "evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 60}, "eval-1")
    writer.mark_last_alias_removed(second.experience_id)
    writer.publish_generation("generation-1", population_fingerprint="population-1", metadata={"trust_policy_version": "trust-policy.v1"})
    database_before = (tmp_path / "experience" / "catalog.sqlite3").read_bytes()
    readonly = ReadonlyExperienceCatalog(tmp_path / "experience")
    assert {r.experience_id for r in readonly.active_records()} == {first.experience_id, future.experience_id}
    assert tuple(r.experience_id for r in readonly.historical_records()) == (second.experience_id,)
    assert {r.experience_id for r in readonly.active_records()} == {r.experience_id for r in writer.active_records()}
    assert readonly.active_records()[0].source_aliases in [r.source_aliases for r in writer.active_records()]
    assert readonly.evaluation_snapshots(first.experience_id) == writer.evaluation_snapshots(first.experience_id)
    assert readonly.feature_projection(first.experience_id)["values"] == [1.0] * len(FEATURE_NAMES_V1)
    assert readonly.active_generation() == writer.active_generation()
    assert {r.trust for r in readonly.active_records()} == {r.trust for r in writer.active_records()}
    writer_profiles = _catalog_profiles(writer)
    assert set(readonly.profiles) == set(writer_profiles)
    assert {
        cohort: profile.population_fingerprint for cohort, profile in readonly.profiles.items()
    } == {
        cohort: profile.population_fingerprint for cohort, profile in writer_profiles.items()
    }
    query = ExperienceQuery(
        market_state=first.market_state,
        top_k=10,
        symbol="EURUSD",
        analysis_profile="INTRADAY",
        analysis_timeframe="M5",
        as_of=datetime(2026, 1, 2, tzinfo=timezone.utc),
        trust_tiers=(TrustTier.TIER_A_HIGH_TRUST,),
    )
    writer_result = ExperienceQueryService(writer, profiles=readonly.profiles).search(query)
    readonly_result = ExperienceQueryService(readonly, profiles=readonly.profiles).search(query)
    assert tuple(hit.experience_id for hit in readonly_result.hits) == tuple(
        hit.experience_id for hit in writer_result.hits
    )
    assert readonly_result.excluded_counts == writer_result.excluded_counts
    assert future.experience_id not in {hit.experience_id for hit in readonly_result.hits}
    assert readonly_result.excluded_counts["as_of"] == writer_result.excluded_counts["as_of"]
    no_cutoff = replace(query, as_of=None, trust_tiers=(TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED))
    readonly_no_cutoff = ExperienceQueryService(readonly, profiles=readonly.profiles).search(no_cutoff)
    writer_no_cutoff = ExperienceQueryService(writer, profiles=writer_profiles).search(no_cutoff)
    assert second.experience_id not in {hit.experience_id for hit in readonly_no_cutoff.hits}
    assert readonly_no_cutoff.excluded_counts["tombstone"] == writer_no_cutoff.excluded_counts["tombstone"] == 1
    stats_request = type("StatsRequest", (), {"experience_ids": (first.experience_id,), "evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 60, "trust_tiers": (TrustTier.TIER_A_HIGH_TRUST,), "as_of": query.as_of})()
    readonly_stats = OutcomeStatsCalculator(readonly).calculate(stats_request)
    writer_stats = OutcomeStatsCalculator(writer).calculate(stats_request)
    assert readonly_stats == writer_stats
    assert readonly_stats.excluded_counts == writer_stats.excluded_counts
    readonly.close()
    assert (tmp_path / "experience" / "catalog.sqlite3").read_bytes() == database_before


def test_populated_phase7_readonly_adapter_matches_published_generation(tmp_path: Path):
    writer = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    writer.initialize()
    artifact = tmp_path / "published-vector.bin"
    artifact.write_bytes(b"published-read-only-artifact")
    artifact_before = artifact.read_bytes()
    with sqlite3.connect(tmp_path / "catalog.sqlite3") as db:
        spec = EmbeddingSpec().to_dict()
        db.execute("INSERT INTO knowledge_index_generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("g1", "vector/g1", "keyword/g1", json.dumps(spec), "fts5-v1", "{}", "index-v1", "pop", "identity", 1, 1, 1, 1, "VALIDATED", None, None, "{}", 1))
        db.execute("INSERT INTO knowledge_documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("doc", "hash", "{}", "parser", "1", "cfg", "PARSED", 1, 1, 1, "g1", "pop", "{}", None, "", ""))
        db.execute("INSERT INTO knowledge_resources VALUES (?,?,?,?,?,?,?,?,?,?,?)", ("res", None, None, None, "hash", None, None, "PARSED", None, "", ""))
        db.execute("INSERT INTO knowledge_aliases VALUES (?,?,?,?,?,?)", ("res", "doc", "hash", "CURRENT", None, ""))
    readonly = ReadonlyKnowledgeCatalog(tmp_path)
    assert readonly.active_generation().generation_id == writer.active_generation().generation_id
    assert readonly.document_is_retrieval_ready("doc") == writer.document_is_retrieval_ready("doc") is True
    query_service = KnowledgeQueryService(
        _ReadonlyVectorReader(),
        _ReadonlyLexicalReader(),
        readonly,
        _ReadonlyEmbeddingProvider(),
    )
    assert query_service.search(KnowledgeQuery(text="published evidence")) == ()
    assert readonly.active_generation().embedding_spec.to_dict() == EmbeddingSpec().to_dict()
    assert artifact.read_bytes() == artifact_before


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
    process_factory = _process_factory
    if orchestrator is not None:
        def bound_process_factory(*, target, args):
            return _BoundProcess(target=target, args=args, orchestrator=orchestrator)

        process_factory = bound_process_factory
    return EvidenceIntegrationService(
        policy=EvidenceQueryPolicy() if enabled else None,
        orchestrator_factory=factory or _empty_factory,
        generation_provider=lambda: ("p7", "p8"),
        provider_endpoint=endpoint,
        clock=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
        process_factory=process_factory,
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
    service = _service(orchestrator)
    context = service.retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert service.retrieval_count == 1
    assert len(orchestrator.calls) == 1
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK


def test_enabled_request_uses_snapshot_as_of():
    service = _service(_FakeOrchestrator())
    snapshot = _snapshot()
    service.retrieve(snapshot, resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert service.last_request.as_of == snapshot.timestamp


def test_partial_bundle_maps_to_fallback():
    context = _service(
        _FakeOrchestrator(
            EvidenceBundle(status="PARTIAL", source_status={"knowledge": "COMPLETE", "experience": "FAILED"})
        )
    ).retrieve(_snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5")
    assert context.integration_status is EvidenceIntegrationStatus.FALLBACK
    assert context.bundle_status is EvidenceBundleStatus.PARTIAL


def test_orchestrator_exception_maps_to_fallback():
    context = _service(_FakeOrchestrator(error=RuntimeError("boom"))).retrieve(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M5"
    )
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


def test_readonly_catalog_close_releases_windows_file_handle(tmp_path: Path):
    db = tmp_path / "catalog.sqlite3"
    sqlite3.connect(db).close()
    catalog = ReadonlyKnowledgeCatalog(tmp_path)
    catalog.close()
    db.unlink()
    assert not db.exists()


def test_close_orchestrator_recurses_through_service_catalog_idempotently():
    class Closed:
        def __init__(self):
            self.count = 0

        def close(self):
            self.count += 1

    knowledge = Closed()
    experience = Closed()
    service_a = type("Service", (), {"catalog": knowledge})()
    service_b = type("Service", (), {"catalog": experience})()
    orchestrator = type("Orchestrator", (), {"knowledge_service": service_a, "experience_service": service_b})()
    _close_orchestrator(orchestrator)
    _close_orchestrator(orchestrator)
    assert knowledge.count == 1
    assert experience.count == 1


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
