from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradingagents.knowledge.models import ContentType, EmbeddingSpec, KnowledgeHit
from tradingagents.self_enhancement.book_atomic_extraction import (
    AtomicExtractionReport,
    ConceptFamily,
    EvidenceSentence,
    ModelCallTelemetry,
    PresenceClassificationReport,
    PresenceGroupResult,
    PresenceStatus,
    ResolvedAtomicRule,
    StrategyConcept,
)
from tradingagents.self_enhancement.book_rule_grammar import parse_supported_rule_quote
from tradingagents.self_enhancement.phase14c_models import (
    CandidateEvaluationRecord,
    CorpusInventory,
    Phase14CCandidateReason,
    Phase14CCandidateStatus,
    Phase14CGenerationPin,
    Phase14CPreflightReason,
    Phase14CReplayPreflight,
    Phase14CSourcePaths,
)
from tradingagents.self_enhancement.phase14c_runner import (
    Phase14CResumeError,
    load_complete_discovery_artifacts,
    run_phase14c_discovery,
)
from tradingagents.self_enhancement.strategy_specs import RuleStage

GENERATION_ID = "gen_phase14c_integration_fixture"
GENERATION_FINGERPRINT = "c" * 64
POPULATION_HASH = "sha256:" + "b" * 64

COMPLETE_QUOTES = {
    RuleStage.ENTRY: "LONG when momentum > 2 points after confirmation",
    RuleStage.CONFIRMATION: (
        "CONFIRMATION: LONG when direction_persistence >= 0.6 fraction after three ticks"
    ),
    RuleStage.INVALIDATION: "INVALIDATION: LONG when momentum <= 0 points after reversal",
    RuleStage.EXPECTED_MOVE: "EXPECTED_MOVE: LONG target 5 points within 30 seconds after entry",
    RuleStage.EXIT: "EXIT: LONG when momentum <= 0 points after reversal",
    RuleStage.PROFIT_PROTECTION: (
        "PROFIT_PROTECTION: LONG when momentum <= 0 points after target retracement"
    ),
    RuleStage.STOP_BEHAVIOR: "STOP_BEHAVIOR: LONG when momentum <= -3 points after adverse move",
    RuleStage.HORIZON: "HORIZON: hold no longer than 30 seconds after entry",
}


def _hit(document_id: str, chunk_id: str, text: str, *, source_hash: str | None = None) -> KnowledgeHit:
    digest = source_hash or hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return KnowledgeHit(
        chunk_id=chunk_id,
        document_id=document_id,
        content_type=ContentType.PROSE,
        text=text,
        score=0.9,
        semantic_score=0.8,
        lexical_score=0.7,
        fused_score=0.6,
        rerank_score=0.9,
        source_filename=f"{document_id}.pdf",
        source_relative_path=f"books/{document_id}.pdf",
        source_hash=digest,
        title=f"Fixture {document_id}",
        page=4,
        section="Rules",
        parser_version="fixture-parser-v1",
        chunker_version="fixture-chunker-v1",
        index_version="fixture-index-v1",
        extra={
            "projection_generation": GENERATION_ID,
            "projection_population_hash": POPULATION_HASH,
        },
    )


class _FixtureCatalog:
    def __init__(self, hits: tuple[KnowledgeHit, ...], embedding_spec: EmbeddingSpec) -> None:
        self._hits_by_document: dict[str, tuple[KnowledgeHit, ...]] = {}
        grouped: dict[str, list[KnowledgeHit]] = {}
        for hit in hits:
            grouped.setdefault(hit.document_id, []).append(hit)
        self._hits_by_document = {key: tuple(values) for key, values in grouped.items()}
        self._generation = SimpleNamespace(
            generation_id=GENERATION_ID,
            population_hash=POPULATION_HASH,
            status="VALIDATED",
            vector_ready=True,
            lexical_ready=True,
            embedding_spec=embedding_spec,
        )

    def active_generation(self):
        return self._generation

    def get_document(self, document_id):
        hits = self._hits_by_document.get(document_id)
        if not hits:
            return None
        hit = hits[0]
        return SimpleNamespace(
            document_id=hit.document_id,
            source_hash=hit.source_hash,
            active=True,
            vector_ready=True,
            lexical_ready=True,
            projection_generation=GENERATION_ID,
            projection_population_hash=POPULATION_HASH,
        )

    def chunks_for_document(self, document_id):
        hits = self._hits_by_document.get(document_id)
        if not hits:
            return ()
        return tuple(
            SimpleNamespace(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                source_hash=hit.source_hash,
                text=hit.text,
                content_type=hit.content_type,
                source_filename=hit.source_filename,
                source_relative_path=hit.source_relative_path,
                title=hit.title,
                authors=hit.authors,
                publication_year=hit.publication_year,
                page=hit.page,
                page_start=hit.page_start,
                page_end=hit.page_end,
                chapter=hit.chapter,
                section_path=hit.section_path,
                epub_spine_item=hit.epub_spine_item,
                anchor=hit.anchor,
                parser_version=hit.parser_version,
                chunker_version=hit.chunker_version,
                index_version=hit.index_version,
                table_metadata=hit.table_metadata,
                equation_metadata=hit.equation_metadata,
                active=True,
            )
            for hit in hits
        )

    def document_is_retrieval_ready(self, document_id):
        return document_id in self._hits_by_document


class _FixtureQueryService:
    def __init__(self, hits: tuple[KnowledgeHit, ...], embedding_spec: EmbeddingSpec) -> None:
        self.catalog = _FixtureCatalog(hits, embedding_spec)
        self.hits = hits

    def search(self, _request):
        by_chunk = {hit.chunk_id: hit for hit in self.hits}
        if "momentum continuation entry" in _request.text:
            return (by_chunk["chunk-rules-1"], by_chunk["chunk-rules-2"])
        if "price breakout strategy" in _request.text:
            return (by_chunk["chunk-conflict"],)
        if "failed breakout reversal" in _request.text:
            return (by_chunk["chunk-rules-3"],)
        if "range edge rejection" in _request.text:
            return (by_chunk["chunk-rules-3"],)
        if "mean-reversion entry" in _request.text:
            return (by_chunk["chunk-duplicate"],)
        return ()


def _rule(stage: RuleStage, sentence: EvidenceSentence) -> ResolvedAtomicRule:
    parsed = parse_supported_rule_quote(sentence.text, stage)
    assert parsed is not None
    return ResolvedAtomicRule(stage, sentence, parsed)


class _FixtureTeacher:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls = []
        self.cache = None
        self.endpoint = "http://localhost:11434"
        self.model = "qwen3.5:2b"
        self.model_version = "fixture-model-sha256"
        self.timeout_seconds = 10.0
        self.max_output_tokens = 512
        self.context_tokens = 4096

    def classify_presence(self, groups):

        self.events.append("presence")
        results = []
        telemetry = []
        for index, group in enumerate(groups):
            if group.family_id == "failed_breakout":
                status = PresenceStatus.AMBIGUOUS
            elif group.family_id == "mean_reversion":
                status = PresenceStatus.NONE
            else:
                status = PresenceStatus.ACTIONABLE
            results.append(PresenceGroupResult(group, status))
            telemetry.append(
                ModelCallTelemetry(
                    "PRESENCE",
                    len(group.sentences),
                    20,
                    2,
                    0.01 + index / 100000,
                    "stop",
                    "VALIDATED",
                    0,
                )
            )
        return PresenceClassificationReport(tuple(results), tuple(telemetry))

    def extract_actionable(self, sentences):
        self.events.append("extract")
        by_text: dict[str, EvidenceSentence] = {}
        for sentence in sentences:
            by_text.setdefault(sentence.text, sentence)
        missing = set(COMPLETE_QUOTES.values()) - set(by_text)
        assert not missing, f"fixture did not select required source sentences: {sorted(missing)}"

        complete = StrategyConcept(
            ConceptFamily.MOMENTUM_CONTINUATION,
            tuple(_rule(stage, by_text[quote]) for stage, quote in COMPLETE_QUOTES.items()),
        )
        partial_entry = "SHORT when rolling_range > 2 points after edge_test"
        partial = StrategyConcept(
            ConceptFamily.RANGE_REJECTION,
            (_rule(RuleStage.ENTRY, by_text[partial_entry]),),
        )
        breakout_entry = "SHORT when rolling_range > 2 points after breakout_test"
        breakout_exit_a = "EXIT: SHORT when rolling_range <= 0 points after failed_breakout"
        breakout_exit_b = "EXIT: SHORT when rolling_range <= -1 points after failed_breakout"
        breakout_left = StrategyConcept(
            ConceptFamily.BREAKOUT,
            (
                _rule(RuleStage.ENTRY, by_text[breakout_entry]),
                _rule(RuleStage.EXIT, by_text[breakout_exit_a]),
            ),
        )
        breakout_right = StrategyConcept(
            ConceptFamily.BREAKOUT,
            (
                _rule(RuleStage.ENTRY, by_text[breakout_entry]),
                _rule(RuleStage.EXIT, by_text[breakout_exit_b]),
            ),
        )
        return AtomicExtractionReport(
            (complete, partial, breakout_left, breakout_right),
            (
                ModelCallTelemetry("CONCEPT_GROUPING", len(sentences), 80, 9, 0.03, "stop", "VALIDATED", 0),
                ModelCallTelemetry(
                    "ATOMIC_RULE_EXTRACTION", len(sentences), 90, 22, 0.04, "stop", "VALIDATED", 0
                ),
            ),
            len(sentences),
            0,
            0,
            4,
            len(COMPLETE_QUOTES) + 5,
            0,
            0,
            0,
        )


def _fixture_hits() -> tuple[KnowledgeHit, ...]:
    first = "\n".join(
        (
            COMPLETE_QUOTES[RuleStage.ENTRY],
            COMPLETE_QUOTES[RuleStage.CONFIRMATION],
            COMPLETE_QUOTES[RuleStage.INVALIDATION],
        )
    )
    second = "\n".join(
        (
            COMPLETE_QUOTES[RuleStage.EXPECTED_MOVE],
            COMPLETE_QUOTES[RuleStage.EXIT],
            COMPLETE_QUOTES[RuleStage.PROFIT_PROTECTION],
        )
    )
    third = "\n".join(
        (
            COMPLETE_QUOTES[RuleStage.STOP_BEHAVIOR],
            COMPLETE_QUOTES[RuleStage.HORIZON],
            "SHORT when rolling_range > 2 points after edge_test",
        )
    )
    fourth = "\n".join(
        (
            "SHORT when rolling_range > 2 points after breakout_test",
            "EXIT: SHORT when rolling_range <= 0 points after failed_breakout",
            "EXIT: SHORT when rolling_range <= -1 points after failed_breakout",
        )
    )
    return (
        _hit("doc-rules", "chunk-rules-1", first),
        _hit("doc-rules", "chunk-rules-2", second),
        _hit("doc-rules", "chunk-rules-3", third),
        _hit("doc-conflict", "chunk-conflict", fourth),
        _hit("doc-duplicate", "chunk-duplicate", first),
    )


def _paths(tmp_path: Path) -> tuple[Phase14CSourcePaths, tuple[Path, Path, Path]]:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    paths = tuple(source_root / name for name in ("phase14a.db", "hft.db", "demo.db"))
    for index, path in enumerate(paths):
        path.write_bytes(f"fixture-source-{index}".encode())
    return Phase14CSourcePaths(*paths), paths


def test_fake_boundary_runs_ordered_pipeline_and_persists_auditable_safe_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradingagents.self_enhancement import phase14c_runner as runner

    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    embedding_root = tmp_path / "embedding"
    embedding_root.mkdir()
    (embedding_root / "local-model.marker").write_text("fixture", encoding="utf-8")
    cache_root = tmp_path / "atomic-cache"
    cache_root.mkdir()
    source_paths, files = _paths(tmp_path)
    artifact_root = tmp_path / "artifacts" / "run-1"
    artifact_root.parent.mkdir()
    embedding_spec = EmbeddingSpec(
        model_id="fixture/bge",
        resolved_model_version="fixture-v1",
        runtime="onnx-cpu",
        artifact_hash="e" * 64,
        dimensions=3,
        tokenizer_fingerprint="f" * 64,
    )
    hits = _fixture_hits()
    query_service = _FixtureQueryService(hits, embedding_spec)
    pin = Phase14CGenerationPin(
        GENERATION_ID,
        GENERATION_FINGERPRINT,
        POPULATION_HASH,
        "VALIDATED",
        True,
        True,
        knowledge_root / "vectors",
        knowledge_root / "lexical.sqlite3",
        "phase14c-inventory.v1",
        4,
        len(hits),
    )
    inventory = CorpusInventory(
        unique_source_count=6,
        indexed_document_count=5,
        queryable_document_count=5,
        alias_count=1,
        needs_ocr_count=1,
        unavailable_resource_count=1,
        active_chunk_count=len(hits),
        source_manifest_fingerprint="d" * 64,
    )
    fingerprints = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in zip(("phase14a", "hft", "demo"), files, strict=True)}
    preflight = Phase14CReplayPreflight(
        source_paths=source_paths,
        source_fingerprints=fingerprints,
        verified_experience_count=1,
        quarantined_experience_count=0,
        causal_tick_count=96,
        causal_segment_count=1,
        commission_status="UNKNOWN",
        ready=True,
        reason_code=Phase14CPreflightReason.READY,
    )
    events: list[str] = []
    inventory_reads = 0

    def read_inventory(*_args, **_kwargs):
        nonlocal inventory_reads
        inventory_reads += 1
        events.append("inventory")
        return pin, inventory

    def read_sources(*_args, **_kwargs):
        events.append("source_preflight")
        return preflight

    def build_query_service(*_args, **_kwargs):
        events.append("query_service")
        return query_service, SimpleNamespace(spec=embedding_spec)

    selected_formulations = {
        "momentum_continuation": "base",
        "breakout": "base",
        "failed_breakout": "reversal",
        "range_rejection": "range_edge",
        "mean_reversion": "base",
    }
    fixture_query_bank = tuple(
        query
        for query in runner.build_discovery_query_bank()
        if selected_formulations.get(query.family_id) == query.formulation_id
    )
    real_retrieve = runner.retrieve_discovery_evidence

    def retrieve(*args, **kwargs):
        events.append("retrieval")
        return real_retrieve(*args, **kwargs)

    real_select = runner.select_diverse_evidence

    def select(*args, **kwargs):
        events.append("selection")
        return real_select(*args, **kwargs)

    real_assemble = runner.assemble_atomic_concepts

    def assemble(*args, **kwargs):
        events.append("assembly")
        return real_assemble(*args, **kwargs)

    real_map = runner.map_current_strategies

    def map_specs(*args, **kwargs):
        events.append("mapping")
        return real_map(*args, **kwargs)

    def teacher_factory():
        events.append("teacher_constructed")
        return _FixtureTeacher(events)

    monkeypatch.setattr(runner, "read_phase7_inventory", read_inventory)
    monkeypatch.setattr(runner, "preflight_phase14_sources", read_sources)
    monkeypatch.setattr(runner, "_build_local_query_service", build_query_service, raising=False)
    monkeypatch.setattr(runner, "build_discovery_query_bank", lambda: fixture_query_bank)
    monkeypatch.setattr(runner, "retrieve_discovery_evidence", retrieve)
    monkeypatch.setattr(runner, "select_diverse_evidence", select)
    monkeypatch.setattr(runner, "assemble_atomic_concepts", assemble)
    monkeypatch.setattr(runner, "map_current_strategies", map_specs, raising=False)

    discovery_args = {
        "artifact_root": artifact_root,
        "knowledge_root": knowledge_root,
        "embedding_model_path": embedding_root,
        "atomic_cache_path": cache_root / "atomic.sqlite3",
        "source_paths": source_paths,
        "source_fingerprints": fingerprints,
        "generation_pin": pin,
        "inventory": inventory,
        "expected_generation_id": GENERATION_ID,
        "expected_generation_fingerprint": GENERATION_FINGERPRINT,
        "expected_population_hash": POPULATION_HASH,
        "endpoint": "http://localhost:11434",
        "model": "qwen3.5:2b",
        "model_version": "fixture-model-sha256",
        "timeout_seconds": 10,
        "max_output_tokens": 512,
        "context_tokens": 4096,
        "resume": False,
        "teacher_factory": teacher_factory,
    }
    changed_fingerprints = dict(fingerprints, phase14a="0" * 64)
    with pytest.raises(ValueError, match="fingerprints changed"):
        run_phase14c_discovery(
            **{
                **discovery_args,
                "artifact_root": artifact_root.parent / "fingerprint-mismatch",
                "source_fingerprints": changed_fingerprints,
            }
        )
    assert events == ["inventory", "source_preflight"]
    assert not (artifact_root.parent / "fingerprint-mismatch").exists()
    events.clear()
    inventory_reads = 0

    report = run_phase14c_discovery(**discovery_args)

    assert report["status"] == "COMPLETE", (report["outcome"], report["classification"], report["coverage"])
    assert report["outcome"] in {
        "CANDIDATES_READY_FOR_EVALUATION",
        "DISCOVERY_COMPLETE_NO_HFT_SUITABLE_SPEC",
    }
    assert report["coverage"]["coverage_complete"] is True
    assert report["inventory"]["alias_count"] == 1
    assert report["inventory"]["needs_ocr_count"] == 1
    assert report["retrieval"]["duplicate_chunk_count"] >= 1
    assert report["classification"]["status_counts"]["AMBIGUOUS"] > 0
    assert report["classification"]["status_counts"]["NONE"] > 0
    assert report["assembly"]["conflict_count"] > 0
    assert report["assembly"]["partial_count"] > 0
    assert report["evaluation"]["status"] == "NOT_RUN"
    assert report["evaluation"]["candidate_count"] == 0
    assert report["source_unchanged"] is True
    assert report["phase7_unchanged"] is True
    assert report["artifact_integrity"] == "VALID"
    assert inventory_reads >= 2

    assert events.index("inventory") < events.index("source_preflight")
    assert events.index("source_preflight") < events.index("query_service")
    assert events.index("query_service") < events.index("retrieval")
    assert events.index("retrieval") < events.index("selection")
    assert events.index("selection") < events.index("teacher_constructed")
    assert events.index("teacher_constructed") < events.index("assembly")
    assert events.index("assembly") < events.index("mapping")
    assert events.index("mapping") < events.index("inventory", events.index("mapping"))
    assert events.index("teacher_constructed") > events.index("selection")

    manifest = json.loads((artifact_root / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    specs, loaded_report = load_complete_discovery_artifacts(artifact_root)
    assert loaded_report["artifact_integrity"] == "VALID"
    assert any(spec.is_executable for spec in specs)

    retrieval_rows = [
        json.loads(line)
        for line in (artifact_root / "retrieval" / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert retrieval_rows
    assert all(row["source_hash"] and row["document_id"] and row["chunk_id"] for row in retrieval_rows)
    assert all(row["page"] == 4 and row["text"] for row in retrieval_rows)
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in artifact_root.rglob("*") if path.is_file())
    for forbidden in ("raw completion body", "hidden reasoning text", "fixture-secret"):
        assert forbidden not in serialized.lower()

    after = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in zip(("phase14a", "hft", "demo"), files, strict=True)
    }
    assert after == fingerprints

    evidence_path = artifact_root / "retrieval" / "evidence.jsonl"
    evidence_path.write_bytes(evidence_path.read_bytes() + b"tampered")
    with pytest.raises(Phase14CResumeError, match="artifact.*fingerprint"):
        load_complete_discovery_artifacts(artifact_root)


def test_incomplete_or_unclassified_selection_cannot_claim_a_complete_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full selected batch is required before artifacts become evaluation-ready."""

    from tradingagents.self_enhancement.phase14c_runner import classify_discovery_outcome

    assert classify_discovery_outcome(
        complete_spec_count=0,
        hft_suitable_spec_count=0,
        deferred_group_count=0,
        classification_failure_count=0,
    ) == ("COMPLETE", "DISCOVERY_COMPLETE_NO_COMPLETE_SPEC")
    assert classify_discovery_outcome(
        complete_spec_count=2,
        hft_suitable_spec_count=0,
        deferred_group_count=0,
        classification_failure_count=0,
    ) == ("COMPLETE", "DISCOVERY_COMPLETE_NO_HFT_SUITABLE_SPEC")
    assert classify_discovery_outcome(
        complete_spec_count=1,
        hft_suitable_spec_count=1,
        deferred_group_count=1,
        classification_failure_count=0,
    ) == ("INCOMPLETE", "DISCOVERY_INCOMPLETE_BUDGET_LIMITED")
    assert classify_discovery_outcome(
        complete_spec_count=1,
        hft_suitable_spec_count=1,
        deferred_group_count=0,
        classification_failure_count=1,
    ) == ("INCOMPLETE", "DISCOVERY_INCOMPLETE_CLASSIFICATION")


def test_blocked_evaluation_does_not_claim_a_replay_result() -> None:
    from tradingagents.self_enhancement.phase14c_runner import summarize_phase14c_evaluation

    blocked = CandidateEvaluationRecord(
        spec_id="spec-blocked",
        candidate_id=None,
        status=Phase14CCandidateStatus.NOT_RUN,
        reason_code=Phase14CPreflightReason.INVALID_CAUSAL_HFT_DATA,
        artifact_path=None,
    )
    summary = summarize_phase14c_evaluation((blocked,))

    assert summary["status"] == "BLOCKED"
    assert summary["outcome"] == "REPLAY_PREFLIGHT_BLOCKED"
    assert summary["evaluated_candidate_count"] == 0


def test_completed_replays_with_no_pass_remain_at_shadow_ceiling() -> None:
    from tradingagents.self_enhancement.phase14c_models import CandidateState
    from tradingagents.self_enhancement.phase14c_runner import summarize_phase14c_evaluation

    rejected = CandidateEvaluationRecord(
        spec_id="spec-rejected",
        candidate_id="candidate-rejected",
        status=Phase14CCandidateStatus.COMPLETED,
        reason_code=Phase14CCandidateReason.REPLAY_COMPLETED,
        artifact_path=None,
        candidate_state=CandidateState.REJECTED.value,
    )
    summary = summarize_phase14c_evaluation((rejected,))

    assert summary["status"] == "EVALUATED"
    assert summary["outcome"] == "CANDIDATES_EVALUATED_NONE_PASSED"
    assert summary["evaluated_candidate_count"] == 1
    assert summary["max_candidate_state"] is None
    with pytest.raises(ValueError, match="shadow ceiling"):
        CandidateEvaluationRecord(
            spec_id="spec-promoted",
            candidate_id="candidate-promoted",
            status=Phase14CCandidateStatus.COMPLETED,
            reason_code=Phase14CCandidateReason.REPLAY_COMPLETED,
            artifact_path=None,
            candidate_state="PROMOTED",
        )


def test_local_query_config_derives_embedding_identity_from_local_files(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from tradingagents.knowledge.models import EmbeddingSpec
    from tradingagents.self_enhancement.phase14c_runner import _knowledge_config_for_generation

    spec = EmbeddingSpec(
        model_id="BAAI/bge-small-en-v1.5",
        resolved_model_version="fastembed-0.8.0",
        runtime="onnx-cpu",
        artifact_hash="sha256:" + "a" * 64,
        dimensions=384,
        normalization_policy="l2",
        tokenizer_fingerprint="sha256:" + "b" * 64,
        model_max_input_tokens=512,
        special_token_budget=2,
        effective_corpus_content_token_limit=510,
        corpus_instruction_policy="none-v1",
        corpus_instruction_version="v1",
        query_instruction_policy="bge-search-prefix-v1",
        query_instruction_version="v1",
        truncation=False,
    )
    knowledge_root = tmp_path / "knowledge"
    model_path = tmp_path / "local-model"
    knowledge_root.mkdir()
    model_path.mkdir()

    config = _knowledge_config_for_generation(
        SimpleNamespace(embedding_spec=spec),
        knowledge_root=knowledge_root,
        embedding_model_path=model_path,
    )

    assert config.embedding_artifact_hash == ""
    assert config.embedding_tokenizer_fingerprint == ""
