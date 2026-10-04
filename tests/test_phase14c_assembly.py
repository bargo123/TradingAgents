from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from tradingagents.knowledge.models import KnowledgeHit
from tradingagents.self_enhancement.book_atomic_extraction import (
    ConceptFamily,
    EvidenceSentence,
    ResolvedAtomicRule,
    StrategyConcept,
    UnresolvedAtomicRule,
)
from tradingagents.self_enhancement.book_rule_grammar import parse_supported_rule_quote
from tradingagents.self_enhancement.phase14c_assembly import assemble_atomic_concepts
from tradingagents.self_enhancement.phase14c_models import (
    AssemblyMode,
    ChunkIdentity,
    DiscoveryHitRecord,
    DuplicateCluster,
    DuplicateClusterIndex,
    QueryReference,
)
from tradingagents.self_enhancement.strategy_specs import RuleStage

GENERATION = "gen-phase14c-fixture"
FINGERPRINT = "c" * 64
POPULATION_HASH = "sha256:" + "b" * 64
FAMILY = ConceptFamily.MOMENTUM_CONTINUATION

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


def _concept(
    document_id: str,
    quotes: dict[RuleStage, str],
    *,
    parsed_overrides: dict[RuleStage, dict] | None = None,
    bad_offset_stage: RuleStage | None = None,
) -> StrategyConcept:
    chunk_id = f"chunk-{document_id}"
    text = "\n".join(quotes.values())
    source_hash = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    hit = KnowledgeHit(
        document_id=document_id,
        chunk_id=chunk_id,
        source_hash=source_hash,
        text=text,
        source_filename=f"{document_id}.pdf",
        title="Fixture trading source",
        page=4,
        section="Rules",
        extra={
            "projection_generation": GENERATION,
            "projection_population_hash": POPULATION_HASH,
        },
    )
    rules = []
    unresolved_rules = []
    offset = 0
    for stage, quote in quotes.items():
        start = text.index(quote, offset)
        end = start + len(quote)
        if stage is bad_offset_stage:
            start += 1
        evidence = EvidenceSentence(
            f"ev-{document_id}-{stage.value.lower()}",
            hit,
            1,
            1,
            start,
            end,
            quote,
            0,
            1,
        )
        parsed = parse_supported_rule_quote(quote, stage)
        if stage in (parsed_overrides or {}):
            parsed = parsed_overrides[stage]
        if parsed is None:
            unresolved_rules.append(
                UnresolvedAtomicRule(stage, evidence, "UNSUPPORTED_RULE_GRAMMAR")
            )
        else:
            rules.append(ResolvedAtomicRule(stage, evidence, parsed))
        offset = end
    return StrategyConcept(FAMILY, tuple(rules), tuple(unresolved_rules))


def _duplicate_index(
    concepts: tuple[StrategyConcept, ...],
    *,
    duplicate_document_groups: tuple[tuple[str, ...], ...] = (),
) -> DuplicateClusterIndex:
    hits = {}
    for concept in concepts:
        for rule in (*concept.rules, *concept.unresolved_rules):
            hit = rule.evidence.hit
            hits[ChunkIdentity.from_hit(hit)] = hit
    assigned: set[ChunkIdentity] = set()
    clusters = []
    groups = list(duplicate_document_groups)
    grouped_docs = {document for group in groups for document in group}
    groups.extend((document,) for document in sorted({key.document_id for key in hits}) if document not in grouped_docs)
    for documents in groups:
        records = [
            DiscoveryHitRecord(
                hit,
                (QueryReference("momentum_continuation", "fixture", 1),),
            )
            for identity, hit in hits.items()
            if identity.document_id in documents
        ]
        records.sort(key=lambda item: ChunkIdentity.from_hit(item.hit))
        if not records:
            continue
        identities = tuple(ChunkIdentity.from_hit(item.hit) for item in records)
        assigned.update(identities)
        digest = hashlib.sha256("|".join(item.key for item in identities).encode("utf-8")).hexdigest()
        clusters.append(DuplicateCluster(f"dup-{digest}", tuple(records)))
    assert assigned == set(hits)
    return DuplicateClusterIndex(tuple(sorted(clusters, key=lambda item: item.cluster_id)))


def _assemble(
    *concepts: StrategyConcept,
    duplicate_document_groups: tuple[tuple[str, ...], ...] = (),
):
    values = tuple(concepts)
    return assemble_atomic_concepts(
        values,
        generation_id=GENERATION,
        generation_fingerprint=FINGERPRINT,
        model_id="qwen3.5:2b",
        available_features=("momentum", "direction_persistence"),
        duplicate_clusters=_duplicate_index(values, duplicate_document_groups=duplicate_document_groups),
    )


def test_exact_atomic_quotes_keep_complete_provenance_and_zero_confidence() -> None:
    concept = _concept("doc-complete", COMPLETE_QUOTES)

    report = _assemble(concept)

    assert report.complete_count == 1
    assert report.partial_count == report.rejected_count == 0
    assert report.to_dict()["complete_count"] == 1
    assert report.to_dict()["partial_count"] == 0
    assert report.to_dict()["rejected_count"] == 0
    record = report.records[0]
    assert record.semantic_anchor.stage is RuleStage.ENTRY
    assert record.assembly_mode is AssemblyMode.SINGLE_SOURCE_COMPLETE
    assert record.executable_eligible is True
    assert record.spec is not None and record.spec.is_executable
    assert record.spec.implementation_confidence == 0.0
    assert record.independent_document_ids == ("doc-complete",)
    assert {claim.evidence.quote for claim in record.rule_claims} == set(COMPLETE_QUOTES.values())
    for claim in record.rule_claims:
        span = claim.evidence
        assert span is not None
        assert span.generation_id == GENERATION
        assert span.document_id == "doc-complete"
        assert span.chunk_id == "chunk-doc-complete"
        assert span.source_hash == hashlib.sha256(b"doc-complete").hexdigest()
        assert concept.rules[0].evidence.hit.text[span.start_offset:span.end_offset] == span.quote
    restored = type(record.spec).from_dict(record.spec.to_dict())
    assert restored.content_hash == record.spec.content_hash


def test_generation_fingerprint_is_not_confused_with_population_hash() -> None:
    concept = _concept("doc-distinct-pin-values", COMPLETE_QUOTES)

    report = _assemble(concept)

    assert report.complete_count == 1
    assert report.records[0].spec is not None
    assert report.records[0].spec.knowledge_fingerprint == FINGERPRINT


def test_complementary_sources_combine_only_under_a_shared_supported_anchor() -> None:
    left = _concept(
        "doc-left",
        {
            RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY],
            RuleStage.CONFIRMATION: COMPLETE_QUOTES[RuleStage.CONFIRMATION],
            RuleStage.INVALIDATION: COMPLETE_QUOTES[RuleStage.INVALIDATION],
            RuleStage.EXIT: COMPLETE_QUOTES[RuleStage.EXIT],
        },
    )
    right = _concept(
        "doc-right",
        {
            RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY],
            RuleStage.EXPECTED_MOVE: COMPLETE_QUOTES[RuleStage.EXPECTED_MOVE],
            RuleStage.PROFIT_PROTECTION: COMPLETE_QUOTES[RuleStage.PROFIT_PROTECTION],
            RuleStage.STOP_BEHAVIOR: COMPLETE_QUOTES[RuleStage.STOP_BEHAVIOR],
            RuleStage.HORIZON: COMPLETE_QUOTES[RuleStage.HORIZON],
        },
    )

    report = _assemble(left, right)

    assert len(report.records) == 1
    record = report.records[0]
    assert record.assembly_mode is AssemblyMode.COMPOSITE_RESEARCH_HYPOTHESIS
    assert record.independent_document_ids == ("doc-left", "doc-right")
    assert record.executable_eligible is False
    assert record.spec is None or not record.spec.is_executable


def test_name_or_family_similarity_without_shared_anchor_never_merges() -> None:
    left = _concept("doc-entry-a", {RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY]})
    other_entry = "LONG when momentum > 3 points after confirmation"
    right = _concept("doc-entry-b", {RuleStage.ENTRY: other_entry})

    report = _assemble(left, right)

    assert len(report.records) == 2
    assert report.partial_count == 2
    assert {record.semantic_anchor.value for record in report.records} == {"2", "3"}
    assert all(record.spec is None for record in report.records)
    assert all(record.assembly_mode is None for record in report.records)


@pytest.mark.parametrize("anchor_stage", (RuleStage.CONFIRMATION, RuleStage.INVALIDATION))
def test_confirmation_or_invalidation_is_also_a_valid_shared_anchor(anchor_stage: RuleStage) -> None:
    quote = COMPLETE_QUOTES[anchor_stage]
    left = _concept("doc-anchor-left", {anchor_stage: quote})
    right = _concept("doc-anchor-right", {anchor_stage: quote})

    report = _assemble(left, right)

    assert len(report.records) == 1
    assert report.records[0].semantic_anchor.stage is anchor_stage
    assert report.records[0].independent_document_ids == ("doc-anchor-left", "doc-anchor-right")
    assert report.records[0].status.value == "PARTIAL"


def test_incompatible_same_stage_rules_become_conflict_not_a_winner() -> None:
    left = _concept(
        "doc-exit-a",
        {
            RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY],
            RuleStage.EXIT: COMPLETE_QUOTES[RuleStage.EXIT],
        },
    )
    right = _concept(
        "doc-exit-b",
        {
            RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY],
            RuleStage.EXIT: "EXIT: LONG when momentum <= -1 points after reversal",
        },
    )

    report = _assemble(left, right)

    assert len(report.conflicts) == 1
    conflict = report.conflicts[0]
    assert conflict.stage is RuleStage.EXIT
    assert conflict.family == FAMILY.value
    assert len(conflict.evidence_refs) == 2
    assert report.records[0].executable_eligible is False
    assert report.records[0].spec is None


def test_missing_numeric_parameter_is_separate_and_never_becomes_a_rule_claim() -> None:
    concept = _concept(
        "doc-missing-value",
        {
            RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY],
            RuleStage.EXIT: "EXIT: LONG when momentum > points after reversal",
        },
    )

    report = _assemble(concept)

    assert len(report.parameter_requests) == 1
    request = report.parameter_requests[0]
    assert request.reason_code == "RESEARCH_PARAMETER_REQUIRED"
    assert request.parameter_kind.value == "NUMERIC_THRESHOLD"
    assert request.stage is RuleStage.EXIT
    assert request.unit == "points"
    assert request.evidence_refs[0].quote == "EXIT: LONG when momentum > points after reversal"
    assert all(claim.stage is not RuleStage.EXIT for claim in report.records[0].rule_claims)
    assert report.records[0].executable_eligible is False


def test_malformed_selected_evidence_rejects_the_concept_instead_of_partial_salvage() -> None:
    concept = _concept(
        "doc-malformed-extra",
        {
            RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY],
            RuleStage.EXIT: "The source gives no parseable exit rule.",
        },
    )

    report = _assemble(concept)

    assert report.rejected_count == 1
    assert report.records[0].status.value == "REJECTED"
    assert report.records[0].spec is None
    assert report.records[0].executable_eligible is False


def test_duplicate_passages_do_not_inflate_independent_document_support() -> None:
    left = _concept("doc-copy-a", {RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY]})
    right = _concept("doc-copy-b", {RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY]})

    report = _assemble(left, right, duplicate_document_groups=(("doc-copy-a", "doc-copy-b"),))

    assert len(report.records) == 1
    assert report.records[0].independent_document_ids == ("doc-copy-a",)
    assert {claim.evidence.document_id for claim in report.records[0].rule_claims} == {
        "doc-copy-a",
        "doc-copy-b",
    }


def test_every_duplicate_cluster_collapses_its_documents_independently() -> None:
    concepts = tuple(
        _concept(document_id, {RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY]})
        for document_id in ("doc-a-copy", "doc-a-original", "doc-b-copy", "doc-b-original")
    )

    report = _assemble(
        *concepts,
        duplicate_document_groups=(
            ("doc-a-copy", "doc-a-original"),
            ("doc-b-copy", "doc-b-original"),
        ),
    )

    assert len(report.records) == 1
    assert report.records[0].independent_document_ids == (
        "doc-a-copy",
        "doc-b-copy",
    )


@pytest.mark.parametrize(
    "corruption",
    ("offset", "parsed_value"),
)
def test_corrupt_offset_or_untrusted_parsed_value_is_rejected(corruption: str) -> None:
    if corruption == "offset":
        concept = _concept(
            "doc-corrupt-offset",
            {RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY]},
            bad_offset_stage=RuleStage.ENTRY,
        )
    else:
        concept = _concept(
            "doc-corrupt-parsed",
            {RuleStage.ENTRY: COMPLETE_QUOTES[RuleStage.ENTRY]},
            parsed_overrides={
                RuleStage.ENTRY: {
                    "stage": RuleStage.ENTRY,
                    "direction": "LONG",
                    "feature": "momentum",
                    "operator": ">",
                    "value": Decimal("99"),
                    "unit": "points",
                    "condition": "after confirmation",
                    "horizon_seconds": None,
                }
            },
        )

    report = _assemble(concept)

    assert report.rejected_count == 1
    assert report.rejections[0].reason_code in {"SOURCE_SPAN_MISMATCH", "PARSED_RULE_MISMATCH"}
    assert report.records[0].rule_claims == ()
