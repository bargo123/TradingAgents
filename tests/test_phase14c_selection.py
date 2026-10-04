from __future__ import annotations

from dataclasses import replace

import pytest

from tradingagents.knowledge.models import KnowledgeHit
from tradingagents.self_enhancement.phase14c_models import (
    ChunkIdentity,
    DiscoveryHitRecord,
    DiscoveryRetrievalReport,
    DiscoverySelectionLimits,
    QueryReference,
)
from tradingagents.self_enhancement.phase14c_selection import (
    DEFAULT_DISCOVERY_LIMITS,
    select_diverse_evidence,
)

GENERATION_ID = "gen-phase14c-fixture"
POPULATION_HASH = "sha256:" + "p" * 64


def _hit(
    document_id: str,
    chunk_id: str,
    text: str,
    *,
    source_hash: str | None = None,
    section: str | None = "Section 1",
    score: float = 1.0,
) -> KnowledgeHit:
    digest = source_hash or ("a" if document_id.endswith("a") else "b") * 64
    return KnowledgeHit(
        document_id=document_id,
        chunk_id=chunk_id,
        source_hash=digest,
        text=text,
        score=score,
        fused_score=score,
        rerank_score=score,
        source_filename=f"{document_id}.pdf",
        source_relative_path=f"{document_id}.pdf",
        title=f"Book {document_id}",
        page=1,
        section=section,
    )


def _record(
    hit: KnowledgeHit,
    family: str = "momentum",
    formulation: str = "base",
    *,
    rank: int = 1,
    score: float | None = None,
) -> DiscoveryHitRecord:
    return DiscoveryHitRecord(
        hit,
        (
            QueryReference(
                family,
                formulation,
                rank,
                fused_score=hit.fused_score if score is None else score,
                rerank_score=hit.rerank_score if score is None else score,
            ),
        ),
    )


def _report(records: tuple[DiscoveryHitRecord, ...]) -> DiscoveryRetrievalReport:
    return DiscoveryRetrievalReport(
        generation_id=GENERATION_ID,
        population_hash=POPULATION_HASH,
        query_bank_version="fixture-query-bank-v1",
        query_bank_fingerprint="q" * 64,
        status="COMPLETE",
        records=records,
        raw_hit_count=len(records),
        unique_hit_count=len(records),
        completed_query_count=1,
    )


def _rule_text(marker: str) -> str:
    return (
        f"LONG when momentum exceeds 2 points after confirmation for setup {marker}. "
        f"Exit when momentum falls below 1 point for setup {marker}. "
        + " ".join(f"{marker}detail{index}" for index in range(30))
    )


def test_same_chunk_from_sibling_formulations_is_one_group_and_keeps_all_refs():
    hit = _hit("doc-a", "chunk-1", _rule_text("alpha"))
    retrieval = _report(
        (
            _record(hit, formulation="base", rank=1),
            _record(hit, formulation="pullback", rank=2),
        )
    )

    result = select_diverse_evidence(retrieval)

    assert result.unique_hit_count == 1
    assert result.selected_group_count == 1
    identity = ChunkIdentity(hit.document_id, hit.chunk_id, hit.source_hash)
    retained = result.duplicate_index.record_for_identity(identity)
    assert {reference.formulation_id for reference in retained.references} == {
        "base",
        "pullback",
    }
    assert result.groups[0].family_id == "momentum"


def test_selection_order_and_report_are_deterministic_round_robin():
    records = (
        _record(_hit("doc-a", "a1", _rule_text("a1"), section="Section A"), "momentum"),
        _record(_hit("doc-a", "a2", _rule_text("a2"), section="Section B"), "momentum"),
        _record(_hit("doc-b", "b1", _rule_text("b1"), section="Section A"), "momentum"),
        _record(_hit("doc-c", "c1", _rule_text("c1")), "breakout"),
    )
    retrieval = _report(records)

    first = select_diverse_evidence(retrieval)
    second = select_diverse_evidence(retrieval)

    assert first.to_json() == second.to_json()
    assert [(group.family_id, group.sentences[0].hit.document_id) for group in first.groups] == [
        ("breakout", "doc-c"),
        ("momentum", "doc-a"),
        ("momentum", "doc-b"),
        ("momentum", "doc-a"),
    ]


def test_equal_scores_use_stable_chunk_identity_not_input_order():
    records = (
        _record(_hit("doc-a", "chunk-b", _rule_text("b"), score=1.0)),
        _record(_hit("doc-a", "chunk-a", _rule_text("a"), score=1.0)),
    )
    limits = replace(
        DEFAULT_DISCOVERY_LIMITS,
        max_chunks_per_document_family=2,
        max_chunks_per_section_family=2,
        max_groups_per_document=8,
        max_groups_per_family=40,
        max_total_groups=10,
    )

    result = select_diverse_evidence(_report(records), limits=limits)

    assert [group.sentences[0].hit.chunk_id for group in result.groups] == ["chunk-a", "chunk-b"]


def test_document_section_and_family_caps_are_reported_as_deferred():
    records = (
        _record(_hit("doc-a", "a1", _rule_text("a1"), section="Section A", score=4.0)),
        _record(_hit("doc-a", "a2", _rule_text("a2"), section="Section A", score=3.0)),
        _record(_hit("doc-a", "a3", _rule_text("a3"), section="Section B", score=2.0)),
        _record(_hit("doc-a", "a4", _rule_text("a4"), section="Section B", score=1.0)),
        _record(_hit("doc-b", "b1", _rule_text("b1"), section="Section A")),
        _record(_hit("doc-c", "c1", _rule_text("c1"), section="Section A"), "breakout"),
    )
    limits = replace(
        DEFAULT_DISCOVERY_LIMITS,
        max_chunks_per_section_family=1,
        max_groups_per_family=5,
        max_total_groups=10,
    )

    result = select_diverse_evidence(_report(records), limits=limits)

    assert result.selected_group_count == 4
    assert result.deferred_group_count == 2
    assert result.coverage_complete is False
    assert result.eligible_groups_by_family == {"breakout": 1, "momentum": 5}
    assert result.selected_groups_by_family == {"breakout": 1, "momentum": 3}
    assert result.deferred_groups_by_family == {"breakout": 0, "momentum": 2}
    selected_a = [
        group for group in result.groups if group.family_id == "momentum" and group.sentences[0].hit.document_id == "doc-a"
    ]
    assert len(selected_a) == 2
    assert {group.sentences[0].hit.section for group in selected_a} == {"Section A", "Section B"}


def test_document_family_cap_keeps_at_most_two_distinct_chunks():
    records = tuple(
        _record(
            _hit("doc-a", f"chunk-{index}", _rule_text(f"doc-family-{index}"), section=f"Section {index}"),
            "momentum",
        )
        for index in range(3)
    )

    result = select_diverse_evidence(_report(records))

    assert result.eligible_groups_by_document == {"doc-a": 3}
    assert result.selected_groups_by_document == {"doc-a": 2}
    assert result.deferred_groups_by_document == {"doc-a": 1}


def test_family_cap_and_document_group_cap_are_separately_enforced():
    family_limited = select_diverse_evidence(
        _report(
            (
                _record(_hit("doc-a", "a1", _rule_text("family-a")), "momentum"),
                _record(_hit("doc-b", "b1", _rule_text("family-b")), "momentum"),
            )
        ),
        limits=replace(DEFAULT_DISCOVERY_LIMITS, max_groups_per_family=1),
    )
    document_limited = select_diverse_evidence(
        _report(
            (
                _record(_hit("doc-a", "a1", _rule_text("document-breakout")), "breakout"),
                _record(_hit("doc-a", "a2", _rule_text("document-momentum")), "momentum"),
            )
        ),
        limits=replace(DEFAULT_DISCOVERY_LIMITS, max_groups_per_document=1),
    )

    assert family_limited.eligible_group_count == 2
    assert family_limited.selected_groups_by_family == {"momentum": 1}
    assert family_limited.deferred_groups_by_family == {"momentum": 1}
    assert document_limited.eligible_group_count == 2
    assert document_limited.selected_group_count == 1
    assert document_limited.deferred_group_count == 1
    assert document_limited.selected_groups_by_document == {"doc-a": 1}


def test_duplicate_cluster_retains_all_records_and_counts_copies_as_one_source():
    base_tokens = [f"token{index:03d}" for index in range(120)]
    base = "LONG when momentum exceeds 2 points. " + " ".join(base_tokens) + "."
    near_tokens = list(base_tokens)
    near_tokens[60] = "differenttoken"
    near = "LONG when momentum exceeds 2 points. " + " ".join(near_tokens) + "."
    exact_normalized = "  " + base.replace("LONG", "ＬＯＮＧ").lower().replace(" ", "\u00a0") + "  "
    hits = (
        _hit("doc-a", "chunk-a", base),
        _hit("doc-b", "chunk-b", exact_normalized),
        _hit("doc-c", "chunk-c", near),
    )

    result = select_diverse_evidence(_report(tuple(_record(hit) for hit in hits)))

    identities = tuple(ChunkIdentity(hit.document_id, hit.chunk_id, hit.source_hash) for hit in hits)
    cluster_ids = {result.duplicate_index.cluster_for_identity(identity) for identity in identities}
    assert len(cluster_ids) == 1
    cluster_id = cluster_ids.pop()
    assert len(result.duplicate_index.records_for_cluster(cluster_id)) == 3
    assert result.duplicate_index.independent_source_count(identities) == 1
    assert result.duplicate_cluster_count == 1
    assert result.duplicate_chunk_count == 2
    assert result.duplicate_group_suppressed_count == 2
    assert result.selected_group_count == 1


def test_two_distinct_chunks_from_one_document_are_one_independent_source():
    first = _hit("doc-a", "chunk-a", _rule_text("first"))
    second = _hit("doc-a", "chunk-b", _rule_text("second") + " additional unrelated terminology.")
    result = select_diverse_evidence(_report((_record(first), _record(second))))
    identities = (
        ChunkIdentity.from_hit(first),
        ChunkIdentity.from_hit(second),
    )

    assert identities[0] != identities[1]
    assert result.duplicate_index.independent_source_count(identities) == 1


def test_repeated_exact_identity_with_different_source_text_fails_closed():
    first = _hit("doc-a", "chunk-1", _rule_text("first"))
    conflicting = _hit(
        "doc-a",
        "chunk-1",
        _rule_text("different"),
        source_hash=first.source_hash,
    )

    with pytest.raises(ValueError, match="conflicting source content"):
        select_diverse_evidence(_report((_record(first), _record(conflicting))))


def test_no_rule_cue_is_counted_unclassified_not_forced_into_a_group():
    retrieval = _report(
        (
            _record(_hit("doc-a", "a1", "This chapter discusses market history and terminology.")),
            _record(_hit("doc-b", "b1", _rule_text("usable"))),
        )
    )

    result = select_diverse_evidence(retrieval)

    assert result.unclassified_chunk_count == 1
    assert result.unclassified_chunks_by_family == {"momentum": 1}
    assert result.eligible_group_count == 1
    assert result.selected_group_count == 1
    assert result.coverage_complete is True


def test_group_751_is_deferred_without_raising_the_v1_hard_cap():
    records = []
    for family_index in range(19):
        family = f"family{family_index:02d}"
        for item_index in range(40):
            marker = f"family{family_index:02d}passage{item_index:03d}"
            repeated = " ".join(f"unique{family_index}_{item_index}_{word}" for word in range(40))
            text = _rule_text(marker) + " " + repeated
            hit = _hit(
                f"doc-{family_index:02d}-{item_index:03d}",
                f"chunk-{family_index:02d}-{item_index:03d}",
                text,
            )
            records.append(_record(hit, family))
    limits = replace(DEFAULT_DISCOVERY_LIMITS, max_groups_per_family=40, max_total_groups=750)

    result = select_diverse_evidence(_report(tuple(records)), limits=limits)

    assert len(result.groups) == 750
    assert result.eligible_group_count == 760
    assert result.deferred_group_count == 10
    assert result.coverage_complete is False
    assert all(1 <= len(group.sentences) <= 3 for group in result.groups)
    assert sum(result.deferred_groups_by_family.values()) == 10
    assert all(result.deferred_groups_by_family[f"family{index:02d}"] == 1 for index in range(9, 19))


def test_invalid_selection_limits_fail_closed():
    with pytest.raises(ValueError):
        DiscoverySelectionLimits(max_total_groups=751)

    with pytest.raises(ValueError):
        replace(DEFAULT_DISCOVERY_LIMITS, max_chunks_per_document_family=3)
