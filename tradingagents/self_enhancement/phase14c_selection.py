"""Deterministic, provenance-preserving Phase 14C evidence selection."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tradingagents.self_enhancement.book_atomic_extraction import (
    EvidenceGroup,
    EvidenceSentence,
    prepare_evidence_sentences,
)

from .phase14c_models import (
    ChunkIdentity,
    DiscoveryHitRecord,
    DiscoveryRetrievalReport,
    DiscoverySelectionLimits,
    DuplicateCluster,
    DuplicateClusterIndex,
    EvidenceSelectionReport,
)

DISCOVERY_SELECTION_VERSION = "phase14c-diverse-evidence.v1"
_NEAR_DUPLICATE_THRESHOLD = 0.90
_TOKEN = re.compile(r"\w+", re.UNICODE)

DEFAULT_DISCOVERY_LIMITS = DiscoverySelectionLimits()


@dataclass(frozen=True, slots=True)
class _Candidate:
    family_id: str
    group: EvidenceGroup
    identity: ChunkIdentity
    cluster_id: str
    section_id: str
    score_key: tuple[Any, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _identity(hit: Any) -> ChunkIdentity:
    return ChunkIdentity.from_hit(hit)


def _same_source_content(left: Any, right: Any) -> bool:
    left_value = left.to_dict()
    right_value = right.to_dict()
    score_fields = {"score", "semantic_score", "lexical_score", "fused_score", "rerank_score"}
    return (
        {key: value for key, value in left_value.items() if key not in score_fields}
        == {key: value for key, value in right_value.items() if key not in score_fields}
    )


def _merge_exact_records(
    report: DiscoveryRetrievalReport,
) -> tuple[DiscoveryHitRecord, ...]:
    grouped: dict[ChunkIdentity, list[DiscoveryHitRecord]] = defaultdict(list)
    for record in report.records:
        grouped[_identity(record.hit)].append(record)

    merged: list[DiscoveryHitRecord] = []
    for identity in sorted(grouped):
        records = grouped[identity]
        base_hit = records[0].hit
        if any(not _same_source_content(base_hit, record.hit) for record in records[1:]):
            raise ValueError("repeated Phase 7 chunk identity has conflicting source content or provenance")

        references = {reference for record in records for reference in record.references}
        ordered_references = tuple(
            sorted(
                references,
                key=lambda item: (
                    item.family_id,
                    item.formulation_id,
                    item.rank,
                    item.rerank_score if item.rerank_score is not None else -math.inf,
                    item.fused_score if item.fused_score is not None else -math.inf,
                    item.semantic_score if item.semantic_score is not None else -math.inf,
                    item.lexical_score if item.lexical_score is not None else -math.inf,
                ),
            )
        )
        # The hit itself is source/provenance-identical across records. Select a
        # deterministic representative independent of sibling-query order.
        representative = min(records, key=lambda item: item.hit.to_json()).hit
        merged.append(DiscoveryHitRecord(representative, ordered_references))
    return tuple(merged)


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _five_token_shingles(normalized: str) -> frozenset[tuple[str, ...]]:
    tokens = tuple(_TOKEN.findall(normalized))
    if len(tokens) < 5:
        return frozenset()
    return frozenset(tuple(tokens[index : index + 5]) for index in range(len(tokens) - 4))


def _duplicate_index(records: Sequence[DiscoveryHitRecord]) -> DuplicateClusterIndex:
    ordered = tuple(sorted(records, key=lambda record: _identity(record.hit)))
    identities = tuple(_identity(record.hit) for record in ordered)
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Stable roots make clustering independent of traversal details.
        if identities[left_root] <= identities[right_root]:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root

    normalized = tuple(_normalized_text(record.hit.text) for record in ordered)
    first_exact: dict[str, int] = {}
    for index, value in enumerate(normalized):
        if not value:
            continue
        prior = first_exact.setdefault(value, index)
        union(prior, index)

    shingles = tuple(_five_token_shingles(value) for value in normalized)
    for left in range(len(ordered)):
        if not shingles[left]:
            continue
        for right in range(left + 1, len(ordered)):
            if find(left) == find(right) or not shingles[right]:
                continue
            smaller = min(len(shingles[left]), len(shingles[right]))
            larger = max(len(shingles[left]), len(shingles[right]))
            if smaller / larger < _NEAR_DUPLICATE_THRESHOLD:
                continue
            intersection = len(shingles[left].intersection(shingles[right]))
            union_count = len(shingles[left]) + len(shingles[right]) - intersection
            if union_count and intersection / union_count >= _NEAR_DUPLICATE_THRESHOLD:
                union(left, right)

    members: dict[int, list[DiscoveryHitRecord]] = defaultdict(list)
    for index, record in enumerate(ordered):
        members[find(index)].append(record)

    clusters: list[DuplicateCluster] = []
    for group in members.values():
        group.sort(key=lambda record: _identity(record.hit))
        member_keys = [_identity(record.hit).to_dict() for record in group]
        cluster_id = "dup-" + hashlib.sha256(_canonical_json(member_keys).encode("utf-8")).hexdigest()
        clusters.append(DuplicateCluster(cluster_id, tuple(group)))
    return DuplicateClusterIndex(tuple(sorted(clusters, key=lambda item: item.cluster_id)))


def _section_id(hit: Any) -> str:
    section = getattr(hit, "section", None)
    if not isinstance(section, str) or not section.strip():
        path = getattr(hit, "section_path", ())
        section = path[-1] if isinstance(path, (tuple, list)) and path else getattr(hit, "chapter", None)
    if not isinstance(section, str) or not section.strip():
        return "__unsectioned__"
    return " ".join(unicodedata.normalize("NFKC", section).casefold().split())


def _negated_score(values: Sequence[float | None]) -> float:
    available = [value for value in values if value is not None]
    return -max(available) if available else math.inf


def _candidate_score_key(record: DiscoveryHitRecord, family_id: str, identity: ChunkIdentity) -> tuple[Any, ...]:
    references = tuple(reference for reference in record.references if reference.family_id == family_id)
    hit = record.hit
    return (
        _negated_score(tuple(reference.rerank_score for reference in references) + (hit.rerank_score,)),
        _negated_score(tuple(reference.fused_score for reference in references) + (hit.fused_score,)),
        _negated_score(tuple(reference.semantic_score for reference in references) + (hit.semantic_score,)),
        _negated_score(tuple(reference.lexical_score for reference in references) + (hit.lexical_score,)),
        min(reference.rank for reference in references),
        -hit.score,
        identity.document_id,
        _section_id(hit),
        identity.chunk_id,
        identity.source_hash,
    )


def _make_candidate(
    record: DiscoveryHitRecord,
    family_id: str,
    cluster_id: str,
    sentences: Sequence[EvidenceSentence],
) -> _Candidate | None:
    sentences = tuple(item for item in sentences if item.relevance_score > 0)
    if not sentences:
        return None
    identity = _identity(record.hit)
    group_payload = [family_id, identity.to_dict(), [item.evidence_id for item in sentences]]
    suffix = hashlib.sha256(_canonical_json(group_payload).encode("utf-8")).hexdigest()
    group = EvidenceGroup(family_id, f"group-{suffix}", sentences)
    return _Candidate(
        family_id,
        group,
        identity,
        cluster_id,
        _section_id(record.hit),
        _candidate_score_key(record, family_id, identity),
    )


def _round_robin(queues: Mapping[str, Sequence[_Candidate]]) -> tuple[_Candidate, ...]:
    ordered_keys = tuple(sorted(queues))
    offsets = dict.fromkeys(ordered_keys, 0)
    result: list[_Candidate] = []
    while True:
        advanced = False
        for key in ordered_keys:
            offset = offsets[key]
            queue = queues[key]
            if offset < len(queue):
                result.append(queue[offset])
                offsets[key] = offset + 1
                advanced = True
        if not advanced:
            return tuple(result)


def _ordered_candidates(candidates: Sequence[_Candidate]) -> tuple[_Candidate, ...]:
    family_documents: dict[str, dict[str, dict[str, list[_Candidate]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for candidate in candidates:
        family_documents[candidate.family_id][candidate.identity.document_id][candidate.section_id].append(candidate)

    family_queues: dict[str, tuple[_Candidate, ...]] = {}
    for family_id, document_map in family_documents.items():
        document_queues: dict[str, tuple[_Candidate, ...]] = {}
        for document_id, section_map in document_map.items():
            section_queues = {
                section_id: tuple(sorted(queue, key=lambda item: item.score_key))
                for section_id, queue in section_map.items()
            }
            document_queues[document_id] = _round_robin(section_queues)
        family_queues[family_id] = _round_robin(document_queues)
    return _round_robin(family_queues)


def _freeze_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(sorted(values.items())))


def select_diverse_evidence(
    report: DiscoveryRetrievalReport,
    *,
    limits: DiscoverySelectionLimits = DEFAULT_DISCOVERY_LIMITS,
) -> EvidenceSelectionReport:
    """Create exact-source groups with bounded family/book/section diversity."""

    if not isinstance(report, DiscoveryRetrievalReport):
        raise TypeError("report must be DiscoveryRetrievalReport")
    if report.status != "COMPLETE":
        raise ValueError("selection requires a complete retrieval report")
    if not isinstance(limits, DiscoverySelectionLimits):
        raise TypeError("limits must be DiscoverySelectionLimits")

    records = _merge_exact_records(report)
    duplicate_index = _duplicate_index(records)
    identity_to_cluster = {
        identity: cluster.cluster_id
        for cluster in duplicate_index.clusters
        for identity in cluster.identities
    }

    cues_by_identity: dict[ChunkIdentity, tuple[EvidenceSentence, ...]] = {}
    unclassified_families: dict[str, set[ChunkIdentity]] = defaultdict(set)
    for record in records:
        identity = _identity(record.hit)
        cues = tuple(
            item
            for item in prepare_evidence_sentences((record.hit,), max_selected_sentences=3)
            if item.relevance_score > 0
        )
        if cues:
            cues_by_identity[identity] = cues
        else:
            for family_id in {reference.family_id for reference in record.references}:
                unclassified_families[family_id].add(identity)

    unclassified_ids = {
        identity
        for identity in ( _identity(record.hit) for record in records )
        if identity not in cues_by_identity
    }
    raw_candidates: list[_Candidate] = []
    for record in records:
        identity = _identity(record.hit)
        if identity not in cues_by_identity:
            continue
        for family_id in sorted({reference.family_id for reference in record.references}):
            candidate = _make_candidate(
                record,
                family_id,
                identity_to_cluster[identity],
                cues_by_identity[identity],
            )
            if candidate is not None:
                raw_candidates.append(candidate)

    best_by_family_cluster: dict[tuple[str, str], _Candidate] = {}
    for candidate in raw_candidates:
        key = (candidate.family_id, candidate.cluster_id)
        current = best_by_family_cluster.get(key)
        if current is None or candidate.score_key < current.score_key:
            best_by_family_cluster[key] = candidate
    candidates = tuple(best_by_family_cluster.values())
    duplicate_group_suppressed_count = len(raw_candidates) - len(candidates)
    ordered_candidates = _ordered_candidates(candidates)

    eligible_by_family = Counter(candidate.family_id for candidate in candidates)
    eligible_by_document = Counter(candidate.identity.document_id for candidate in candidates)
    selected: list[_Candidate] = []
    selected_by_family: Counter[str] = Counter()
    selected_by_document: Counter[str] = Counter()
    document_family_chunks: dict[tuple[str, str], set[ChunkIdentity]] = defaultdict(set)
    section_family_chunks: dict[tuple[str, str, str], set[ChunkIdentity]] = defaultdict(set)

    for candidate in ordered_candidates:
        identity = candidate.identity
        document_id = identity.document_id
        family_id = candidate.family_id
        document_family = (document_id, family_id)
        section_family = (document_id, candidate.section_id, family_id)
        violates_cap = (
            (
                identity not in document_family_chunks[document_family]
                and len(document_family_chunks[document_family]) >= limits.max_chunks_per_document_family
            )
            or (
                identity not in section_family_chunks[section_family]
                and len(section_family_chunks[section_family]) >= limits.max_chunks_per_section_family
            )
            or selected_by_document[document_id] >= limits.max_groups_per_document
            or selected_by_family[family_id] >= limits.max_groups_per_family
            or len(selected) >= limits.max_total_groups
        )
        if violates_cap:
            continue
        selected.append(candidate)
        document_family_chunks[document_family].add(identity)
        section_family_chunks[section_family].add(identity)
        selected_by_document[document_id] += 1
        selected_by_family[family_id] += 1

    eligible_family_keys = sorted(eligible_by_family)
    eligible_document_keys = sorted(eligible_by_document)
    eligible_groups = len(candidates)
    selected_groups = len(selected)
    deferred_groups = eligible_groups - selected_groups
    selected_by_family_full = {
        key: selected_by_family[key] for key in eligible_family_keys
    }
    selected_by_document_full = {
        key: selected_by_document[key] for key in eligible_document_keys
    }
    deferred_by_family = {
        key: eligible_by_family[key] - selected_by_family_full[key]
        for key in eligible_family_keys
    }
    deferred_by_document = {
        key: eligible_by_document[key] - selected_by_document_full[key]
        for key in eligible_document_keys
    }
    selected_identities = {
        candidate.identity for candidate in selected
    }

    return EvidenceSelectionReport(
        generation_id=report.generation_id,
        population_hash=report.population_hash,
        selection_version=DISCOVERY_SELECTION_VERSION,
        limits=limits,
        groups=tuple(candidate.group for candidate in selected),
        duplicate_index=duplicate_index,
        raw_hit_count=report.raw_hit_count,
        unique_hit_count=len(records),
        duplicate_cluster_count=duplicate_index.duplicate_cluster_count,
        duplicate_chunk_count=duplicate_index.duplicate_chunk_count,
        duplicate_group_suppressed_count=duplicate_group_suppressed_count,
        selected_chunk_count=len(selected_identities),
        unclassified_chunk_count=len(unclassified_ids),
        unclassified_chunks_by_family=_freeze_counts(
            {family: len(identities) for family, identities in unclassified_families.items()}
        ),
        eligible_group_count=eligible_groups,
        selected_group_count=selected_groups,
        deferred_group_count=deferred_groups,
        eligible_groups_by_family=_freeze_counts(dict(eligible_by_family)),
        selected_groups_by_family=_freeze_counts(selected_by_family_full),
        deferred_groups_by_family=_freeze_counts(deferred_by_family),
        eligible_groups_by_document=_freeze_counts(dict(eligible_by_document)),
        selected_groups_by_document=_freeze_counts(selected_by_document_full),
        deferred_groups_by_document=_freeze_counts(deferred_by_document),
        coverage_complete=deferred_groups == 0,
    )


__all__ = [
    "DEFAULT_DISCOVERY_LIMITS",
    "DISCOVERY_SELECTION_VERSION",
    "select_diverse_evidence",
]
