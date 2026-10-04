"""Immutable value objects shared by offline Phase 14C discovery stages."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from tradingagents.knowledge.models import ContentType, KnowledgeHit, SerializableModel
from tradingagents.self_enhancement.book_atomic_extraction import EvidenceGroup
from tradingagents.self_enhancement.models import CandidateState
from tradingagents.self_enhancement.strategy_specs import (
    EvidenceSpan,
    RuleDirection,
    RuleOperator,
    RuleStage,
    StrategyRuleClaim,
    StrategySpec,
)


@dataclass(frozen=True, slots=True)
class Phase14CGenerationPin:
    generation_id: str
    generation_fingerprint: str
    population_hash: str
    status: str
    vector_ready: bool
    lexical_ready: bool
    vector_location: Path
    lexical_location: Path
    inventory_schema_version: str
    document_count: int
    chunk_count: int

    def __post_init__(self) -> None:
        for name in ("generation_id", "generation_fingerprint", "population_hash", "status"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(self, "vector_location", Path(self.vector_location))
        object.__setattr__(self, "lexical_location", Path(self.lexical_location))
        if self.document_count < 0 or self.chunk_count < 0:
            raise ValueError("generation counts must be non-negative")


@dataclass(frozen=True, slots=True)
class CorpusInventory:
    unique_source_count: int
    indexed_document_count: int
    queryable_document_count: int
    alias_count: int
    needs_ocr_count: int
    unavailable_resource_count: int
    active_chunk_count: int
    source_manifest_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "unique_source_count",
            "indexed_document_count",
            "queryable_document_count",
            "alias_count",
            "needs_ocr_count",
            "unavailable_resource_count",
            "active_chunk_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if len(self.source_manifest_fingerprint) != 64:
            raise ValueError("source_manifest_fingerprint must be a SHA-256 hex digest")


def _stable_code(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value):
        raise ValueError(f"{field_name} must be a stable lowercase code")
    return value


def _score(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class DiscoveryQuery:
    family_id: str
    formulation_id: str
    text: str
    content_types: tuple[ContentType, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _stable_code(self.family_id, "family_id"))
        object.__setattr__(self, "formulation_id", _stable_code(self.formulation_id, "formulation_id"))
        text = str(self.text).strip()
        if not text:
            raise ValueError("query text must be non-empty")
        object.__setattr__(self, "text", text)
        try:
            content_types = tuple(ContentType(value) for value in self.content_types)
        except ValueError as exc:
            raise ValueError("query contains an unsupported content type") from exc
        if len(content_types) != len(set(content_types)):
            raise ValueError("query content_types must not contain duplicates")
        object.__setattr__(self, "content_types", content_types)


@dataclass(frozen=True, slots=True)
class QueryReference:
    family_id: str
    formulation_id: str
    rank: int
    semantic_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _stable_code(self.family_id, "family_id"))
        object.__setattr__(self, "formulation_id", _stable_code(self.formulation_id, "formulation_id"))
        rank = int(self.rank)
        if rank < 1:
            raise ValueError("query reference rank must be positive")
        object.__setattr__(self, "rank", rank)
        for name in ("semantic_score", "lexical_score", "fused_score", "rerank_score"):
            object.__setattr__(self, name, _score(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class DiscoveryHitRecord:
    hit: KnowledgeHit
    references: tuple[QueryReference, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hit, KnowledgeHit):
            raise TypeError("discovery record hit must be KnowledgeHit")
        references = tuple(self.references)
        if not references or any(not isinstance(ref, QueryReference) for ref in references):
            raise ValueError("discovery record requires QueryReference provenance")
        object.__setattr__(self, "references", references)


@dataclass(frozen=True, slots=True)
class DiscoveryQueryFailure:
    family_id: str
    formulation_id: str
    code: str
    exception_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _stable_code(self.family_id, "family_id"))
        object.__setattr__(self, "formulation_id", _stable_code(self.formulation_id, "formulation_id"))
        object.__setattr__(self, "code", _stable_code(self.code.lower(), "code"))
        exception_type = str(self.exception_type).strip()
        if not exception_type or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", exception_type):
            raise ValueError("exception_type must be a safe type name")
        object.__setattr__(self, "exception_type", exception_type)


@dataclass(frozen=True, slots=True)
class DiscoveryRetrievalReport:
    generation_id: str
    population_hash: str
    query_bank_version: str
    query_bank_fingerprint: str
    status: str
    records: tuple[DiscoveryHitRecord, ...] = ()
    raw_hit_count: int = 0
    unique_hit_count: int = 0
    completed_query_count: int = 0
    query_failures: tuple[DiscoveryQueryFailure, ...] = ()

    def __post_init__(self) -> None:
        for name in ("generation_id", "population_hash", "query_bank_version", "query_bank_fingerprint"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.status not in {"COMPLETE", "FAILED"}:
            raise ValueError("retrieval status must be COMPLETE or FAILED")
        records = tuple(self.records)
        failures = tuple(self.query_failures)
        if any(not isinstance(record, DiscoveryHitRecord) for record in records):
            raise TypeError("records must contain DiscoveryHitRecord values")
        if any(not isinstance(failure, DiscoveryQueryFailure) for failure in failures):
            raise TypeError("query_failures must contain DiscoveryQueryFailure values")
        counts = (int(self.raw_hit_count), int(self.unique_hit_count), int(self.completed_query_count))
        if any(value < 0 for value in counts):
            raise ValueError("retrieval counts must be non-negative")
        if self.status == "FAILED" and (records or counts[0] or counts[1]):
            raise ValueError("failed retrieval must not claim partial hit results")
        if self.status == "COMPLETE" and failures:
            raise ValueError("complete retrieval cannot contain query failures")
        if counts[1] != len(records) or counts[0] < counts[1]:
            raise ValueError("retrieval hit counts do not match records")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "query_failures", failures)
        object.__setattr__(self, "raw_hit_count", counts[0])
        object.__setattr__(self, "unique_hit_count", counts[1])
        object.__setattr__(self, "completed_query_count", counts[2])


@dataclass(frozen=True, slots=True, order=True)
class ChunkIdentity(SerializableModel):
    """Exact Phase 7 identity required to associate a source chunk safely."""

    document_id: str
    chunk_id: str
    source_hash: str

    def __post_init__(self) -> None:
        for name in ("document_id", "chunk_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        digest = str(self.source_hash).strip()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("source_hash must be a lowercase SHA-256 digest")
        object.__setattr__(self, "source_hash", digest)

    @classmethod
    def from_hit(cls, hit: KnowledgeHit) -> ChunkIdentity:
        if not isinstance(hit, KnowledgeHit):
            raise TypeError("hit must be KnowledgeHit")
        return cls(hit.document_id, hit.chunk_id, str(hit.source_hash or ""))

    @property
    def key(self) -> str:
        return json.dumps(
            [self.document_id, self.chunk_id, self.source_hash],
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class DuplicateCluster(SerializableModel):
    cluster_id: str
    records: tuple[DiscoveryHitRecord, ...]

    def __post_init__(self) -> None:
        cluster_id = str(self.cluster_id).strip()
        if not re.fullmatch(r"dup-[0-9a-f]{64}", cluster_id):
            raise ValueError("cluster_id must be a deterministic SHA-256 cluster ID")
        records = tuple(self.records)
        if not records or any(not isinstance(record, DiscoveryHitRecord) for record in records):
            raise ValueError("duplicate cluster must contain retrieval records")
        identities = tuple(ChunkIdentity.from_hit(record.hit) for record in records)
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate cluster contains repeated chunk identities")
        if tuple(sorted(identities)) != identities:
            raise ValueError("duplicate cluster records must be identity ordered")
        object.__setattr__(self, "cluster_id", cluster_id)
        object.__setattr__(self, "records", records)

    @property
    def identities(self) -> tuple[ChunkIdentity, ...]:
        return tuple(ChunkIdentity.from_hit(record.hit) for record in self.records)


@dataclass(frozen=True, slots=True)
class DuplicateClusterIndex(SerializableModel):
    """Stable exact-identity to text-cluster lookup retaining every hit/reference."""

    clusters: tuple[DuplicateCluster, ...]

    def __post_init__(self) -> None:
        clusters = tuple(self.clusters)
        if any(not isinstance(cluster, DuplicateCluster) for cluster in clusters):
            raise TypeError("clusters must contain DuplicateCluster values")
        cluster_ids = tuple(cluster.cluster_id for cluster in clusters)
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("duplicate cluster IDs must be unique")
        identities = tuple(identity for cluster in clusters for identity in cluster.identities)
        if len(identities) != len(set(identities)):
            raise ValueError("a chunk identity may occur in only one duplicate cluster")
        if tuple(sorted(clusters, key=lambda item: item.cluster_id)) != clusters:
            raise ValueError("duplicate clusters must be ID ordered")
        object.__setattr__(self, "clusters", clusters)

    @property
    def identities(self) -> tuple[ChunkIdentity, ...]:
        return tuple(identity for cluster in self.clusters for identity in cluster.identities)

    @property
    def duplicate_cluster_count(self) -> int:
        return sum(len(cluster.records) > 1 for cluster in self.clusters)

    @property
    def duplicate_chunk_count(self) -> int:
        return sum(len(cluster.records) - 1 for cluster in self.clusters if len(cluster.records) > 1)

    def cluster_for_identity(self, identity: ChunkIdentity) -> str:
        if not isinstance(identity, ChunkIdentity):
            raise TypeError("identity must be ChunkIdentity")
        for cluster in self.clusters:
            if identity in cluster.identities:
                return cluster.cluster_id
        raise KeyError("chunk identity is not present in the duplicate index")

    def records_for_cluster(self, cluster_id: str) -> tuple[DiscoveryHitRecord, ...]:
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster.records
        raise KeyError("duplicate cluster is not present in the index")

    def record_for_identity(self, identity: ChunkIdentity) -> DiscoveryHitRecord:
        cluster_id = self.cluster_for_identity(identity)
        for record in self.records_for_cluster(cluster_id):
            if ChunkIdentity.from_hit(record.hit) == identity:
                return record
        raise KeyError("chunk identity has no retained retrieval record")

    def independent_source_count(self, identities: Sequence[ChunkIdentity]) -> int:
        return len(self.independent_document_ids(identities))

    def independent_document_ids(self, identities: Sequence[ChunkIdentity]) -> tuple[str, ...]:
        """Return stable representative document IDs after passage-cluster collapse."""

        values = tuple(identities)
        if any(not isinstance(identity, ChunkIdentity) for identity in values):
            raise TypeError("identities must contain ChunkIdentity values")
        documents = {identity.document_id for identity in values}
        parents = {document_id: document_id for document_id in documents}

        def find(document_id: str) -> str:
            while parents[document_id] != document_id:
                parents[document_id] = parents[parents[document_id]]
                document_id = parents[document_id]
            return document_id

        by_cluster: dict[str, set[str]] = defaultdict(set)
        for identity in values:
            by_cluster[self.cluster_for_identity(identity)].add(identity.document_id)
        for source_documents in by_cluster.values():
            ordered = sorted(source_documents)
            if not ordered:
                continue
            root = find(ordered[0])
            for document_id in ordered[1:]:
                root = find(root)
                other = find(document_id)
                if root != other:
                    parents[max(root, other)] = min(root, other)
                    root = min(root, other)
        return tuple(sorted({find(document_id) for document_id in documents}))


@dataclass(frozen=True, slots=True)
class DiscoverySelectionLimits(SerializableModel):
    max_chunks_per_document_family: int = 2
    max_chunks_per_section_family: int = 2
    max_groups_per_document: int = 8
    max_groups_per_family: int = 40
    max_total_groups: int = 750

    def __post_init__(self) -> None:
        ceilings = {
            "max_chunks_per_document_family": 2,
            "max_chunks_per_section_family": 2,
            "max_groups_per_document": 8,
            "max_groups_per_family": 40,
            "max_total_groups": 750,
        }
        for name, maximum in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from 1 to {maximum}")


def _count_mapping(value: Mapping[str, int], field_name: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a string-to-count mapping")
    normalized: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = str(raw_key).strip()
        if not key or type(raw_count) is not int or raw_count < 0:
            raise ValueError(f"{field_name} contains an invalid key or count")
        if key in normalized:
            raise ValueError(f"{field_name} contains duplicate normalized keys")
        normalized[key] = raw_count
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class EvidenceSelectionReport(SerializableModel):
    generation_id: str
    population_hash: str
    selection_version: str
    limits: DiscoverySelectionLimits
    groups: tuple[EvidenceGroup, ...]
    duplicate_index: DuplicateClusterIndex
    raw_hit_count: int
    unique_hit_count: int
    duplicate_cluster_count: int
    duplicate_chunk_count: int
    duplicate_group_suppressed_count: int
    selected_chunk_count: int
    unclassified_chunk_count: int
    unclassified_chunks_by_family: Mapping[str, int]
    eligible_group_count: int
    selected_group_count: int
    deferred_group_count: int
    eligible_groups_by_family: Mapping[str, int]
    selected_groups_by_family: Mapping[str, int]
    deferred_groups_by_family: Mapping[str, int]
    eligible_groups_by_document: Mapping[str, int]
    selected_groups_by_document: Mapping[str, int]
    deferred_groups_by_document: Mapping[str, int]
    coverage_complete: bool

    def __post_init__(self) -> None:
        for name in ("generation_id", "population_hash", "selection_version"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        if not isinstance(self.limits, DiscoverySelectionLimits):
            raise TypeError("limits must be DiscoverySelectionLimits")
        if not isinstance(self.duplicate_index, DuplicateClusterIndex):
            raise TypeError("duplicate_index must be DuplicateClusterIndex")
        groups = tuple(self.groups)
        if any(not isinstance(group, EvidenceGroup) for group in groups):
            raise TypeError("groups must contain EvidenceGroup values")
        group_ids = tuple(group.group_id for group in groups)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("selected evidence group IDs must be unique")
        object.__setattr__(self, "groups", groups)

        count_names = (
            "raw_hit_count",
            "unique_hit_count",
            "duplicate_cluster_count",
            "duplicate_chunk_count",
            "duplicate_group_suppressed_count",
            "selected_chunk_count",
            "unclassified_chunk_count",
            "eligible_group_count",
            "selected_group_count",
            "deferred_group_count",
        )
        counts: dict[str, int] = {}
        for name in count_names:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            counts[name] = value
        if counts["raw_hit_count"] < counts["unique_hit_count"]:
            raise ValueError("raw hit count cannot be below unique hit count")
        if counts["unique_hit_count"] != len(self.duplicate_index.identities):
            raise ValueError("unique hit count differs from duplicate-index population")
        if counts["duplicate_cluster_count"] != self.duplicate_index.duplicate_cluster_count:
            raise ValueError("duplicate cluster count differs from duplicate index")
        if counts["duplicate_chunk_count"] != self.duplicate_index.duplicate_chunk_count:
            raise ValueError("duplicate chunk count differs from duplicate index")
        if counts["selected_group_count"] != len(groups):
            raise ValueError("selected group count differs from the group list")
        if counts["selected_group_count"] + counts["deferred_group_count"] != counts["eligible_group_count"]:
            raise ValueError("eligible groups must partition into selected and deferred groups")
        if type(self.coverage_complete) is not bool or self.coverage_complete != (counts["deferred_group_count"] == 0):
            raise ValueError("coverage_complete must reflect the deferred group count")

        map_names = (
            "unclassified_chunks_by_family",
            "eligible_groups_by_family",
            "selected_groups_by_family",
            "deferred_groups_by_family",
            "eligible_groups_by_document",
            "selected_groups_by_document",
            "deferred_groups_by_document",
        )
        for name in map_names:
            object.__setattr__(self, name, _count_mapping(getattr(self, name), name))
        family_keys = set(self.eligible_groups_by_family)
        document_keys = set(self.eligible_groups_by_document)
        if any(set(getattr(self, name)) != family_keys for name in (
            "selected_groups_by_family",
            "deferred_groups_by_family",
        )):
            raise ValueError("family group count maps must have identical keys")
        if any(set(getattr(self, name)) != document_keys for name in (
            "selected_groups_by_document",
            "deferred_groups_by_document",
        )):
            raise ValueError("document group count maps must have identical keys")

        for eligible, selected, deferred, total, label in (
            (
                self.eligible_groups_by_family,
                self.selected_groups_by_family,
                self.deferred_groups_by_family,
                counts["eligible_group_count"],
                "family",
            ),
            (
                self.eligible_groups_by_document,
                self.selected_groups_by_document,
                self.deferred_groups_by_document,
                counts["eligible_group_count"],
                "document",
            ),
        ):
            if sum(eligible.values()) != total:
                raise ValueError(f"{label} eligible counts do not sum to total")
            if sum(selected.values()) != counts["selected_group_count"]:
                raise ValueError(f"{label} selected counts do not sum to total")
            if sum(deferred.values()) != counts["deferred_group_count"]:
                raise ValueError(f"{label} deferred counts do not sum to total")
            if any(eligible[key] != selected[key] + deferred[key] for key in eligible):
                raise ValueError(f"{label} group counts do not partition")

        if sum(self.unclassified_chunks_by_family.values()) < counts["unclassified_chunk_count"]:
            raise ValueError("per-family unclassified counts cannot be below unique unclassified chunks")
        selected_by_family: dict[str, int] = dict.fromkeys(family_keys, 0)
        selected_by_document: dict[str, int] = dict.fromkeys(document_keys, 0)
        selected_identities: set[ChunkIdentity] = set()
        for group in groups:
            if not group.sentences:
                raise ValueError("selected evidence groups cannot be empty")
            identity = ChunkIdentity.from_hit(group.sentences[0].hit)
            if any(ChunkIdentity.from_hit(sentence.hit) != identity for sentence in group.sentences):
                raise ValueError("selected groups must preserve one exact chunk identity")
            selected_identities.add(identity)
            if group.family_id not in selected_by_family or identity.document_id not in selected_by_document:
                raise ValueError("selected group is missing from its count maps")
            selected_by_family[group.family_id] += 1
            selected_by_document[identity.document_id] += 1
        if any(selected_by_family[key] != self.selected_groups_by_family[key] for key in family_keys):
            raise ValueError("selected family counts differ from the selected groups")
        if any(selected_by_document[key] != self.selected_groups_by_document[key] for key in document_keys):
            raise ValueError("selected document counts differ from the selected groups")
        if len(selected_identities) != counts["selected_chunk_count"]:
            raise ValueError("selected chunk count differs from selected group provenance")
        if counts["selected_chunk_count"] > counts["unique_hit_count"]:
            raise ValueError("selected chunk count exceeds unique retrieval population")

        for name, value in counts.items():
            object.__setattr__(self, name, value)


class AssemblyMode(str, Enum):
    SINGLE_SOURCE_COMPLETE = "SINGLE_SOURCE_COMPLETE"
    MULTI_SOURCE_CONSISTENT = "MULTI_SOURCE_CONSISTENT"
    COMPOSITE_RESEARCH_HYPOTHESIS = "COMPOSITE_RESEARCH_HYPOTHESIS"


class AssemblyStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    CONFLICTED = "CONFLICTED"
    REJECTED = "REJECTED"


class ResearchParameterKind(str, Enum):
    NUMERIC_THRESHOLD = "NUMERIC_THRESHOLD"


@dataclass(frozen=True, slots=True)
class SemanticAnchor(SerializableModel):
    """Exact supported entry/confirmation/invalidation semantics used for joins."""

    stage: RuleStage
    direction: RuleDirection
    operator: RuleOperator
    feature: str | None
    value: str
    unit: str
    condition: str
    horizon_seconds: int | None = None

    def __post_init__(self) -> None:
        stage = RuleStage(self.stage)
        if stage not in {RuleStage.ENTRY, RuleStage.CONFIRMATION, RuleStage.INVALIDATION}:
            raise ValueError("semantic anchor stage must be entry, confirmation, or invalidation")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "direction", RuleDirection(self.direction))
        object.__setattr__(self, "operator", RuleOperator(self.operator))
        for name in ("value", "unit", "condition"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        if self.feature is not None:
            feature = str(self.feature).strip()
            if not re.fullmatch(r"[a-z][a-z0-9_]*", feature):
                raise ValueError("feature must be a normalized feature name")
            object.__setattr__(self, "feature", feature)
        if self.horizon_seconds is not None and (
            type(self.horizon_seconds) is not int or self.horizon_seconds < 1
        ):
            raise ValueError("horizon_seconds must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class SourceConflict(SerializableModel):
    family: str
    anchor: SemanticAnchor
    stage: RuleStage
    evidence_refs: tuple[EvidenceSpan, ...]
    reason_code: str = "CONTRADICTORY_SAME_STAGE_RULES"

    def __post_init__(self) -> None:
        family = str(self.family).strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", family):
            raise ValueError("family must be a closed uppercase code")
        if not isinstance(self.anchor, SemanticAnchor):
            raise TypeError("anchor must be SemanticAnchor")
        object.__setattr__(self, "stage", RuleStage(self.stage))
        refs = tuple(self.evidence_refs)
        if len(refs) < 2 or any(not isinstance(item, EvidenceSpan) for item in refs):
            raise ValueError("conflict must retain at least two exact EvidenceSpan references")
        refs = tuple(sorted(set(refs), key=_evidence_sort_key))
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "evidence_refs", refs)
        if self.reason_code != "CONTRADICTORY_SAME_STAGE_RULES":
            raise ValueError("reason_code is not a supported source conflict code")


def _evidence_sort_key(value: EvidenceSpan) -> tuple[str, str, str, int, int, str]:
    return (
        value.document_id,
        value.chunk_id,
        value.source_hash,
        value.start_offset,
        value.end_offset,
        value.quote,
    )


@dataclass(frozen=True, slots=True)
class ResearchParameterRequest(SerializableModel):
    family: str
    semantic_anchor: SemanticAnchor | None
    stage: RuleStage
    parameter_kind: ResearchParameterKind
    unit: str
    evidence_refs: tuple[EvidenceSpan, ...]
    reason_code: str = "RESEARCH_PARAMETER_REQUIRED"

    def __post_init__(self) -> None:
        family = str(self.family).strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", family):
            raise ValueError("family must be a closed uppercase code")
        if self.semantic_anchor is not None and not isinstance(self.semantic_anchor, SemanticAnchor):
            raise TypeError("semantic_anchor must be SemanticAnchor or None")
        object.__setattr__(self, "stage", RuleStage(self.stage))
        object.__setattr__(self, "parameter_kind", ResearchParameterKind(self.parameter_kind))
        unit = str(self.unit).strip()
        if not unit:
            raise ValueError("unit must be non-empty")
        refs = tuple(self.evidence_refs)
        if not refs or any(not isinstance(item, EvidenceSpan) for item in refs):
            raise ValueError("parameter request requires exact EvidenceSpan references")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "evidence_refs", tuple(sorted(set(refs), key=_evidence_sort_key)))
        if self.reason_code != "RESEARCH_PARAMETER_REQUIRED":
            raise ValueError("reason_code is not a supported research parameter request code")


@dataclass(frozen=True, slots=True)
class AssemblyRejection(SerializableModel):
    family: str
    evidence_id: str
    document_id: str
    chunk_id: str
    reason_code: str

    def __post_init__(self) -> None:
        for name in ("family", "evidence_id", "document_id", "chunk_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", self.reason_code):
            raise ValueError("reason_code must be a closed uppercase code")


@dataclass(frozen=True, slots=True)
class AssembledConcept(SerializableModel):
    family: str
    semantic_anchor: SemanticAnchor | None
    independent_document_ids: tuple[str, ...]
    rule_claims: tuple[StrategyRuleClaim, ...]
    assembly_mode: AssemblyMode | None
    status: AssemblyStatus
    executable_eligible: bool
    spec: StrategySpec | None
    missing_stages: tuple[RuleStage, ...] = ()

    def __post_init__(self) -> None:
        family = str(self.family).strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", family):
            raise ValueError("family must be a closed uppercase code")
        if self.semantic_anchor is not None and not isinstance(self.semantic_anchor, SemanticAnchor):
            raise TypeError("semantic_anchor must be SemanticAnchor or None")
        documents = tuple(sorted({str(value).strip() for value in self.independent_document_ids}))
        if any(not value for value in documents):
            raise ValueError("independent_document_ids must be non-empty strings")
        claims = tuple(self.rule_claims)
        if any(not isinstance(item, StrategyRuleClaim) for item in claims):
            raise TypeError("rule_claims must contain StrategyRuleClaim values")
        mode = AssemblyMode(self.assembly_mode) if self.assembly_mode is not None else None
        status = AssemblyStatus(self.status)
        if self.spec is not None and not isinstance(self.spec, StrategySpec):
            raise TypeError("spec must be StrategySpec or None")
        missing = tuple(sorted({RuleStage(item) for item in self.missing_stages}, key=lambda item: item.value))
        if type(self.executable_eligible) is not bool:
            raise TypeError("executable_eligible must be a boolean")
        if self.executable_eligible and (
            status is not AssemblyStatus.COMPLETE
            or mode is AssemblyMode.COMPOSITE_RESEARCH_HYPOTHESIS
            or self.spec is None
            or not self.spec.is_executable
        ):
            raise ValueError("only complete, non-composite executable specs may be eligible")
        if status is AssemblyStatus.COMPLETE and (mode is None or self.spec is None or missing):
            raise ValueError("complete assembly requires a mode, spec, and no missing stages")
        if status is AssemblyStatus.CONFLICTED and (self.spec is not None or self.executable_eligible):
            raise ValueError("conflicted assembly cannot expose an executable spec")
        if status is AssemblyStatus.REJECTED and self.executable_eligible:
            raise ValueError("rejected assembly cannot be executable")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "independent_document_ids", documents)
        object.__setattr__(self, "rule_claims", claims)
        object.__setattr__(self, "assembly_mode", mode)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "missing_stages", missing)


@dataclass(frozen=True, slots=True)
class AssemblyReport(SerializableModel):
    records: tuple[AssembledConcept, ...]
    conflicts: tuple[SourceConflict, ...] = ()
    parameter_requests: tuple[ResearchParameterRequest, ...] = ()
    rejections: tuple[AssemblyRejection, ...] = ()
    complete_count: int = field(init=False, default=0)
    partial_count: int = field(init=False, default=0)
    rejected_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        records = tuple(self.records)
        conflicts = tuple(self.conflicts)
        requests = tuple(self.parameter_requests)
        rejections = tuple(self.rejections)
        if any(not isinstance(item, AssembledConcept) for item in records):
            raise TypeError("records must contain AssembledConcept values")
        if any(not isinstance(item, SourceConflict) for item in conflicts):
            raise TypeError("conflicts must contain SourceConflict values")
        if any(not isinstance(item, ResearchParameterRequest) for item in requests):
            raise TypeError("parameter_requests must contain ResearchParameterRequest values")
        if any(not isinstance(item, AssemblyRejection) for item in rejections):
            raise TypeError("rejections must contain AssemblyRejection values")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "parameter_requests", requests)
        object.__setattr__(self, "rejections", rejections)
        object.__setattr__(self, "complete_count", sum(item.status is AssemblyStatus.COMPLETE for item in records))
        object.__setattr__(self, "partial_count", sum(item.status is AssemblyStatus.PARTIAL for item in records))
        object.__setattr__(
            self,
            "rejected_count",
            sum(item.status in {AssemblyStatus.CONFLICTED, AssemblyStatus.REJECTED} for item in records),
        )


CURRENT_STRATEGY_COMPONENTS = (
    "entry",
    "confirmation",
    "expected_move",
    "exit",
    "risk",
    "holding_horizon",
)
CURRENT_STRATEGY_IDS = ("momentum_continuation", "range_rejection")


class CurrentStrategyMappingStatus(str, Enum):
    MATCHED = "MATCHED"
    PARTIALLY_MATCHED = "PARTIALLY_MATCHED"
    NO_DIRECT_MATCH = "NO_DIRECT_MATCH"


def _freeze_contract_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized = {str(key): _freeze_contract_value(item) for key, item in value.items()}
        if len(normalized) != len(value):
            raise ValueError("contract value contains duplicate normalized keys")
        return MappingProxyType(dict(sorted(normalized.items())))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_contract_value(item) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("contract values must be finite")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError("contract values must be JSON-safe primitives, mappings, or sequences")


@dataclass(frozen=True, slots=True)
class CurrentStrategyRuleSignature(SerializableModel):
    stage: RuleStage
    direction: RuleDirection
    operator: RuleOperator
    feature: str | None
    value: str
    unit: str
    condition: str
    horizon_seconds: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", RuleStage(self.stage))
        object.__setattr__(self, "direction", RuleDirection(self.direction))
        object.__setattr__(self, "operator", RuleOperator(self.operator))
        for name in ("value", "unit", "condition"):
            normalized = str(getattr(self, name)).strip()
            if not normalized:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, normalized)
        if self.feature is not None:
            feature = str(self.feature).strip()
            if not re.fullmatch(r"[a-z][a-z0-9_]*", feature):
                raise ValueError("feature must be a normalized feature name")
            object.__setattr__(self, "feature", feature)
        if self.horizon_seconds is not None and (
            type(self.horizon_seconds) is not int or self.horizon_seconds < 1
        ):
            raise ValueError("horizon_seconds must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class CurrentStrategyComponentContract(SerializableModel):
    component: str
    canonical_values: Mapping[str, Any]
    matchable_rules: tuple[CurrentStrategyRuleSignature, ...] = ()

    def __post_init__(self) -> None:
        component = str(self.component).strip()
        if component not in CURRENT_STRATEGY_COMPONENTS:
            raise ValueError("component is not part of the current-strategy mapping contract")
        if not isinstance(self.canonical_values, Mapping):
            raise TypeError("canonical_values must be a mapping")
        values = _freeze_contract_value(self.canonical_values)
        rules = tuple(self.matchable_rules)
        if any(not isinstance(item, CurrentStrategyRuleSignature) for item in rules):
            raise TypeError("matchable_rules must contain CurrentStrategyRuleSignature values")
        if len(set(rules)) != len(rules):
            raise ValueError("matchable_rules must not contain duplicates")
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "canonical_values", values)
        object.__setattr__(self, "matchable_rules", rules)


@dataclass(frozen=True, slots=True)
class CurrentStrategyContract(SerializableModel):
    strategy_id: str
    components: tuple[CurrentStrategyComponentContract, ...]

    def __post_init__(self) -> None:
        strategy_id = str(self.strategy_id).strip()
        if strategy_id not in CURRENT_STRATEGY_IDS:
            raise ValueError("strategy_id is not a reviewed current strategy")
        components = tuple(self.components)
        if any(not isinstance(item, CurrentStrategyComponentContract) for item in components):
            raise TypeError("components must contain CurrentStrategyComponentContract values")
        if tuple(item.component for item in components) != CURRENT_STRATEGY_COMPONENTS:
            raise ValueError("current strategy contract must contain all six components in order")
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "components", components)

    def component(self, name: str) -> CurrentStrategyComponentContract:
        for item in self.components:
            if item.component == name:
                return item
        raise KeyError(f"unknown strategy component: {name}")


@dataclass(frozen=True, slots=True)
class CurrentStrategyContractSnapshot(SerializableModel):
    schema_version: str
    contract_version: str
    strategies: tuple[CurrentStrategyContract, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("schema_version", "contract_version"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        strategies = tuple(sorted(self.strategies, key=lambda item: item.strategy_id))
        if any(not isinstance(item, CurrentStrategyContract) for item in strategies):
            raise TypeError("strategies must contain CurrentStrategyContract values")
        if tuple(item.strategy_id for item in strategies) != CURRENT_STRATEGY_IDS:
            raise ValueError("snapshot must contain each reviewed strategy exactly once")
        payload = {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "strategies": [item.to_dict() for item in strategies],
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "strategies", strategies)
        object.__setattr__(self, "fingerprint", fingerprint)

    def strategy(self, strategy_id: str) -> CurrentStrategyContract:
        for item in self.strategies:
            if item.strategy_id == strategy_id:
                return item
        raise KeyError(f"unknown current strategy: {strategy_id}")


@dataclass(frozen=True, slots=True)
class CurrentStrategyComponentMapping(SerializableModel):
    component: str
    status: CurrentStrategyMappingStatus
    matched_source_rule_fingerprints: tuple[str, ...] = ()
    reason_code: str = "NO_EXACT_COMPONENT_MATCH"

    def __post_init__(self) -> None:
        component = str(self.component).strip()
        if component not in CURRENT_STRATEGY_COMPONENTS:
            raise ValueError("component is not part of the current-strategy mapping contract")
        status = CurrentStrategyMappingStatus(self.status)
        fingerprints = tuple(sorted(set(self.matched_source_rule_fingerprints)))
        if any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in fingerprints):
            raise ValueError("matched source rule fingerprints must be SHA-256 hex values")
        if bool(fingerprints) != (status is not CurrentStrategyMappingStatus.NO_DIRECT_MATCH):
            raise ValueError("only directly matched components may cite source rule fingerprints")
        reason = str(self.reason_code).strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", reason):
            raise ValueError("reason_code must be a closed uppercase code")
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "matched_source_rule_fingerprints", fingerprints)
        object.__setattr__(self, "reason_code", reason)


@dataclass(frozen=True, slots=True)
class CurrentStrategyMapping(SerializableModel):
    strategy_id: str
    components: tuple[CurrentStrategyComponentMapping, ...]

    def __post_init__(self) -> None:
        strategy_id = str(self.strategy_id).strip()
        if strategy_id not in CURRENT_STRATEGY_IDS:
            raise ValueError("strategy_id is not a reviewed current strategy")
        components = tuple(self.components)
        if any(not isinstance(item, CurrentStrategyComponentMapping) for item in components):
            raise TypeError("components must contain CurrentStrategyComponentMapping values")
        if tuple(item.component for item in components) != CURRENT_STRATEGY_COMPONENTS:
            raise ValueError("current strategy mapping must contain all six components in order")
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "components", components)


class Phase14CPreflightReason(str, Enum):
    READY = "READY"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    INVALID_PHASE14A = "INVALID_PHASE14A"
    NO_VERIFIED_PHASE14A_EXPERIENCE = "NO_VERIFIED_PHASE14A_EXPERIENCE"
    INVALID_DEMO_SOURCE = "INVALID_DEMO_SOURCE"
    DEMO_RECONCILIATION_UNCERTAIN = "DEMO_RECONCILIATION_UNCERTAIN"
    INVALID_CAUSAL_HFT_DATA = "INVALID_CAUSAL_HFT_DATA"
    SOURCE_MUTATED_DURING_PREFLIGHT = "SOURCE_MUTATED_DURING_PREFLIGHT"


class Phase14CCandidateStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Phase14CCandidateReason(str, Enum):
    SPEC_NOT_EXECUTABLE = "SPEC_NOT_EXECUTABLE"
    UNIMPLEMENTED_SPEC = "UNIMPLEMENTED_SPEC"
    DUPLICATE_SPEC = "DUPLICATE_SPEC"
    MAX_CANDIDATE_RUNS_REACHED = "MAX_CANDIDATE_RUNS_REACHED"
    REPLAY_COMPLETED = "REPLAY_COMPLETED"
    REPLAY_FAILED = "REPLAY_FAILED"
    SOURCE_MUTATED_DURING_EVALUATION = "SOURCE_MUTATED_DURING_EVALUATION"
    STATE_EXCEEDS_SHADOW_CEILING = "STATE_EXCEEDS_SHADOW_CEILING"


def _phase14c_path(value: Path | str, field_name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field_name} must be an explicit path")
    return Path(value).expanduser().resolve()


def _phase14c_fingerprints(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("source_fingerprints must be a mapping")
    allowed = {"phase14a", "hft", "demo"}
    if not set(value).issubset(allowed):
        raise ValueError("source_fingerprints contains an unknown source")
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        name = str(raw_name)
        digest = str(raw_digest)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("source fingerprint must be a SHA-256 hex digest")
        normalized[name] = digest
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class Phase14CSourcePaths:
    """Explicit local Phase 14A, causal HFT, and DEMO evidence sources."""

    phase14a_path: Path
    hft_path: Path
    demo_path: Path

    def __post_init__(self) -> None:
        for name in ("phase14a_path", "hft_path", "demo_path"):
            object.__setattr__(self, name, _phase14c_path(getattr(self, name), name))

    def to_dict(self) -> dict[str, str]:
        return {
            "phase14a_path": str(self.phase14a_path),
            "hft_path": str(self.hft_path),
            "demo_path": str(self.demo_path),
        }


def _identity_sources(value: Mapping[str, str], *, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = {str(key): str(item) for key, item in value.items()}
    if set(normalized) != {"phase14a", "hft", "demo"}:
        raise ValueError(f"{field_name} must identify Phase 14A, HFT, and DEMO explicitly")
    if any(not item.strip() for item in normalized.values()):
        raise ValueError(f"{field_name} values must be non-empty")
    if field_name == "source_paths":
        try:
            normalized = {
                key: str(Path(item).expanduser().resolve(strict=False))
                for key, item in normalized.items()
            }
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("source_paths must be resolvable local paths") from exc
        if any(not Path(item).is_absolute() for item in normalized.values()):
            raise ValueError("source_paths must be absolute")
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("source_paths must identify three distinct source files")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class Phase14CRunIdentity(SerializableModel):
    """Complete safe semantic identity required to resume a discovery run."""

    generation_id: str
    generation_fingerprint: str
    population_hash: str
    embedding_spec_fingerprint: str
    query_bank_version: str
    query_bank_fingerprint: str
    top_k_per_formulation: int
    selection_version: str
    selection_config_fingerprint: str
    assembly_version: str
    assembly_schema_version: str
    artifact_schema_fingerprint: str
    provider: str
    ollama_endpoint: str
    model_id: str
    model_version: str
    temperature: float
    timeout_seconds: float
    max_output_tokens: int
    context_tokens: int
    atomic_cache_path: Path
    atomic_cache_schema_version: str
    presence_prompt_version: str
    presence_schema_version: str
    extraction_prompt_version: str
    segmentation_version: str
    stage_budgets_fingerprint: str
    resume_schema_version: str
    source_paths: Mapping[str, str]
    source_fingerprints: Mapping[str, str]
    identity_schema_version: str = "phase14c-run-identity.v1"

    def __post_init__(self) -> None:
        for name in (
            "generation_id",
            "query_bank_version",
            "selection_config_fingerprint",
            "selection_version",
            "assembly_version",
            "assembly_schema_version",
            "artifact_schema_fingerprint",
            "provider",
            "ollama_endpoint",
            "model_id",
            "model_version",
            "atomic_cache_schema_version",
            "presence_prompt_version",
            "presence_schema_version",
            "extraction_prompt_version",
            "segmentation_version",
            "resume_schema_version",
            "identity_schema_version",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        if self.provider != "ollama-local":
            raise ValueError("Phase 14C provider must be local Ollama")
        if not self.model_id.startswith("qwen3.5:2b"):
            raise ValueError("Phase 14C model must be qwen3.5:2b-compatible")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", self.model_version):
            raise ValueError("model_version must be a safe bounded local identifier")
        for name in (
            "generation_fingerprint",
            "embedding_spec_fingerprint",
            "query_bank_fingerprint",
            "selection_config_fingerprint",
            "artifact_schema_fingerprint",
            "stage_budgets_fingerprint",
        ):
            digest = str(getattr(self, name)).strip()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            object.__setattr__(self, name, digest)
        population = str(self.population_hash).strip()
        if not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", population):
            raise ValueError("population_hash must be a SHA-256 identity")
        object.__setattr__(self, "population_hash", population)
        if isinstance(self.temperature, bool) or float(self.temperature) != 0.0:
            raise ValueError("Phase 14C local teacher temperature must be exactly zero")
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ValueError("timeout_seconds must be bounded between 0 and 600")
        if type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= 8192:
            raise ValueError("max_output_tokens must be an integer from 1 through 8192")
        if type(self.context_tokens) is not int or not 512 <= self.context_tokens <= 32768:
            raise ValueError("context_tokens must be an integer from 512 through 32768")
        object.__setattr__(self, "temperature", 0.0)
        object.__setattr__(self, "timeout_seconds", timeout)
        if type(self.top_k_per_formulation) is not int or not 1 <= self.top_k_per_formulation <= 50:
            raise ValueError("top_k_per_formulation must be an integer from 1 through 50")
        try:
            endpoint = urlsplit(self.ollama_endpoint)
            port = endpoint.port
        except ValueError as exc:
            raise ValueError("ollama_endpoint must be a native local Ollama endpoint") from exc
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"localhost", "127.0.0.1"}
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path not in {"", "/"}
            or endpoint.query
            or endpoint.fragment
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError("ollama_endpoint must be a native local Ollama endpoint")
        object.__setattr__(self, "ollama_endpoint", f"http://{endpoint.netloc}")
        cache = _phase14c_path(self.atomic_cache_path, "atomic_cache_path")
        if not cache.is_absolute():
            raise ValueError("atomic_cache_path must resolve to an absolute path")
        object.__setattr__(self, "atomic_cache_path", cache)
        source_paths = _identity_sources(self.source_paths, field_name="source_paths")
        fingerprints = _identity_sources(
            self.source_fingerprints,
            field_name="source_fingerprints",
        )
        for name, digest in fingerprints.items():
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{name} source fingerprint must be a lowercase SHA-256 digest")
        object.__setattr__(self, "source_paths", source_paths)
        object.__setattr__(self, "source_fingerprints", fingerprints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "generation_id": self.generation_id,
            "generation_fingerprint": self.generation_fingerprint,
            "population_hash": self.population_hash,
            "embedding_spec_fingerprint": self.embedding_spec_fingerprint,
            "query_bank_version": self.query_bank_version,
            "query_bank_fingerprint": self.query_bank_fingerprint,
            "top_k_per_formulation": self.top_k_per_formulation,
            "selection_version": self.selection_version,
            "selection_config_fingerprint": self.selection_config_fingerprint,
            "assembly_version": self.assembly_version,
            "assembly_schema_version": self.assembly_schema_version,
            "artifact_schema_fingerprint": self.artifact_schema_fingerprint,
            "provider": self.provider,
            "ollama_endpoint": self.ollama_endpoint,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": self.max_output_tokens,
            "context_tokens": self.context_tokens,
            "atomic_cache_path": str(self.atomic_cache_path),
            "atomic_cache_schema_version": self.atomic_cache_schema_version,
            "presence_prompt_version": self.presence_prompt_version,
            "presence_schema_version": self.presence_schema_version,
            "extraction_prompt_version": self.extraction_prompt_version,
            "segmentation_version": self.segmentation_version,
            "stage_budgets_fingerprint": self.stage_budgets_fingerprint,
            "resume_schema_version": self.resume_schema_version,
            "source_paths": dict(self.source_paths),
            "source_fingerprints": dict(self.source_fingerprints),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Phase14CReplayPreflight:
    source_paths: Phase14CSourcePaths
    source_fingerprints: Mapping[str, str]
    verified_experience_count: int
    quarantined_experience_count: int
    causal_tick_count: int
    causal_segment_count: int
    commission_status: str
    ready: bool
    reason_code: Phase14CPreflightReason | str

    def __post_init__(self) -> None:
        if not isinstance(self.source_paths, Phase14CSourcePaths):
            raise TypeError("source_paths must be Phase14CSourcePaths")
        counts = (
            self.verified_experience_count,
            self.quarantined_experience_count,
            self.causal_tick_count,
            self.causal_segment_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("preflight counts must be non-negative integers")
        reason = Phase14CPreflightReason(self.reason_code)
        if type(self.ready) is not bool:
            raise TypeError("ready must be bool")
        if self.ready != (reason is Phase14CPreflightReason.READY):
            raise ValueError("preflight readiness must match its reason code")
        if self.ready and (
            self.verified_experience_count < 1
            or self.causal_tick_count < 1
            or self.causal_segment_count < 1
        ):
            raise ValueError("ready preflight requires verified experience and causal data")
        if self.commission_status != "UNKNOWN":
            raise ValueError("Phase 14C replay commission status must remain UNKNOWN")
        fingerprints = _phase14c_fingerprints(self.source_fingerprints)
        if self.ready and set(fingerprints) != {"phase14a", "hft", "demo"}:
            raise ValueError("ready preflight requires all source fingerprints")
        object.__setattr__(self, "source_fingerprints", fingerprints)
        object.__setattr__(self, "reason_code", reason)


@dataclass(frozen=True, slots=True)
class CandidateEvaluationRecord:
    spec_id: str
    candidate_id: str | None
    status: Phase14CCandidateStatus | str
    reason_code: Phase14CCandidateReason | Phase14CPreflightReason | str
    artifact_path: Path | None
    gate_decision: str | None = None
    gate_reasons: tuple[str, ...] = ()
    source_fingerprints: Mapping[str, str] = field(default_factory=dict)
    candidate_state: str | None = None

    def __post_init__(self) -> None:
        spec_id = str(self.spec_id).strip()
        if not spec_id:
            raise ValueError("spec_id must be non-empty")
        if self.candidate_id is not None and not str(self.candidate_id).strip():
            raise ValueError("candidate_id must be non-empty when present")
        status = Phase14CCandidateStatus(self.status)
        reason_value = (
            self.reason_code.value
            if isinstance(self.reason_code, Enum)
            else str(self.reason_code)
        )
        allowed_reasons = {item.value for item in Phase14CCandidateReason} | {
            item.value for item in Phase14CPreflightReason
        }
        if reason_value not in allowed_reasons:
            raise ValueError("candidate reason_code is not a closed Phase 14C code")
        if status is Phase14CCandidateStatus.COMPLETED and self.candidate_id is None:
            raise ValueError("completed candidate record requires candidate_id")
        if self.artifact_path is not None:
            object.__setattr__(self, "artifact_path", _phase14c_path(self.artifact_path, "artifact_path"))
        if self.gate_decision is not None:
            gate_decision = str(self.gate_decision).strip()
            if not gate_decision:
                raise ValueError("gate_decision must be non-empty when present")
            object.__setattr__(self, "gate_decision", gate_decision)
        gate_reasons = tuple(str(item).strip() for item in self.gate_reasons)
        if any(not item for item in gate_reasons):
            raise ValueError("gate_reasons must contain non-empty values")
        candidate_state = self.candidate_state
        if candidate_state is not None:
            candidate_state = str(candidate_state).strip()
            if candidate_state not in {
                CandidateState.INSUFFICIENT_EVIDENCE.value,
                CandidateState.REJECTED.value,
                CandidateState.SHADOW_CHALLENGER.value,
            }:
                raise ValueError("candidate_state exceeds the Phase 14C shadow ceiling")
        object.__setattr__(self, "spec_id", spec_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason_value)
        object.__setattr__(self, "gate_reasons", gate_reasons)
        object.__setattr__(self, "source_fingerprints", _phase14c_fingerprints(self.source_fingerprints))
        object.__setattr__(self, "candidate_state", candidate_state)
