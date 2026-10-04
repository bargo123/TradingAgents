"""Immutable value objects shared by offline Phase 14C discovery stages."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from tradingagents.knowledge.models import ContentType, KnowledgeHit, SerializableModel
from tradingagents.self_enhancement.book_atomic_extraction import EvidenceGroup


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
                other = find(document_id)
                if root != other:
                    parents[max(root, other)] = min(root, other)
                    root = min(root, other)
        return len({find(document_id) for document_id in documents})


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
