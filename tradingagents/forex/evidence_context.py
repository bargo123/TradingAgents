"""Immutable Phase 9 evidence-context contracts and closed vocabularies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

from tradingagents.experience.models import EvidenceBundle, TrustTier


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class EvidenceIntegrationStatus(_ValueEnum):
    DISABLED = "DISABLED"
    INJECTED = "INJECTED"
    FALLBACK = "FALLBACK"


class EvidenceBundleStatus(_ValueEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    EMPTY = "EMPTY"
    FAILED = "FAILED"


class EvidenceUseStatus(_ValueEnum):
    USED = "USED"
    NONE_RELEVANT = "NONE_RELEVANT"
    UNAVAILABLE = "UNAVAILABLE"
    DISABLED = "DISABLED"


class EvidenceAuditStatus(_ValueEnum):
    VALID = "VALID"
    INVALID_REFERENCE = "INVALID_REFERENCE"
    NOT_RECORDED = "NOT_RECORDED"
    WRITE_FAILED = "WRITE_FAILED"


class EvidenceReferenceRejectionReason(_ValueEnum):
    CONFLICTS_WITH_CURRENT_STATE = "CONFLICTS_WITH_CURRENT_STATE"
    LOW_RELEVANCE = "LOW_RELEVANCE"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    REDUNDANT = "REDUNDANT"


class EvidenceSourceKind(_ValueEnum):
    KNOWLEDGE = "KNOWLEDGE"
    EXPERIENCE = "EXPERIENCE"
    STATISTICS = "STATISTICS"


def _freeze(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)
    if is_dataclass(value) and not isinstance(value, type):
        return MappingProxyType({field.name: _freeze(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    if isinstance(value, Collection) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(v) for v in value)
    return value


def _json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        values = [_json(v) for v in value]
        if isinstance(value, (set, frozenset)):
            values.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return values
    return value


def _utc(value: datetime, name: str = "as_of") -> datetime:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value.astimezone(timezone.utc)


def _freeze_items(values: Any) -> tuple[Any, ...]:
    """Freeze arbitrary item entries while preserving canonical contract items."""
    return tuple(value if isinstance(value, CanonicalEvidenceItem) else _freeze(value) for value in values)


@dataclass(frozen=True, slots=True)
class CanonicalKnowledgeQuery:
    text: str = ""
    fingerprint: str = ""
    policy_version: str = ""


@dataclass(frozen=True, slots=True)
class EvidenceSnapshotAdapter:
    def to_market_state(
        self, snapshot: Any, *, resolved_symbol: str, analysis_profile: str, analysis_timeframe: str
    ) -> Mapping[str, Any]:
        raise NotImplementedError("snapshot adapters must implement to_market_state")


@dataclass(frozen=True, slots=True)
class EvidenceQueryPolicy:
    query_policy_version: str = "v1"
    budget_policy_version: str = "v1"
    knowledge_top_k: int = 10
    experience_top_k: int = 50
    max_knowledge_items: int = 4
    max_experience_items: int = 4
    max_statistics_items: int = 2
    max_rendered_characters: int = 6000
    trust_tiers: tuple[TrustTier, ...] = (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED)
    evaluation_basis: str = "ANALYSIS_SNAPSHOT"
    statistics_horizon_seconds: int | None = None
    evidence_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name in ("knowledge_top_k", "experience_top_k", "max_knowledge_items", "max_experience_items", "max_statistics_items", "max_rendered_characters"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.evidence_timeout_seconds < 0:
            raise ValueError("evidence_timeout_seconds must be non-negative")
        object.__setattr__(self, "trust_tiers", tuple(TrustTier(tier) for tier in self.trust_tiers))
        if any(tier not in (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED) for tier in self.trust_tiers):
            raise ValueError("trust_tiers contains an unapproved tier")
        if self.statistics_horizon_seconds is not None and (self.statistics_horizon_seconds <= 0 or not self.evaluation_basis):
            raise ValueError("statistics horizon requires a valid evaluation_basis")


@dataclass(frozen=True, slots=True)
class CanonicalEvidenceItem:
    display_id: str
    source_kind: EvidenceSourceKind | str
    authoritative_id: str
    text: str
    content_type: str
    score: float
    provenance: Mapping[str, Any]
    metadata: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_kind", EvidenceSourceKind(self.source_kind))
        if not self.authoritative_id:
            raise ValueError("authoritative_id must be non-empty")
        object.__setattr__(self, "provenance", _freeze(self.provenance or {}))
        object.__setattr__(self, "metadata", _freeze(self.metadata or {}))


@dataclass(frozen=True, slots=True)
class EvidenceReferenceRejection:
    ref: str
    reason: EvidenceReferenceRejectionReason | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", EvidenceReferenceRejectionReason(self.reason))


@dataclass(frozen=True, slots=True)
class EvidenceReferenceValidation:
    evidence_use_status: EvidenceUseStatus | str = EvidenceUseStatus.NONE_RELEVANT
    evidence_refs_used: tuple[str, ...] = ()
    evidence_refs_rejected: tuple[EvidenceReferenceRejection, ...] = ()
    evidence_audit_status: EvidenceAuditStatus | str = EvidenceAuditStatus.NOT_RECORDED

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_use_status", EvidenceUseStatus(self.evidence_use_status))
        object.__setattr__(self, "evidence_audit_status", EvidenceAuditStatus(self.evidence_audit_status))
        object.__setattr__(self, "evidence_refs_used", tuple(self.evidence_refs_used))
        object.__setattr__(self, "evidence_refs_rejected", tuple(self.evidence_refs_rejected))


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    version: str = "v1"
    integration_status: EvidenceIntegrationStatus | str = EvidenceIntegrationStatus.DISABLED
    bundle_status: EvidenceBundleStatus | str = EvidenceBundleStatus.EMPTY
    as_of: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
    knowledge_generation_id: str | None = None
    experience_generation_id: str | None = None
    query_normalization_fingerprint: str | None = None
    knowledge_query: CanonicalKnowledgeQuery | None = None
    knowledge_query_fingerprint: str | None = None
    knowledge_query_policy_version: str | None = None
    knowledge_items: tuple[CanonicalEvidenceItem, ...] = ()
    experience_items: tuple[CanonicalEvidenceItem, ...] = ()
    statistics_items: tuple[CanonicalEvidenceItem, ...] = ()
    statistics_status: str = "NOT_REQUESTED"
    diagnostics: Mapping[str, Any] = None
    source_errors: Mapping[str, Any] = None
    rendered_context: str = ""
    rendered_context_hash: str | None = None
    selected_knowledge_count: int = 0
    selected_experience_count: int = 0
    selected_statistics_count: int = 0
    dropped_knowledge_count: int = 0
    dropped_experience_count: int = 0
    dropped_statistics_count: int = 0
    rendered_character_count: int = 0
    budget_policy_version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "integration_status", EvidenceIntegrationStatus(self.integration_status))
        object.__setattr__(self, "bundle_status", EvidenceBundleStatus(self.bundle_status))
        object.__setattr__(self, "as_of", _utc(self.as_of))
        for name in ("knowledge_items", "experience_items", "statistics_items"):
            object.__setattr__(self, name, _freeze_items(getattr(self, name)))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics or {}))
        object.__setattr__(self, "source_errors", _freeze(self.source_errors or {}))

    def to_dict(self) -> dict[str, Any]:
        return _json(self)

    def canonical_payload_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class Phase9EvidenceContextBuilder:
    """Build one bounded, deterministic context from an already ordered bundle."""

    @staticmethod
    def _provenance(item: Any, kind: EvidenceSourceKind, authoritative_id: str) -> dict[str, Any]:
        value = dict(_field(item, "provenance", {}) or {})
        value.setdefault("source_kind", kind.value)
        value.setdefault("authoritative_id", authoritative_id)
        if kind is EvidenceSourceKind.KNOWLEDGE:
            for name in ("document_id", "chunk_id"):
                candidate = _field(item, name)
                if candidate:
                    value.setdefault(name, str(candidate))
        elif kind is EvidenceSourceKind.EXPERIENCE:
            for name in ("source_database_id", "source_decision_id", "experience_id"):
                candidate = _field(item, name)
                if candidate:
                    value.setdefault(name, str(candidate))
        return value

    @classmethod
    def _item(cls, value: Any, kind: EvidenceSourceKind, display_id: str) -> CanonicalEvidenceItem:
        existing_provenance = dict(_field(value, "provenance", {}) or {})
        if kind is EvidenceSourceKind.KNOWLEDGE:
            authoritative_value = (
                _field(value, "chunk_id")
                or _field(value, "document_id")
                or existing_provenance.get("chunk_id")
                or existing_provenance.get("document_id")
            )
        elif kind is EvidenceSourceKind.EXPERIENCE:
            authoritative_value = _field(value, "experience_id") or existing_provenance.get("experience_id")
        else:
            authoritative_value = (
                _field(value, "authoritative_id")
                or _field(value, "statistics_id")
                or _field(value, "experience_id")
                or existing_provenance.get("authoritative_id")
                or existing_provenance.get("statistics_id")
                or existing_provenance.get("experience_id")
            )
        if not authoritative_value:
            raise ValueError(f"{kind.value} item is missing an authoritative identifier")
        authoritative_id = str(authoritative_value)
        score = _field(value, "score", _field(value, "similarity_score", 0.0))
        metadata = _field(value, "metadata", _field(value, "extra", {})) or {}
        return CanonicalEvidenceItem(
            display_id=display_id,
            source_kind=kind,
            authoritative_id=authoritative_id,
            text=str(_field(value, "text", "") or ""),
            content_type=str(_field(value, "content_type", "") or ""),
            score=float(score or 0.0),
            provenance=cls._provenance(value, kind, authoritative_id),
            metadata=metadata,
        )

    @staticmethod
    def _statistics_items(value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (tuple, list)):
            return tuple(value)
        return (value,)

    @staticmethod
    def _rendered_item(item: CanonicalEvidenceItem) -> dict[str, Any]:
        return {
            "authoritative_id": item.authoritative_id,
            "content_type": item.content_type,
            "display_id": item.display_id,
            "metadata": _json(item.metadata),
            "provenance": _json(item.provenance),
            "score": item.score,
            "source_kind": item.source_kind.value,
            "text": item.text,
        }

    def build(
        self,
        bundle: EvidenceBundle,
        *,
        policy: EvidenceQueryPolicy,
        as_of: datetime,
        knowledge_query: CanonicalKnowledgeQuery | None = None,
        phase7_generation_id: str | None = None,
        phase8_generation_id: str | None = None,
    ) -> EvidenceContext:
        if not isinstance(bundle, EvidenceBundle):
            raise TypeError("bundle must be an EvidenceBundle")
        raw = {
            EvidenceSourceKind.KNOWLEDGE: tuple(bundle.knowledge),
            EvidenceSourceKind.EXPERIENCE: tuple(bundle.experience),
            EvidenceSourceKind.STATISTICS: self._statistics_items(bundle.statistics),
        }
        caps = {
            EvidenceSourceKind.KNOWLEDGE: policy.max_knowledge_items,
            EvidenceSourceKind.EXPERIENCE: policy.max_experience_items,
            EvidenceSourceKind.STATISTICS: policy.max_statistics_items,
        }
        capped: dict[EvidenceSourceKind, list[CanonicalEvidenceItem]] = {}
        dropped = {kind: max(0, len(values) - caps[kind]) for kind, values in raw.items()}
        prefixes = {
            EvidenceSourceKind.KNOWLEDGE: "K",
            EvidenceSourceKind.EXPERIENCE: "E",
            EvidenceSourceKind.STATISTICS: "S",
        }
        for kind in (EvidenceSourceKind.KNOWLEDGE, EvidenceSourceKind.EXPERIENCE, EvidenceSourceKind.STATISTICS):
            capped[kind] = [self._item(v, kind, f"{prefixes[kind]}{i}") for i, v in enumerate(raw[kind][: caps[kind]], 1)]

        selected: dict[EvidenceSourceKind, list[CanonicalEvidenceItem]] = {kind: [] for kind in capped}
        rendered: list[dict[str, Any]] = []
        as_of_payload = _json(_utc(as_of))
        for kind in (EvidenceSourceKind.KNOWLEDGE, EvidenceSourceKind.EXPERIENCE, EvidenceSourceKind.STATISTICS):
            for item in capped[kind]:
                candidate = rendered + [self._rendered_item(item)]
                payload = json.dumps(
                    {
                        "as_of": as_of_payload,
                        "budget_policy_version": policy.budget_policy_version,
                        "query_policy_version": policy.query_policy_version,
                        "items": candidate,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if len(payload) > policy.max_rendered_characters:
                    dropped[kind] += 1
                    continue
                rendered.append(self._rendered_item(item))
                selected[kind].append(item)
        rendered_payload = {
            "as_of": _json(_utc(as_of)),
            "budget_policy_version": policy.budget_policy_version,
            "query_policy_version": policy.query_policy_version,
            "items": rendered,
        }
        rendered_context = json.dumps(rendered_payload, sort_keys=True, separators=(",", ":"))
        rendered_hash = hashlib.sha256(rendered_context.encode("utf-8")).hexdigest()
        source_errors = {
            str(_field(error, "source", i)): _json(error) for i, error in enumerate(bundle.errors)
        }
        status = EvidenceBundleStatus(str(bundle.status))
        return EvidenceContext(
            integration_status=EvidenceIntegrationStatus.INJECTED,
            bundle_status=status,
            as_of=as_of,
            knowledge_generation_id=phase7_generation_id,
            experience_generation_id=phase8_generation_id,
            query_normalization_fingerprint=(bundle.provenance or {}).get("query_normalization_fingerprint"),
            knowledge_query=knowledge_query,
            knowledge_query_fingerprint=knowledge_query.fingerprint if knowledge_query else None,
            knowledge_query_policy_version=knowledge_query.policy_version if knowledge_query else None,
            knowledge_items=tuple(selected[EvidenceSourceKind.KNOWLEDGE]),
            experience_items=tuple(selected[EvidenceSourceKind.EXPERIENCE]),
            statistics_items=tuple(selected[EvidenceSourceKind.STATISTICS]),
            statistics_status="COMPLETE" if raw[EvidenceSourceKind.STATISTICS] else "NOT_REQUESTED",
            diagnostics={"source_status": _json(bundle.source_status), "warnings": _json(bundle.warnings)},
            source_errors=source_errors,
            rendered_context=rendered_context,
            rendered_context_hash=rendered_hash,
            selected_knowledge_count=len(selected[EvidenceSourceKind.KNOWLEDGE]),
            selected_experience_count=len(selected[EvidenceSourceKind.EXPERIENCE]),
            selected_statistics_count=len(selected[EvidenceSourceKind.STATISTICS]),
            dropped_knowledge_count=dropped[EvidenceSourceKind.KNOWLEDGE],
            dropped_experience_count=dropped[EvidenceSourceKind.EXPERIENCE],
            dropped_statistics_count=dropped[EvidenceSourceKind.STATISTICS],
            rendered_character_count=len(rendered_context),
            budget_policy_version=policy.budget_policy_version,
        )


__all__ = [
    name
    for name in globals()
    if name.startswith("Evidence") or name.startswith("Canonical") or name == "Phase9EvidenceContextBuilder"
]
