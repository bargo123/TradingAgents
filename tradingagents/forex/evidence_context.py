"""Immutable Phase 9 evidence-context contracts and closed vocabularies."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

from tradingagents.experience.models import TrustTier


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


__all__ = [name for name in globals() if name.startswith("Evidence") or name.startswith("Canonical")]
