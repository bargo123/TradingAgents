"""Immutable public value objects for Phase 8."""
from __future__ import annotations

import json
from types import MappingProxyType
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TrustTier(str, Enum):
    TIER_A_HIGH_TRUST = "TIER_A_HIGH_TRUST"
    TIER_B_LIMITED = "TIER_B_LIMITED"
    TIER_C_DIAGNOSTIC_ONLY = "TIER_C_DIAGNOSTIC_ONLY"


class SourceAliasState(str, Enum):
    CURRENT = "CURRENT"
    RETAINED_PREVIOUS = "RETAINED_PREVIOUS"
    REMOVED = "REMOVED"


class EvaluationStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    INELIGIBLE = "INELIGIBLE"


class ImportState(str, Enum):
    ACCEPTED = "ACCEPTED"
    QUARANTINED = "QUARANTINED"
    SOURCE_DECISION_CONFLICT = "SOURCE_DECISION_CONFLICT"
    SOURCE_MISSING = "SOURCE_MISSING"
    REMOVED = "REMOVED"


def _json(value: Any) -> Any:
    if isinstance(value, Enum): return value.value
    if isinstance(value, datetime): return value.isoformat()
    if is_dataclass(value): return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping): return {str(_json(k)): _json(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json(v) for v in sorted(value, key=lambda item: json.dumps(_json(item), sort_keys=True, default=str))]
    if isinstance(value, (tuple, list)): return [_json(v) for v in value]
    return value


class Serializable:
    def to_dict(self) -> dict[str, Any]: return _json(self)
    def as_dict(self) -> dict[str, Any]: return self.to_dict()
    def to_json(self) -> str: return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _utc(value: datetime | None, name: str = "timestamp") -> datetime | None:
    if value is None: return None
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value


def _tiers(value: Sequence[TrustTier | str] | None) -> tuple[TrustTier, ...]:
    if value is None: return (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED)
    return tuple(TrustTier(v) for v in value)


def _freeze(value: Any) -> Any:
    """Deep-copy contract payloads into immutable deterministic containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    return value


def _reject_reserved(value: Any) -> None:
    if isinstance(value, Mapping):
        if "training_eligible" in value or "training_eligibility_reason" in value:
            raise ValueError("training_eligible is reserved for provenance")
        for item in value.values(): _reject_reserved(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value: _reject_reserved(item)


@dataclass(frozen=True, slots=True)
class ExperienceRecord(Serializable):
    experience_id: str
    source_database_id: str
    source_decision_id: str
    symbol: str
    analysis_snapshot_timestamp: datetime
    source_aliases: Mapping[str, SourceAliasState | str] = None
    source_run_id: str | None = None
    requested_symbol: str | None = None
    analysis_profile: str | None = None
    analysis_timeframe: str | None = None
    decision_completed_timestamp: datetime | None = None
    decision_reference_timestamp: datetime | None = None
    market_state: Mapping[str, Any] = None
    decision_evidence: Mapping[str, Any] = None
    outcome_evidence_by_basis_horizon: Mapping[str, Any] = None
    trust: TrustTier | str = TrustTier.TIER_A_HIGH_TRUST
    provenance: Mapping[str, Any] = None
    experience_schema_version: str = "phase8.experience.v1"
    feature_schema_version: str = "experience-features.v1"
    feature_extractor_version: str = "phase8-feature-extractor.v1"
    similarity_profile_version: str = "similarity-profile.v1"
    trust_policy_version: str = "trust-policy.v1"
    statistics_policy_version: str = "statistics-policy.v1"
    source_decision_fingerprint: str | None = None
    source_evaluation_fingerprints: Mapping[str, str] = None

    def __post_init__(self) -> None:
        for name in ("experience_id", "source_database_id", "source_decision_id", "symbol"):
            if not str(getattr(self, name)).strip(): raise ValueError(f"{name} must be non-empty")
        object.__setattr__(self, "analysis_snapshot_timestamp", _utc(self.analysis_snapshot_timestamp, "analysis_snapshot_timestamp"))
        object.__setattr__(self, "decision_completed_timestamp", _utc(self.decision_completed_timestamp, "decision_completed_timestamp"))
        object.__setattr__(self, "decision_reference_timestamp", _utc(self.decision_reference_timestamp, "decision_reference_timestamp"))
        object.__setattr__(self, "trust", TrustTier(self.trust))
        object.__setattr__(self, "source_aliases", _freeze({str(k): SourceAliasState(v) for k, v in (self.source_aliases or {}).items()}))
        for name in ("market_state", "decision_evidence", "outcome_evidence_by_basis_horizon", "provenance", "source_evaluation_fingerprints"):
            value = getattr(self, name) or {}
            if name != "provenance": _reject_reserved(value)
            object.__setattr__(self, name, _freeze(value))


@dataclass(frozen=True, slots=True)
class ExperienceQuery(Serializable):
    market_state: Mapping[str, Any]
    top_k: int = 50
    symbol: str | None = None
    analysis_profile: str | None = None
    analysis_timeframe: str | None = None
    as_of: datetime | None = None
    trust_tiers: tuple[TrustTier, ...] = (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED)
    action_filter: str | None = None
    def __post_init__(self) -> None:
        if not isinstance(self.market_state, Mapping): raise ValueError("market_state must be a mapping")
        _reject_reserved(self.market_state)
        if self.top_k < 1: raise ValueError("top_k must be positive")
        object.__setattr__(self, "market_state", _freeze(self.market_state))
        object.__setattr__(self, "trust_tiers", _tiers(self.trust_tiers)); object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))


@dataclass(frozen=True, slots=True)
class ExperienceHit(Serializable):
    experience_id: str
    source_decision_id: str | None = None
    similarity_score: float | None = None
    distance: float | None = None
    comparable_feature_count: int = 0
    trust_tier: TrustTier | str | None = None
    market_state: Mapping[str, Any] = None
    action: str | None = None
    timestamps: Mapping[str, datetime] = None
    outcome_availability: Mapping[str, Any] = None
    provenance: Mapping[str, Any] = None
    feature_schema_version: str | None = None
    similarity_profile_version: str | None = None
    currently_tombstoned: bool = False
    def __post_init__(self) -> None:
        if not str(self.experience_id).strip(): raise ValueError("experience_id must be non-empty")
        if self.trust_tier is not None: object.__setattr__(self, "trust_tier", TrustTier(self.trust_tier))
        for n in ("market_state", "timestamps", "outcome_availability", "provenance"):
            value = getattr(self, n) or {}
            if n != "provenance": _reject_reserved(value)
            if n == "timestamps":
                for timestamp in value.values(): _utc(timestamp, "timestamp")
            object.__setattr__(self, n, _freeze(value))


@dataclass(frozen=True, slots=True)
class ExperienceSearchResult(Serializable):
    hits: tuple[ExperienceHit, ...] = ()
    query_normalization_fingerprint: str | None = None
    active_generation_id: str | None = None
    candidate_count: int = 0
    excluded_counts: Mapping[str, int] = None
    def __post_init__(self) -> None:
        object.__setattr__(self, "hits", tuple(self.hits)); object.__setattr__(self, "excluded_counts", _freeze(self.excluded_counts or {}))


@dataclass(frozen=True, slots=True)
class OutcomeStatsRequest(Serializable):
    experience_ids: tuple[str, ...] = ()
    evaluation_basis: str = "ANALYSIS_SNAPSHOT"
    horizon_seconds: int = 0
    trust_tiers: tuple[TrustTier, ...] = (TrustTier.TIER_A_HIGH_TRUST,)
    as_of: datetime | None = None
    def __post_init__(self) -> None:
        object.__setattr__(self, "experience_ids", tuple(self.experience_ids)); object.__setattr__(self, "trust_tiers", _tiers(self.trust_tiers)); object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        if self.horizon_seconds < 0: raise ValueError("horizon_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class OutcomeStatistics(Serializable):
    eligible_sample_denominator: int = 0
    evaluation_basis: str | None = None
    horizon_seconds: int | None = None
    excluded_counts: Mapping[str, int] = None
    source_evaluation_fingerprints: Mapping[str, str] = None
    def __post_init__(self) -> None:
        object.__setattr__(self, "excluded_counts", _freeze(self.excluded_counts or {})); object.__setattr__(self, "source_evaluation_fingerprints", _freeze(self.source_evaluation_fingerprints or {}))


@dataclass(frozen=True, slots=True)
class EvidenceRequest(Serializable):
    research_question: str | None = None
    market_state: Mapping[str, Any] | None = None
    knowledge_top_k: int = 10
    experience_top_k: int = 50
    content_types: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    symbol: str | None = None
    analysis_profile: str | None = None
    analysis_timeframe: str | None = None
    evaluation_basis: str | None = None
    horizon_seconds: int | None = None
    as_of: datetime | None = None
    trust_tiers: tuple[TrustTier, ...] = ()
    action_filter: str | None = None
    def __post_init__(self) -> None:
        if (self.evaluation_basis is None) != (self.horizon_seconds is None): raise ValueError("evaluation_basis and horizon_seconds must be paired")
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of")); object.__setattr__(self, "trust_tiers", tuple(TrustTier(v) for v in self.trust_tiers)); object.__setattr__(self, "market_state", _freeze(self.market_state) if self.market_state is not None else None)


@dataclass(frozen=True, slots=True)
class EvidenceBundle(Serializable):
    status: str = "EMPTY"
    knowledge: tuple[Any, ...] = ()
    experience: tuple[ExperienceHit, ...] = ()
    statistics: OutcomeStatistics | None = None
    source_status: Mapping[str, Any] = None
    warnings: tuple[Any, ...] = ()
    errors: tuple[Any, ...] = ()
    provenance: Mapping[str, Any] = None
    def __post_init__(self) -> None:
        object.__setattr__(self, "knowledge", tuple(self.knowledge)); object.__setattr__(self, "experience", tuple(self.experience)); object.__setattr__(self, "source_status", _freeze(self.source_status or {})); object.__setattr__(self, "provenance", _freeze(self.provenance or {}))
