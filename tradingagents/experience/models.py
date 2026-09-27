"""Immutable public value objects for Phase 8."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
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
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(_json(k)): _json(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [
            _json(v)
            for v in sorted(
                value, key=lambda item: json.dumps(_json(item), sort_keys=True, default=str)
            )
        ]
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return _json(self)

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _utc(value: datetime | None, name: str = "timestamp") -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value


def _tiers(value: Sequence[TrustTier | str] | None) -> tuple[TrustTier, ...]:
    if value is None:
        return (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED)
    return tuple(TrustTier(v) for v in value)


def _freeze(value: Any) -> Any:
    """Deep-copy contract payloads into immutable deterministic containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(k): _freeze(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    return value


def _mapping_or_empty(value: Any, name: str) -> Mapping:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_reserved(value: Any) -> None:
    if isinstance(value, Mapping):
        if "training_eligible" in value or "training_eligibility_reason" in value:
            raise ValueError("training_eligible is reserved for provenance")
        for item in value.values():
            _reject_reserved(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _reject_reserved(item)


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
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.analysis_snapshot_timestamp, datetime):
            raise ValueError("analysis_snapshot_timestamp is required")
        object.__setattr__(
            self,
            "analysis_snapshot_timestamp",
            _utc(self.analysis_snapshot_timestamp, "analysis_snapshot_timestamp"),
        )
        object.__setattr__(
            self,
            "decision_completed_timestamp",
            _utc(self.decision_completed_timestamp, "decision_completed_timestamp"),
        )
        object.__setattr__(
            self,
            "decision_reference_timestamp",
            _utc(self.decision_reference_timestamp, "decision_reference_timestamp"),
        )
        object.__setattr__(self, "trust", TrustTier(self.trust))
        aliases = {} if self.source_aliases is None else self.source_aliases
        if not isinstance(aliases, Mapping):
            raise ValueError("source_aliases must be a mapping")
        if any(not isinstance(key, str) or not key.strip() for key in aliases):
            raise ValueError("source_aliases keys must be non-empty strings")
        object.__setattr__(
            self,
            "source_aliases",
            _freeze({str(k): SourceAliasState(v) for k, v in aliases.items()}),
        )
        for name in (
            "market_state",
            "decision_evidence",
            "outcome_evidence_by_basis_horizon",
            "provenance",
            "source_evaluation_fingerprints",
        ):
            raw_value = getattr(self, name)
            value = {} if raw_value is None else raw_value
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            if name != "provenance":
                _reject_reserved(value)
            if name == "source_evaluation_fingerprints" and any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(item, str)
                or not item.strip()
                for key, item in value.items()
            ):
                raise ValueError(
                    "source_evaluation_fingerprints keys and values must be non-empty strings"
                )
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
        if not isinstance(self.market_state, Mapping):
            raise ValueError("market_state must be a mapping")
        _reject_reserved(self.market_state)
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValueError("top_k must be a positive integer")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        object.__setattr__(self, "market_state", _freeze(self.market_state))
        object.__setattr__(self, "trust_tiers", _tiers(self.trust_tiers))
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))


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
        if not isinstance(self.experience_id, str) or not self.experience_id.strip():
            raise ValueError("experience_id must be a non-empty string")
        if not isinstance(self.currently_tombstoned, bool):
            raise ValueError("currently_tombstoned must be a boolean")
        if self.trust_tier is not None:
            object.__setattr__(self, "trust_tier", TrustTier(self.trust_tier))
        for n in ("market_state", "timestamps", "outcome_availability", "provenance"):
            raw_value = getattr(self, n)
            value = {} if raw_value is None else raw_value
            if not isinstance(value, Mapping):
                raise ValueError(f"{n} must be a mapping")
            if n != "provenance":
                _reject_reserved(value)
            if n == "timestamps":
                for timestamp in value.values():
                    _utc(timestamp, "timestamp")
            object.__setattr__(self, n, _freeze(value))


@dataclass(frozen=True, slots=True)
class ExperienceSearchResult(Serializable):
    hits: tuple[ExperienceHit, ...] = ()
    query_normalization_fingerprint: str | None = None
    active_generation_id: str | None = None
    candidate_count: int = 0
    excluded_counts: Mapping[str, int] = None

    def __post_init__(self) -> None:
        hits = tuple(self.hits)
        if any(not isinstance(hit, ExperienceHit) for hit in hits):
            raise TypeError("hits must contain ExperienceHit values")
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "excluded_counts", _freeze(_mapping_or_empty(self.excluded_counts, "excluded_counts")))


@dataclass(frozen=True, slots=True)
class OutcomeStatsRequest(Serializable):
    experience_ids: tuple[str, ...] = ()
    evaluation_basis: str = "ANALYSIS_SNAPSHOT"
    horizon_seconds: int = 0
    trust_tiers: tuple[TrustTier, ...] = (TrustTier.TIER_A_HIGH_TRUST,)
    as_of: datetime | None = None

    def __post_init__(self) -> None:
        if isinstance(self.experience_ids, (str, bytes)):
            raise ValueError("experience_ids must be a sequence of strings")
        try:
            experience_ids = tuple(self.experience_ids)
        except TypeError as exc:
            raise ValueError("experience_ids must be a sequence of strings") from exc
        if any(not isinstance(value, str) or not value.strip() for value in experience_ids):
            raise ValueError("experience_ids must contain non-empty strings")
        object.__setattr__(self, "experience_ids", experience_ids)
        object.__setattr__(self, "trust_tiers", _tiers(self.trust_tiers))
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        if isinstance(self.horizon_seconds, bool) or not isinstance(self.horizon_seconds, int):
            raise ValueError("horizon_seconds must be a non-negative integer")
        if self.horizon_seconds < 0:
            raise ValueError("horizon_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class OutcomeDirectionStatistics(Serializable):
    net_points: tuple[float, ...] = ()
    win_rate: float | None = None
    count: int = 0
    positive_net_count: int = 0
    positive_net_rate: float | None = None
    negative_net_count: int = 0
    negative_net_rate: float | None = None
    zero_net_count: int = 0
    zero_net_rate: float | None = None
    mean_net_points: float | None = None
    median_net_points: float | None = None
    mfe_points: tuple[float, ...] = ()
    mae_points: tuple[float, ...] = ()
    mfe_mean: float | None = None
    mfe_median: float | None = None
    mae_mean: float | None = None
    mae_median: float | None = None
    mfe_quantiles: Mapping[str, float] = None
    mae_quantiles: Mapping[str, float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "net_points", tuple(self.net_points))
        object.__setattr__(self, "mfe_points", tuple(self.mfe_points))
        object.__setattr__(self, "mae_points", tuple(self.mae_points))
        object.__setattr__(self, "mfe_quantiles", _freeze(_mapping_or_empty(self.mfe_quantiles, "mfe_quantiles")))
        object.__setattr__(self, "mae_quantiles", _freeze(_mapping_or_empty(self.mae_quantiles, "mae_quantiles")))

    @property
    def positive_count(self) -> int:
        return self.positive_net_count

    @property
    def negative_count(self) -> int:
        return self.negative_net_count


@dataclass(frozen=True, slots=True)
class OutcomeHoldStatistics(Serializable):
    opportunity_cost_points: tuple[float, ...] = ()
    normal_win_rate: float | None = None
    count: int = 0
    missed_buy_opportunity_points: tuple[float, ...] = ()
    missed_sell_opportunity_points: tuple[float, ...] = ()
    best_counterfactual_counts: Mapping[str, int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "opportunity_cost_points", tuple(self.opportunity_cost_points))
        object.__setattr__(
            self, "missed_buy_opportunity_points", tuple(self.missed_buy_opportunity_points)
        )
        object.__setattr__(
            self, "missed_sell_opportunity_points", tuple(self.missed_sell_opportunity_points)
        )
        object.__setattr__(
            self, "best_counterfactual_counts", _freeze(
                _mapping_or_empty(self.best_counterfactual_counts, "best_counterfactual_counts")
            )
        )

    @property
    def buy_missed_opportunity_points(self) -> tuple[float, ...]:
        return self.missed_buy_opportunity_points

    @property
    def sell_missed_opportunity_points(self) -> tuple[float, ...]:
        return self.missed_sell_opportunity_points


@dataclass(frozen=True, slots=True)
class OutcomeStatistics(Serializable):
    eligible_sample_denominator: int = 0
    evaluation_basis: str | None = None
    horizon_seconds: int | None = None
    excluded_counts: Mapping[str, int] = None
    source_evaluation_fingerprints: Mapping[str, str] = None
    eligible_count: int = 0
    requested_basis: str | None = None
    requested_horizon_seconds: int | None = None
    exclusions_by_status: Mapping[str, int] = None
    exclusions_by_tier: Mapping[str, int] = None
    buy: OutcomeDirectionStatistics = None
    sell: OutcomeDirectionStatistics = None
    hold: OutcomeHoldStatistics = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "excluded_counts", _freeze(_mapping_or_empty(self.excluded_counts, "excluded_counts")))
        object.__setattr__(
            self,
            "source_evaluation_fingerprints",
            _freeze(
                _mapping_or_empty(
                    self.source_evaluation_fingerprints, "source_evaluation_fingerprints"
                )
            ),
        )
        # Zero is a meaningful result when every candidate was excluded.  Do
        # not replace an explicit zero with the denominator via truthiness.
        object.__setattr__(self, "eligible_count", self.eligible_count)
        object.__setattr__(
            self,
            "requested_basis",
            self.requested_basis if self.requested_basis is not None else self.evaluation_basis,
        )
        object.__setattr__(
            self,
            "requested_horizon_seconds",
            self.requested_horizon_seconds
            if self.requested_horizon_seconds is not None
            else self.horizon_seconds,
        )
        object.__setattr__(
            self,
            "exclusions_by_status",
            _freeze(_mapping_or_empty(self.exclusions_by_status, "exclusions_by_status")),
        )
        object.__setattr__(
            self,
            "exclusions_by_tier",
            _freeze(_mapping_or_empty(self.exclusions_by_tier, "exclusions_by_tier")),
        )
        object.__setattr__(self, "buy", self.buy or OutcomeDirectionStatistics())
        object.__setattr__(self, "sell", self.sell or OutcomeDirectionStatistics())
        object.__setattr__(self, "hold", self.hold or OutcomeHoldStatistics())


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
        for name in ("knowledge_top_k", "experience_top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000:
                raise ValueError(f"{name} must be between 1 and 1000")
        if self.evaluation_basis is not None and (
            not isinstance(self.evaluation_basis, str) or not self.evaluation_basis.strip()
        ):
            raise ValueError("evaluation_basis must be a non-empty string")
        if (self.evaluation_basis is None) != (self.horizon_seconds is None):
            raise ValueError("evaluation_basis and horizon_seconds must be paired")
        if self.horizon_seconds is not None and (
            isinstance(self.horizon_seconds, bool)
            or not isinstance(self.horizon_seconds, int)
            or self.horizon_seconds < 0
        ):
            raise ValueError("horizon_seconds must be a non-negative integer")
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        object.__setattr__(self, "trust_tiers", tuple(TrustTier(v) for v in self.trust_tiers))
        if self.market_state is not None and not isinstance(self.market_state, Mapping):
            raise ValueError("market_state must be a mapping")
        for name in ("content_types", "document_ids"):
            raw = getattr(self, name)
            if raw is None:
                values = ()
            elif isinstance(raw, (str, bytes, bytearray, Mapping)):
                raise ValueError(f"{name} must be a sequence of strings")
            else:
                try:
                    values = tuple(raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be a sequence of strings") from exc
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} entries must be non-empty strings")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "market_state",
            _freeze(self.market_state) if self.market_state is not None else None,
        )


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
        object.__setattr__(self, "knowledge", tuple(self.knowledge))
        object.__setattr__(self, "experience", tuple(self.experience))
        object.__setattr__(
            self, "source_status", _freeze(_mapping_or_empty(self.source_status, "source_status"))
        )
        object.__setattr__(
            self, "provenance", _freeze(_mapping_or_empty(self.provenance, "provenance"))
        )


@dataclass(frozen=True, slots=True)
class EvidenceSourceError(Serializable):
    """Bounded, source-qualified failure returned by the evidence boundary."""

    source: str
    error_type: str
    message: str
    fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceWarning(Serializable):
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class OrchestrationProvenance(Serializable):
    """Small immutable provenance envelope; source details stay on each hit."""

    query_normalization_fingerprint: str | None = None
    knowledge_requested: bool = False
    experience_requested: bool = False
