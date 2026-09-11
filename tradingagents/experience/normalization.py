"""Deterministic, cohort-local robust normalization for numeric experience features."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, NamedTuple

from .features import FEATURE_NAMES_V1
from .models import TrustTier

SIMILARITY_PROFILE_VERSION = "similarity-profile.v1"
FALLBACK_POLICY_VERSION = "mad-then-feature-fallback.v1"
DEFAULT_EPSILON = 1e-9
DEFAULT_CLIPPING = (-8.0, 8.0)


class NormalizationCohortV1(NamedTuple):
    """The complete compatibility key for a normalization population."""

    resolved_symbol: str
    analysis_profile: str
    analysis_timeframe: str
    feature_schema_version: str
    feature_extractor_version: str


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """Small adapter contract accepted by :func:`build_profile`.

    Mapping rows are also accepted so catalog/importer projections can be used
    without coupling this module to their storage classes.
    """

    values: tuple[float, ...] | Mapping[str, float]
    mask: tuple[bool, ...] | Mapping[str, bool] | None = None
    feature_names: tuple[str, ...] = FEATURE_NAMES_V1
    cohort: NormalizationCohortV1 | tuple[str, ...] | None = None
    trust_tier: TrustTier | str = TrustTier.TIER_A_HIGH_TRUST
    analysis_snapshot_timestamp: datetime | None = None
    decision_completed_timestamp: datetime | None = None
    source_aliases: Mapping[str, Any] | None = None
    feature_fingerprint: str | None = "valid"
    accepted: bool = True
    provenance_valid: bool = True
    conflict: bool = False


@dataclass(frozen=True, slots=True)
class SimilarityProfileV1:
    medians: Mapping[str, float]
    iqr: Mapping[str, float]
    mad_scales: Mapping[str, float]
    scales: Mapping[str, float]
    fallback_scales: Mapping[str, float]
    epsilon: float
    weights: Mapping[str, float]
    clipping: tuple[float, float]
    feature_order: tuple[str, ...]
    mask_policy: str
    trust_tiers: tuple[TrustTier, ...]
    cohort: NormalizationCohortV1
    similarity_profile_version: str = SIMILARITY_PROFILE_VERSION
    fallback_policy_version: str = FALLBACK_POLICY_VERSION
    normalization_cutoff: datetime | None = None
    population_fingerprint: str = ""
    population_count: int = 0

    @property
    def version(self) -> str:
        return self.similarity_profile_version

    @property
    def cutoff(self) -> datetime | None:
        return self.normalization_cutoff

    @property
    def iqrs(self) -> Mapping[str, float]:
        return self.iqr

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, TrustTier):
                return value.value
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Mapping):
                return {str(k): convert(v) for k, v in sorted(value.items())}
            if isinstance(value, (tuple, list)):
                return [convert(v) for v in value]
            return value

        return {"medians": convert(self.medians), "iqr": convert(self.iqr), "mad_scales": convert(self.mad_scales),
                "scales": convert(self.scales), "fallback_scales": convert(self.fallback_scales), "epsilon": self.epsilon,
                "weights": convert(self.weights), "clipping": convert(self.clipping), "feature_order": convert(self.feature_order),
                "mask_policy": self.mask_policy, "trust_tiers": convert(self.trust_tiers), "cohort": convert(self.cohort),
                "similarity_profile_version": self.similarity_profile_version, "fallback_policy_version": self.fallback_policy_version,
                "normalization_cutoff": convert(self.normalization_cutoff), "population_fingerprint": self.population_fingerprint,
                "population_count": self.population_count}

    def to_fingerprint(self) -> str:
        return _digest(self.to_dict())


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        return None
    return value


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        value = row.get(name, default)
    else:
        value = getattr(row, name, default)
    if value is default and name in {"values", "mask", "feature_names", "cohort"}:
        vector = row.get("vector") if isinstance(row, Mapping) else getattr(row, "vector", None)
        if vector is not None:
            value = getattr(vector, name, default)
    return value


def _cohort(row: Any) -> NormalizationCohortV1 | None:
    value = _value(row, "cohort")
    if value is None:
        return None
    try:
        return value if isinstance(value, NormalizationCohortV1) else NormalizationCohortV1(*tuple(value))
    except (TypeError, ValueError):
        return None


def _eligible(row: Any, cohort: NormalizationCohortV1, tiers: tuple[TrustTier, ...], as_of: datetime | None) -> bool:
    if _cohort(row) != cohort or TrustTier(_value(row, "trust_tier", _value(row, "trust", TrustTier.TIER_C_DIAGNOSTIC_ONLY))) not in tiers:
        return False
    provenance = _value(row, "provenance", {}) or {}
    provenance_ok = _value(row, "provenance_valid", provenance.get("valid", True) if isinstance(provenance, Mapping) else True)
    conflict = _value(row, "conflict", _value(row, "source_conflict", False))
    if not _value(row, "accepted", True) or conflict or not provenance_ok:
        return False
    if not (_value(row, "feature_fingerprint", _value(row, "fingerprint", ""))):
        return False
    aliases = _value(row, "source_aliases", {}) or {}
    if as_of is None:
        return any(str(v) == "CURRENT" or getattr(v, "value", None) == "CURRENT" for v in aliases.values())
    analysis = _utc(_value(row, "analysis_snapshot_timestamp", _value(row, "snapshot_timestamp")))
    completed = _utc(_value(row, "decision_completed_timestamp"))
    return analysis is not None and completed is not None and analysis < as_of and completed < as_of


def _vectors(rows: Sequence[Any], names: tuple[str, ...]) -> tuple[list[list[float]], list[Any]]:
    vectors: list[list[float]] = []
    selected: list[Any] = []
    for row in rows:
        values = _value(row, "values", {})
        mask = _value(row, "mask", None)
        if isinstance(values, Mapping):
            row_values = [values.get(name, math.nan) for name in names]
            row_mask = [bool(mask.get(name, v is not None)) if isinstance(mask, Mapping) else v is not None for name, v in zip(names, row_values)]
        else:
            row_values = list(values)
            row_mask = list(mask) if mask is not None else [True] * len(row_values)
            if len(row_values) != len(names):
                continue
        vectors.append([float(v) if m and math.isfinite(float(v)) else math.nan for v, m in zip(row_values, row_mask)])
        selected.append(row)
    return vectors, selected


def build_profile(rows: Sequence[Any], cohort: NormalizationCohortV1, trust_tiers: Sequence[TrustTier | str] | None = None, as_of: datetime | None = None, *, epsilon: float = DEFAULT_EPSILON) -> SimilarityProfileV1:
    cohort = cohort if isinstance(cohort, NormalizationCohortV1) else NormalizationCohortV1(*tuple(cohort))
    tiers = tuple(TrustTier(v) for v in (trust_tiers or (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED)))
    if TrustTier.TIER_C_DIAGNOSTIC_ONLY in tiers:
        raise ValueError("Tier C is diagnostic-only and cannot define numeric normalization")
    cutoff = _utc(as_of)
    eligible = [r for r in rows if _eligible(r, cohort, tiers, cutoff)]
    names = tuple(_value(eligible[0], "feature_names", FEATURE_NAMES_V1)) if eligible else FEATURE_NAMES_V1
    vectors, selected = _vectors(eligible, names)
    medians: dict[str, float] = {}; iqrs: dict[str, float] = {}; mads: dict[str, float] = {}; fallbacks: dict[str, float] = {}; scales: dict[str, float] = {}
    for i, name in enumerate(names):
        vals = sorted(v[i] for v in vectors if math.isfinite(v[i]))
        median = float(__import__("numpy").median(vals)) if vals else 0.0
        q25, q75 = (__import__("numpy").percentile(vals, [25, 75]) if vals else (0.0, 0.0))
        iqr = float(q75 - q25)
        mad = float(1.4826 * __import__("numpy").median([abs(v - median) for v in vals])) if vals else 0.0
        fallback = 1.0
        medians[name], iqrs[name], fallbacks[name] = median, iqr, fallback
        # ``mad_scales`` records the persisted non-IQR scale selected by the
        # versioned MAD-then-feature-fallback policy.  This makes the effective
        # choice auditable even when a zero MAD requires the feature fallback.
        mad_effective = mad if mad > epsilon else fallback
        mads[name] = mad_effective
        scales[name] = iqr if iqr > epsilon else mad_effective
    weights = {name: (2.0 if name == "spread_points" else 0.5 if name.startswith("utc_hour_") else 1.0) for name in names}
    population = []
    for row in selected:
        population.append({"id": _value(row, "experience_id", _value(row, "row_id", None)), "fingerprint": _value(row, "feature_fingerprint", "valid"), "values": _value(row, "values", {}), "mask": _value(row, "mask", None)})
    population_fingerprint = _digest({"cohort": cohort, "tiers": tiers, "as_of": cutoff, "rows": sorted(population, key=lambda x: json.dumps(x, sort_keys=True, default=str))})
    return SimilarityProfileV1(medians, iqrs, mads, scales, fallbacks, epsilon, weights, DEFAULT_CLIPPING, names, "exclude-missing.v1", tiers, cohort, normalization_cutoff=cutoff, population_fingerprint=population_fingerprint, population_count=len(selected))


def query_normalization_fingerprint(profile: SimilarityProfileV1, cohort: NormalizationCohortV1, trust_tiers: Sequence[TrustTier | str] | None = None, as_of: datetime | None = None) -> str:
    tiers = tuple(TrustTier(v) for v in (trust_tiers or profile.trust_tiers))
    payload = {"cohort": cohort, "trust_tiers": tiers, "normalization_cutoff": _utc(as_of), "population_fingerprint": profile.population_fingerprint,
               "similarity_profile_version": profile.similarity_profile_version, "feature_order": profile.feature_order, "mask_policy": profile.mask_policy,
               "epsilon": profile.epsilon, "fallback_policy_version": profile.fallback_policy_version, "medians": profile.medians, "iqr": profile.iqr,
               "scales": profile.scales, "weights": profile.weights, "clipping": profile.clipping}
    return _digest(payload)


__all__ = ["FeatureRow", "NormalizationCohortV1", "SimilarityProfileV1", "build_profile", "query_normalization_fingerprint"]
