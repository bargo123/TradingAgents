"""Read-only compatibility gates and query envelope for experience search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .models import ExperienceHit, ExperienceQuery, ExperienceSearchResult, TrustTier
from .normalization import NormalizationCohortV1, query_normalization_fingerprint
from .similarity import ExactSimilarityIndex


def _get(row: Any, name: str, default: Any = None) -> Any:
    return row.get(name, default) if isinstance(row, Mapping) else getattr(row, name, default)


def _utc(v):
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (TypeError, ValueError, OverflowError):
            return None
    if (
        not isinstance(v, datetime)
        or v.tzinfo is None
        or v.utcoffset() != timezone.utc.utcoffset(v)
    ):
        return None
    return v


def _action(row):
    direct = _get(row, "action")
    if direct is not None:
        return direct
    evidence = _get(row, "decision_evidence", {})
    if isinstance(evidence, Mapping):
        return evidence.get("action", evidence.get("chosen_action"))
    return None


class ExperienceQueryService:
    def __init__(
        self,
        records: Sequence[Any] | Any,
        index: ExactSimilarityIndex | None = None,
        profile=None,
        profiles=None,
        feature_vectors: Mapping[str, Any] | None = None,
        generation_id: str | None = None,
        active_generation_id: str | None = None,
    ):
        if hasattr(records, "active_records"):
            self.catalog = records
            records = tuple(records.active_records()) + tuple(records.historical_records())
        else:
            self.catalog = None
        self.records = tuple(records)
        self.profile = profile
        self.profiles = profiles or {}
        self.generation_id = generation_id or active_generation_id
        vectors = feature_vectors or {str(_get(r, "experience_id")): r for r in self.records}
        self.index = index or ExactSimilarityIndex(
            vectors, {str(_get(r, "experience_id")): r for r in self.records}
        )

    def _profile(self, record, query):
        if self.profile is not None:
            return self.profile
        cohort = self._cohort(record)
        return self.profiles.get(cohort) or self.profiles.get(tuple(cohort))

    @staticmethod
    def _cohort(row):
        c = _get(row, "cohort")
        if c is not None:
            return c if isinstance(c, NormalizationCohortV1) else NormalizationCohortV1(*tuple(c))
        return NormalizationCohortV1(
            str(_get(row, "symbol")),
            str(_get(row, "analysis_profile")),
            str(_get(row, "analysis_timeframe")),
            str(_get(row, "feature_schema_version", "")),
            str(_get(row, "feature_extractor_version", "")),
        )

    def search(self, query: ExperienceQuery) -> ExperienceSearchResult:
        if TrustTier.TIER_C_DIAGNOSTIC_ONLY in query.trust_tiers:
            raise ValueError("Tier C is diagnostic-only and cannot be used for numeric similarity")
        rows = self.records
        exclusions: dict[str, int] = {}
        eligible = []
        for row in rows:
            reason = None
            aliases = _get(row, "source_aliases", {}) or {}
            tombstoned = bool(_get(row, "tombstoned", False)) or (
                bool(aliases)
                and not any(
                    str(v) == "CURRENT" or getattr(v, "value", None) == "CURRENT"
                    for v in aliases.values()
                )
            )
            if query.symbol and _get(row, "symbol", _get(row, "resolved_symbol")) != query.symbol:
                reason = "symbol"
            elif query.analysis_profile and _get(row, "analysis_profile") != query.analysis_profile:
                reason = "profile"
            elif (
                query.analysis_timeframe
                and _get(row, "analysis_timeframe") != query.analysis_timeframe
            ):
                reason = "timeframe"
            elif (
                TrustTier(
                    _get(row, "trust_tier", _get(row, "trust", TrustTier.TIER_C_DIAGNOSTIC_ONLY))
                )
                not in query.trust_tiers
            ):
                reason = "trust_tier"
            elif (
                not _get(row, "accepted", True)
                or _get(row, "conflict", False)
                or not _get(row, "provenance_valid", True)
            ):
                reason = "provenance"
            elif query.action_filter is not None and _action(row) != query.action_filter:
                reason = "action"
            elif tombstoned and query.as_of is None:
                reason = "tombstone"
            elif query.as_of is not None and (
                _utc(_get(row, "analysis_snapshot_timestamp")) is None
                or _utc(_get(row, "decision_completed_timestamp")) is None
                or _utc(_get(row, "analysis_snapshot_timestamp")) >= query.as_of
                or _utc(_get(row, "decision_completed_timestamp")) >= query.as_of
            ):
                reason = "as_of"
            if reason:
                exclusions[reason] = exclusions.get(reason, 0) + 1
            else:
                eligible.append(row)
        if not eligible:
            return ExperienceSearchResult(
                candidate_count=len(rows),
                excluded_counts=exclusions,
                active_generation_id=self.generation_id,
            )
        profile = self._profile(eligible[0], query)
        if profile is None:
            return ExperienceSearchResult(
                candidate_count=len(rows),
                excluded_counts=exclusions,
                active_generation_id=self.generation_id,
            )
        for row in tuple(eligible):
            cohort = self._cohort(row)
            if cohort != profile.cohort:
                eligible.remove(row)
                exclusions["cohort"] = exclusions.get("cohort", 0) + 1
            elif (
                _get(row, "feature_schema_version", profile.cohort.feature_schema_version)
                != profile.cohort.feature_schema_version
            ):
                eligible.remove(row)
                exclusions["feature_schema"] = exclusions.get("feature_schema", 0) + 1
            elif (
                _get(row, "feature_extractor_version", profile.cohort.feature_extractor_version)
                != profile.cohort.feature_extractor_version
            ):
                eligible.remove(row)
                exclusions["feature_extractor"] = exclusions.get("feature_extractor", 0) + 1
        if not eligible:
            return ExperienceSearchResult(
                candidate_count=len(rows),
                excluded_counts=exclusions,
                active_generation_id=self.generation_id,
            )
        # A persisted profile is reusable only when its population policy is
        # exactly this query's policy. Otherwise rebuild a deterministic,
        # query-local profile from the already-gated projections.
        cohort = self._cohort(eligible[0])
        if profile.trust_tiers != query.trust_tiers or profile.normalization_cutoff != query.as_of:
            from .normalization import build_profile

            profile = build_profile(eligible, cohort, query.trust_tiers, query.as_of)
        state = query.market_state
        state_cohort = state.get("cohort") if isinstance(state, Mapping) else None
        if state_cohort is not None:
            invalid_state_cohort = False
            try:
                state_cohort = (
                    state_cohort
                    if isinstance(state_cohort, NormalizationCohortV1)
                    else NormalizationCohortV1(*tuple(state_cohort))
                )
            except (TypeError, ValueError):
                invalid_state_cohort = True
            if invalid_state_cohort:
                exclusions["query_cohort_invalid"] = 1
                return ExperienceSearchResult(
                    candidate_count=len(rows),
                    excluded_counts=exclusions,
                    active_generation_id=self.generation_id,
                )
            if state_cohort != profile.cohort:
                exclusions["query_cohort"] = 1
                return ExperienceSearchResult(
                    candidate_count=len(rows),
                    excluded_counts=exclusions,
                    active_generation_id=self.generation_id,
                )
        state_names = state.get("feature_names") if isinstance(state, Mapping) else None
        if state_names is not None and tuple(state_names) != tuple(profile.feature_order):
            exclusions["query_feature_names"] = 1
            return ExperienceSearchResult(
                candidate_count=len(rows),
                excluded_counts=exclusions,
                active_generation_id=self.generation_id,
            )
        values = state.get("values") if isinstance(state, Mapping) else None
        mask = state.get("mask") if isinstance(state, Mapping) else None
        if values is None:
            values = [state.get(n) for n in profile.feature_order]
            mask = [v is not None for v in values]
        cohort = self._cohort(eligible[0])
        hits = self.index.search(
            values, mask, [str(_get(r, "experience_id")) for r in eligible], query.top_k, profile
        )
        by_id = {str(_get(row, "experience_id")): row for row in eligible}
        result_hits = tuple(
            ExperienceHit(
                experience_id=h.experience_id,
                distance=h.distance,
                similarity_score=h.similarity_score,
                comparable_feature_count=h.comparable_feature_count,
                trust_tier=_get(
                    by_id[h.experience_id], "trust_tier", _get(by_id[h.experience_id], "trust")
                ),
                market_state=_get(by_id[h.experience_id], "market_state", {}),
                action=_get(by_id[h.experience_id], "action"),
                provenance=_get(by_id[h.experience_id], "provenance", {}),
                feature_schema_version=_get(by_id[h.experience_id], "feature_schema_version"),
                similarity_profile_version=_get(
                    by_id[h.experience_id], "similarity_profile_version"
                ),
                currently_tombstoned=bool(_get(by_id[h.experience_id], "tombstoned", False))
                or (
                    bool(_get(by_id[h.experience_id], "source_aliases", {}))
                    and not any(
                        str(v) == "CURRENT" or getattr(v, "value", None) == "CURRENT"
                        for v in (_get(by_id[h.experience_id], "source_aliases", {}) or {}).values()
                    )
                ),
                timestamps={"analysis_snapshot": h.analysis_snapshot_timestamp}
                if h.analysis_snapshot_timestamp
                else {},
            )
            for h in hits
        )
        return ExperienceSearchResult(
            result_hits,
            query_normalization_fingerprint(profile, cohort, query.trust_tiers, query.as_of),
            self.generation_id,
            len(rows),
            exclusions,
        )


__all__ = ["ExperienceQueryService"]
