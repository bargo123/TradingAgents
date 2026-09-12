"""Deterministic, descriptive outcome statistics for retained Phase 8 evidence."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .models import (
    OutcomeDirectionStatistics,
    OutcomeHoldStatistics,
    OutcomeStatistics,
    OutcomeStatsRequest,
    TrustTier,
)


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _utc(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError, OverflowError):
            return None
    if not isinstance(value, datetime) or value.tzinfo is None:
        return None
    if value.utcoffset() != timezone.utc.utcoffset(value):
        return None
    return value


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


class OutcomeStatsCalculator:
    """Calculate statistics from records or an ``ExperienceCatalog`` only.

    The calculator deliberately accepts a small record protocol so callers can
    use immutable catalog records as well as deterministic test projections.
    It never reads or mutates similarity features.
    """

    def __init__(self, records: Sequence[Any] | Any):
        if hasattr(records, "active_records"):
            catalog = records
            self.records = tuple(catalog.active_records()) + tuple(catalog.historical_records())
            self._snapshots = catalog.evaluation_snapshots
        else:
            self.records = tuple(records)
            self._snapshots = None

    def _record_snapshots(self, record: Any) -> tuple[dict[str, Any], ...]:
        if self._snapshots is not None:
            return tuple(self._snapshots(str(_get(record, "experience_id"))))
        snapshots = _get(record, "outcome_snapshots", None)
        if snapshots is None:
            snapshots = _get(record, "evaluation_snapshots", None)
        if snapshots is None:
            snapshots = _get(record, "outcome_evidence_by_basis_horizon", ())
            if isinstance(snapshots, Mapping):
                normalized = []
                for key, value in snapshots.items():
                    if not isinstance(value, Mapping):
                        continue
                    if isinstance(key, tuple) and len(key) == 2:
                        basis, horizon = key
                    elif isinstance(key, str) and "/" in key:
                        basis, horizon = key.split("/", 1)
                    else:
                        continue
                    normalized.append(
                        dict(value, evaluation_basis=basis, horizon_seconds=int(horizon))
                    )
                snapshots = tuple(normalized)
        return tuple(snapshots or ())

    @staticmethod
    def _availability(snapshot: Mapping[str, Any]) -> datetime | None:
        return (
            _utc(snapshot.get("recovered_from_unavailable_at"))
            or _utc(snapshot.get("evaluated_at"))
            or _utc(snapshot.get("created_at"))
        )

    @staticmethod
    def _snapshot_key(snapshot: Mapping[str, Any]) -> tuple[str, str]:
        available = OutcomeStatsCalculator._availability(snapshot)
        return (available.isoformat() if available else "", str(snapshot.get("fingerprint", "")))

    def _select(
        self, snapshots: tuple[dict[str, Any], ...], request: OutcomeStatsRequest
    ) -> tuple[dict[str, Any] | None, str | None]:
        exact = [
            s
            for s in snapshots
            if s.get("evaluation_basis") == request.evaluation_basis
            and int(s.get("horizon_seconds") or 0) == request.horizon_seconds
        ]
        if not exact:
            return None, "BASIS_OR_HORIZON_MISMATCH" if snapshots else "EVALUATION_NOT_AVAILABLE"
        if request.as_of is None:
            return max(exact, key=self._snapshot_key), None
        observed = []
        for snapshot in exact:
            observation = _utc(snapshot.get("observation_timestamp"))
            available = self._availability(snapshot)
            if (
                observation is not None
                and observation <= request.as_of
                and available is not None
                and available <= request.as_of
            ):
                observed.append(snapshot)
        if observed:
            return max(observed, key=self._snapshot_key), None
        # A DATA_UNAVAILABLE state observed before the cutoff is meaningful
        # history. A recovered row first seen after recovery has no such state.
        if any(
            s.get("evaluation_status") == "DATA_UNAVAILABLE"
            and _utc(s.get("observation_timestamp")) is not None
            and _utc(s.get("observation_timestamp")) <= request.as_of
            and self._availability(s) is not None
            and self._availability(s) <= request.as_of
            for s in exact
        ):
            prior = [
                s
                for s in exact
                if s.get("evaluation_status") == "DATA_UNAVAILABLE"
                and _utc(s.get("observation_timestamp")) <= request.as_of
                and self._availability(s) <= request.as_of
            ]
            return max(prior, key=self._snapshot_key), None
        return None, "EVALUATION_NOT_YET_AVAILABLE"

    def calculate(self, request: OutcomeStatsRequest) -> OutcomeStatistics:
        by_reason: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_tier: dict[str, int] = {}
        fingerprints: dict[str, str] = {}
        buy: list[float] = []
        sell: list[float] = []
        hold: list[float] = []
        buy_mfe: list[float] = []
        buy_mae: list[float] = []
        sell_mfe: list[float] = []
        sell_mae: list[float] = []
        hold_buy_missed: list[float] = []
        hold_sell_missed: list[float] = []
        best_counterfactuals: dict[str, int] = {}
        eligible_count = 0
        records = {str(_get(record, "experience_id")): record for record in self.records}
        for experience_id in request.experience_ids:
            record = records.get(str(experience_id))
            if record is None:
                reason, snapshot = "EXPERIENCE_NOT_FOUND", None
            else:
                tier = TrustTier(
                    _get(
                        record,
                        "trust",
                        _get(record, "trust_tier", TrustTier.TIER_C_DIAGNOSTIC_ONLY),
                    )
                )
                snapshots = self._record_snapshots(record)
                snapshot, reason = self._select(snapshots, request)
                if reason is None and tier not in request.trust_tiers:
                    reason = "TRUST_TIER_NOT_ALLOWED"
                if reason is None:
                    status = str(snapshot.get("evaluation_status", ""))
                    if status != "COMPLETE":
                        reason = status or "EVALUATION_STATUS_UNKNOWN"
                    elif self._availability(snapshot) is None:
                        reason = "EVALUATION_AVAILABILITY_UNKNOWN"
                    elif snapshot.get("source_context_eligible") != 1:
                        reason = "SOURCE_CONTEXT_INELIGIBLE"
                    elif (
                        request.evaluation_basis == "DECISION_REFERENCE"
                        and str(snapshot.get("decision_reference_status", "AVAILABLE")).upper()
                        != "AVAILABLE"
                    ):
                        reason = "REFERENCE_UNAVAILABLE"
                    elif (
                        not all(
                            _finite(snapshot.get(field))
                            for field in ("buy_net_points", "sell_net_points")
                        )
                        or str(snapshot.get("selected_action", "")).upper() == "HOLD"
                        and not _finite(snapshot.get("hold_opportunity_cost_points"))
                    ):
                        reason = "REQUIRED_FIELD_MISSING_OR_NONFINITE"
                if reason is None:
                    eligible_count += 1
                    fingerprints[str(experience_id)] = str(snapshot.get("fingerprint", ""))
                    action = str(snapshot.get("selected_action", "")).upper()
                    if _finite(snapshot.get("buy_mfe_points")):
                        buy_mfe.append(float(snapshot["buy_mfe_points"]))
                    if _finite(snapshot.get("buy_mae_points")):
                        buy_mae.append(float(snapshot["buy_mae_points"]))
                    if _finite(snapshot.get("sell_mfe_points")):
                        sell_mfe.append(float(snapshot["sell_mfe_points"]))
                    if _finite(snapshot.get("sell_mae_points")):
                        sell_mae.append(float(snapshot["sell_mae_points"]))
                    buy.append(float(snapshot["buy_net_points"]))
                    sell.append(float(snapshot["sell_net_points"]))
                    if action == "HOLD":
                        hold.append(float(snapshot["hold_opportunity_cost_points"]))
                        if float(snapshot["buy_net_points"]) > 0:
                            hold_buy_missed.append(float(snapshot["buy_net_points"]))
                        if float(snapshot["sell_net_points"]) > 0:
                            hold_sell_missed.append(float(snapshot["sell_net_points"]))
                        best = str(snapshot.get("best_counterfactual_action", "")).upper()
                        if best in {"BUY", "SELL", "TIE"}:
                            best_counterfactuals[best] = best_counterfactuals.get(best, 0) + 1
                    continue
                status = str(snapshot.get("evaluation_status", "")) if snapshot is not None else ""
                if status:
                    by_status[status] = by_status.get(status, 0) + 1
                by_tier[tier.value] = by_tier.get(tier.value, 0) + 1
            by_reason[reason] = by_reason.get(reason, 0) + 1

        def directional(
            values: list[float], mfe: list[float], mae: list[float]
        ) -> OutcomeDirectionStatistics:
            n = len(values)
            positive = sum(v > 0 for v in values)
            negative = sum(v < 0 for v in values)
            zero = sum(v == 0 for v in values)

            def quantiles(points: list[float]) -> dict[str, float]:
                if len(points) < 2:
                    return {}
                ordered = sorted(points)
                return {
                    "p50": float(statistics.median(ordered)),
                    "p95": float(statistics.quantiles(ordered, n=100, method="inclusive")[94]),
                }

            return OutcomeDirectionStatistics(
                tuple(values),
                positive / n if n else None,
                n,
                positive,
                positive / n if n else None,
                negative,
                negative / n if n else None,
                zero,
                zero / n if n else None,
                statistics.mean(values) if values else None,
                statistics.median(values) if values else None,
                tuple(mfe),
                tuple(mae),
                statistics.mean(mfe) if mfe else None,
                statistics.median(mfe) if mfe else None,
                statistics.mean(mae) if mae else None,
                statistics.median(mae) if mae else None,
                quantiles(mfe),
                quantiles(mae),
            )

        return OutcomeStatistics(
            eligible_sample_denominator=eligible_count,
            eligible_count=eligible_count,
            evaluation_basis=request.evaluation_basis,
            horizon_seconds=request.horizon_seconds,
            requested_basis=request.evaluation_basis,
            requested_horizon_seconds=request.horizon_seconds,
            excluded_counts=by_reason,
            exclusions_by_status=by_status,
            exclusions_by_tier=by_tier,
            source_evaluation_fingerprints=fingerprints,
            buy=directional(buy, buy_mfe, buy_mae),
            sell=directional(sell, sell_mfe, sell_mae),
            hold=OutcomeHoldStatistics(
                tuple(hold),
                None,
                len(hold),
                tuple(hold_buy_missed),
                tuple(hold_sell_missed),
                best_counterfactuals,
            ),
        )


__all__ = ["OutcomeStatsCalculator"]
