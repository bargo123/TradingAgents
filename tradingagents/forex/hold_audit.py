"""Read-only audit of realized outcomes after Portfolio Manager HOLD decisions.

The audit deliberately lives outside the collector, evaluator, and dashboard.
It reads the deterministic Phase 5 ``DECISION_REFERENCE`` rows and never
recomputes outcomes from prices, calls MT5/Ollama, or writes to SQLite.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

UTC = timezone.utc
EVALUATION_HORIZONS = (300, 900, 1800, 3600)
HORIZON_LABELS = {300: "5m", 900: "15m", 1800: "30m", 3600: "60m"}
MISSED_MOVE_THRESHOLDS = (0, 1, 2, 5, 10, 20, 50)
EPSILON = 1e-9

CATEGORY_HOLD_BEST = "HOLD_BEST"
CATEGORY_BUY_BETTER = "BUY_BETTER"
CATEGORY_SELL_BETTER = "SELL_BETTER"
CATEGORY_DIRECTIONAL_TIE = "DIRECTIONAL_TIE"
CATEGORIES = (
    CATEGORY_HOLD_BEST,
    CATEGORY_BUY_BETTER,
    CATEGORY_SELL_BETTER,
    CATEGORY_DIRECTIONAL_TIE,
)


class HoldAuditError(RuntimeError):
    """Base class for deterministic, user-visible audit failures."""


class HoldAuditReadError(HoldAuditError):
    """The source database could not be read within the bounded policy."""


class HoldAuditSchemaError(HoldAuditError):
    """The source database does not satisfy the Phase 5 read contract."""


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed.astimezone(UTC)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_true(value: Any) -> bool:
    return value is True or value == 1 or value == "1" or value == "true"


def percentile(values: Iterable[float], percent: float) -> float | None:
    """Return a deterministic nearest-rank percentile without numpy."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 < percent <= 100:
        raise ValueError("percent must be greater than 0 and at most 100")
    rank = max(1, math.ceil(percent / 100.0 * len(ordered)))
    return ordered[rank - 1]


def classify_hold_outcome(
    buy_net_points: float,
    sell_net_points: float,
    hold_opportunity_cost_points: float,
    *,
    epsilon: float = EPSILON,
    best_counterfactual_action: str | None = None,
) -> str:
    """Classify one persisted COMPLETE HOLD outcome.

    ``best_counterfactual_action`` is intentionally accepted only for API
    compatibility/diagnostics and never drives classification.  A negative
    BUY/SELL pair remains ``HOLD_BEST`` even if Phase 5 stored SELL as the
    best (least-negative) counterfactual.
    """
    del best_counterfactual_action
    buy = float(buy_net_points)
    sell = float(sell_net_points)
    opportunity_cost = float(hold_opportunity_cost_points)
    if opportunity_cost <= epsilon:
        return CATEGORY_HOLD_BEST
    if buy > 0 and buy > sell + epsilon:
        return CATEGORY_BUY_BETTER
    if sell > 0 and sell > buy + epsilon:
        return CATEGORY_SELL_BETTER
    if buy > 0 and sell > 0 and abs(buy - sell) <= epsilon:
        return CATEGORY_DIRECTIONAL_TIE
    # A valid Phase 5 row normally cannot reach this branch.  Keep the audit
    # conservative if a damaged row has inconsistent persisted fields.
    if buy > sell and buy > 0:
        return CATEGORY_BUY_BETTER
    if sell > buy and sell > 0:
        return CATEGORY_SELL_BETTER
    return CATEGORY_HOLD_BEST


def classify_cross_horizon(
    opportunity_costs: Iterable[float], *, strong_miss_points: float = 5.0, epsilon: float = EPSILON
) -> str:
    values = tuple(float(value) for value in opportunity_costs)
    positive = sum(value > epsilon for value in values)
    if positive == 0:
        return "CONSISTENT_HOLD"
    if positive < 3:
        return "TRANSIENT_MISS"
    if sum(value > epsilon and value >= strong_miss_points for value in values) >= 3:
        return "STRONG_PERSISTENT_MISS"
    return "PERSISTENT_DIRECTIONAL_MISS"


def _stats(values: Iterable[float]) -> dict[str, float | None]:
    numbers = [float(value) for value in values]
    return {
        "average": sum(numbers) / len(numbers) if numbers else None,
        "median": percentile(numbers, 50),
        "p75": percentile(numbers, 75),
        "p90": percentile(numbers, 90),
        "p95": percentile(numbers, 95),
        "maximum": max(numbers) if numbers else None,
    }


@dataclass(frozen=True, slots=True)
class HoldObservation:
    decision_id: str
    horizon_seconds: int
    symbol: str
    decision_timestamp: datetime | None
    buy_net_points: float
    sell_net_points: float
    hold_opportunity_cost_points: float
    category: str


@dataclass(frozen=True, slots=True)
class HorizonAudit:
    horizon_seconds: int
    complete_hold_samples: int
    category_counts: Mapping[str, int]
    category_percentages: Mapping[str, float]
    opportunity_cost_statistics: Mapping[str, float | None]
    thresholds: Mapping[int, Mapping[str, float | int]]
    missed_buy: Mapping[str, float | int | None]
    missed_sell: Mapping[str, float | int | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "label": HORIZON_LABELS[self.horizon_seconds],
            "complete_hold_samples": self.complete_hold_samples,
            "category_counts": dict(self.category_counts),
            "category_percentages": dict(self.category_percentages),
            "opportunity_cost_points": dict(self.opportunity_cost_statistics),
            "thresholds": {str(key): dict(value) for key, value in self.thresholds.items()},
            "missed_buy": dict(self.missed_buy),
            "missed_sell": dict(self.missed_sell),
        }


@dataclass(frozen=True, slots=True)
class CrossHorizonAudit:
    fully_evaluated_hold_decisions: int
    classification_counts: Mapping[str, int]
    direction_counts: Mapping[str, int]

    @property
    def persistent_directional_misses(self) -> int:
        return self.classification_counts.get("PERSISTENT_DIRECTIONAL_MISS", 0) + self.classification_counts.get(
            "STRONG_PERSISTENT_MISS", 0
        )

    @property
    def strong_persistent_misses(self) -> int:
        return self.classification_counts.get("STRONG_PERSISTENT_MISS", 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fully_evaluated_hold_decisions": self.fully_evaluated_hold_decisions,
            "classification_counts": dict(self.classification_counts),
            "persistent_directional_misses": self.persistent_directional_misses,
            "strong_persistent_misses": self.strong_persistent_misses,
            "direction_counts": dict(self.direction_counts),
        }


@dataclass(frozen=True, slots=True)
class PopulationAudit:
    horizons: Mapping[int, HorizonAudit]
    cross_horizon: CrossHorizonAudit

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizons": {str(key): value.to_dict() for key, value in self.horizons.items()},
            "cross_horizon": self.cross_horizon.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class HoldAuditReport:
    database_path: str
    generated_at: datetime
    recent_hours: float
    strong_miss_points: float
    fully_evaluated_source_eligible_decisions: int
    fully_evaluated_source_eligible_hold_decisions: int
    fully_evaluated_source_eligible_buy_decisions: int
    fully_evaluated_source_eligible_sell_decisions: int
    all_population: PopulationAudit
    recent_population: PopulationAudit
    data_quality: Mapping[str, int]
    data_quality_unavailable_reasons: Mapping[str, int]
    invalid_complete_hold_rows: int
    top_missed_opportunities: tuple[Mapping[str, Any], ...]
    review_status: str = "OBSERVATION ONLY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_path": self.database_path,
            "generated_at": self.generated_at.isoformat().replace("+00:00", "Z"),
            "recent_hours": self.recent_hours,
            "strong_miss_points": self.strong_miss_points,
            "review_status": self.review_status,
            "fully_evaluated_source_eligible_decisions": self.fully_evaluated_source_eligible_decisions,
            "fully_evaluated_source_eligible_hold_decisions": self.fully_evaluated_source_eligible_hold_decisions,
            "fully_evaluated_source_eligible_buy_decisions": self.fully_evaluated_source_eligible_buy_decisions,
            "fully_evaluated_source_eligible_sell_decisions": self.fully_evaluated_source_eligible_sell_decisions,
            "all": self.all_population.to_dict(),
            "recent": self.recent_population.to_dict(),
            "data_quality": dict(self.data_quality),
            "data_quality_unavailable_reasons": dict(self.data_quality_unavailable_reasons),
            "invalid_complete_hold_rows": self.invalid_complete_hold_rows,
            "top_missed_opportunities": [dict(item) for item in self.top_missed_opportunities],
            "interpretation": {
                "status": "OBSERVATION ONLY",
                "caution": (
                    "A positive counterfactual outcome after the fact does not prove that HOLD was "
                    "an incorrect ex-ante decision."
                ),
                "limitations": [
                    "This audit measures realized opportunity cost and directional outcome only.",
                    "It does not prove predictability, strategy error, live profitability, or whether a risk-adjusted entry was justified.",
                ],
            },
        }


@dataclass(frozen=True, slots=True)
class _EvaluationRow:
    decision_id: str
    horizon_seconds: int
    symbol: str
    source_context_eligible: bool
    status: str
    selected_action: str | None
    buy_net_points: float | None
    sell_net_points: float | None
    opportunity_cost: float | None
    best_counterfactual_action: str | None
    unavailable_reason: str | None
    decision_timestamp: datetime | None


def _readonly_connection(path: Path, busy_timeout_seconds: float) -> sqlite3.Connection:
    if busy_timeout_seconds <= 0 or not math.isfinite(busy_timeout_seconds):
        raise ValueError("busy_timeout_seconds must be a positive finite number")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise HoldAuditReadError(f"database does not exist: {resolved}")
    uri = "file:" + quote(resolved.as_posix(), safe="/:\\") + "?mode=ro"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=busy_timeout_seconds,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout={max(1, int(busy_timeout_seconds * 1000))}")
            connection.execute("PRAGMA query_only=ON")
            return connection
        except sqlite3.Error as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
    raise HoldAuditReadError(f"bounded read failed for {resolved}: {last_error}") from last_error


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchall()
    if not rows:
        return set()
    escaped = table.replace('"', '""')
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{escaped}")')}


def _load_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    evaluation_columns = _table_columns(connection, "shadow_decision_evaluations")
    required = {
        "decision_id",
        "evaluation_basis",
        "horizon_seconds",
        "resolved_symbol",
        "source_context_eligible",
        "training_eligible",
        "evaluation_status",
        "selected_action",
        "buy_net_points",
        "sell_net_points",
        "hold_opportunity_cost_points",
        "created_at",
    }
    missing = sorted(required - evaluation_columns)
    if missing:
        raise HoldAuditSchemaError(
            "shadow_decision_evaluations is missing required columns: " + ", ".join(missing)
        )
    decision_columns = _table_columns(connection, "shadow_decisions")
    if "decision_id" in decision_columns:
        selected = (
            "e.*, d.created_at AS decision_created_at, "
            "d.decision_completed_timestamp AS decision_completed_timestamp, "
            "d.resolved_symbol AS decision_resolved_symbol"
        )
        query = (
            f"SELECT {selected} FROM shadow_decision_evaluations AS e "
            "LEFT JOIN shadow_decisions AS d ON d.decision_id=e.decision_id"
        )
    else:
        query = "SELECT * FROM shadow_decision_evaluations"
    return [dict(row) for row in connection.execute(query).fetchall()]


def _row_to_evaluation(row: Mapping[str, Any]) -> _EvaluationRow:
    timestamp = (
        _parse_utc(row.get("decision_completed_timestamp"))
        or _parse_utc(row.get("decision_created_at"))
        or _parse_utc(row.get("created_at"))
    )
    return _EvaluationRow(
        decision_id=str(row.get("decision_id") or ""),
        horizon_seconds=int(row.get("horizon_seconds") or 0),
        symbol=str(row.get("decision_resolved_symbol") or row.get("resolved_symbol") or ""),
        source_context_eligible=_is_true(row.get("source_context_eligible")),
        status=str(row.get("evaluation_status") or "UNKNOWN"),
        selected_action=(str(row["selected_action"]) if row.get("selected_action") is not None else None),
        buy_net_points=_finite(row.get("buy_net_points")),
        sell_net_points=_finite(row.get("sell_net_points")),
        opportunity_cost=_finite(row.get("hold_opportunity_cost_points")),
        best_counterfactual_action=(
            str(row["best_counterfactual_action"])
            if row.get("best_counterfactual_action") is not None
            else None
        ),
        unavailable_reason=(str(row["unavailable_reason"]) if row.get("unavailable_reason") else None),
        decision_timestamp=timestamp,
    )


def _observation(row: _EvaluationRow) -> HoldObservation | None:
    if (
        row.status != "COMPLETE"
        or not row.source_context_eligible
        or row.selected_action != "HOLD"
        or row.buy_net_points is None
        or row.sell_net_points is None
        or row.opportunity_cost is None
    ):
        return None
    return HoldObservation(
        decision_id=row.decision_id,
        horizon_seconds=row.horizon_seconds,
        symbol=row.symbol,
        decision_timestamp=row.decision_timestamp,
        buy_net_points=row.buy_net_points,
        sell_net_points=row.sell_net_points,
        hold_opportunity_cost_points=row.opportunity_cost,
        category=classify_hold_outcome(
            row.buy_net_points,
            row.sell_net_points,
            row.opportunity_cost,
            best_counterfactual_action=row.best_counterfactual_action,
        ),
    )


def _horizon_audit(horizon: int, observations: Iterable[HoldObservation]) -> HorizonAudit:
    rows = tuple(observations)
    count = len(rows)
    category_counts = {category: sum(row.category == category for row in rows) for category in CATEGORIES}
    category_percentages = {
        category: (value * 100.0 / count if count else 0.0)
        for category, value in category_counts.items()
    }
    opportunity_costs = [row.hold_opportunity_cost_points for row in rows]
    thresholds: dict[int, dict[str, float | int]] = {}
    for threshold in MISSED_MOVE_THRESHOLDS:
        matched = sum(value > threshold for value in opportunity_costs)
        thresholds[threshold] = {
            "count": matched,
            "percent": matched * 100.0 / count if count else 0.0,
        }
    buy_misses = [row.buy_net_points for row in rows if row.category == CATEGORY_BUY_BETTER]
    sell_misses = [row.sell_net_points for row in rows if row.category == CATEGORY_SELL_BETTER]
    buy_stats = _stats(buy_misses)
    sell_stats = _stats(sell_misses)
    return HorizonAudit(
        horizon_seconds=horizon,
        complete_hold_samples=count,
        category_counts=category_counts,
        category_percentages=category_percentages,
        opportunity_cost_statistics=_stats(opportunity_costs),
        thresholds=thresholds,
        missed_buy={"count": len(buy_misses), **buy_stats},
        missed_sell={"count": len(sell_misses), **sell_stats},
    )


def _cross_horizon(
    observations: Iterable[HoldObservation], *, strong_miss_points: float
) -> CrossHorizonAudit:
    grouped: dict[str, dict[int, HoldObservation]] = defaultdict(dict)
    for observation in observations:
        grouped[observation.decision_id][observation.horizon_seconds] = observation
    complete = {
        decision_id: rows
        for decision_id, rows in grouped.items()
        if all(horizon in rows for horizon in EVALUATION_HORIZONS)
    }
    classification_counts = Counter()
    direction_counts = Counter()
    for rows in complete.values():
        ordered = [rows[horizon] for horizon in EVALUATION_HORIZONS]
        classification_counts[
            classify_cross_horizon(
                (row.hold_opportunity_cost_points for row in ordered),
                strong_miss_points=strong_miss_points,
            )
        ] += 1
        buy_count = sum(row.category == CATEGORY_BUY_BETTER for row in ordered)
        sell_count = sum(row.category == CATEGORY_SELL_BETTER for row in ordered)
        if all(row.category == CATEGORY_HOLD_BEST for row in ordered):
            direction_counts["HOLD_BEST_ALL_FOUR"] += 1
        elif buy_count >= 3:
            direction_counts["CONSISTENT_BUY_BETTER"] += 1
        elif sell_count >= 3:
            direction_counts["CONSISTENT_SELL_BETTER"] += 1
        else:
            direction_counts["MIXED_DIRECTION"] += 1
    return CrossHorizonAudit(
        fully_evaluated_hold_decisions=len(complete),
        classification_counts={
            key: int(classification_counts.get(key, 0))
            for key in (
                "CONSISTENT_HOLD",
                "TRANSIENT_MISS",
                "PERSISTENT_DIRECTIONAL_MISS",
                "STRONG_PERSISTENT_MISS",
            )
        },
        direction_counts={
            key: int(direction_counts.get(key, 0))
            for key in (
                "CONSISTENT_BUY_BETTER",
                "CONSISTENT_SELL_BETTER",
                "MIXED_DIRECTION",
                "HOLD_BEST_ALL_FOUR",
            )
        },
    )


def _population(
    observations: Iterable[HoldObservation], *, strong_miss_points: float
) -> PopulationAudit:
    by_horizon: dict[int, list[HoldObservation]] = {horizon: [] for horizon in EVALUATION_HORIZONS}
    for observation in observations:
        if observation.horizon_seconds in by_horizon:
            by_horizon[observation.horizon_seconds].append(observation)
    return PopulationAudit(
        horizons={horizon: _horizon_audit(horizon, rows) for horizon, rows in by_horizon.items()},
        cross_horizon=_cross_horizon(observations, strong_miss_points=strong_miss_points),
    )


def _full_source_eligible_groups(rows: Iterable[_EvaluationRow]) -> dict[str, dict[int, _EvaluationRow]]:
    grouped: dict[str, dict[int, _EvaluationRow]] = defaultdict(dict)
    for row in rows:
        if row.decision_id and row.horizon_seconds in EVALUATION_HORIZONS:
            grouped[row.decision_id][row.horizon_seconds] = row
    return {
        decision_id: values
        for decision_id, values in grouped.items()
        if all(
            horizon in values
            and values[horizon].status == "COMPLETE"
            and values[horizon].source_context_eligible
            for horizon in EVALUATION_HORIZONS
        )
    }


def _top_missed(observations: Iterable[HoldObservation]) -> tuple[Mapping[str, Any], ...]:
    grouped: dict[str, dict[int, HoldObservation]] = defaultdict(dict)
    for observation in observations:
        grouped[observation.decision_id][observation.horizon_seconds] = observation
    entries: list[dict[str, Any]] = []
    for decision_id, values in grouped.items():
        if not all(horizon in values for horizon in EVALUATION_HORIZONS):
            continue
        ordered = [values[horizon] for horizon in EVALUATION_HORIZONS]
        maximum = max(ordered, key=lambda row: (row.hold_opportunity_cost_points, -row.horizon_seconds))
        if maximum.hold_opportunity_cost_points <= EPSILON:
            direction = "HOLD"
        elif abs(maximum.buy_net_points - maximum.sell_net_points) <= EPSILON:
            direction = "TIE"
        else:
            direction = "BUY" if maximum.buy_net_points > maximum.sell_net_points else "SELL"
        entry: dict[str, Any] = {
            "decision_timestamp": (
                ordered[0].decision_timestamp.isoformat().replace("+00:00", "Z")
                if ordered[0].decision_timestamp
                else None
            ),
            "decision_id_short": decision_id[:8],
            "symbol": maximum.symbol,
            "largest_missed_direction": direction,
            "max_missed_points": maximum.hold_opportunity_cost_points,
        }
        for row in ordered:
            entry[f"{HORIZON_LABELS[row.horizon_seconds]}_opportunity_cost_points"] = row.hold_opportunity_cost_points
        entries.append(entry)
    entries.sort(
        key=lambda item: (
            -float(item["max_missed_points"]),
            item["decision_timestamp"] or "",
            item["decision_id_short"],
        )
    )
    return tuple(entries[:10])


def audit_hold_outcomes(
    db_path: str | Path,
    *,
    strong_miss_points: float = 5.0,
    recent_hours: float = 24.0,
    busy_timeout_seconds: float = 0.25,
) -> HoldAuditReport:
    """Build one read-only HOLD outcome report from a Phase 5/6 database."""
    if strong_miss_points < 0 or not math.isfinite(strong_miss_points):
        raise ValueError("strong_miss_points must be a non-negative finite number")
    if recent_hours <= 0 or not math.isfinite(recent_hours):
        raise ValueError("recent_hours must be a positive finite number")
    generated_at = datetime.now(UTC)
    connection = _readonly_connection(Path(db_path), busy_timeout_seconds)
    try:
        raw_rows = _load_rows(connection)
    except sqlite3.Error as exc:
        raise HoldAuditReadError(f"failed to read evaluation rows: {exc}") from exc
    finally:
        connection.close()

    rows = [_row_to_evaluation(row) for row in raw_rows if row.get("evaluation_basis") == "DECISION_REFERENCE"]
    data_quality_counter = Counter(row.status for row in rows)
    data_quality = {
        status: int(data_quality_counter.get(status, 0))
        for status in ("COMPLETE", "DATA_UNAVAILABLE", "PENDING", "INELIGIBLE")
    }
    unknown_statuses = {
        status: count
        for status, count in data_quality_counter.items()
        if status not in data_quality
    }
    data_quality.update(unknown_statuses)
    unavailable_reasons = Counter(
        row.unavailable_reason or "UNKNOWN"
        for row in rows
        if row.status == "DATA_UNAVAILABLE"
    )
    observations = tuple(
        observation
        for row in rows
        if (observation := _observation(row)) is not None
    )
    invalid_complete_hold_rows = sum(
        row.status == "COMPLETE"
        and row.source_context_eligible
        and row.selected_action == "HOLD"
        and _observation(row) is None
        for row in rows
    )
    all_population = _population(observations, strong_miss_points=strong_miss_points)
    cutoff = generated_at - timedelta(hours=recent_hours)
    recent_observations = tuple(
        observation
        for observation in observations
        if observation.decision_timestamp is not None and observation.decision_timestamp >= cutoff
    )
    recent_population = _population(recent_observations, strong_miss_points=strong_miss_points)

    full_groups = _full_source_eligible_groups(rows)
    full_action_counts = Counter()
    for values in full_groups.values():
        actions = {values[horizon].selected_action for horizon in EVALUATION_HORIZONS}
        if len(actions) == 1:
            full_action_counts[next(iter(actions))] += 1
    return HoldAuditReport(
        database_path=str(Path(db_path).expanduser().resolve()),
        generated_at=generated_at,
        recent_hours=recent_hours,
        strong_miss_points=strong_miss_points,
        fully_evaluated_source_eligible_decisions=len(full_groups),
        fully_evaluated_source_eligible_hold_decisions=int(full_action_counts.get("HOLD", 0)),
        fully_evaluated_source_eligible_buy_decisions=int(full_action_counts.get("BUY", 0)),
        fully_evaluated_source_eligible_sell_decisions=int(full_action_counts.get("SELL", 0)),
        all_population=all_population,
        recent_population=recent_population,
        data_quality=data_quality,
        data_quality_unavailable_reasons=dict(unavailable_reasons),
        invalid_complete_hold_rows=invalid_complete_hold_rows,
        top_missed_opportunities=_top_missed(observations),
    )


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _render_horizon_table(population: PopulationAudit) -> list[str]:
    lines = [
        "HORIZON SUMMARY",
        "horizon | samples | HOLD_BEST | BUY_BETTER | SELL_BETTER | TIE | avg cost | median | p75 | p90 | p95 | max",
    ]
    for horizon in EVALUATION_HORIZONS:
        row = population.horizons[horizon]
        stats = row.opportunity_cost_statistics
        lines.append(
            f"{HORIZON_LABELS[horizon]:>7} | {row.complete_hold_samples:7d} | "
            f"{row.category_counts[CATEGORY_HOLD_BEST]:9d} ({row.category_percentages[CATEGORY_HOLD_BEST]:5.1f}%) | "
            f"{row.category_counts[CATEGORY_BUY_BETTER]:10d} ({row.category_percentages[CATEGORY_BUY_BETTER]:5.1f}%) | "
            f"{row.category_counts[CATEGORY_SELL_BETTER]:11d} ({row.category_percentages[CATEGORY_SELL_BETTER]:5.1f}%) | "
            f"{row.category_counts[CATEGORY_DIRECTIONAL_TIE]:3d} | {_fmt(stats['average'])} | {_fmt(stats['median'])} | "
            f"{_fmt(stats['p75'])} | {_fmt(stats['p90'])} | {_fmt(stats['p95'])} | {_fmt(stats['maximum'])}"
        )
    return lines


def render_text(report: HoldAuditReport) -> str:
    all_cross = report.all_population.cross_horizon
    lines = [
        "HOLD OUTCOME AUDIT",
        f"Database: {report.database_path}",
        f"Review status: {report.review_status}",
        f"Fully evaluated source-eligible decisions: {report.fully_evaluated_source_eligible_decisions}",
        f"Fully evaluated source-eligible HOLD decisions: {report.fully_evaluated_source_eligible_hold_decisions}",
        f"Fully evaluated source-eligible BUY decisions: {report.fully_evaluated_source_eligible_buy_decisions}",
        f"Fully evaluated source-eligible SELL decisions: {report.fully_evaluated_source_eligible_sell_decisions}",
        "No completed BUY/SELL comparison sample is available yet."
        if report.fully_evaluated_source_eligible_buy_decisions == 0
        and report.fully_evaluated_source_eligible_sell_decisions == 0
        else "",
        "",
        *_render_horizon_table(report.all_population),
        "",
        "MISSED POSITIVE DIRECTIONAL MOVE (points; strict > threshold)",
    ]
    for horizon in EVALUATION_HORIZONS:
        row = report.all_population.horizons[horizon]
        values = ", ".join(
            f">{threshold}: {int(row.thresholds[threshold]['count'])} ({float(row.thresholds[threshold]['percent']):.1f}%)"
            for threshold in MISSED_MOVE_THRESHOLDS
        )
        lines.append(f"{HORIZON_LABELS[horizon]}: {values}")
    lines.extend(["", "DIRECTIONAL MISSES (positive net points only)"])
    for horizon in EVALUATION_HORIZONS:
        row = report.all_population.horizons[horizon]
        lines.append(
            f"{HORIZON_LABELS[horizon]} BUY: {row.missed_buy['count']} avg={_fmt(row.missed_buy['average'])} "
            f"median={_fmt(row.missed_buy['median'])} max={_fmt(row.missed_buy['maximum'])}; "
            f"SELL: {row.missed_sell['count']} avg={_fmt(row.missed_sell['average'])} "
            f"median={_fmt(row.missed_sell['median'])} max={_fmt(row.missed_sell['maximum'])}"
        )
    lines.extend(
        [
            "",
            "CROSS-HORIZON HOLD AUDIT",
            f"Fully evaluated HOLD decisions: {all_cross.fully_evaluated_hold_decisions}",
            f"CONSISTENT_HOLD: {all_cross.classification_counts['CONSISTENT_HOLD']}",
            f"TRANSIENT_MISS: {all_cross.classification_counts['TRANSIENT_MISS']}",
            f"PERSISTENT_DIRECTIONAL_MISS: {all_cross.persistent_directional_misses}",
            f"STRONG_PERSISTENT_MISS >= {report.strong_miss_points:g} points: {all_cross.strong_persistent_misses}",
            "Direction consistency: "
            + ", ".join(f"{key}={value}" for key, value in all_cross.direction_counts.items()),
            "",
            "DATA QUALITY (DECISION_REFERENCE)",
            ", ".join(f"{key}={value}" for key, value in report.data_quality.items()),
            "Unavailable reasons: "
            + (", ".join(f"{key}={value}" for key, value in report.data_quality_unavailable_reasons.items()) or "-"),
            f"Invalid COMPLETE HOLD rows: {report.invalid_complete_hold_rows}",
            "",
            "RECENT QUALIFYING HOLD EVALUATIONS",
            f"Window: last {report.recent_hours:g} UTC hours",
            *_render_horizon_table(report.recent_population),
            "",
            "TOP MISSED OPPORTUNITIES",
        ]
    )
    if report.top_missed_opportunities:
        lines.append("UTC | decision | symbol | 5m | 15m | 30m | 60m | direction | max points")
        for row in report.top_missed_opportunities:
            lines.append(
                f"{row['decision_timestamp'] or '-'} | {row['decision_id_short']} | {row['symbol']} | "
                + " | ".join(_fmt(row.get(f"{label}_opportunity_cost_points")) for label in ("5m", "15m", "30m", "60m"))
                + f" | {row['largest_missed_direction']} | {_fmt(row['max_missed_points'])}"
            )
    else:
        lines.append("No fully evaluated HOLD decisions with four horizons are available.")
    lines.extend(
        [
            "",
            "INTERPRETATION",
            "A positive counterfactual outcome after the fact does NOT prove that HOLD was an incorrect ex-ante decision.",
            "This audit measures realized opportunity cost and directional outcome only.",
            "It does not prove predictability, strategy error, live profitability, or whether a risk-adjusted entry was justified.",
            "Review status: OBSERVATION ONLY",
        ]
    )
    return "\n".join(line for line in lines if line is not None)


def report_as_json(report: HoldAuditReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CATEGORY_BUY_BETTER",
    "CATEGORY_DIRECTIONAL_TIE",
    "CATEGORY_HOLD_BEST",
    "CATEGORY_SELL_BETTER",
    "EVALUATION_HORIZONS",
    "HoldAuditError",
    "HoldAuditReadError",
    "HoldAuditReport",
    "audit_hold_outcomes",
    "classify_cross_horizon",
    "classify_hold_outcome",
    "percentile",
    "render_text",
    "report_as_json",
]
