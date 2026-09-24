"""Read-only Trader-to-Portfolio-Manager decision-path diagnostics.

This module is deliberately outside the collector, evaluator, dashboard, and
strategy paths.  It reads an immutable Phase 5/6 SQLite snapshot through a
strict ``mode=ro`` connection and reports where directional recommendations
change.  It never calls MT5/Ollama and never writes the source database.
"""

from __future__ import annotations

import json
import math
import re
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
TRADING_ACTIONS = ("BUY", "HOLD", "SELL")
TRADER_ACTION_UNAVAILABLE = "TRADER_ACTION_UNAVAILABLE"
FINAL_ACTION_UNAVAILABLE = "FINAL_ACTION_UNAVAILABLE"
RESEARCH_RECOMMENDATIONS = ("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL")
RESEARCH_RECOMMENDATION_UNAVAILABLE = "RESEARCH_RECOMMENDATION_UNAVAILABLE"
RESEARCH_MANAGER_UNAVAILABLE = (
    "Research Manager structured recommendation is unavailable in the immutable dataset."
)
EPSILON = 1e-9

_TRADER_MARKER = re.compile(
    r"(?m)^FINAL TRANSACTION PROPOSAL:\s+\*\*(BUY|HOLD|SELL)\*\*[ \t]*$"
)
class DecisionPathAuditError(RuntimeError):
    """Base class for deterministic, user-visible audit failures."""


class DecisionPathAuditReadError(DecisionPathAuditError):
    """The source database could not be read within the bounded policy."""


class DecisionPathAuditSchemaError(DecisionPathAuditError):
    """The source database does not satisfy the audit read contract."""


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


def _stats(values: Iterable[float]) -> dict[str, float | None]:
    numbers = [float(value) for value in values]
    return {
        "average": sum(numbers) / len(numbers) if numbers else None,
        "median": percentile(numbers, 50),
        "p90": percentile(numbers, 90),
        "p95": percentile(numbers, 95),
    }


def extract_trader_action(trader_summary: Any) -> str:
    """Extract one exact bounded Trader marker; arbitrary prose is ignored."""
    if not isinstance(trader_summary, str):
        return TRADER_ACTION_UNAVAILABLE
    matches = _TRADER_MARKER.findall(trader_summary)
    if len(matches) != 1:
        return TRADER_ACTION_UNAVAILABLE
    return matches[0]


def _action_from_persisted(value: Any) -> str:
    if isinstance(value, str) and value.upper() in TRADING_ACTIONS:
        return value.upper()
    return FINAL_ACTION_UNAVAILABLE


def _rating_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return {
        "buy": "BUY",
        "overweight": "BUY",
        "hold": "HOLD",
        "underweight": "SELL",
        "sell": "SELL",
    }.get(value.strip().casefold())


def _transition_key(trader_action: str, final_action: str) -> str:
    return f"{trader_action}->{final_action}"


def _decision_transition_key(decision: _Decision) -> str:
    return _transition_key(decision.trader_action, decision.final_action)


def _transition_keys() -> tuple[str, ...]:
    return tuple(
        _transition_key(trader, final)
        for trader in (*TRADING_ACTIONS, TRADER_ACTION_UNAVAILABLE)
        for final in (*TRADING_ACTIONS, FINAL_ACTION_UNAVAILABLE)
    )


def _research_transition_key(recommendation: str | None, trader_action: str) -> str:
    return f"{recommendation or RESEARCH_RECOMMENDATION_UNAVAILABLE}->{trader_action}"


def _research_transition_keys() -> tuple[str, ...]:
    return tuple(
        _research_transition_key(recommendation, trader)
        for recommendation in (*RESEARCH_RECOMMENDATIONS, None)
        for trader in (*TRADING_ACTIONS, TRADER_ACTION_UNAVAILABLE)
    )


def _research_trader_pm_key(decision: _Decision) -> str:
    recommendation = decision.research_manager_recommendation or RESEARCH_RECOMMENDATION_UNAVAILABLE
    return f"{recommendation}->{decision.trader_action}->{decision.final_action}"


def _research_trader_pm_keys() -> tuple[str, ...]:
    return tuple(
        f"{recommendation or RESEARCH_RECOMMENDATION_UNAVAILABLE}->{trader}->{final}"
        for recommendation in (*RESEARCH_RECOMMENDATIONS, None)
        for trader in (*TRADING_ACTIONS, TRADER_ACTION_UNAVAILABLE)
        for final in (*TRADING_ACTIONS, FINAL_ACTION_UNAVAILABLE)
    )


def classify_cross_horizon(
    opportunity_costs: Iterable[float], *, strong_miss_points: float = 5.0, epsilon: float = EPSILON
) -> str:
    """Use the same four-horizon persistent-miss definitions as ``hold_audit``."""
    values = tuple(float(value) for value in opportunity_costs)
    positive = sum(value > epsilon for value in values)
    if positive == 0:
        return "CONSISTENT_HOLD"
    if positive < 3:
        return "TRANSIENT_MISS"
    if sum(value > epsilon and value >= strong_miss_points for value in values) >= 3:
        return "STRONG_PERSISTENT_MISS"
    return "PERSISTENT_DIRECTIONAL_MISS"


@dataclass(frozen=True, slots=True)
class DecisionPathPopulation:
    population_count: int
    transition_matrix: Mapping[str, Mapping[str, float | int]]
    final_hold_origins: Mapping[str, int]
    outcome_by_transition: Mapping[str, Mapping[str, float | int | None]]
    persistent_by_transition: Mapping[str, Mapping[str, int]]
    fully_evaluated_source_eligible_count: int
    research_to_trader_matrix: Mapping[str, Mapping[str, float | int]]
    research_to_trader_pm_matrix: Mapping[str, Mapping[str, float | int]]
    research_to_trader_suppression: Mapping[str, int]
    research_to_trader_outcomes: Mapping[str, Mapping[str, float | int | None]]
    research_to_trader_persistent: Mapping[str, Mapping[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_count": self.population_count,
            "transition_matrix": {key: dict(value) for key, value in self.transition_matrix.items()},
            "final_hold_origins": dict(self.final_hold_origins),
            "outcome_by_transition": {
                key: dict(value) for key, value in self.outcome_by_transition.items()
            },
            "persistent_by_transition": {
                key: dict(value) for key, value in self.persistent_by_transition.items()
            },
            "fully_evaluated_source_eligible_count": self.fully_evaluated_source_eligible_count,
            "research_to_trader_matrix": {
                key: dict(value) for key, value in self.research_to_trader_matrix.items()
            },
            "research_to_trader_pm_matrix": {
                key: dict(value) for key, value in self.research_to_trader_pm_matrix.items()
            },
            "research_to_trader_suppression": dict(self.research_to_trader_suppression),
            "research_to_trader_outcomes": {
                key: dict(value) for key, value in self.research_to_trader_outcomes.items()
            },
            "research_to_trader_persistent": {
                key: dict(value) for key, value in self.research_to_trader_persistent.items()
            },
        }


@dataclass(frozen=True, slots=True)
class DecisionPathAuditReport:
    database_path: str
    generated_at: datetime
    recent_hours: float
    strong_miss_points: float
    all_population: DecisionPathPopulation
    recent_population: DecisionPathPopulation
    pm_rating_consistency: Mapping[str, Any]
    research_manager_audit: Mapping[str, Any]
    data_quality: Mapping[str, Any]
    review_status: str = "DIAGNOSTIC ONLY — NO STRATEGY CHANGE PERFORMED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_path": self.database_path,
            "generated_at": self.generated_at.isoformat().replace("+00:00", "Z"),
            "recent_hours": self.recent_hours,
            "strong_miss_points": self.strong_miss_points,
            "review_status": self.review_status,
            "all": self.all_population.to_dict(),
            "recent": self.recent_population.to_dict(),
            "pm_rating_consistency": dict(self.pm_rating_consistency),
            "research_manager_audit": dict(self.research_manager_audit),
            "data_quality": dict(self.data_quality),
        }


@dataclass(frozen=True, slots=True)
class _Decision:
    decision_id: str
    trader_action: str
    final_action: str
    decision_timestamp: datetime | None
    raw_pm_result: Any
    research_manager_recommendation: str | None
    row: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _Evaluation:
    decision_id: str
    horizon_seconds: int
    status: str
    source_context_eligible: bool
    selected_action: str | None
    buy_net_points: float | None
    sell_net_points: float | None
    opportunity_cost: float | None


def _readonly_connection(path: Path, busy_timeout_seconds: float) -> sqlite3.Connection:
    if busy_timeout_seconds <= 0 or not math.isfinite(busy_timeout_seconds):
        raise ValueError("busy_timeout_seconds must be a positive finite number")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise DecisionPathAuditReadError(f"database does not exist: {resolved}")
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
    raise DecisionPathAuditReadError(f"bounded read failed for {resolved}: {last_error}") from last_error


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if exists is None:
        return set()
    escaped = table.replace('"', '""')
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{escaped}")')}


def _require_columns(connection: sqlite3.Connection, table: str, required: set[str]) -> None:
    columns = _table_columns(connection, table)
    if not columns:
        raise DecisionPathAuditSchemaError(f"required table is missing: {table}")
    missing = sorted(required - columns)
    if missing:
        raise DecisionPathAuditSchemaError(
            f"{table} is missing required columns: {', '.join(missing)}"
        )


def _persisted_research_recommendation(row: sqlite3.Row) -> str | None:
    try:
        value = row["research_manager_recommendation"]
    except (IndexError, KeyError):
        return None
    return value if isinstance(value, str) and value in RESEARCH_RECOMMENDATIONS else None


def _load_decisions(connection: sqlite3.Connection) -> list[_Decision]:
    _require_columns(
        connection,
        "shadow_decisions",
        {
            "decision_id",
            "created_at",
            "decision_completed_timestamp",
            "action",
            "decision_context_status",
            "normalization_status",
            "decision_reference_status",
            "raw_portfolio_manager_result",
            "trader_summary",
        },
    )
    _require_columns(connection, "forex_watch_runs", {"decision_id", "run_status", "stale_by_completion"})
    rows = connection.execute(
        """
        SELECT d.*
        FROM shadow_decisions AS d
        WHERE d.decision_context_status = 'COMPLETE'
          AND d.normalization_status = 'NORMALIZED'
          AND d.decision_reference_status = 'AVAILABLE'
          AND EXISTS (
              SELECT 1
              FROM forex_watch_runs AS r
              WHERE r.decision_id = d.decision_id
                AND r.run_status IN ('SUCCEEDED', 'SUCCEEDED_SLOW')
                AND r.stale_by_completion = 0
          )
        ORDER BY COALESCE(d.decision_completed_timestamp, d.created_at), d.decision_id
        """
    ).fetchall()
    return [
        _Decision(
            decision_id=str(row["decision_id"]),
            trader_action=extract_trader_action(row["trader_summary"]),
            final_action=_action_from_persisted(row["action"]),
            decision_timestamp=(
                _parse_utc(row["decision_completed_timestamp"]) or _parse_utc(row["created_at"])
            ),
            raw_pm_result=row["raw_portfolio_manager_result"],
            research_manager_recommendation=_persisted_research_recommendation(row),
            row=dict(row),
        )
        for row in rows
    ]


def _load_evaluations(
    connection: sqlite3.Connection, decision_ids: set[str]
) -> tuple[list[_Evaluation], Counter[str], int]:
    evaluation_columns = _table_columns(connection, "shadow_decision_evaluations")
    _require_columns(
        connection,
        "shadow_decision_evaluations",
        {
            "decision_id",
            "evaluation_basis",
            "horizon_seconds",
            "source_context_eligible",
            "training_eligible",
            "evaluation_status",
            "selected_action",
            "buy_net_points",
            "sell_net_points",
            "hold_opportunity_cost_points",
            "created_at",
        },
    )
    if not decision_ids:
        return [], Counter(), 0
    order_column = "evaluated_at" if "evaluated_at" in evaluation_columns else "created_at"
    rows = connection.execute(
        f"""
        SELECT *
        FROM shadow_decision_evaluations
        WHERE evaluation_basis = 'DECISION_REFERENCE'
        ORDER BY decision_id, horizon_seconds, {order_column}
        """
    ).fetchall()
    status_counts: Counter[str] = Counter()
    training_null_count = 0
    deduplicated: dict[tuple[str, int], _Evaluation] = {}
    for row in rows:
        decision_id = str(row["decision_id"])
        if decision_id not in decision_ids:
            continue
        status = str(row["evaluation_status"] or "UNKNOWN")
        status_counts[status] += 1
        if row["training_eligible"] is None:
            training_null_count += 1
        horizon = int(row["horizon_seconds"])
        deduplicated[(decision_id, horizon)] = _Evaluation(
            decision_id=decision_id,
            horizon_seconds=horizon,
            status=status,
            source_context_eligible=_is_true(row["source_context_eligible"]),
            selected_action=(
                str(row["selected_action"]).upper() if row["selected_action"] is not None else None
            ),
            buy_net_points=_finite(row["buy_net_points"]),
            sell_net_points=_finite(row["sell_net_points"]),
            opportunity_cost=_finite(row["hold_opportunity_cost_points"]),
        )
    return list(deduplicated.values()), status_counts, training_null_count


def _empty_outcome() -> dict[str, float | int | None]:
    return {
        "samples": 0,
        "average_hold_opportunity_cost_points": None,
        "median_hold_opportunity_cost_points": None,
        "p90_hold_opportunity_cost_points": None,
        "p95_hold_opportunity_cost_points": None,
        "buy_better_count": 0,
        "buy_better_frequency": 0.0,
        "sell_better_count": 0,
        "sell_better_frequency": 0.0,
        "positive_directional_count": 0,
        "positive_directional_frequency": 0.0,
    }


def _outcomes(
    decisions: Iterable[_Decision],
    evaluations: Iterable[_Evaluation],
    *,
    key_factory: Any = _decision_transition_key,
) -> dict[str, dict[str, float | int | None]]:
    decision_map = {decision.decision_id: decision for decision in decisions}
    values: dict[str, list[float]] = defaultdict(list)
    buy_counts: Counter[str] = Counter()
    sell_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    for row in evaluations:
        decision = decision_map.get(row.decision_id)
        if decision is None or row.horizon_seconds not in EVALUATION_HORIZONS:
            continue
        if (
            row.status != "COMPLETE"
            or not row.source_context_eligible
            or decision.final_action != "HOLD"
            or row.selected_action != "HOLD"
            or row.buy_net_points is None
            or row.sell_net_points is None
            or row.opportunity_cost is None
        ):
            continue
        key = key_factory(decision)
        values[key].append(row.opportunity_cost)
        if row.buy_net_points > EPSILON and row.buy_net_points > row.sell_net_points + EPSILON:
            buy_counts[key] += 1
        if row.sell_net_points > EPSILON and row.sell_net_points > row.buy_net_points + EPSILON:
            sell_counts[key] += 1
        if row.opportunity_cost > EPSILON:
            positive_counts[key] += 1

    output_keys = (
        _transition_keys()
        if key_factory is _decision_transition_key
        else _research_transition_keys()
    )
    output = {key: _empty_outcome() for key in output_keys}
    for key, costs in values.items():
        stats = _stats(costs)
        count = len(costs)
        output[key].update(
            {
                "samples": count,
                "average_hold_opportunity_cost_points": stats["average"],
                "median_hold_opportunity_cost_points": stats["median"],
                "p90_hold_opportunity_cost_points": stats["p90"],
                "p95_hold_opportunity_cost_points": stats["p95"],
                "buy_better_count": buy_counts[key],
                "buy_better_frequency": buy_counts[key] * 100.0 / count,
                "sell_better_count": sell_counts[key],
                "sell_better_frequency": sell_counts[key] * 100.0 / count,
                "positive_directional_count": positive_counts[key],
                "positive_directional_frequency": positive_counts[key] * 100.0 / count,
            }
        )
    return output


def _persistent_breakdown(
    decisions: Iterable[_Decision],
    evaluations: Iterable[_Evaluation],
    strong_miss_points: float,
    *,
    key_factory: Any = _decision_transition_key,
) -> dict[str, dict[str, int]]:
    decision_map = {decision.decision_id: decision for decision in decisions}
    grouped: dict[str, dict[int, _Evaluation]] = defaultdict(dict)
    for row in evaluations:
        if (
            row.status == "COMPLETE"
            and row.source_context_eligible
            and row.selected_action == "HOLD"
            and row.horizon_seconds in EVALUATION_HORIZONS
            and row.opportunity_cost is not None
        ):
            grouped[row.decision_id][row.horizon_seconds] = row
    output = {
        key: {
            "total": 0,
            "CONSISTENT_HOLD": 0,
            "TRANSIENT_MISS": 0,
            "PERSISTENT_DIRECTIONAL_MISS": 0,
            "STRONG_PERSISTENT_MISS": 0,
            "persistent_directional_miss_total": 0,
        }
        for key in (
            _transition_keys()
            if key_factory is _decision_transition_key
            else _research_transition_keys()
        )
    }
    for decision_id, rows in grouped.items():
        decision = decision_map.get(decision_id)
        if decision is None or decision.final_action != "HOLD" or len(rows) != len(EVALUATION_HORIZONS):
            continue
        key = key_factory(decision)
        classification = classify_cross_horizon(
            (rows[horizon].opportunity_cost for horizon in EVALUATION_HORIZONS),
            strong_miss_points=strong_miss_points,
        )
        output[key]["total"] += 1
        output[key][classification] += 1
        if classification in {"PERSISTENT_DIRECTIONAL_MISS", "STRONG_PERSISTENT_MISS"}:
            output[key]["persistent_directional_miss_total"] += 1
    return output


def _research_key_for_decision(decision: _Decision) -> str:
    return _research_transition_key(decision.research_manager_recommendation, decision.trader_action)


def _research_to_trader_matrix(
    decisions: Iterable[_Decision],
) -> dict[str, dict[str, float | int]]:
    rows = tuple(decisions)
    matrix = {
        key: {"count": 0, "percentage": 0.0}
        for key in _research_transition_keys()
    }
    for decision in rows:
        matrix[_research_key_for_decision(decision)]["count"] += 1
    total = len(rows)
    for values in matrix.values():
        values["percentage"] = values["count"] * 100.0 / total if total else 0.0
    return matrix


def _research_to_trader_pm_matrix(
    decisions: Iterable[_Decision],
) -> dict[str, dict[str, float | int]]:
    rows = tuple(decisions)
    matrix = {
        key: {"count": 0, "percentage": 0.0}
        for key in _research_trader_pm_keys()
    }
    for decision in rows:
        matrix[_research_trader_pm_key(decision)]["count"] += 1
    total = len(rows)
    for values in matrix.values():
        values["percentage"] = values["count"] * 100.0 / total if total else 0.0
    return matrix


def _research_to_trader_suppression(decisions: Iterable[_Decision]) -> dict[str, int]:
    output = {
        _research_transition_key(recommendation, "HOLD"): 0
        for recommendation in RESEARCH_RECOMMENDATIONS
        if recommendation != "HOLD"
    }
    for decision in decisions:
        if (
            decision.research_manager_recommendation in {"BUY", "OVERWEIGHT", "SELL", "UNDERWEIGHT"}
            and decision.trader_action == "HOLD"
        ):
            output[_research_key_for_decision(decision)] += 1
    return output


def _research_suppressed_decisions(decisions: Iterable[_Decision]) -> tuple[_Decision, ...]:
    return tuple(
        decision
        for decision in decisions
        if decision.research_manager_recommendation
        in {"BUY", "OVERWEIGHT", "SELL", "UNDERWEIGHT"}
        and decision.trader_action == "HOLD"
    )


def _population(
    decisions: Iterable[_Decision], evaluations: Iterable[_Evaluation], strong_miss_points: float
) -> DecisionPathPopulation:
    rows = tuple(decisions)
    research_suppressed_rows = _research_suppressed_decisions(rows)
    matrix = {
        key: {"count": 0, "percentage": 0.0}
        for key in _transition_keys()
    }
    for decision in rows:
        matrix[_transition_key(decision.trader_action, decision.final_action)]["count"] += 1
    total = len(rows)
    for value in matrix.values():
        value["percentage"] = value["count"] * 100.0 / total if total else 0.0

    hold_origins = Counter(
        _transition_key(decision.trader_action, decision.final_action)
        for decision in rows
        if decision.final_action == "HOLD"
    )
    by_decision: dict[str, dict[int, _Evaluation]] = defaultdict(dict)
    for evaluation in evaluations:
        if (
            evaluation.status == "COMPLETE"
            and evaluation.source_context_eligible
            and evaluation.selected_action == "HOLD"
            and evaluation.horizon_seconds in EVALUATION_HORIZONS
        ):
            by_decision[evaluation.decision_id][evaluation.horizon_seconds] = evaluation
    fully_evaluated = sum(
        all(horizon in by_decision[decision.decision_id] for horizon in EVALUATION_HORIZONS)
        for decision in rows
    )
    return DecisionPathPopulation(
        population_count=total,
        transition_matrix=matrix,
        final_hold_origins=dict(sorted(hold_origins.items())),
        outcome_by_transition=_outcomes(rows, evaluations),
        persistent_by_transition=_persistent_breakdown(rows, evaluations, strong_miss_points),
        fully_evaluated_source_eligible_count=fully_evaluated,
        research_to_trader_matrix=_research_to_trader_matrix(rows),
        research_to_trader_pm_matrix=_research_to_trader_pm_matrix(rows),
        research_to_trader_suppression=_research_to_trader_suppression(rows),
        research_to_trader_outcomes=_outcomes(
            research_suppressed_rows,
            evaluations,
            key_factory=_research_key_for_decision,
        ),
        research_to_trader_persistent=_persistent_breakdown(
            research_suppressed_rows,
            evaluations,
            strong_miss_points,
            key_factory=_research_key_for_decision,
        ),
    )


def _pm_rating_consistency(decisions: Iterable[_Decision]) -> dict[str, Any]:
    valid_count = mismatch_count = malformed_count = missing_rating_count = invalid_rating_count = 0
    mismatches: list[dict[str, str]] = []
    for decision in decisions:
        raw = decision.raw_pm_result
        if raw is None:
            missing_rating_count += 1
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                malformed_count += 1
                continue
        if not isinstance(raw, Mapping):
            malformed_count += 1
            continue
        if "rating" not in raw:
            missing_rating_count += 1
            continue
        rating_action = _rating_action(raw.get("rating"))
        if rating_action is None:
            invalid_rating_count += 1
            continue
        valid_count += 1
        if decision.final_action != rating_action:
            mismatch_count += 1
            mismatches.append(
                {
                    "decision_id_short": decision.decision_id[:8],
                    "rating_action": rating_action,
                    "final_action": decision.final_action,
                }
            )
    return {
        "valid_count": valid_count,
        "mismatch_count": mismatch_count,
        "malformed_count": malformed_count,
        "missing_rating_count": missing_rating_count,
        "invalid_rating_count": invalid_rating_count,
        "mismatches": mismatches,
    }


def _research_manager_audit(decisions: Iterable[_Decision]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for decision in decisions:
        if decision.research_manager_recommendation in RESEARCH_RECOMMENDATIONS:
            counts[decision.research_manager_recommendation] += 1
    if not counts:
        return {
            "status": "UNAVAILABLE",
            "message": RESEARCH_MANAGER_UNAVAILABLE,
            "counts": {},
        }
    return {
        "status": "AVAILABLE",
        "message": "Research Manager recommendations were counted only from the persisted structured field.",
        "counts": {key: int(counts[key]) for key in RESEARCH_RECOMMENDATIONS},
    }


def audit_decision_path(
    db_path: str | Path,
    *,
    strong_miss_points: float = 5.0,
    recent_hours: float = 24.0,
    busy_timeout_seconds: float = 0.25,
) -> DecisionPathAuditReport:
    """Build one read-only decision-path report from a Phase 5/6 database."""
    if strong_miss_points < 0 or not math.isfinite(strong_miss_points):
        raise ValueError("strong_miss_points must be a non-negative finite number")
    if recent_hours <= 0 or not math.isfinite(recent_hours):
        raise ValueError("recent_hours must be a positive finite number")
    generated_at = datetime.now(UTC)
    connection = _readonly_connection(Path(db_path), busy_timeout_seconds)
    try:
        decisions = _load_decisions(connection)
        evaluations, status_counts, training_null_count = _load_evaluations(
            connection, {decision.decision_id for decision in decisions}
        )
    except sqlite3.Error as exc:
        raise DecisionPathAuditReadError(f"failed to read audit rows: {exc}") from exc
    finally:
        connection.close()

    cutoff = generated_at - timedelta(hours=recent_hours)
    recent_decisions = tuple(
        decision
        for decision in decisions
        if decision.decision_timestamp is not None and decision.decision_timestamp >= cutoff
    )
    recent_ids = {decision.decision_id for decision in recent_decisions}
    all_evaluations = tuple(evaluations)
    recent_evaluations = tuple(row for row in all_evaluations if row.decision_id in recent_ids)
    data_quality: dict[str, Any] = {
        "primary_population": len(decisions),
        "evaluation_rows_by_status": {
            key: int(value) for key, value in sorted(status_counts.items())
        },
        "training_eligible_null_rows": training_null_count,
        "source_context_eligible_complete_rows": sum(
            row.status == "COMPLETE" and row.source_context_eligible for row in all_evaluations
        ),
        "recent_population": len(recent_decisions),
    }
    return DecisionPathAuditReport(
        database_path=str(Path(db_path).expanduser().resolve()),
        generated_at=generated_at,
        recent_hours=recent_hours,
        strong_miss_points=strong_miss_points,
        all_population=_population(decisions, all_evaluations, strong_miss_points),
        recent_population=_population(
            recent_decisions, recent_evaluations, strong_miss_points
        ),
        pm_rating_consistency=_pm_rating_consistency(decisions),
        research_manager_audit=_research_manager_audit(decisions),
        data_quality=data_quality,
    )


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _render_population(population: DecisionPathPopulation) -> list[str]:
    lines = [
        f"population decisions: {population.population_count}",
        f"fully evaluated source-eligible: {population.fully_evaluated_source_eligible_count}",
        "transition | count | percent",
    ]
    for key, values in population.transition_matrix.items():
        lines.append(f"{key} | {values['count']} | {float(values['percentage']):.1f}%")
    return lines


def _render_outcome_map(
    outcome_by_transition: Mapping[str, Mapping[str, float | int | None]],
) -> list[str]:
    lines = [
        "transition | samples | avg cost | median | p90 | p95 | BUY_BETTER | SELL_BETTER | positive directional"
    ]
    for key, values in outcome_by_transition.items():
        if int(values["samples"]) == 0:
            continue
        lines.append(
            f"{key} | {values['samples']} | {_fmt(values['average_hold_opportunity_cost_points'])} | "
            f"{_fmt(values['median_hold_opportunity_cost_points'])} | "
            f"{_fmt(values['p90_hold_opportunity_cost_points'])} | "
            f"{_fmt(values['p95_hold_opportunity_cost_points'])} | "
            f"{values['buy_better_count']} ({float(values['buy_better_frequency']):.1f}%) | "
            f"{values['sell_better_count']} ({float(values['sell_better_frequency']):.1f}%) | "
            f"{values['positive_directional_count']} ({float(values['positive_directional_frequency']):.1f}%)"
        )
    return lines


def _render_outcomes(population: DecisionPathPopulation) -> list[str]:
    return _render_outcome_map(population.outcome_by_transition)


def _render_persistent_map(
    persistent_by_transition: Mapping[str, Mapping[str, int]],
) -> list[str]:
    lines = ["transition | total | CONSISTENT_HOLD | TRANSIENT_MISS | PERSISTENT_DIRECTIONAL_MISS | STRONG"]
    for key, values in persistent_by_transition.items():
        if values["total"] == 0:
            continue
        lines.append(
            f"{key} | {values['total']} | {values['CONSISTENT_HOLD']} | "
            f"{values['TRANSIENT_MISS']} | {values['PERSISTENT_DIRECTIONAL_MISS']} | "
            f"{values['STRONG_PERSISTENT_MISS']}"
        )
    if len(lines) == 1:
        lines.append("No final HOLD transition has all four complete source-eligible horizons.")
    return lines


def _render_persistent(population: DecisionPathPopulation) -> list[str]:
    return _render_persistent_map(population.persistent_by_transition)


def _render_research_matrix(population: DecisionPathPopulation) -> list[str]:
    lines = ["research recommendation -> trader action | count | percent"]
    for key, values in population.research_to_trader_matrix.items():
        lines.append(f"{key} | {values['count']} | {float(values['percentage']):.1f}%")
    return lines


def _render_research_trader_pm_matrix(population: DecisionPathPopulation) -> list[str]:
    lines = ["research recommendation -> trader action -> PM action | count | percent"]
    for key, values in population.research_to_trader_pm_matrix.items():
        lines.append(f"{key} | {values['count']} | {float(values['percentage']):.1f}%")
    return lines


def _render_research_suppression(population: DecisionPathPopulation) -> list[str]:
    lines = ["research directional recommendation -> Trader HOLD | count"]
    for key, value in population.research_to_trader_suppression.items():
        lines.append(f"{key} | {value}")
    return lines


def render_text(report: DecisionPathAuditReport) -> str:
    lines = [
        "DECISION PATH AUDIT",
        f"Database: {report.database_path}",
        f"Review status: {report.review_status}",
        "",
        "Population",
        "ALL",
        *_render_population(report.all_population),
        "",
        "Trader -> PM transition matrix",
        *_render_population(report.all_population)[2:],
        "",
        "Research Manager -> Trader transition matrix",
        *_render_research_matrix(report.all_population),
        "",
        "Research Manager -> Trader -> PM transition matrix",
        *_render_research_trader_pm_matrix(report.all_population),
        "",
        "Research -> Trader suppression",
        *_render_research_suppression(report.all_population),
        "",
        "Final HOLD origins",
    ]
    if report.all_population.final_hold_origins:
        lines.extend(
            f"{key}: {value}" for key, value in report.all_population.final_hold_origins.items()
        )
    else:
        lines.append("None")
    lines.extend(
        [
            "",
            "Outcome opportunity cost by transition",
            *_render_outcomes(report.all_population),
            "",
            "Persistent misses by transition",
            *_render_persistent(report.all_population),
            "",
            "Research -> Trader outcome opportunity cost",
            *_render_outcome_map(report.all_population.research_to_trader_outcomes),
            "Research -> Trader persistent misses",
            *_render_persistent_map(report.all_population.research_to_trader_persistent),
            "",
            "Recent-vs-all comparison",
            f"recent window: last {report.recent_hours:g} UTC hours",
            f"all decisions: {report.all_population.population_count}",
            f"recent decisions: {report.recent_population.population_count}",
            *_render_population(report.recent_population)[2:],
            "recent Research Manager -> Trader transition matrix",
            *_render_research_matrix(report.recent_population),
            "recent Research -> Trader suppression",
            *_render_research_suppression(report.recent_population),
            "recent outcome opportunity cost by transition",
            *_render_outcomes(report.recent_population),
            "recent persistent misses by transition",
            *_render_persistent(report.recent_population),
            "",
            "PM rating consistency",
            ", ".join(
                f"{key}={value}"
                for key, value in report.pm_rating_consistency.items()
                if key != "mismatches"
            ),
            "",
            "Research Manager audit",
            str(report.research_manager_audit.get("message")),
            "",
            "Data quality",
            json.dumps(report.data_quality, sort_keys=True),
            "",
            "DIAGNOSTIC ONLY — NO STRATEGY CHANGE PERFORMED",
        ]
    )
    return "\n".join(lines)


def report_as_json(report: DecisionPathAuditReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "DecisionPathAuditError",
    "DecisionPathAuditReadError",
    "DecisionPathAuditReport",
    "DecisionPathAuditSchemaError",
    "EVALUATION_HORIZONS",
    "FINAL_ACTION_UNAVAILABLE",
    "RESEARCH_MANAGER_UNAVAILABLE",
    "RESEARCH_RECOMMENDATION_UNAVAILABLE",
    "RESEARCH_RECOMMENDATIONS",
    "TRADER_ACTION_UNAVAILABLE",
    "audit_decision_path",
    "classify_cross_horizon",
    "extract_trader_action",
    "percentile",
    "render_text",
    "report_as_json",
]
