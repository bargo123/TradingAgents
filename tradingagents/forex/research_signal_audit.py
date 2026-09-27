"""Read-only forensic audit of the forex research signal path.

The audit consumes only persisted Phase 5/6 metadata.  It never starts an
MT5 session, calls an LLM, accesses the network, or writes the source DB.
Bull/Bear reports are intentionally treated as prose presence metadata: this
module does not infer a directional stance by searching report text.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .decision_path_audit import extract_trader_action

RESEARCH_RECOMMENDATIONS = ("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL")
UNAVAILABLE = "UNAVAILABLE"
MACRO_UNAVAILABLE = "MACRO/EVENT DATA UNAVAILABLE"
TIMEFRAMES = ("M1", "M5", "M15", "H1")
_SUCCESS_STATUSES = ("SUCCEEDED", "SUCCEEDED_SLOW")


class ResearchSignalAuditError(RuntimeError):
    """Base class for deterministic research-signal audit failures."""


class ResearchSignalAuditReadError(ResearchSignalAuditError):
    """The source database could not be read within the bounded policy."""


class ResearchSignalAuditSchemaError(ResearchSignalAuditError):
    """The source database is missing the metadata read contract."""


@dataclass(frozen=True, slots=True)
class ResearchSignalObservation:
    """Metadata-only observation for one successful shadow decision."""

    decision_id: str
    research_recommendation: str | None
    trader_action: str
    portfolio_manager_action: str
    market_present: bool
    news_present: bool
    bull_present: bool
    bear_present: bool
    macro_event_status: str
    evidence_integration_status: str
    decision_context_status: str
    normalization_status: str
    timeframe_directions: Mapping[str, str | None] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "research_recommendation": self.research_recommendation or UNAVAILABLE,
            "trader_action": self.trader_action,
            "portfolio_manager_action": self.portfolio_manager_action,
            "market_present": self.market_present,
            "news_present": self.news_present,
            "bull_present": self.bull_present,
            "bear_present": self.bear_present,
            "timeframe_directions": dict(self.timeframe_directions or {}),
            "macro_event_status": self.macro_event_status,
            "evidence_integration_status": self.evidence_integration_status,
            "decision_context_status": self.decision_context_status,
            "normalization_status": self.normalization_status,
        }


@dataclass(frozen=True, slots=True)
class ResearchSignalAuditReport:
    database_path: str
    population_count: int
    recommendation_counts: Mapping[str, int]
    trader_action_counts: Mapping[str, int]
    portfolio_manager_action_counts: Mapping[str, int]
    pipeline_counts: Mapping[str, int]
    timeframe_pattern_recommendations: Mapping[str, Mapping[str, int]]
    availability: Mapping[str, Mapping[str, int | float]]
    structured_upstream: Mapping[str, Mapping[str, Any]]
    observations: tuple[ResearchSignalObservation, ...]
    limitations: tuple[str, ...]
    review_status: str = "DIAGNOSTIC ONLY — NO STRATEGY CHANGE PERFORMED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_path": self.database_path,
            "population_count": self.population_count,
            "recommendation_counts": dict(self.recommendation_counts),
            "trader_action_counts": dict(self.trader_action_counts),
            "portfolio_manager_action_counts": dict(self.portfolio_manager_action_counts),
            "pipeline_counts": dict(self.pipeline_counts),
            "timeframe_pattern_recommendations": {
                key: dict(value) for key, value in self.timeframe_pattern_recommendations.items()
            },
            "availability": {key: dict(value) for key, value in self.availability.items()},
            "structured_upstream": {
                key: dict(value) for key, value in self.structured_upstream.items()
            },
            "observations": [observation.to_dict() for observation in self.observations],
            "limitations": list(self.limitations),
            "review_status": self.review_status,
        }


def _readonly_connection(path: Path, busy_timeout_seconds: float) -> sqlite3.Connection:
    if busy_timeout_seconds <= 0 or not math.isfinite(busy_timeout_seconds):
        raise ValueError("busy_timeout_seconds must be a positive finite number")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ResearchSignalAuditReadError(f"database does not exist: {resolved}")
    uri = "file:" + quote(resolved.as_posix(), safe="/:\\") + "?mode=ro"
    last_error: Exception | None = None
    for attempt in range(3):
        connection: sqlite3.Connection | None = None
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
            if connection is not None:
                connection.close()
            last_error = exc
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
    raise ResearchSignalAuditReadError(f"bounded read failed for {resolved}: {last_error}") from last_error


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
        raise ResearchSignalAuditSchemaError(f"required table is missing: {table}")
    missing = sorted(required - columns)
    if missing:
        raise ResearchSignalAuditSchemaError(
            f"{table} is missing required columns: {', '.join(missing)}"
        )


def _canonical_recommendation(value: Any) -> str | None:
    if isinstance(value, str) and value in RESEARCH_RECOMMENDATIONS:
        return value
    return None


def _canonical_action(value: Any) -> str:
    if isinstance(value, str) and value in {"BUY", "HOLD", "SELL"}:
        return value
    return UNAVAILABLE


def _metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _artifact(
    metrics: Mapping[str, Any], node: str, key: str
) -> Mapping[str, Any]:
    result: Mapping[str, Any] = {}
    boundaries = metrics.get("state_boundaries")
    if not isinstance(boundaries, list):
        return result
    for entry in boundaries:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("node") != node or entry.get("phase") != "after":
            continue
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, Mapping):
            candidate = artifacts.get(key)
            if isinstance(candidate, Mapping):
                result = candidate
    return result


def _present(artifact: Mapping[str, Any]) -> bool:
    value = artifact.get("present")
    return value is True or (type(value) is int and value == 1)


def _timeframe_directions(metrics: Mapping[str, Any]) -> dict[str, str | None]:
    features = metrics.get("market_features")
    features = features if isinstance(features, Mapping) else {}
    directions: dict[str, str | None] = {}
    for timeframe in TIMEFRAMES:
        section = features.get(timeframe)
        direction = section.get("direction") if isinstance(section, Mapping) else None
        directions[timeframe] = (
            str(direction).strip().upper()
            if isinstance(direction, str) and direction.strip()
            else None
        )
    return directions


def _timeframe_pattern(directions: Mapping[str, str | None]) -> str:
    return " / ".join(
        str(directions.get(timeframe) or UNAVAILABLE) for timeframe in TIMEFRAMES
    )


def _load_observations(connection: sqlite3.Connection) -> tuple[ResearchSignalObservation, ...]:
    _require_columns(
        connection,
        "shadow_decisions",
        {
            "decision_id",
            "action",
            "research_manager_recommendation",
            "trader_summary",
            "decision_context_status",
            "normalization_status",
        },
    )
    _require_columns(connection, "forex_watch_runs", {"decision_id", "run_status", "metrics_json"})
    rows = connection.execute(
        """
        SELECT d.decision_id, d.action, d.research_manager_recommendation,
               d.trader_summary, d.decision_context_status, d.normalization_status,
               r.metrics_json
        FROM shadow_decisions AS d
        JOIN forex_watch_runs AS r ON r.decision_id = d.decision_id
        WHERE r.run_status IN ('SUCCEEDED', 'SUCCEEDED_SLOW')
        ORDER BY d.decision_id, r.rowid DESC
        """
    ).fetchall()
    observations: list[ResearchSignalObservation] = []
    seen: set[str] = set()
    for row in rows:
        decision_id = str(row["decision_id"])
        if decision_id in seen:
            continue
        seen.add(decision_id)
        metrics = _metrics(row["metrics_json"])
        recommendation = _canonical_recommendation(row["research_manager_recommendation"])
        observations.append(
            ResearchSignalObservation(
                decision_id=decision_id,
                research_recommendation=recommendation,
                trader_action=extract_trader_action(row["trader_summary"]),
                portfolio_manager_action=_canonical_action(row["action"]),
                market_present=_present(_artifact(metrics, "Market Analyst", "market")),
                news_present=_present(_artifact(metrics, "News Analyst", "news")),
                bull_present=_present(_artifact(metrics, "Bull Researcher", "bull")),
                bear_present=_present(_artifact(metrics, "Bear Researcher", "bear")),
                timeframe_directions=_timeframe_directions(metrics),
                macro_event_status=str(metrics.get("macro_event_status") or UNAVAILABLE),
                evidence_integration_status=str(
                    metrics.get("evidence_integration_status") or UNAVAILABLE
                ),
                decision_context_status=str(row["decision_context_status"] or UNAVAILABLE),
                normalization_status=str(row["normalization_status"] or UNAVAILABLE),
            )
        )
    return tuple(observations)


def _condition_stats(
    observations: tuple[ResearchSignalObservation, ...],
    predicate,
) -> dict[str, int | float]:
    rows = [
        observation
        for observation in observations
        if observation.research_recommendation is not None and predicate(observation)
    ]
    holds = sum(observation.research_recommendation == "HOLD" for observation in rows)
    return {
        "samples": len(rows),
        "holds": holds,
        "hold_rate": (holds / len(rows)) if rows else 0.0,
    }


def audit_research_signals(
    db_path: str | Path,
    *,
    busy_timeout_seconds: float = 0.25,
) -> ResearchSignalAuditReport:
    """Read persisted signal metadata using a bounded, query-only connection."""
    connection = _readonly_connection(Path(db_path), busy_timeout_seconds)
    try:
        observations = _load_observations(connection)
    except sqlite3.Error as exc:
        raise ResearchSignalAuditReadError(f"failed to read audit rows: {exc}") from exc
    finally:
        connection.close()

    recommendation_counts = Counter(
        observation.research_recommendation or UNAVAILABLE for observation in observations
    )
    trader_counts = Counter(observation.trader_action for observation in observations)
    pm_counts = Counter(observation.portfolio_manager_action for observation in observations)
    pipeline_counts = Counter(
        f"{observation.research_recommendation or UNAVAILABLE}"
        f"->{observation.trader_action}->{observation.portfolio_manager_action}"
        for observation in observations
    )
    timeframe_pattern_recommendations: dict[str, Counter[str]] = {}
    for observation in observations:
        pattern = _timeframe_pattern(observation.timeframe_directions or {})
        counts = timeframe_pattern_recommendations.setdefault(pattern, Counter())
        counts[observation.research_recommendation or UNAVAILABLE] += 1
    availability = {
        "macro_unavailable": _condition_stats(
            observations,
            lambda observation: observation.macro_event_status.casefold()
            == MACRO_UNAVAILABLE.casefold(),
        ),
        "macro_available": _condition_stats(
            observations,
            lambda observation: observation.macro_event_status.casefold()
            != MACRO_UNAVAILABLE.casefold()
            and observation.macro_event_status != UNAVAILABLE,
        ),
        "news_unavailable": _condition_stats(observations, lambda observation: not observation.news_present),
        "news_available": _condition_stats(observations, lambda observation: observation.news_present),
        "market_unavailable": _condition_stats(
            observations, lambda observation: not observation.market_present
        ),
        "market_available": _condition_stats(
            observations, lambda observation: observation.market_present
        ),
        "all_expected_inputs_present": _condition_stats(
            observations,
            lambda observation: observation.market_present
            and observation.news_present
            and observation.macro_event_status.casefold() != MACRO_UNAVAILABLE.casefold()
            and observation.macro_event_status != UNAVAILABLE,
        ),
    }
    structured_upstream = {
        key: {
            "available": False,
            "reason": (
                "The persisted graph contract stores Bull/Bear as prose reports; "
                "no structured directional stance is available and prose keywords are not parsed."
            ),
        }
        for key in ("bull", "bear")
    }
    limitations = (
        "Historical rows expose report presence/size metadata, not report text or token finish status.",
        "Macro/event availability is persisted as a run-level status; correlation is not causation.",
    )
    return ResearchSignalAuditReport(
        database_path=str(Path(db_path).expanduser().resolve()),
        population_count=len(observations),
        recommendation_counts={
            key: int(recommendation_counts.get(key, 0))
            for key in (*RESEARCH_RECOMMENDATIONS, UNAVAILABLE)
        },
        trader_action_counts={key: int(trader_counts.get(key, 0)) for key in ("BUY", "HOLD", "SELL", UNAVAILABLE)},
        portfolio_manager_action_counts={
            key: int(pm_counts.get(key, 0)) for key in ("BUY", "HOLD", "SELL", UNAVAILABLE)
        },
        pipeline_counts=dict(sorted(pipeline_counts.items())),
        timeframe_pattern_recommendations={
            key: {label: int(count) for label, count in sorted(values.items())}
            for key, values in sorted(timeframe_pattern_recommendations.items())
        },
        availability=availability,
        structured_upstream=structured_upstream,
        observations=observations,
        limitations=limitations,
    )


def report_as_json(report: ResearchSignalAuditReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def render_text(report: ResearchSignalAuditReport) -> str:
    lines = [
        "FOREX RESEARCH SIGNAL AUDIT (READ ONLY)",
        f"database: {report.database_path}",
        f"population: {report.population_count}",
        f"research recommendations: {dict(report.recommendation_counts)}",
        f"trader actions: {dict(report.trader_action_counts)}",
        f"portfolio-manager actions: {dict(report.portfolio_manager_action_counts)}",
        "pipeline: " + ", ".join(f"{key}={value}" for key, value in report.pipeline_counts.items()),
        "timeframe pattern -> Research recommendation: "
        + "; ".join(
            f"{pattern}={dict(values)}"
            for pattern, values in report.timeframe_pattern_recommendations.items()
        ),
        "availability (structured Research Manager rows only):",
    ]
    for key, value in report.availability.items():
        lines.append(
            f"  {key}: samples={value['samples']} holds={value['holds']} "
            f"hold_rate={float(value['hold_rate']):.3f}"
        )
    lines.extend(
        [
            "structured Bull/Bear stance: unavailable (prose is not keyword-parsed)",
            "limitations: " + " | ".join(report.limitations),
            report.review_status,
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ResearchSignalAuditError",
    "ResearchSignalAuditReadError",
    "ResearchSignalAuditSchemaError",
    "ResearchSignalObservation",
    "ResearchSignalAuditReport",
    "audit_research_signals",
    "report_as_json",
    "render_text",
]
