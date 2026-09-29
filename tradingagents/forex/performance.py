"""Read-only replay and performance benchmark contracts for Forex shadow runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot

from .evidence_replay import SavedSnapshotCodec, SnapshotReplayError
from .runner import ForexShadowRunner


class BenchmarkError(RuntimeError):
    """Raised when a replay case cannot be loaded or benchmarked safely."""


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """One immutable, causally bounded source snapshot for replay."""

    source_database_path: Path
    source_run_id: str
    snapshot: ForexMarketSnapshot
    snapshot_bytes: bytes
    symbol: str
    analysis_date: str
    case_fingerprint: str


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Explicit options for one read-only replay benchmark."""

    source_database_path: Path
    source_run_id: str
    output_path: Path
    provider: str | None = None
    models: Mapping[str, Any] = field(default_factory=dict)
    model_settings: Mapping[str, Any] = field(default_factory=dict)
    runner_config: Mapping[str, Any] = field(default_factory=dict)
    evidence_enabled: bool = False
    runner_factory: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        for name in ("source_database_path", "output_path"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        if not isinstance(self.source_run_id, str) or not self.source_run_id.strip():
            raise ValueError("source_run_id must be a non-empty string")
        if not isinstance(self.models, Mapping) or not isinstance(self.model_settings, Mapping):
            raise ValueError("models and model_settings must be mappings")
        if not isinstance(self.runner_config, Mapping):
            raise ValueError("runner_config must be a mapping")
        if self.runner_factory is not None and not callable(self.runner_factory):
            raise TypeError("runner_factory must be callable")


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Scalar-only output from one replay invocation."""

    report_version: int
    run_kind: str
    source_run_id: str
    case_fingerprint: str
    configuration_fingerprint: str
    symbol: str
    analysis_timestamp: str
    replay_status: str
    normalized_action: str | None
    normalization_status: str | None
    decision_context_status: str | None
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelBenchmarkResult:
    """One factual model candidate result; candidates are never auto-selected."""

    candidate: str
    model_config: Mapping[str, Any]
    case_fingerprint: str
    configuration_fingerprint: str
    elapsed_seconds: float
    replay_status: str
    normalized_action: str | None
    normalization_status: str | None
    decision_context_status: str | None
    metrics: Mapping[str, Any]
    failure_category: str | None = None
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Scalar same-case comparison; invalid inputs never produce deltas."""

    valid: bool
    reason: str | None
    case_fingerprint: str | None
    configuration_fingerprint: str | None
    deltas: Mapping[str, float]
    validity: Mapping[str, Any]
    safety: Mapping[str, Any]
    action_transition: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_only_row(database: Path, run_id: str) -> dict[str, Any]:
    if not database.is_file():
        raise BenchmarkError(f"source database does not exist: {database}")
    resolved = database.resolve()
    uri = f"file:{resolved.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            tables = {
                str(item[0])
                for item in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            row = None
            if "forex_watch_runs" in tables:
                columns = {
                    str(item[1])
                    for item in connection.execute('PRAGMA table_info("forex_watch_runs")')
                }
                if "snapshot_json" in columns:
                    row = connection.execute(
                        "SELECT * FROM forex_watch_runs WHERE run_id = ?", (run_id,)
                    ).fetchone()
                elif "shadow_decisions" in tables and "decision_id" in columns:
                    order_clause = (
                        "ORDER BY r.attempt_number DESC"
                        if "attempt_number" in columns
                        else ""
                    )
                    row = connection.execute(
                        f"""
                        SELECT d.*, r.run_id AS replay_run_id, r.run_status AS replay_run_status
                        FROM forex_watch_runs AS r
                        JOIN shadow_decisions AS d ON d.decision_id = r.decision_id
                        WHERE r.run_id = ? OR r.decision_id = ? OR d.decision_id = ?
                        {order_clause}
                        LIMIT 1
                        """,
                        (run_id, run_id, run_id),
                    ).fetchone()
            if row is not None:
                result = dict(row)
                if "replay_run_status" in result:
                    result["run_status"] = result.pop("replay_run_status")
                if "replay_run_id" in result:
                    result["run_id"] = result.pop("replay_run_id")
                return result
    except sqlite3.Error as exc:
        raise BenchmarkError("could not read forex_watch_runs in read-only mode") from exc
    raise BenchmarkError(f"source run not found: {run_id}")


def load_replay_case(database: str | Path, run_id: str) -> ReplayCase:
    """Load one completed causal snapshot without opening a writable connection."""

    database_path = Path(database)
    row = _read_only_row(database_path, run_id)
    if str(row.get("run_status", "")).upper() != "SUCCEEDED":
        raise BenchmarkError("replay source run must be SUCCEEDED")
    encoded = row.get("snapshot_json")
    if isinstance(encoded, bytes):
        snapshot_bytes = encoded
    elif isinstance(encoded, str):
        snapshot_bytes = encoded.encode("utf-8")
    else:
        raise BenchmarkError("source snapshot_json must be text or bytes")
    try:
        snapshot = SavedSnapshotCodec.from_source_row(row)
    except SnapshotReplayError as exc:
        raise BenchmarkError("source snapshot failed causal replay validation") from exc
    if snapshot.timestamp > datetime.now(timezone.utc):
        raise BenchmarkError("future snapshot cannot be used for replay")
    return ReplayCase(
        source_database_path=database_path,
        source_run_id=run_id,
        snapshot=snapshot,
        snapshot_bytes=snapshot_bytes,
        symbol=snapshot.symbol,
        analysis_date=snapshot.timestamp.date().isoformat(),
        case_fingerprint=hashlib.sha256(snapshot_bytes).hexdigest(),
    )


def _scalar_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    allowed = {
        "elapsed_seconds",
        "llm_calls",
        "tool_calls",
        "tokens_in",
        "tokens_out",
        "reasoning_tokens",
        "telemetry_status",
        "stage_timings",
        "critical_path",
        "agents",
        "signal_path",
        "reference_poll_attempts",
        "reference_wait_seconds",
        "final_reference_delay_seconds",
        "reference_status",
    }
    result: dict[str, Any] = {}
    for key in allowed:
        item = value.get(key)
        if key == "critical_path" and isinstance(item, Mapping):
            result[key] = {
                str(metric): metric_value
                for metric, metric_value in item.items()
                if isinstance(metric_value, (str, int, float, bool)) or metric_value is None
            }
        elif key in {"stage_timings", "agents"} and isinstance(item, Mapping):
            result[key] = {
                str(name): {
                    str(metric): metric_value
                    for metric, metric_value in details.items()
                    if isinstance(metric_value, (str, int, float, bool)) or metric_value is None
                }
                for name, details in item.items()
                if isinstance(name, str) and isinstance(details, Mapping)
            }
        elif key == "signal_path" and isinstance(item, Mapping):
            result[key] = {
                str(name): str(signal)
                for name, signal in item.items()
                if isinstance(name, str) and isinstance(signal, (str, int, float, bool))
            }
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = item
    return result


def _write_report(path: Path, report: BenchmarkReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":"))
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        temporary = Path(handle.name)
    temporary.replace(path)


def _configuration_fingerprint(config: BenchmarkConfig, model_config: Mapping[str, Any]) -> str:
    """Hash only replay/model scalar settings; never persist prompts or responses."""

    payload = {
        "provider": config.provider,
        "models": {str(key): str(value) for key, value in sorted(model_config.items())},
        "model_settings": {
            str(key): value
            for key, value in sorted(config.model_settings.items())
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report_number(report: BenchmarkReport | ModelBenchmarkResult, key: str) -> float | None:
    value = report.metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _report_critical_path(report: BenchmarkReport | ModelBenchmarkResult) -> float | None:
    value = report.metrics.get("critical_path")
    if not isinstance(value, Mapping):
        return None
    candidate = value.get("critical_path_seconds")
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return None
    if not math.isfinite(float(candidate)):
        return None
    return float(candidate)


def compare_benchmarks(
    baseline: BenchmarkReport | ModelBenchmarkResult,
    candidate: BenchmarkReport | ModelBenchmarkResult,
) -> ComparisonReport:
    """Compare two reports only when causal case/configuration identity matches."""

    if baseline.case_fingerprint != candidate.case_fingerprint:
        return ComparisonReport(
            valid=False,
            reason="CASE_FINGERPRINT_MISMATCH",
            case_fingerprint=None,
            configuration_fingerprint=None,
            deltas={},
            validity={},
            safety={"replay_only": True},
            action_transition=None,
        )
    if baseline.configuration_fingerprint != candidate.configuration_fingerprint:
        return ComparisonReport(
            valid=False,
            reason="CONFIGURATION_FINGERPRINT_MISMATCH",
            case_fingerprint=baseline.case_fingerprint,
            configuration_fingerprint=None,
            deltas={},
            validity={},
            safety={"replay_only": True},
            action_transition=None,
        )
    deltas: dict[str, float] = {}
    for key in ("elapsed_seconds", "llm_calls", "tool_calls", "tokens_in", "tokens_out"):
        before = _report_number(baseline, key)
        after = _report_number(candidate, key)
        if before is not None and after is not None:
            deltas[key] = after - before
    baseline_path = _report_critical_path(baseline)
    candidate_path = _report_critical_path(candidate)
    if baseline_path is not None and candidate_path is not None:
        deltas["critical_path_seconds"] = candidate_path - baseline_path
    validity = {
        "baseline_replay": baseline.replay_status,
        "candidate_replay": candidate.replay_status,
        "baseline_context": baseline.decision_context_status,
        "candidate_context": candidate.decision_context_status,
        "baseline_normalization": baseline.normalization_status,
        "candidate_normalization": candidate.normalization_status,
    }
    valid = baseline.replay_status == candidate.replay_status == "COMPLETE"
    reason = None if valid else "REPLAY_STATUS_MISMATCH"
    transition = f"{baseline.normalized_action or 'UNAVAILABLE'}->{candidate.normalized_action or 'UNAVAILABLE'}"
    return ComparisonReport(
        valid=valid,
        reason=reason,
        case_fingerprint=baseline.case_fingerprint,
        configuration_fingerprint=baseline.configuration_fingerprint,
        deltas=deltas,
        validity=validity,
        safety={"replay_only": True, "source_writes": False, "order_calls": False},
        action_transition=transition,
    )


def directional_distribution(
    reports: Sequence[BenchmarkReport | ModelBenchmarkResult],
) -> dict[str, Any]:
    """Summarize scalar research/trader/PM signals without outcomes or labels."""

    from collections import Counter

    research: Counter[str] = Counter()
    trader: Counter[str] = Counter()
    portfolio: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    skipped = 0
    for report in reports:
        signal_path = report.metrics.get("signal_path")
        if not isinstance(signal_path, Mapping):
            skipped += 1
            continue
        values = {
            "research": signal_path.get("research_manager_recommendation"),
            "trader": signal_path.get("trader_action"),
            "portfolio": signal_path.get("portfolio_manager_rating"),
        }
        if isinstance(values["research"], str) and values["research"]:
            research[values["research"]] += 1
        if isinstance(values["trader"], str) and values["trader"]:
            trader[values["trader"]] += 1
        if isinstance(values["portfolio"], str) and values["portfolio"]:
            portfolio[values["portfolio"]] += 1
        if report.normalized_action:
            actions[report.normalized_action] += 1
        transitions[
            "->".join(str(values[key]) if values[key] is not None else "UNAVAILABLE" for key in ("research", "trader", "portfolio"))
        ] += 1
    return {
        "count": len(reports),
        "skipped": skipped,
        "research_manager_recommendations": dict(sorted(research.items())),
        "trader_actions": dict(sorted(trader.items())),
        "portfolio_manager_ratings": dict(sorted(portfolio.items())),
        "normalized_actions": dict(sorted(actions.items())),
        "transitions": dict(sorted(transitions.items())),
        "hold_count": actions.get("HOLD", 0),
    }


def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    """Run one explicit replay and write only a scalar benchmark artifact."""

    case = load_replay_case(config.source_database_path, config.source_run_id)
    runner = (
        config.runner_factory(config=config, case=case)
        if config.runner_factory is not None
        else ForexShadowRunner(config=config.runner_config)
    )
    result = runner.analyze(
        symbol=case.symbol,
        analysis_date=case.analysis_date,
        source_run_id=case.source_run_id,
        snapshot=case.snapshot,
        snapshot_bytes=case.snapshot_bytes,
        evidence_enabled=config.evidence_enabled,
        provider=config.provider,
        models=config.models,
        model_settings=config.model_settings,
        persist=False,
    )
    raw_metrics = getattr(result, "metrics", {})
    metrics = _scalar_metrics(raw_metrics)
    report = BenchmarkReport(
        report_version=1,
        run_kind="REPLAY",
        source_run_id=case.source_run_id,
        case_fingerprint=case.case_fingerprint,
        configuration_fingerprint=_configuration_fingerprint(config, config.models),
        symbol=case.symbol,
        analysis_timestamp=case.snapshot.timestamp.isoformat(),
        replay_status="COMPLETE",
        normalized_action=getattr(result, "normalized_action", None),
        normalization_status=getattr(result, "normalization_status", None),
        decision_context_status=getattr(result, "decision_context_status", None),
        metrics=metrics,
    )
    _write_report(config.output_path, report)
    return report


def benchmark_models(
    config: BenchmarkConfig,
    candidates: Mapping[str, Mapping[str, Any]],
) -> tuple[ModelBenchmarkResult, ...]:
    """Run explicit model candidates sequentially on one replay case."""

    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates must be a non-empty mapping")
    case_fingerprint = load_replay_case(
        config.source_database_path, config.source_run_id
    ).case_fingerprint
    results: list[ModelBenchmarkResult] = []
    for raw_candidate, model_config in candidates.items():
        candidate = str(raw_candidate).strip()
        if not candidate:
            raise ValueError("candidate names must be non-empty")
        if not isinstance(model_config, Mapping):
            raise ValueError("each candidate model config must be a mapping")
        candidate_output = config.output_path.with_name(
            f"{config.output_path.stem}.{re.sub(r'[^A-Za-z0-9_.-]+', '_', candidate)}.json"
        )
        candidate_config = replace(
            config,
            output_path=candidate_output,
            models=dict(model_config),
        )
        started = time.perf_counter()
        try:
            report = run_benchmark(candidate_config)
        except Exception as exc:  # noqa: BLE001 — candidate failures are reported, not retried
            results.append(
                ModelBenchmarkResult(
                    candidate=candidate,
                    model_config=dict(model_config),
                    case_fingerprint=case_fingerprint,
                    configuration_fingerprint=_configuration_fingerprint(config, model_config),
                    elapsed_seconds=max(0.0, time.perf_counter() - started),
                    replay_status="FAILED",
                    normalized_action=None,
                    normalization_status=None,
                    decision_context_status=None,
                    metrics={},
                    failure_category=type(exc).__name__,
                    selected=False,
                )
            )
            continue
        results.append(
            ModelBenchmarkResult(
                candidate=candidate,
                model_config=dict(model_config),
                case_fingerprint=report.case_fingerprint,
                configuration_fingerprint=report.configuration_fingerprint,
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                replay_status=report.replay_status,
                normalized_action=report.normalized_action,
                normalization_status=report.normalization_status,
                decision_context_status=report.decision_context_status,
                metrics=report.metrics,
                selected=False,
            )
        )
    return tuple(results)


__all__ = [
    "BenchmarkConfig",
    "BenchmarkError",
    "BenchmarkReport",
    "ComparisonReport",
    "ModelBenchmarkResult",
    "ReplayCase",
    "benchmark_models",
    "compare_benchmarks",
    "directional_distribution",
    "load_replay_case",
    "run_benchmark",
]
