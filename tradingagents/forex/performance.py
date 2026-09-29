"""Read-only replay and performance benchmark contracts for Forex shadow runs."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import time
from collections.abc import Callable, Mapping
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


def _read_only_row(database: Path, run_id: str) -> dict[str, Any]:
    if not database.is_file():
        raise BenchmarkError(f"source database does not exist: {database}")
    resolved = database.resolve()
    uri = f"file:{resolved.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM forex_watch_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
    except sqlite3.Error as exc:
        raise BenchmarkError("could not read forex_watch_runs in read-only mode") from exc
    if row is None:
        raise BenchmarkError(f"source run not found: {run_id}")
    return dict(row)


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
        if key in {"stage_timings", "agents"} and isinstance(item, Mapping):
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
                configuration_fingerprint=_configuration_fingerprint(config, model_config),
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
    "ModelBenchmarkResult",
    "ReplayCase",
    "benchmark_models",
    "load_replay_case",
    "run_benchmark",
]
