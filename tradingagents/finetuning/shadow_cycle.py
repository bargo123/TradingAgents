"""Bounded Phase 11 shadow-data lifecycle orchestration."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tradingagents.datasets.factory import DatasetFactory
from tradingagents.datasets.models import DatasetConfig
from tradingagents.datasets.sources import (
    ReadonlyExperienceSource,
    ReadonlyPhase9AuditSource,
    ReadonlyPhase56Source,
)
from tradingagents.experience.catalog import ExperienceCatalog
from tradingagents.experience.importer import ExperienceImporter, ExperienceRebuilder

from .shadow_audit import build_exclusion_audit, summarize_report

UTC = timezone.utc
CommandRunner = Callable[[Sequence[str], Mapping[str, str]], Any]


class ShadowCycleError(RuntimeError):
    """A bounded collection lifecycle failed closed."""


@dataclass(frozen=True, slots=True)
class ShadowCycleConfig:
    db_path: Path
    phase8_root: Path
    phase10_output_root: Path
    phase9_audit_path: Path | None = None
    phase7_root: Path | None = None
    embedding_model_path: Path | None = None
    runtime_cache_dir: Path | None = None
    terminal_path: str | None = None
    symbol: str = "EURUSD"
    analysis_profile: str = "INTRADAY"
    analysts: tuple[str, ...] = ("market", "news")
    schedule_timeframe: str = "M15"
    horizon_seconds: int = 300
    observation_tolerance_seconds: int = 30
    mode: str = "collect"

    def __post_init__(self) -> None:
        object.__setattr__(self, "db_path", Path(self.db_path).resolve())
        object.__setattr__(self, "phase8_root", Path(self.phase8_root).resolve())
        object.__setattr__(
            self,
            "phase10_output_root",
            Path(self.phase10_output_root).resolve(),
        )
        if self.phase9_audit_path is not None:
            object.__setattr__(self, "phase9_audit_path", Path(self.phase9_audit_path).resolve())
        if self.phase7_root is not None:
            object.__setattr__(self, "phase7_root", Path(self.phase7_root).resolve())
        if self.embedding_model_path is not None:
            object.__setattr__(
                self, "embedding_model_path", Path(self.embedding_model_path).resolve()
            )
        if self.runtime_cache_dir is not None:
            object.__setattr__(self, "runtime_cache_dir", Path(self.runtime_cache_dir).resolve())
        if self.symbol.upper() != "EURUSD":
            raise ValueError("the bounded first lifecycle supports EURUSD only")
        if self.analysts != ("market", "news"):
            raise ValueError("the bounded lifecycle requires market,news analysts")
        if self.horizon_seconds <= 0 or self.observation_tolerance_seconds < 0:
            raise ValueError("horizon and tolerance must be non-negative/positive")
        if self.mode not in {"collect", "evaluate"}:
            raise ValueError("mode must be collect or evaluate")


def _subprocess_runner(command: Sequence[str], env: Mapping[str, str]) -> Any:
    return subprocess.run(
        list(command),
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )


def _fresh_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ShadowCycleError(f"Phase 10 output root must be fresh: {path}")


def _sync_phase8(config: ShadowCycleConfig) -> dict[str, Any]:
    """Import the source DB append-only and publish the active Phase 8 generation."""
    catalog = ExperienceCatalog(config.phase8_root)
    imported = None
    if config.db_path.is_file():
        imported = ExperienceImporter(catalog).import_sources((config.db_path,))
    generation = ExperienceRebuilder(catalog).rebuild()
    records = catalog.active_records()
    with catalog._connect() as db:
        feature_projection_count = int(
            db.execute("SELECT COUNT(*) FROM experience_feature_projections").fetchone()[0]
        )
        evaluation_snapshot_count = int(
            db.execute("SELECT COUNT(*) FROM experience_outcome_snapshots").fetchone()[0]
        )
    return {
        "import": None if imported is None else asdict(imported),
        "generation_id": generation.generation_id,
        "population_fingerprint": generation.population_fingerprint,
        "experience_count": generation.experience_count,
        "trust_counts": dict(sorted(Counter(str(item.trust) for item in records).items())),
        "feature_projection_count": feature_projection_count,
        "evaluation_snapshot_count": evaluation_snapshot_count,
    }


def _environment(config: ShadowCycleConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "KNOWLEDGE_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TRADINGAGENTS_FOREX_EVIDENCE_ENABLED": "1",
            "TRADINGAGENTS_FOREX_EVIDENCE_TIMEOUT_SECONDS": "30",
            "TRADINGAGENTS_FOREX_EVIDENCE_STATISTICS_HORIZON_SECONDS": str(
                config.horizon_seconds
            ),
            "TRADINGAGENTS_FOREX_EVIDENCE_EXPERIENCE_ARTIFACT_ROOT": str(
                config.phase8_root
            ),
        }
    )
    if config.runtime_cache_dir is not None:
        env["TRADINGAGENTS_CACHE_DIR"] = str(config.runtime_cache_dir)
    if config.phase7_root is not None:
        env["TRADINGAGENTS_FOREX_EVIDENCE_KNOWLEDGE_ARTIFACT_ROOT"] = str(
            config.phase7_root
        )
    if config.embedding_model_path is not None:
        env["TRADINGAGENTS_FOREX_EVIDENCE_KNOWLEDGE_EMBEDDING_MODEL_PATH"] = str(
            config.embedding_model_path
        )
    return env


def _watch_command(config: ShadowCycleConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "cli.forex_watch",
        "once",
        "--db-path",
        str(config.db_path),
        "--symbols",
        config.symbol,
        "--analysis-profile",
        config.analysis_profile,
        "--analysts",
        ",".join(config.analysts),
        "--schedule-timeframe",
        config.schedule_timeframe,
    ]
    if config.terminal_path:
        command.extend(("--terminal-path", config.terminal_path))
    return command


def _evaluate_command(config: ShadowCycleConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "cli.forex_evaluate",
        "--pending",
        "--db-path",
        str(config.db_path),
        "--observation-tolerance-seconds",
        str(config.observation_tolerance_seconds),
    ]
    if config.terminal_path:
        command.extend(("--terminal-path", config.terminal_path))
    return command


def _run_checked(
    command: Sequence[str],
    env: Mapping[str, str],
    runner: CommandRunner,
    label: str,
) -> Any:
    result = runner(command, env)
    if getattr(result, "returncode", 1) != 0:
        raise ShadowCycleError(f"{label} failed with exit code {getattr(result, 'returncode', 1)}")
    return result


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _evaluation_due(config: ShadowCycleConfig, now: datetime) -> tuple[bool, str | None]:
    if not config.db_path.is_file():
        raise ShadowCycleError("shadow database is missing")
    try:
        db = sqlite3.connect(
            "file:" + config.db_path.as_posix() + "?mode=ro", uri=True
        )
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT analysis_snapshot_timestamp, decision_reference_timestamp "
            "FROM shadow_decisions ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    except sqlite3.Error as exc:
        raise ShadowCycleError("shadow database cannot be inspected") from exc
    finally:
        with suppress(UnboundLocalError):
            db.close()
    if row is None:
        raise ShadowCycleError("shadow database contains no decision")
    anchor = _parse_timestamp(row["decision_reference_timestamp"])
    if anchor is None:
        anchor = _parse_timestamp(row["analysis_snapshot_timestamp"])
    if anchor is None:
        raise ShadowCycleError("decision has no valid UTC evaluation anchor")
    deadline = anchor + timedelta(
        seconds=config.horizon_seconds + config.observation_tolerance_seconds
    )
    return now >= deadline, deadline.isoformat().replace("+00:00", "Z")


def _build_phase10(config: ShadowCycleConfig) -> dict[str, Any]:
    _fresh_output_root(config.phase10_output_root)
    report = DatasetFactory().build(
        DatasetConfig(
            source_db_paths=(config.db_path,),
            phase8_root=config.phase8_root,
            phase9_audit_path=config.phase9_audit_path,
            output_root=config.phase10_output_root,
            evaluation_basis="ANALYSIS_SNAPSHOT",
            horizon_seconds=config.horizon_seconds,
            allow_empty=True,
        )
    )
    phase56 = ReadonlyPhase56Source(config.db_path).read()
    phase8 = ReadonlyExperienceSource(config.phase8_root / "catalog.sqlite3").read()
    phase9 = (
        ReadonlyPhase9AuditSource(config.phase9_audit_path).read()
        if config.phase9_audit_path is not None
        else ReadonlyPhase9AuditSource(config.phase10_output_root / "missing-audit.sqlite3").read()
    )
    audit = build_exclusion_audit(report, phase56, phase8, phase9)
    summary = summarize_report(report, audit)
    summary["safety"] = DatasetFactory.safety_counters()
    return summary


def run_shadow_cycle(
    config: ShadowCycleConfig,
    *,
    command_runner: CommandRunner | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one bounded collection/evaluation stage with no execution path."""
    if not isinstance(config, ShadowCycleConfig):
        raise TypeError("config must be ShadowCycleConfig")
    _fresh_output_root(config.phase10_output_root)
    runner = command_runner or _subprocess_runner
    env = _environment(config)
    phase8_before = _sync_phase8(config)
    if config.mode == "collect":
        _run_checked(_watch_command(config), env, runner, "forex-watch")
        phase8_after = _sync_phase8(config)
        return {
            "status": "COLLECTED",
            "executed": False,
            "symbol": config.symbol,
            "horizon_seconds": config.horizon_seconds,
            "phase8_before": phase8_before,
            "phase8_after": phase8_after,
            "evaluation_status": "PENDING",
        }

    current = (now or datetime.now(UTC)).astimezone(UTC)
    due, deadline = _evaluation_due(config, current)
    if not due:
        return {
            "status": "PENDING",
            "executed": False,
            "symbol": config.symbol,
            "horizon_seconds": config.horizon_seconds,
            "evaluation_status": "PENDING",
            "evaluation_deadline": deadline,
            "phase8": phase8_before,
        }
    _run_checked(_evaluate_command(config), env, runner, "forex-evaluate")
    phase8_after = _sync_phase8(config)
    phase10 = _build_phase10(config)
    return {
        "status": "EVALUATED",
        "executed": False,
        "symbol": config.symbol,
        "horizon_seconds": config.horizon_seconds,
        "phase8": phase8_after,
        "phase10": phase10,
    }


__all__ = ["ShadowCycleConfig", "ShadowCycleError", "run_shadow_cycle"]
