"""Run one bounded, read-only Phase 11 shadow collection/evaluation stage."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tradingagents.finetuning.shadow_cycle import (
    ShadowCycleConfig,
    ShadowCycleError,
    run_shadow_cycle,
)


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase11-shadow-cycle",
        description="Bounded EURUSD shadow collection and Phase 10 readiness audit.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("collect", "evaluate"):
        command = subparsers.add_parser(mode, help=f"run one {mode} stage")
        command.add_argument("--db-path", required=True)
        command.add_argument("--phase8-root", required=True)
        command.add_argument("--phase10-output-root", required=True)
        command.add_argument("--phase9-audit", default=None)
        command.add_argument("--phase7-root", default=None)
        command.add_argument("--embedding-model-path", default=None)
        command.add_argument("--runtime-cache-dir", default=None)
        command.add_argument("--terminal-path", default=None)
        command.add_argument("--symbol", default="EURUSD")
        command.add_argument("--analysis-profile", default="INTRADAY")
        command.add_argument("--analysts", default="market,news")
        command.add_argument("--schedule-timeframe", default="M15")
        command.add_argument("--horizon-seconds", type=_positive, default=300)
        command.add_argument(
            "--observation-tolerance-seconds", type=_nonnegative, default=30
        )
    return parser


def _paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    phase7 = Path(args.phase7_root).resolve() if args.phase7_root else None
    embedding = (
        Path(args.embedding_model_path).resolve()
        if args.embedding_model_path
        else None
    )
    if phase7 is not None and not phase7.is_dir():
        raise ValueError(f"Phase 7 root does not exist: {phase7}")
    if embedding is not None and not embedding.is_dir():
        raise ValueError(f"embedding model path does not exist: {embedding}")
    phase8 = Path(args.phase8_root).resolve()
    if phase8.exists() and not phase8.is_dir():
        raise ValueError(f"Phase 8 root is not a directory: {phase8}")
    return phase8, phase7, embedding


def _config(args: argparse.Namespace) -> ShadowCycleConfig:
    phase8, phase7, embedding = _paths(args)
    analysts = tuple(item.strip() for item in args.analysts.split(",") if item.strip())
    return ShadowCycleConfig(
        db_path=Path(args.db_path),
        phase8_root=phase8,
        phase10_output_root=Path(args.phase10_output_root),
        phase9_audit_path=Path(args.phase9_audit) if args.phase9_audit else None,
        phase7_root=phase7,
        embedding_model_path=embedding,
        runtime_cache_dir=(
            Path(args.runtime_cache_dir) if args.runtime_cache_dir else None
        ),
        terminal_path=args.terminal_path,
        symbol=args.symbol,
        analysis_profile=args.analysis_profile,
        analysts=analysts,
        schedule_timeframe=args.schedule_timeframe,
        horizon_seconds=args.horizon_seconds,
        observation_tolerance_seconds=args.observation_tolerance_seconds,
        mode=args.mode,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        config = _config(args)
        print("MT5 FOREX — PHASE 11 SHADOW CYCLE")
        print("SHADOW ONLY — NO ORDER WILL BE SENT")
        result = run_shadow_cycle(config)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        phase10 = result.get("phase10") if isinstance(result, dict) else None
        if isinstance(phase10, dict) and phase10.get("status") == "FAILED":
            return 1
        return 0
    except (ShadowCycleError, OSError, ValueError, TypeError) as exc:
        print(
            json.dumps(
                {"status": "FAILED", "error_type": type(exc).__name__},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
