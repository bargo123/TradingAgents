"""Run one explicit read-only Forex replay performance benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tradingagents.forex.performance import (
    BenchmarkConfig,
    BenchmarkError,
    benchmark_models,
    run_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex-benchmark",
        description="Benchmark one saved Forex snapshot without writing live state.",
    )
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--evidence", action="store_true")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=ROLE:MODEL",
        help="Run sequential model candidate(s), for example 2b=quick:qwen3.5:2b",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = BenchmarkConfig(
            source_database_path=Path(args.db_path),
            source_run_id=args.run_id,
            output_path=Path(args.output),
            provider=args.provider,
            evidence_enabled=args.evidence,
        )
        candidates: dict[str, dict[str, str]] = {}
        for value in args.candidate:
            try:
                label, remainder = value.split("=", 1)
                role, model = remainder.split(":", 1)
            except ValueError as exc:
                raise ValueError("--candidate must be LABEL=ROLE:MODEL") from exc
            if not label or not role or not model:
                raise ValueError("--candidate must be LABEL=ROLE:MODEL")
            candidates[label] = {role: model}
        if candidates:
            results = benchmark_models(config, candidates)
            payload = [result.to_dict() for result in results]
            if args.json:
                print(json.dumps(payload, sort_keys=True, default=str))
            else:
                print("FOREX REPLAY MODEL BENCHMARK")
                for result in results:
                    print(f"{result.candidate}: {result.replay_status} {result.elapsed_seconds:.3f}s")
                print("AUTO-SELECTION: false")
                print("READ-ONLY: true")
            return 0
        report = run_benchmark(config)
    except (BenchmarkError, ValueError, TypeError) as exc:
        print(f"FOREX BENCHMARK ERROR: {exc}")
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), sort_keys=True, default=str))
    else:
        print("FOREX REPLAY BENCHMARK")
        print(f"RUN: {report.source_run_id}")
        print(f"STATUS: {report.replay_status}")
        print(f"RUNTIME: {report.metrics.get('elapsed_seconds', 'unknown')}")
        print(f"ACTION: {report.normalized_action or 'none'}")
        print("READ-ONLY: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
