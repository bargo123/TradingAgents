"""Read-only revision-scoped forex validation CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from tradingagents.forex.revision_validation import validate_revision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex-validate-revision",
        description="Report only read-only forex runs produced by one commit.",
    )
    parser.add_argument("--db-path", default="data_cache/shadow_decisions.db")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_revision(args.db_path, args.commit)
    except Exception as exc:  # CLI should be deterministic and visible
        print(f"FOREX REVISION VALIDATION ERROR: {exc}")
        return 1
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print("FOREX REVISION VALIDATION")
        print(f"COMMIT: {report['commit']}")
        print(f"RUNS: {report['run_count']}")
        print(f"STATUS: {report['run_status_counts']}")
        print(f"RUNTIME: {report['runtime_seconds']}")
        print(f"RESEARCH: {report['research_recommendations']}")
        print(f"TRADER: {report['trader_actions']}")
        print(f"PORTFOLIO MANAGER: {report['portfolio_manager_actions']}")
        print(f"TEMPORAL: {report['temporal_status_counts']}")
        print(f"AGENT METRICS: {report['agent_metrics']}")
        print(f"LATENCY BOTTLENECKS: {report['latency_bottlenecks']}")
        print("READ-ONLY: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
