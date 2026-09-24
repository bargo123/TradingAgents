"""One-shot, read-only Trader-to-Portfolio-Manager decision-path audit."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

from tradingagents.forex.decision_path_audit import (
    DecisionPathAuditError,
    audit_decision_path,
    render_text,
    report_as_json,
)


def _finite_non_negative(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _finite_positive(value: str) -> float:
    parsed = _finite_non_negative(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex_decision_path_audit",
        description="Read-only Trader-to-Portfolio-Manager signal-loss audit.",
    )
    parser.add_argument("--db-path", required=True, help="Phase 5/6 shadow SQLite database")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--strong-miss-points",
        type=_finite_non_negative,
        default=5.0,
        help="minimum points for a strong persistent miss (default: 5)",
    )
    parser.add_argument(
        "--recent-hours",
        type=_finite_positive,
        default=24.0,
        help="UTC window for the recent population (default: 24)",
    )
    parser.add_argument(
        "--busy-timeout-seconds",
        type=_finite_positive,
        default=0.25,
        help="bounded SQLite busy timeout (default: 0.25)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        report = audit_decision_path(
            args.db_path,
            strong_miss_points=args.strong_miss_points,
            recent_hours=args.recent_hours,
            busy_timeout_seconds=args.busy_timeout_seconds,
        )
    except (DecisionPathAuditError, OSError, ValueError, TypeError) as exc:
        if argv is not None and "--json" in argv:
            import json

            print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        else:
            print(f"DECISION PATH AUDIT ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(report_as_json(report))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
