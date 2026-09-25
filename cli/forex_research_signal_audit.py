"""CLI for the read-only forex research-signal audit."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

from tradingagents.forex.research_signal_audit import (
    ResearchSignalAuditError,
    audit_research_signals,
    render_text,
    report_as_json,
)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex_research_signal_audit",
        description="Read-only audit of the forex analyst-to-research signal path.",
    )
    parser.add_argument("--db-path", required=True, help="Phase 5/6 shadow SQLite database")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--busy-timeout-seconds",
        type=_positive_float,
        default=0.25,
        help="bounded SQLite busy timeout (default: 0.25)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        report = audit_research_signals(
            args.db_path,
            busy_timeout_seconds=args.busy_timeout_seconds,
        )
    except (ResearchSignalAuditError, OSError, ValueError, TypeError) as exc:
        if argv is not None and "--json" in argv:
            import json

            print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        else:
            print(f"RESEARCH SIGNAL AUDIT ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(report_as_json(report) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
