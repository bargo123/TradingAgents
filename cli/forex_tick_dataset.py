"""Read-only audit for the durable Phase 12 tick dataset."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence

from tradingagents.forex.hft.dataset import read_tick_dataset


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex_tick_dataset",
        description="Read-only quality audit for a persisted Phase 12 HFT tick ledger.",
    )
    parser.add_argument("audit", nargs="?", default="audit", choices=("audit",))
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--large-gap-seconds", type=_positive_float, default=300.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = read_tick_dataset(
            args.db_path,
            symbol=args.symbol,
            large_gap_seconds=args.large_gap_seconds,
        )
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        print(f"FOREX TICK DATASET ERROR: {type(exc).__name__}: {exc}")
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), sort_keys=True))
        return 0
    print("FOREX TICK DATASET AUDIT")
    print(f"DATABASE: {report.database_path}")
    print(f"SYMBOL: {report.symbol or '-'}")
    print(f"STATUS: {report.quality_status}")
    print(f"TOTAL/UNIQUE/DUPLICATES: {report.total_ticks}/{report.unique_ticks}/{report.duplicate_ticks}")
    print(f"VALID/INVALID: {report.valid_ticks}/{report.invalid_ticks}")
    print(f"FIRST/LAST UTC: {report.first_timestamp or '-'}/{report.last_timestamp or '-'}")
    print(f"DURATION/TICKS PER SECOND: {report.duration_seconds or '-'} / {report.ticks_per_second or '-'}")
    print(f"DAYS: {', '.join(report.days) or '-'}")
    print(f"SESSIONS: {', '.join(report.sessions) or '-'}")
    print(f"LARGE GAPS/MAX GAP SECONDS: {report.large_gap_count}/{report.max_gap_seconds or '-'}")
    print(f"EXECUTED ROWS: {report.executed_rows}")
    print("READ-ONLY: true")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
