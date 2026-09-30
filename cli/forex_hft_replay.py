"""Causal read-only Phase 12 tick replay CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tradingagents.forex.hft.replay import (
    StrategyFamily,
    TickReplay,
    build_strategy_plan,
    load_ticks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forex_hft_replay", description="Replay Forex ticks through the deterministic shadow engine.")
    parser.add_argument("--ticks", required=True, help="CSV, JSON, or JSONL normalized tick file")
    parser.add_argument("--output", required=True, help="scalar JSON report path")
    parser.add_argument("--strategy", choices=[item.value for item in StrategyFamily], default=StrategyFamily.MOMENTUM_CONTINUATION.value)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ticks = load_ticks(args.ticks)
    plan = build_strategy_plan(ticks[0], strategy=StrategyFamily(args.strategy))
    report = TickReplay(ticks, plan=plan).run()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
    print("FOREX HFT — REPLAY MODE")
    print(f"TICKS: {report.ticks_processed}")
    print(f"TRADES: {report.trades}")
    print(f"P50/P95/P99 MS: {report.latency_p50_ms:.4f}/{report.latency_p95_ms:.4f}/{report.latency_p99_ms:.4f}")
    print("EXECUTED: False")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
