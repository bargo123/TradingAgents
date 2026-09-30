"""Run an explicitly isolated TEST_ONLY Phase 12C lifecycle canary."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.forex.hft.canary import CanaryConfig, CanaryError, run_canary_side
from tradingagents.forex.hft.models import Direction
from tradingagents.forex.hft.runtime import MT5ReadOnlyTickSource
from tradingagents.forex.hft.store import HftShadowStore
from tradingagents.forex.watch_store import WatcherStore
from tradingagents.forex.watcher import SerializedMt5OperationGate


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex-hft-canary",
        description="Run isolated TEST_ONLY LONG and SHORT shadow lifecycle canaries.",
    )
    parser.add_argument("run", nargs="?", default="run", choices=("run",))
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--watcher-db-path", default="data_cache/live-market-clean-20260923.db")
    parser.add_argument("--production-hft-db-path", default=None)
    parser.add_argument("--terminal-path", default=None)
    parser.add_argument("--max-ticks", type=_positive_int, default=30)
    parser.add_argument("--time-stop-seconds", type=_positive_int, default=5)
    parser.add_argument("--poll-interval-seconds", type=_nonnegative_float, default=1.0)
    return parser


def _make_provider(terminal_path: str | None):
    from tradingagents.dataflows.mt5.provider import MT5Provider

    provider = MT5Provider(terminal_path=terminal_path)
    if not provider.initialize():
        raise RuntimeError("MT5 provider initialization failed")
    return provider


def _symbol_point(provider, symbol: str) -> float:
    resolved = provider.ensure_symbol(symbol)
    for info in provider.get_symbols():
        if info.name == resolved and info.point is not None:
            return float(info.point)
    raise CanaryError("resolved symbol has no positive point metadata")


def _watcher_active(path: Path) -> bool:
    if not path.is_file():
        return False
    lease = WatcherStore(path).read_only_active_lease(datetime.now(timezone.utc))
    return lease is not None and lease.lease_expires_at > datetime.now(timezone.utc)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("MT5 FOREX — TEST_ONLY HFT CANARY")
    print("NO ORDER WILL BE SENT")
    provider = None
    try:
        canary_path = Path(args.db_path).expanduser().resolve()
        watcher_path = Path(args.watcher_db_path).expanduser().resolve()
        production_hft_path = Path(
            args.production_hft_db_path
            if args.production_hft_db_path is not None
            else watcher_path.with_suffix(".hft.sqlite3")
        ).expanduser().resolve()
        if canary_path in {watcher_path, production_hft_path}:
            raise CanaryError("canary database must be separate from production databases")
        config = CanaryConfig(
            symbol=args.symbol,
            artifact_path=canary_path,
            max_ticks=args.max_ticks,
            poll_interval_seconds=args.poll_interval_seconds,
            time_stop_seconds=args.time_stop_seconds,
        )
        config.ensure_fresh_artifact()
        canary_store = HftShadowStore(canary_path)
        active = canary_store.read_only_active_lease(datetime.now(timezone.utc))
        if active is not None and active.lease_expires_at > datetime.now(timezone.utc):
            raise CanaryError("TEST_ONLY canary database is already running")
        if _watcher_active(watcher_path):
            print("PRODUCTION WATCHER: ACTIVE (canary remains isolated)")
        provider = _make_provider(args.terminal_path)
        point = _symbol_point(provider, config.symbol)
        config = CanaryConfig(
            symbol=config.symbol,
            artifact_path=config.artifact_path,
            max_ticks=config.max_ticks,
            poll_interval_seconds=config.poll_interval_seconds,
            point=point,
            time_stop_seconds=config.time_stop_seconds,
        )
        source = MT5ReadOnlyTickSource(provider, point=point)
        gate = SerializedMt5OperationGate()
        long_report = run_canary_side(source, direction=Direction.LONG, config=config, store=canary_store, mt5_gate=gate)
        if long_report.runtime_result.get("entries") != 1 or long_report.runtime_result.get("exits") != 1:
            raise CanaryError("TEST_ONLY LONG canary did not complete one entry and one exit")
        short_report = run_canary_side(source, direction=Direction.SHORT, config=config, store=canary_store, mt5_gate=gate)
        if short_report.runtime_result.get("entries") != 1 or short_report.runtime_result.get("exits") != 1:
            raise CanaryError("TEST_ONLY SHORT canary did not complete one entry and one exit")
        payload = {
            "execution_mode": "TEST_ONLY",
            "excluded_from_performance": True,
            "long": long_report.to_dict(),
            "short": short_report.to_dict(),
            "database": str(canary_path),
            "point": point,
            "executed": False,
        }
        print(json.dumps(payload, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        print(f"FOREX HFT CANARY ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if provider is not None:
            with suppress(Exception):
                provider.shutdown()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
