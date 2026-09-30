"""Explicit read-only MT5 HFT-style shadow CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.forex.watch_store import WatcherStore
from tradingagents.forex.hft.models import Direction, EntryConstraints, RiskPosture, StopPolicy, StrategicExecutionPlan
from tradingagents.forex.hft.plan_store import AtomicPlanStore
from tradingagents.forex.hft.runtime import HftShadowConfig, HftShadowRuntime, MT5ReadOnlyTickSource
from tradingagents.forex.hft.store import HftShadowStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forex_hft_shadow", description="Read-only MT5 HFT-style shadow runtime.")
    parser.add_argument("run", nargs="?", default="run", choices=("run",))
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--watcher-db-path", default="data_cache/shadow_decisions.db")
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--terminal-path", default=None)
    parser.add_argument("--max-ticks", type=int, default=1)
    return parser


def _active_watcher(db_path: str) -> bool:
    store = WatcherStore(Path(db_path))
    lease = store.read_only_active_lease(datetime.now(timezone.utc))
    return lease is not None and lease.lease_expires_at > datetime.now(timezone.utc)


def _make_provider(terminal_path: str | None):
    from tradingagents.dataflows.mt5.provider import MT5Provider

    provider = MT5Provider(terminal_path=terminal_path)
    if not provider.initialize():
        raise RuntimeError("MT5 provider initialization failed")
    return provider


def _load_plan(path: str | Path) -> StrategicExecutionPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = payload["entry_constraints"]
    stop = payload["stop_policy"]
    return StrategicExecutionPlan(
        symbol=payload["symbol"], created_at=datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00")), expires_at=datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00")), allowed_until=datetime.fromisoformat(payload["allowed_until"].replace("Z", "+00:00")), timeframe=payload["timeframe"], regime=payload["regime"], primary_direction=Direction(payload["primary_direction"]), confidence=payload["confidence"], strategy_family=payload["strategy_family"], entry_constraints=EntryConstraints(**entry), risk_posture=RiskPosture(payload["risk_posture"]), stop_policy=StopPolicy(**stop), invalidation=tuple(payload.get("invalidation", ())), session_constraints=tuple(payload.get("session_constraints", ())), plan_id=payload.get("plan_id"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("MT5 FOREX — HFT SHADOW MODE")
    print("NO ORDER WILL BE SENT")
    try:
        if _active_watcher(args.watcher_db_path):
            print("FOREX HFT SHADOW: WATCHER_ALREADY_RUNNING", file=sys.stderr)
            return 1
        plan = _load_plan(args.plan_json)
        provider = _make_provider(args.terminal_path)
        try:
            plans = AtomicPlanStore()
            plans.replace(plan, now=plan.created_at)
            config = HftShadowConfig(symbol=args.symbol, artifact_path=Path(args.db_path), max_ticks=args.max_ticks)
            result = HftShadowRuntime(MT5ReadOnlyTickSource(provider), plans, config=config, store=HftShadowStore(config.artifact_path)).run(max_ticks=args.max_ticks)
        finally:
            provider.shutdown()
    except Exception as exc:
        print(f"FOREX HFT SHADOW ERROR: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"RUN ID: {result['run_id']}")
    print(f"ACTIONS: {result['actions']}")
    print(f"ENTRIES/EXITS: {result['entries']}/{result['exits']}")
    print("EXECUTED: FALSE")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
