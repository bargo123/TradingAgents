"""Autonomous read-only MT5 forex shadow collector CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.forex.watch_store import LeaseStatus, WatcherStore
from tradingagents.forex.watcher import (
    CompletedBarSchedule,
    ReadOnlyMarketProbe,
    SerializedMt5OperationGate,
    SingleSlotAnalysisExecutor,
    WatcherConfig,
    WatcherCoordinator,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _add_schedule_options(parser: argparse.ArgumentParser, *, include_runtime: bool = True) -> None:
    parser.add_argument("--db-path", default="data_cache/shadow_decisions.db")
    parser.add_argument("--terminal-path", default=None)
    parser.add_argument("--symbols", default="EURUSD")
    parser.add_argument("--analysis-profile", default="INTRADAY")
    parser.add_argument("--analysts", default="market,news")
    parser.add_argument("--schedule-timeframe", default="M15")
    if include_runtime:
        parser.add_argument("--poll-interval-seconds", type=_positive_int, default=15)
        parser.add_argument("--cooldown-seconds", type=_nonnegative_int, default=60)
        parser.add_argument("--analysis-timeout-seconds", type=_positive_int, default=7200)
        parser.add_argument("--evaluation-interval-seconds", type=_nonnegative_int, default=60)
        parser.add_argument("--no-evaluate", action="store_true")
        parser.add_argument("--json-logs", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex-watch",
        description="Read-only autonomous MT5 forex shadow collector.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the foreground collector loop")
    _add_schedule_options(run)
    once = subparsers.add_parser("once", help="run one current collector cycle")
    _add_schedule_options(once)
    status = subparsers.add_parser("status", help="show persisted watcher state")
    status.add_argument("--db-path", default="data_cache/shadow_decisions.db")
    status.add_argument("--json", action="store_true")
    status.add_argument("--probe", action="store_true")
    evaluate = subparsers.add_parser("evaluate", help="run the existing Phase 5 evaluator")
    evaluate.add_argument("--db-path", default="data_cache/shadow_decisions.db")
    evaluate.add_argument("--terminal-path", default=None)
    return parser


def _symbols(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one symbol is required")
    if len(set(values)) != len(values):
        raise ValueError("symbols must not contain duplicates")
    return values


def _make_config(args: argparse.Namespace) -> WatcherConfig:
    return WatcherConfig(
        symbols=_symbols(args.symbols),
        analysis_profile=args.analysis_profile,
        analysts=tuple(part.strip() for part in args.analysts.split(",") if part.strip()),
        schedule_timeframe=args.schedule_timeframe,
        poll_interval_seconds=args.poll_interval_seconds,
        cooldown_seconds=args.cooldown_seconds,
        analysis_timeout_seconds=args.analysis_timeout_seconds,
        evaluation_interval_seconds=args.evaluation_interval_seconds,
        evaluation_enabled=not args.no_evaluate,
        db_path=Path(args.db_path),
        terminal_path=args.terminal_path,
    )


def _provider_factory(terminal_path: str | None = None):
    from tradingagents.dataflows.mt5.provider import MT5Provider

    return MT5Provider(terminal_path=terminal_path)


def _make_coordinator(args: argparse.Namespace, config: WatcherConfig) -> WatcherCoordinator:
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.forex.evaluation import (
        EvaluationConfig,
        ShadowEvaluationStore,
        ShadowOutcomeEvaluator,
    )
    from tradingagents.forex.runner import ForexShadowRunner
    from tradingagents.forex.shadow import ShadowDecisionStore
    from tradingagents.forex.watcher import SystemClock

    safe_config = dict(DEFAULT_CONFIG)
    safe_config.update(
        {
            "analysis_profile": config.analysis_profile,
            "analysts": config.analysts,
            "schedule_timeframe": config.schedule_timeframe,
        }
    )
    decision_store = ShadowDecisionStore(config.db_path)
    evaluation_store = ShadowEvaluationStore(config.db_path)
    evaluator = ShadowOutcomeEvaluator(
        decision_store=decision_store,
        evaluation_store=evaluation_store,
        config=EvaluationConfig(),
        provider_factory=_provider_factory,
    )
    runner = ForexShadowRunner(config=safe_config, store=decision_store)
    probe = ReadOnlyMarketProbe(_provider_factory, terminal_path=config.terminal_path)
    schedule = CompletedBarSchedule(
        timeframe=config.schedule_timeframe,
        settle_seconds=config.bar_close_settle_seconds,
        config_fingerprint=config.safe_fingerprint,
        requested_symbols=config.symbols,
        analysis_profile=config.analysis_profile,
        analyst_set=config.analysts,
    )
    return WatcherCoordinator(
        config=config,
        store=WatcherStore(
            config.db_path,
            lease_ttl_seconds=config.lease_ttl_seconds,
            busy_timeout_seconds=config.db_busy_timeout_seconds,
        ),
        clock=SystemClock(),
        schedule=schedule,
        probe=probe,
        runner=runner,
        evaluator=evaluator,
        executor=SingleSlotAnalysisExecutor(),
        gate=SerializedMt5OperationGate(),
    )


def _watcher_lease_is_active(
    db_path: Path,
    now: datetime,
    store_factory: Any | None = None,
) -> bool:
    store = (store_factory or WatcherStore)(db_path)
    lease = store.active_lease(now)
    return lease is not None and lease.lease_expires_at > now


def _print_status(summary: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str))
        return
    print("WATCHER STATUS")
    print(f"LIFECYCLE: {summary.get('lifecycle_status', 'STOPPED')}")
    print(f"LEASE EXPIRES AT: {summary.get('lease_expires_at') or 'none'}")
    print(f"CURRENT RUN: {summary.get('current_run_id') or 'none'}")
    print(f"EVALUATION DUE PENDING: {summary.get('evaluation_due_pending', False)}")
    print(f"LLM CALLS: {summary.get('llm_calls', 0)}")
    print(f"DATABASE: {summary.get('database_path')}")


def _print_banner() -> None:
    print("MT5 FOREX — SHADOW COLLECTOR")
    print("NO ORDER WILL BE SENT")


def _run_once(coordinator: Any) -> int:
    result = coordinator.start()
    if getattr(result, "status", None) is not LeaseStatus.ACQUIRED:
        print(f"FOREX WATCH ERROR: {getattr(result.status, 'value', result.status)}", file=sys.stderr)
        shutdown = getattr(coordinator, "shutdown", None)
        if callable(shutdown):
            shutdown()
        return 1
    try:
        cycle = coordinator.run_once()
        # Production coordinators expose wait_for_active; test doubles can
        # return IDLE immediately without needing a worker implementation.
        waiter = getattr(coordinator, "wait_for_active", None)
        if callable(waiter) and getattr(cycle, "active_run_id", None):
            waiter()
        summary = coordinator.store.summary() if hasattr(coordinator, "store") else {}
        _print_status(summary, False)
        return 0 if getattr(cycle, "error_code", None) is None else 1
    finally:
        coordinator.shutdown()


def main(
    argv: Sequence[str] | None = None,
    *,
    coordinator_factory: Any | None = None,
    store_factory: Any | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        store = (store_factory or WatcherStore)(Path(args.db_path))
        if args.probe:
            now = datetime.now(timezone.utc)
            lease = store.active_lease(now)
            if lease is not None and lease.lease_expires_at > now:
                print("FOREX WATCH ERROR: WATCHER_ALREADY_RUNNING", file=sys.stderr)
                return 1
            try:
                probe = ReadOnlyMarketProbe(_provider_factory)
                for symbol in ("EURUSD",):
                    probe.probe(symbol)
            except Exception as exc:
                print(f"FOREX WATCH ERROR: {exc}", file=sys.stderr)
                return 1
        _print_status(store.summary(), args.json)
        return 0
    if args.command == "evaluate":
        from cli.forex_evaluate import main as evaluate_main

        forwarded = ["--pending", "--db-path", args.db_path]
        if args.terminal_path:
            forwarded.extend(["--terminal-path", args.terminal_path])
        return evaluate_main(forwarded)

    _print_banner()
    try:
        config = _make_config(args)
        if _watcher_lease_is_active(
            config.db_path, datetime.now(timezone.utc), store_factory
        ):
            print("FOREX WATCH ERROR: WATCHER_ALREADY_RUNNING", file=sys.stderr)
            return 1
        coordinator = (coordinator_factory or _make_coordinator)(args, config)
        if args.command == "once":
            return _run_once(coordinator)
        result = coordinator.start()
        if getattr(result, "status", None) is not LeaseStatus.ACQUIRED:
            print(f"FOREX WATCH ERROR: {getattr(result.status, 'value', result.status)}", file=sys.stderr)
            with suppress(Exception):
                coordinator.shutdown()
            return 1
        try:
            coordinator.run_forever()
        except KeyboardInterrupt:
            coordinator.request_shutdown()
        finally:
            coordinator.shutdown()
        return 0
    except (TypeError, ValueError) as exc:
        print(f"FOREX WATCH ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"FOREX WATCH ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
