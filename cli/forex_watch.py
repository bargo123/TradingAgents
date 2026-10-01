"""Autonomous read-only MT5 forex shadow collector CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cli.stats_handler import StatsCallbackHandler
from tradingagents.forex.watch_store import LeaseStatus, WatcherStore
from tradingagents.forex.watcher import (
    CompletedBarSchedule,
    ReadOnlyMarketProbe,
    ReadOnlyMt5ProviderProxy,
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


def _add_hft_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hft-shadow", action="store_true")
    parser.add_argument("--hft-db-path", default=None)
    parser.add_argument("--hft-symbol", default="EURUSD")
    parser.add_argument("--hft-max-ticks", type=_nonnegative_int, default=0)
    parser.add_argument("--hft-poll-interval-seconds", type=float, default=0.05)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex-watch",
        description="Read-only autonomous MT5 forex shadow collector.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the foreground collector loop")
    _add_schedule_options(run)
    _add_hft_options(run)
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


def _make_coordinator(
    args: argparse.Namespace,
    config: WatcherConfig,
    *,
    runtime_config: Any | None = None,
    hft_context: Any | None = None,
) -> WatcherCoordinator:
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.forex.evaluation import (
        EvaluationConfig,
        ShadowEvaluationStore,
        ShadowOutcomeEvaluator,
    )
    from tradingagents.forex.runner import ForexShadowRunner
    from tradingagents.forex.runtime_config import collect_runtime_provenance
    from tradingagents.forex.shadow import ShadowDecisionStore
    from tradingagents.forex.watcher import SystemClock, safe_effective_config

    safe_config = (
        runtime_config.to_tradingagents_config()
        if runtime_config is not None
        else dict(DEFAULT_CONFIG)
    )
    safe_config.update(
        {
            "analysis_profile": config.analysis_profile,
            "analysts": config.analysts,
            "schedule_timeframe": config.schedule_timeframe,
            "freshness_budget_seconds": config.freshness_budget_seconds,
        }
    )
    decision_store = ShadowDecisionStore(config.db_path)
    evaluation_store = ShadowEvaluationStore(config.db_path)
    base_provider_factory = (
        getattr(hft_context, "provider_factory", None)
        if hft_context is not None
        else None
    ) or _provider_factory
    operation_gate = (
        getattr(hft_context, "gate", None)
        if hft_context is not None
        else None
    )

    def gated_provider_factory(terminal_path: str | None = None):
        provider = base_provider_factory(terminal_path=terminal_path)
        if operation_gate is None:
            return provider
        return ReadOnlyMt5ProviderProxy(provider, operation_gate)

    provider_factory = gated_provider_factory if operation_gate is not None else base_provider_factory
    evaluator = ShadowOutcomeEvaluator(
        decision_store=decision_store,
        evaluation_store=evaluation_store,
        config=EvaluationConfig(),
        provider_factory=provider_factory,
    )
    stats_handler = StatsCallbackHandler()
    runner = ForexShadowRunner(
        provider_factory=provider_factory,
        config=safe_config,
        store=decision_store,
    )
    probe = ReadOnlyMarketProbe(provider_factory, terminal_path=config.terminal_path)
    runtime_fingerprint = (
        None if runtime_config is None else runtime_config.fingerprint
    )
    config_fingerprint = hashlib.sha256(
        json.dumps(
            {"watcher": config.safe_fingerprint, "runtime": runtime_fingerprint},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    safe_config_values = safe_effective_config(safe_config)
    provenance = (
        collect_runtime_provenance(runtime_config)
        if runtime_config is not None
        else collect_runtime_provenance(None, safe_config=safe_config_values)
    )
    provenance["config_fingerprint"] = config_fingerprint
    provenance["prompt_config_version"] = str(
        safe_config.get("prompt_config_version", "forex-shadow.v1")
    )
    provenance["collector_contract_version"] = str(
        safe_config.get("collector_contract_version", "forex-watch.v1")
    )
    provenance["safe_config_json"] = json.dumps(
        safe_config_values, sort_keys=True, separators=(",", ":")
    )
    schedule = CompletedBarSchedule(
        timeframe=config.schedule_timeframe,
        settle_seconds=config.bar_close_settle_seconds,
        config_fingerprint=config_fingerprint,
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
            provenance=provenance,
        ),
        clock=SystemClock(),
        schedule=schedule,
        probe=probe,
        runner=runner,
        evaluator=evaluator,
        executor=SingleSlotAnalysisExecutor(),
        gate=operation_gate or SerializedMt5OperationGate(),
        mt5_operations_gated=operation_gate is not None,
        on_decision=(getattr(hft_context, "handle_decision", None) if hft_context is not None else None),
        callbacks=(stats_handler,),
    )


def _watcher_lease_is_active(
    db_path: Path,
    now: datetime,
    store_factory: Any | None = None,
) -> bool:
    factory = WatcherStore if store_factory is None else store_factory
    store = factory(db_path)
    reader = getattr(store, "read_only_active_lease", None)
    lease = reader(now) if callable(reader) else store.active_lease(now)
    return lease is not None and lease.lease_expires_at > now


def _read_only_lease(store: Any, now: datetime) -> Any:
    reader = getattr(store, "read_only_active_lease", None)
    return reader(now) if callable(reader) else store.active_lease(now)


def _read_only_summary(store: Any) -> dict[str, Any]:
    reader = getattr(store, "read_only_summary", None)
    return reader() if callable(reader) else store.summary()


def _print_status(summary: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str))
        return
    print("WATCHER STATUS")
    print(f"LIFECYCLE: {summary.get('lifecycle_status', 'STOPPED')}")
    print(f"LEASE EXPIRES AT: {summary.get('lease_expires_at') or 'none'}")
    print(f"CURRENT RUN: {summary.get('current_run_id') or 'none'}")
    print(f"EVALUATION DUE PENDING: {summary.get('evaluation_due_pending', False)}")
    llm_calls = summary.get("llm_calls")
    print(f"LLM CALLS: {llm_calls if llm_calls is not None else 'unknown'}")
    print(f"DATABASE: {summary.get('database_path')}")


def _print_banner(hft_context: Any | None = None) -> None:
    worker = getattr(hft_context, "worker", None)
    config = getattr(worker, "config", None)
    if getattr(config, "execution_mode", "SHADOW") == "DEMO":
        print("MT5 FOREX — DEMO MODE")
        print("DEMO ACCOUNT ONLY — NO REAL-MONEY ORDERS")
        return
    print("MT5 FOREX — SHADOW COLLECTOR")
    print("NO ORDER WILL BE SENT")


def _run_once(coordinator: Any) -> int:
    try:
        result = coordinator.start()
    except Exception:
        with suppress(Exception):
            coordinator.shutdown()
        raise
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
        wait_error = None
        if callable(waiter) and getattr(cycle, "active_run_id", None):
            wait_error = waiter()
        if not isinstance(wait_error, str):
            wait_error = getattr(wait_error, "error_code", None)
        error_code = getattr(cycle, "error_code", None) or wait_error
        if error_code:
            print(f"FOREX WATCH ERROR: {error_code}", file=sys.stderr)
        summary = coordinator.store.summary() if hasattr(coordinator, "store") else {}
        _print_status(summary, False)
        return 0 if error_code is None else 1
    finally:
        cleanup_failed_during_primary_error = sys.exc_info()[0] is not None
        try:
            coordinator.shutdown()
        except Exception:
            if not cleanup_failed_during_primary_error:
                raise


def main(
    argv: Sequence[str] | None = None,
    *,
    coordinator_factory: Any | None = None,
    store_factory: Any | None = None,
    runtime_config: Any | None = None,
    hft_context: Any | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        try:
            factory = WatcherStore if store_factory is None else store_factory
            store = factory(Path(args.db_path))
            if args.probe:
                now = datetime.now(timezone.utc)
                lease = _read_only_lease(store, now)
                if lease is not None and lease.lease_expires_at > now:
                    print("FOREX WATCH ERROR: WATCHER_ALREADY_RUNNING", file=sys.stderr)
                    return 1
                probe = ReadOnlyMarketProbe(_provider_factory)
                for symbol in ("EURUSD",):
                    probe.probe(symbol)
            _print_status(_read_only_summary(store), args.json)
            return 0
        except Exception as exc:
            print(f"FOREX WATCH ERROR: {exc}", file=sys.stderr)
            return 1
    if args.command == "evaluate":
        from cli.forex_evaluate import main as evaluate_main

        forwarded = ["--pending", "--db-path", args.db_path]
        if args.terminal_path:
            forwarded.extend(["--terminal-path", args.terminal_path])
        return evaluate_main(forwarded)

    _print_banner(hft_context)
    try:
        config = _make_config(args)
        if _watcher_lease_is_active(
            config.db_path, datetime.now(timezone.utc), store_factory
        ):
            print("FOREX WATCH ERROR: WATCHER_ALREADY_RUNNING", file=sys.stderr)
            return 1
        factory = _make_coordinator if coordinator_factory is None else coordinator_factory
        if runtime_config is None and hft_context is None:
            coordinator = factory(args, config)
        elif hft_context is None:
            coordinator = factory(
                args, config, runtime_config=runtime_config
            )
        else:
            coordinator = factory(
                args,
                config,
                runtime_config=runtime_config,
                hft_context=hft_context,
            )
        if args.command == "once":
            return _run_once(coordinator)
        try:
            result = coordinator.start()
        except Exception:
            with suppress(Exception):
                coordinator.shutdown()
            raise
        if getattr(result, "status", None) is not LeaseStatus.ACQUIRED:
            print(f"FOREX WATCH ERROR: {getattr(result.status, 'value', result.status)}", file=sys.stderr)
            with suppress(Exception):
                coordinator.shutdown()
            return 1
        if hft_context is not None:
            hft_context.start()
        try:
            coordinator.run_forever()
        except KeyboardInterrupt:
            coordinator.request_shutdown()
        finally:
            coordinator.shutdown()
            if hft_context is not None:
                hft_context.stop()
        return 0
    except (TypeError, ValueError) as exc:
        print(f"FOREX WATCH ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"FOREX WATCH ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
