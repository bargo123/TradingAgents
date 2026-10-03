"""One-command bounded supervisor for the read-only forex shadow collector."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence

from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig
from tradingagents.forex.supervisor import ForexSupervisor


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forex-supervisor",
        description="Start or inspect the self-supervised read-only forex collector.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="ensure dedicated Ollama and run forex-watch")
    run.add_argument("--db-path", default="data_cache/shadow_decisions.db")
    run.add_argument("--terminal-path", default=None)
    run.add_argument("--no-prewarm", action="store_true")
    run.add_argument(
        "--phase12-strategic",
        action="store_true",
        help="opt into the Phase 12 parallel strategic freshness graph",
    )
    run.add_argument(
        "--phase12-deep-model",
        default=None,
        help="explicit deep model for Phase 12 strategic mode only",
    )
    run.add_argument("--max-restarts", type=int, default=3)
    run.add_argument("--hft-shadow", action="store_true")
    run.add_argument("--hft-db-path", default=None)
    run.add_argument("--hft-symbol", default="EURUSD")
    run.add_argument("--hft-max-ticks", type=_nonnegative_int, default=0)
    run.add_argument(
        "--hft-poll-interval-seconds", type=_nonnegative_float, default=0.05
    )
    run.add_argument(
        "--demo-execute",
        action="store_true",
        help="explicitly enable verified MT5 DEMO execution; requires --hft-shadow",
    )
    run.add_argument("--demo-db-path", default=None)
    run.add_argument(
        "--market-session-calendar",
        default=None,
        help="explicit broker/symbol session calendar required for DEMO HFT operation",
    )
    status = subparsers.add_parser("status", help="show scalar runtime and watcher health")
    status.add_argument("--db-path", default="data_cache/shadow_decisions.db")
    status.add_argument("--hft-db-path", default=None)
    status.add_argument("--demo-db-path", default=None)
    status.add_argument("--phase12-strategic", action="store_true")
    status.add_argument("--phase12-deep-model", default=None)
    status.add_argument("--json", action="store_true")
    control = subparsers.add_parser("control", help="enqueue an owner-side DEMO control request")
    control_subparsers = control.add_subparsers(dest="control_command", required=True)
    reconcile = control_subparsers.add_parser(
        "reconcile-demo-position",
        help="request reconciliation of one DEMO ledger ticket inside the active supervisor",
    )
    reconcile.add_argument("--demo-db-path", required=True)
    reconcile.add_argument("--ticket", type=_nonnegative_int, required=True)
    reconcile.add_argument("--symbol", default="EURUSD")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    supervisor_factory=ForexSupervisor,
    watch_main=None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "control":
        if args.control_command != "reconcile-demo-position":
            raise ValueError("unsupported control command")
        try:
            from tradingagents.forex.hft.demo_store import DemoExecutionStore

            request_id = DemoExecutionStore(args.demo_db_path).request_reconciliation(
                ticket=args.ticket,
                symbol=args.symbol,
            )
            print(f"RECONCILIATION_REQUESTED: {request_id}")
            return 0
        except Exception as exc:
            print(f"FOREX SUPERVISOR ERROR: {exc}", file=sys.stderr)
            return 1
    if args.command == "status":
        try:
            runtime_kwargs = {"phase12_strategic": args.phase12_strategic}
            if args.phase12_deep_model:
                if not args.phase12_strategic:
                    raise ValueError("--phase12-deep-model requires --phase12-strategic")
                runtime_kwargs["deep_model"] = args.phase12_deep_model
            supervisor = supervisor_factory(ForexShadowRuntimeConfig(**runtime_kwargs))
            if args.hft_db_path is None:
                report = supervisor.status(args.db_path, demo_db_path=args.demo_db_path)
            else:
                report = supervisor.status(
                    args.db_path,
                    hft_db_path=args.hft_db_path,
                    demo_db_path=args.demo_db_path,
                )
            if args.json:
                print(json.dumps(report, sort_keys=True, default=str))
            else:
                print("FOREX SUPERVISOR STATUS")
                print(f"HEALTH: {report['health_level']}")
                if report.get("health_reason"):
                    print(f"REASON: {report['health_reason']}")
                ollama = report["ollama"]
                print(f"OLLAMA: {ollama['status']} {ollama['endpoint']}")
                print(f"CONTEXT: {ollama.get('context_length') or 'unknown'}")
                print(f"WATCHER: {report['watcher'].get('lifecycle_status', 'STOPPED')}")
                print(f"HFT SHADOW: {report['hft_engine_health']}")
                print(f"DEMO EXECUTION: {report.get('demo_execution_health', 'DISABLED')}")
                print(f"MT5 READ-ONLY: {report['mt5_read_only_health']}")
                print("NO ORDER WILL BE SENT")
            return 0 if report["health_level"] != "OPERATOR_REVIEW_REQUIRED" else 1
        except Exception as exc:
            print(f"FOREX SUPERVISOR ERROR: {exc}", file=sys.stderr)
            return 1

    if watch_main is None:
        from cli.forex_watch import main as watch_main

    try:
        if args.phase12_deep_model and not args.phase12_strategic:
            raise ValueError("--phase12-deep-model requires --phase12-strategic")
        runtime_kwargs = {"phase12_strategic": args.phase12_strategic}
        if args.phase12_deep_model:
            runtime_kwargs["deep_model"] = args.phase12_deep_model
        supervisor = supervisor_factory(
            ForexShadowRuntimeConfig(**runtime_kwargs)
        )
        return supervisor.run(
            db_path=args.db_path,
            terminal_path=args.terminal_path,
            prewarm=not args.no_prewarm,
            max_restarts=args.max_restarts,
            watch_main=watch_main,
            hft_shadow=args.hft_shadow,
            hft_db_path=args.hft_db_path,
            hft_symbol=args.hft_symbol,
            hft_max_ticks=args.hft_max_ticks,
            hft_poll_interval_seconds=args.hft_poll_interval_seconds,
            demo_execute=args.demo_execute,
            demo_db_path=args.demo_db_path,
            market_session_calendar=args.market_session_calendar,
        )
    except Exception as exc:
        print(f"FOREX SUPERVISOR ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
