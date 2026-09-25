"""One-command bounded supervisor for the read-only forex shadow collector."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig
from tradingagents.forex.supervisor import ForexSupervisor


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
    run.add_argument("--max-restarts", type=int, default=3)
    status = subparsers.add_parser("status", help="show scalar runtime and watcher health")
    status.add_argument("--db-path", default="data_cache/shadow_decisions.db")
    status.add_argument("--json", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    supervisor_factory=ForexSupervisor,
    watch_main=None,
) -> int:
    args = build_parser().parse_args(argv)
    supervisor = supervisor_factory(ForexShadowRuntimeConfig())
    if args.command == "status":
        report = supervisor.status(args.db_path)
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
            print("NO ORDER WILL BE SENT")
        return 0 if report["health_level"] != "OPERATOR_REVIEW_REQUIRED" else 1

    if watch_main is None:
        from cli.forex_watch import main as watch_main

    return supervisor.run(
        db_path=args.db_path,
        terminal_path=args.terminal_path,
        prewarm=not args.no_prewarm,
        max_restarts=args.max_restarts,
        watch_main=watch_main,
    )


if __name__ == "__main__":
    raise SystemExit(main())
