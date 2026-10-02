"""Explicit Phase 14 offline CLI; no MT5 or model construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dashboard import read_dashboard_status
from .importer import import_verified_demo
from .orchestrator import SelfEnhancementOrchestrator
from .store import SelfEnhancementStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="self-enhancement")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--artifact-root", required=True)
    for name in ("import", "experiment"):
        command = sub.add_parser(name)
        command.add_argument("--artifact-root", required=True)
        command.add_argument("--hft-path", required=True)
        command.add_argument("--demo-path", required=True)
        command.add_argument("--source-commit", default="unknown")
        command.add_argument("--minimum-verified-trades", type=int, default=20)
        command.add_argument("--minimum-ticks", type=int, default=100)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--artifact-root", required=True)
    rollback.add_argument("--deployment-id", required=True)
    rollback.add_argument("--reason", required=True)
    trigger = sub.add_parser("trigger")
    trigger.add_argument("--artifact-root", required=True)
    trigger.add_argument("--trigger-type", required=True)
    trigger.add_argument("--payload", default="{}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.artifact_root)
    if args.command == "status":
        print(json.dumps(read_dashboard_status(root), sort_keys=True))
        return 0
    store = SelfEnhancementStore(root / "phase14.sqlite3")
    store.initialize()
    if args.command == "import":
        report = import_verified_demo(args.hft_path, args.demo_path, store)
        print(json.dumps(report.__dict__, sort_keys=True, default=str))
        return 0
    if args.command == "trigger":
        try:
            payload = json.loads(args.payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid trigger payload: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit("trigger payload must be a JSON object")
        trigger_id = store.record_trigger(args.trigger_type, payload)
        print(json.dumps({"trigger_id": trigger_id, "trigger_type": args.trigger_type.upper()}, sort_keys=True))
        return 0
    if args.command == "experiment":
        report = SelfEnhancementOrchestrator(root).run_once(
            hft_path=args.hft_path,
            demo_path=args.demo_path,
            source_commit=args.source_commit,
            minimum_verified_trades=args.minimum_verified_trades,
            minimum_ticks=args.minimum_ticks,
        )
        print(json.dumps(report.to_dict(), sort_keys=True, default=str))
        return 0
    store.rollback(args.deployment_id, args.reason)
    print(json.dumps({"deployment_id": args.deployment_id, "status": "ROLLED_BACK", "real_money": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
