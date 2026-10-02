"""Run one explicit, bounded Phase 14 offline experiment.

The caller must provide immutable source paths and an isolated artifact root.
This script never constructs an MT5 provider and never invokes an LLM.
"""

from __future__ import annotations

import argparse
import json

from tradingagents.self_enhancement.orchestrator import SelfEnhancementOrchestrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase14-first-experiment")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--hft-path", required=True)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--minimum-verified-trades", type=int, default=20)
    parser.add_argument("--minimum-ticks", type=int, default=100)
    args = parser.parse_args(argv)
    report = SelfEnhancementOrchestrator(args.artifact_root).run_once(
        hft_path=args.hft_path,
        demo_path=args.demo_path,
        source_commit=args.source_commit,
        minimum_verified_trades=args.minimum_verified_trades,
        minimum_ticks=args.minimum_ticks,
    )
    print(json.dumps(report.to_dict(), sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
