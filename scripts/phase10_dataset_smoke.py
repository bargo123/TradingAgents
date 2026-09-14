"""Offline Phase 10 acceptance smoke for explicit real source artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingagents.datasets.factory import DatasetFactory
from tradingagents.datasets.models import DatasetConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local-only Phase 10 dataset smoke.")
    parser.add_argument("--source-db", action="append", required=True)
    parser.add_argument("--phase8-root", required=True)
    parser.add_argument("--phase9-audit")
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sources = tuple(Path(item).resolve() for item in args.source_db)
    phase8 = Path(args.phase8_root).resolve()
    audit = Path(args.phase9_audit).resolve() if args.phase9_audit else None
    output = Path(args.output_root).resolve()
    required_paths = (*sources, phase8) + ((audit,) if audit else ())
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        print(json.dumps({"status": "SOURCE_MISSING", "paths": missing}, sort_keys=True))
        return 2
    try:
        report = DatasetFactory().build(DatasetConfig(sources, phase8, audit, output, allow_empty=True))
    except Exception as exc:  # visible, bounded harness failure
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    payload = report.to_dict()
    manifest = payload.get("manifest") or {}
    bounded = {
        "status": report.status,
        "examples": manifest.get("examples", 0),
        "exclusions": manifest.get("exclusions", len(report.exclusions)),
        "split_status": manifest.get("split_status"),
        "reason_counts": (manifest.get("counts") or {}).get("reason_counts", {}),
        "source_fingerprints": manifest.get("source_fingerprints", {}),
        "safety": manifest.get("safety", DatasetFactory.safety_counters()),
    }
    print(json.dumps(bounded, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if report.status in {"PUBLISHED", "EMPTY_ELIGIBLE_SET"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
