"""Standalone, local-only command line interface for the Phase 10 factory."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import DatasetError
from .factory import DatasetFactory
from .models import DatasetConfig
from .writer import validate_generation

_EXIT_OK = 0
_EXIT_USAGE = 2
_EXIT_FAILURE = 1
_MAX_REPORT_ITEMS = 100


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset-factory",
        description="Build and inspect a local Phase 10 dataset generation.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="read sources and publish a generation")
    build.add_argument("--source-db", action="append", required=True, metavar="PATH")
    build.add_argument("--phase8-root", required=True, metavar="PATH")
    build.add_argument("--phase9-audit", metavar="PATH")
    build.add_argument("--output-root", required=True, metavar="PATH")
    build.add_argument("--evaluation-basis", default="ANALYSIS_SNAPSHOT")
    build.add_argument("--horizon-seconds", type=_positive_int, default=300)
    build.add_argument("--allow-empty", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    validate = commands.add_parser("validate", help="validate one published generation")
    validate.add_argument("--generation", required=True, metavar="PATH")
    validate.add_argument("--json", action="store_true")

    status = commands.add_parser("status", help="list published generations")
    status.add_argument("--output-root", required=True, metavar="PATH")
    status.add_argument("--json", action="store_true")
    return parser


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _build(args: argparse.Namespace) -> int:
    config = DatasetConfig(
        source_db_paths=tuple(Path(item) for item in args.source_db),
        phase8_root=Path(args.phase8_root),
        phase9_audit_path=Path(args.phase9_audit) if args.phase9_audit else None,
        output_root=Path(args.output_root),
        evaluation_basis=args.evaluation_basis,
        horizon_seconds=args.horizon_seconds,
        allow_empty=args.allow_empty,
    )
    report = DatasetFactory().build(config)
    payload = report.to_dict()
    # Keep terminal output bounded even when a damaged source produces many rows.
    payload["exclusions"] = payload.get("exclusions", [])[:_MAX_REPORT_ITEMS]
    payload["exclusions_returned"] = len(payload["exclusions"])
    payload["exclusions_total"] = len(report.exclusions)
    payload["errors"] = list(report.errors[:_MAX_REPORT_ITEMS])
    _json(payload)
    return _EXIT_OK if report.status in {"PUBLISHED", "EMPTY_ELIGIBLE_SET"} else _EXIT_FAILURE


def _validate(args: argparse.Namespace) -> int:
    report = validate_generation(Path(args.generation))
    _json(report.to_dict())
    return _EXIT_OK if report.valid else _EXIT_FAILURE


def _status(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve()
    generations = []
    if root.is_dir():
        for candidate in sorted(root.iterdir(), key=lambda item: item.name):
            manifest_path = candidate / "manifest.json"
            if not candidate.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    continue
                generations.append({
                    "generation_id": candidate.name,
                    "status": manifest.get("status"),
                    "examples": manifest.get("examples", 0),
                    "exclusions": manifest.get("exclusions", 0),
                })
            except (OSError, ValueError, TypeError):
                generations.append({"generation_id": candidate.name, "status": "INVALID"})
    _json({
        "status": "OK" if generations else "EMPTY",
        "output_root": str(root),
        "generations": generations[:_MAX_REPORT_ITEMS],
    })
    return _EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "build":
            return _build(args)
        if args.command == "validate":
            return _validate(args)
        return _status(args)
    except (DatasetError, OSError, ValueError, TypeError) as exc:
        print(f"DATASET FACTORY ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_USAGE if isinstance(exc, (ValueError, TypeError)) else _EXIT_FAILURE
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else _EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
