"""Standalone machine-readable Phase 11A knowledge-distillation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .models import SourceBlock, SourcePacket, SourcePlan, SourceRef, canonical_json
from .phase7 import Phase7KnowledgeSource
from .teacher import teacher_from_environment
from .writer import validate_generation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledge-distill")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="create a bounded plan from an existing Phase 7 generation"
    )
    plan.add_argument("--phase7-root", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--topic", action="append", default=[])
    distill = commands.add_parser(
        "distill", help="distill a plan with an explicitly configured teacher"
    )
    distill.add_argument("--plan", required=True)
    distill.add_argument("--output-root", "--output", dest="output_root", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--generation", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--generation", required=True)
    return parser


def _json_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _emit(payload: Any) -> None:
    print(json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":")))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def _ensure_outside(path: str | Path, source_root: str | Path) -> None:
    destination = Path(path).resolve()
    root = Path(source_root).resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        return
    raise ValueError("plan output must be outside the Phase 7 source root")


def _load_plan(path: Path) -> SourcePlan:
    value = json.loads(path.read_text(encoding="utf-8"))

    def ref(row):
        if isinstance(row, SourceRef):
            return row
        return SourceRef.from_dict(row)

    def block(row):
        return SourceBlock(
            ref(row.get("ref", row)),
            row["text"],
            row.get("content_type", "PROSE"),
            row.get("reading_order", 0),
            tuple(row.get("section_path", ())),
            row.get("metadata", {}),
        )

    packets = tuple(
        SourcePacket(
            row["packet_id"],
            tuple(block(item) for item in row["blocks"]),
            row.get("request_fingerprint", ""),
        )
        for row in value.get("packets", ())
    )
    return SourcePlan(
        value.get("generation_id", value.get("phase7_generation_id", "")),
        packets,
        value.get("request_fingerprint", value.get("plan_fingerprint", "")),
        tuple(value.get("diagnostics", ())),
        value.get("source_fingerprints", value.get("phase7_fingerprints", {})),
        value.get("policy_versions", {}),
        value.get("source_root", ""),
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            from .planning import PlannerConfig, SourcePacketPlanner

            source = Phase7KnowledgeSource.open(args.phase7_root)
            value = SourcePacketPlanner.plan(source, tuple(args.topic), PlannerConfig())
            _ensure_outside(args.output, args.phase7_root)
            _write_json(Path(args.output), value)
            _emit(
                {"status": "PLANNED", "plan": str(Path(args.output)), "packets": len(value.packets)}
            )
            return 0
        if args.command == "distill":
            teacher = teacher_from_environment()
            if teacher.__class__.__name__ == "UnconfiguredTeacher":
                _emit({"status": "DISTILLATION_TEACHER_NOT_CONFIGURED"})
                return 1
            from .factory import DistillationFactory

            report = DistillationFactory().distill(
                _load_plan(Path(args.plan)), teacher, args.output_root
            )
            _emit(
                {
                    "status": "PUBLISHED",
                    "generation": str(report.generation),
                    "accepted": report.accepted,
                    "excluded": report.excluded,
                }
            )
            return 0
        if args.command == "validate":
            report = validate_generation(args.generation)
            _emit(
                {
                    "status": "VALID" if report.valid else "INVALID",
                    "valid": report.valid,
                    "errors": list(report.errors),
                }
            )
            return 0 if report.valid else 1
        manifest = json.loads((Path(args.generation) / "manifest.json").read_text(encoding="utf-8"))
        metadata = manifest.get("metadata", {})
        _emit(
            {
                "status": manifest.get("status", "INVALID"),
                "generation_id": manifest.get("generation_id"),
                "counts": manifest.get("counts", {}),
                "split_counts": manifest.get("split_counts", {}),
                "exclusion_reasons": manifest.get("exclusion_reasons", {}),
                "lesson_type_distribution": metadata.get("lesson_type_distribution", {}),
                "topic_distribution": metadata.get("topic_distribution", {}),
                "difficulty_distribution": metadata.get("difficulty_distribution", {}),
                "source_coverage": metadata.get("source_coverage", {}),
            }
        )
        return 0
    except Exception as exc:
        _emit({"status": type(exc).__name__, "error": str(exc)[:512]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["main"]
