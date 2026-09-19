"""Run one bounded, explicitly configured Phase 11A Ollama teacher pilot."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.distillation.factory import DistillationFactory
from tradingagents.distillation.phase7 import Phase7KnowledgeSource
from tradingagents.distillation.pilot import select_representative_packets
from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner
from tradingagents.distillation.teacher import OllamaTeacher, teacher_from_environment
from tradingagents.distillation.writer import validate_generation


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def run_pilot(
    phase7_root: str | Path,
    output_root: str | Path,
    *,
    count: int = 5,
    expected_generation_id: str | None = None,
    planner_config: PlannerConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_root = Path(phase7_root).resolve()
    destination = Path(output_root).resolve()
    if source_root == destination or source_root in destination.parents:
        raise ValueError("pilot output must be outside the Phase 7 source root")
    source = Phase7KnowledgeSource.open(source_root, expected_generation_id)
    fingerprint_before = dict(source.source_fingerprints)
    plan = SourcePacketPlanner.plan(source, (), planner_config or PlannerConfig())
    pilot_plan = select_representative_packets(plan, count)
    teacher = teacher_from_environment()
    if not isinstance(teacher, OllamaTeacher):
        raise RuntimeError("PHASE11A_TEACHER_PROVIDER=ollama is required for the real pilot")
    report = DistillationFactory().distill(
        pilot_plan,
        teacher,
        destination,
        metadata={
            "phase7_generation_id": source.generation_id,
            "source_root": str(source_root),
            "source_fingerprints": fingerprint_before,
            "pilot": True,
            "pilot_requested_packets": count,
            "pilot_selected_packets": len(pilot_plan.packets),
        },
    )
    generation = Path(report.generation)
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    validation = validate_generation(generation)
    fingerprint_after = dict(
        Phase7KnowledgeSource.open(source_root, expected_generation_id=source.generation_id).source_fingerprints
    )
    reason_counts = Counter(manifest.get("exclusion_reasons", {}))
    quality_reasons = {
        reason: int(reason_counts.get(reason, 0))
        for reason in (
            "LESSON_TOO_LONG",
            "VERBATIM_OVERLAP_EXCESSIVE",
            "UNSAFE_FUTURE_OUTCOME_INFERENCE",
            "JUDGE_REJECTED",
        )
        if reason_counts.get(reason, 0)
    }
    grounding_reasons = {
        reason: int(reason_counts.get(reason, 0))
        for reason in ("GROUNDING_FAILED", "UNSUPPORTED_CLAIM", "SOURCE_PROVENANCE_INCOMPLETE")
        if reason_counts.get(reason, 0)
    }
    elapsed = time.perf_counter() - started
    average = teacher.metrics.average_latency_seconds
    teacher_metrics = teacher.metrics.to_dict()
    call_diagnostics = teacher_metrics["call_diagnostics"]
    failure_classes = Counter(
        str(item.get("failure_class"))
        for item in call_diagnostics
        if item.get("failure_class")
    )
    provider_statuses = Counter(
        str(item.get("provider_status"))
        for item in call_diagnostics
        if item.get("provider_status")
    )
    return {
        "status": "VALID" if validation.valid else "INVALID",
        "phase7_root": str(source_root),
        "phase7_generation_id": source.generation_id,
        "phase7_fingerprint_before": fingerprint_before,
        "phase7_fingerprint_after": fingerprint_after,
        "phase7_unchanged": fingerprint_before == fingerprint_after,
        "packets_available": len(plan.packets),
        "packets_attempted": report.teacher_calls,
        "accepted_lessons": report.accepted,
        "excluded_lessons": report.excluded,
        "schema_failures": int(reason_counts.get("SCHEMA_INVALID", 0)),
        "grounding_failures": grounding_reasons,
        "quality_failures": quality_reasons,
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "teacher_failures": int(reason_counts.get("TEACHER_FAILED", 0)),
        "model": teacher.model,
        "provider": teacher.provider,
        "endpoint": teacher.endpoint,
        "temperature": teacher.config.temperature,
        "max_tokens": teacher.config.max_tokens,
        "timeout_seconds": teacher.config.timeout_seconds,
        "retry_budget": teacher.max_retries,
        "generation_path": str(generation),
        "train_count": int(manifest.get("split_counts", {}).get("train", 0)),
        "validation_count": int(manifest.get("split_counts", {}).get("validation", 0)),
        "test_count": int(manifest.get("split_counts", {}).get("test", 0)),
        "validation": {"valid": validation.valid, "errors": list(validation.errors)},
        "average_latency_seconds": round(average, 3),
        "max_latency_seconds": round(teacher.metrics.max_latency_seconds, 3),
        "total_elapsed_seconds": round(elapsed, 3),
        "input_tokens": teacher_metrics["input_tokens"],
        "output_tokens": teacher_metrics["output_tokens"],
        "failure_class_counts": dict(sorted(failure_classes.items())),
        "provider_status_counts": dict(sorted(provider_statuses.items())),
        "call_diagnostics": call_diagnostics,
        "projected_runtime_seconds": {
            str(size): round(average * size, 1) for size in (100, 1000, 4351)
        },
        "llm_calls": report.teacher_calls,
        "mt5_calls": 0,
        "network_attempts": report.network_attempts,
        "tool_calls": report.tool_calls,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--expected-generation-id")
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--max-estimated-tokens", type=int, default=3_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_pilot(
            args.phase7_root,
            args.output_root,
            count=args.count,
            expected_generation_id=args.expected_generation_id,
            planner_config=PlannerConfig(
                max_blocks=args.max_blocks,
                max_chars=args.max_chars,
                max_estimated_tokens=args.max_estimated_tokens,
            ),
        )
        print(_json(result))
        return 0 if result["status"] == "VALID" and result["phase7_unchanged"] else 1
    except Exception as exc:
        print(_json({"status": type(exc).__name__, "error": str(exc)[:512]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
