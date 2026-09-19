"""Profile local Ollama inference without creating a Phase 11A dataset."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.distillation.factory import _coerce_candidate
from tradingagents.distillation.grounding import GroundingValidator
from tradingagents.distillation.ollama_diagnostic import (
    run_native_call,
    stop_model,
)
from tradingagents.distillation.phase7 import Phase7KnowledgeSource
from tradingagents.distillation.pilot import select_representative_packets
from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner
from tradingagents.distillation.quality import QualityPolicy
from tradingagents.distillation.teacher import (
    TeacherLesson,
    _packet_prompt,
    _validate_packet_refs,
)


def _simple_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"answer": {"type": "string", "minLength": 1}},
        "required": ["answer"],
        "additionalProperties": False,
    }


def _validate_simple(value: Any) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
        raise ValueError("simple diagnostic schema mismatch")


def _medium_messages() -> tuple[list[dict[str, str]], int, int]:
    system = "Return only the requested JSON object; use only the supplied synthetic evidence."
    evidence = (
        "Synthetic evidence for latency profiling. This text has no trading decision, future "
        "outcome, or private reasoning content. Summarize the supplied concept faithfully. "
    ) * 45
    evidence = evidence[:3600]
    user = "Answer using only this evidence:\n\n" + evidence
    return (
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        len(evidence),
        len(system) + len("Answer using only this evidence:\n\n"),
    )


def _tiny_messages() -> tuple[list[dict[str, str]], int, int]:
    system = "Return only JSON with an answer string."
    user = "Answer with the word: ready."
    return (
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        len(user),
        len(system),
    )


def _real_validator(packet):
    grounding = GroundingValidator()
    quality = QualityPolicy()

    def validate(value: Any) -> str | None:
        lesson = TeacherLesson.model_validate(value)
        _validate_packet_refs(lesson, packet)
        candidate = _coerce_candidate(lesson.model_dump(mode="json"), packet)
        grounding_report = grounding.validate(candidate, packet)
        if not getattr(grounding_report, "accepted", bool(grounding_report)):
            return "GROUNDING_FAILED"
        quality_report = quality.validate(candidate, packet)
        if not getattr(quality_report, "accepted", bool(quality_report)):
            return "QUALITY_FAILED"
        return None

    return validate


def _context_fit(result: dict[str, Any], contexts: tuple[int, ...]) -> dict[str, str]:
    required = (
        int(result.get("approximate_input_tokens") or 0)
        + math.ceil(int(result.get("schema_chars") or 0) / 4)
        + int(result.get("max_tokens") or 0)
    )
    return {
        str(context): "ELIGIBLE" if required <= context else "INELIGIBLE_CONTEXT_SIZE"
        for context in contexts
    }


def _load_seconds(result: dict[str, Any]) -> float | None:
    value = result.get("load_duration_ns")
    return round(value / 1_000_000_000, 3) if isinstance(value, int) else None


def select_real_packets(packets, indexes: tuple[int, ...]) -> tuple[Any, ...]:
    """Select at most three deterministic real packets for a bounded run."""

    available = tuple(packets)
    if not indexes or len(indexes) > 3:
        raise ValueError("real packet indexes must contain one to three entries")
    if len(set(indexes)) != len(indexes):
        raise ValueError("real packet indexes must be unique")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indexes
    ):
        raise ValueError("real packet indexes must be non-negative integers")
    if any(index >= len(available) for index in indexes):
        raise ValueError("real packet index is outside the selected packet set")
    return tuple(available[index] for index in indexes)


def _run_model(
    model: str,
    *,
    packet,
    real_packets,
    endpoint: str,
    context_tokens: int,
    max_tokens: int,
    timeout_seconds: float,
    success_benchmark: bool,
    context_candidates: tuple[int, ...],
) -> dict[str, Any]:
    stop_model(model)
    diagnostics: list[dict[str, Any]] = []
    real_results: list[dict[str, Any]] = []
    api_base = endpoint.rstrip("/")
    if api_base.endswith("/v1"):
        api_base = api_base[:-3]
    try:
        timeout = httpx.Timeout(timeout_seconds)
        with httpx.Client(
            base_url=api_base,
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            tiny_messages, tiny_source, tiny_instruction = _tiny_messages()
            tiny = run_native_call(
                client,
                model=model,
                test_name="A_TINY",
                messages=tiny_messages,
                schema=_simple_schema(),
                validator=_validate_simple,
                context_tokens=context_tokens,
                max_tokens=max_tokens,
                keep_alive="5m",
                source_chars=tiny_source,
                instruction_chars=tiny_instruction,
                load_phase="cold_candidate",
            )
            tiny["context_candidates"] = _context_fit(tiny, context_candidates)
            diagnostics.append(tiny)

            medium_messages, medium_source, medium_instruction = _medium_messages()
            medium = run_native_call(
                client,
                model=model,
                test_name="B_MEDIUM",
                messages=medium_messages,
                schema=_simple_schema(),
                validator=_validate_simple,
                context_tokens=context_tokens,
                max_tokens=max_tokens,
                keep_alive="5m",
                source_chars=medium_source,
                instruction_chars=medium_instruction,
                load_phase="warm_candidate",
            )
            medium["context_candidates"] = _context_fit(medium, context_candidates)
            diagnostics.append(medium)

            def run_real(name: str, real_packet) -> dict[str, Any]:
                messages = _packet_prompt(real_packet)
                source_chars = sum(len(block.text) for block in real_packet.blocks)
                request_chars = sum(len(item["content"]) for item in messages)
                instruction_chars = max(0, request_chars - source_chars)
                result = run_native_call(
                    client,
                    model=model,
                    test_name=name,
                    messages=messages,
                    schema=TeacherLesson.model_json_schema(),
                    validator=_real_validator(real_packet),
                    context_tokens=context_tokens,
                    max_tokens=max_tokens,
                    keep_alive="5m",
                    source_chars=source_chars,
                    instruction_chars=instruction_chars,
                    load_phase="warm_candidate",
                )
                result["packet_id"] = real_packet.packet_id
                result["context_candidates"] = _context_fit(result, context_candidates)
                return result

            first_real = run_real("C_REAL_1", packet)
            diagnostics.append(first_real)
            real_results.append(first_real)
            if success_benchmark and first_real["status"] == "SUCCESS":
                for index, real_packet in enumerate(real_packets[1:3], start=2):
                    result = run_real(f"REAL_{index}", real_packet)
                    diagnostics.append(result)
                    real_results.append(result)
    finally:
        stop_model(model)

    statuses = Counter(str(item.get("status")) for item in diagnostics)
    real_statuses = Counter(str(item.get("status")) for item in real_results)
    real_latencies = [
        float(item["elapsed_seconds"])
        for item in real_results
        if item.get("status") == "SUCCESS"
    ]
    return {
        "model": model,
        "context_tokens": context_tokens,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "retry_count": 0,
        "diagnostics": diagnostics,
        "real_results": real_results,
        "status_counts": dict(sorted(statuses.items())),
        "real_status_counts": dict(sorted(real_statuses.items())),
        "provider_successes": int(statuses.get("SUCCESS", 0)),
        "timeouts": int(statuses.get("PROVIDER_TIMEOUT", 0)),
        "schema_failures": int(
            statuses.get("JSON_INVALID", 0) + statuses.get("SCHEMA_INVALID", 0)
        ),
        "output_truncations": int(statuses.get("OUTPUT_TRUNCATED", 0)),
        "grounding_failures": int(statuses.get("GROUNDING_FAILED", 0)),
        "quality_failures": int(statuses.get("QUALITY_FAILED", 0)),
        "accepted_lessons": int(real_statuses.get("SUCCESS", 0)),
        "cold_load_seconds": _load_seconds(diagnostics[0]) if diagnostics else None,
        "warm_load_seconds": _load_seconds(diagnostics[1]) if len(diagnostics) > 1 else None,
        "first_success_real_latency_seconds": next(iter(real_latencies), None),
        "real_average_latency_seconds": round(statistics.mean(real_latencies), 3)
        if real_latencies
        else None,
        "real_median_latency_seconds": round(statistics.median(real_latencies), 3)
        if real_latencies
        else None,
        "total_elapsed_seconds": round(
            sum(float(item["elapsed_seconds"]) for item in diagnostics), 3
        ),
        "projected_runtime_seconds": {
            str(size): round(
                (statistics.mean(real_latencies) if real_latencies else timeout_seconds) * size, 1
            )
            for size in (100, 1000, 4351)
        },
    }


def run_diagnostics(
    phase7_root: str | Path,
    *,
    models: tuple[str, ...],
    expected_generation_id: str | None = None,
    endpoint: str = "http://localhost:11434",
    context_tokens: int = 16_384,
    max_tokens: int = 512,
    timeout_seconds: float = 300.0,
    success_benchmark: bool = True,
    real_packet_indexes: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    source = Phase7KnowledgeSource.open(Path(phase7_root), expected_generation_id)
    plan = SourcePacketPlanner.plan(
        source,
        (),
        PlannerConfig(max_blocks=8, max_chars=12_000, max_estimated_tokens=3_000),
    )
    selected = select_representative_packets(plan, 5)
    indexes = real_packet_indexes if real_packet_indexes is not None else (0, 1, 2)
    real_packets = select_real_packets(selected.packets, indexes)
    packet = real_packets[0]
    context_candidates = (8192, 12288, 16384)
    started = time.perf_counter()
    reports = []
    for model in models:
        reports.append(
            _run_model(
                model,
                packet=packet,
                real_packets=real_packets,
                endpoint=endpoint,
                context_tokens=context_tokens,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                success_benchmark=success_benchmark,
                context_candidates=context_candidates,
            )
        )
    fingerprint_after = Phase7KnowledgeSource.open(
        Path(phase7_root), expected_generation_id=source.generation_id
    ).source_fingerprints
    return {
        "status": "VALID",
        "phase7_root": str(Path(phase7_root).resolve()),
        "phase7_generation_id": source.generation_id,
        "phase7_fingerprint_before": dict(source.source_fingerprints),
        "phase7_fingerprint_after": dict(fingerprint_after),
        "phase7_unchanged": dict(source.source_fingerprints) == dict(fingerprint_after),
        "packets_available": len(plan.packets),
        "selected_packet_ids": [item.packet_id for item in selected.packets],
        "real_packet_ids": [item.packet_id for item in real_packets],
        "endpoint": endpoint,
        "context_candidates": list(context_candidates),
        "context_tokens": context_tokens,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "models": reports,
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-root", type=Path, required=True)
    parser.add_argument("--expected-generation-id")
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--models", nargs="+", default=["qwen3.5:2b", "qwen3.5:4b"])
    parser.add_argument("--context", type=int, default=16_384)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--real-indexes",
        nargs="+",
        type=int,
        default=None,
        help="one to three indexes in the five-packet representative set",
    )
    parser.add_argument("--no-success-benchmark", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_diagnostics(
            args.phase7_root,
            models=tuple(args.models),
            expected_generation_id=args.expected_generation_id,
            endpoint=args.endpoint,
            context_tokens=args.context,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            success_benchmark=not args.no_success_benchmark,
            real_packet_indexes=(
                tuple(args.real_indexes) if args.real_indexes is not None else None
            ),
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0 if report["phase7_unchanged"] else 1
    except Exception as error:
        print(json.dumps({"status": type(error).__name__, "error": str(error)[:256]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
