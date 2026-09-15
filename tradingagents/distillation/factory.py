"""Deterministic Phase 7 knowledge-distillation orchestration."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .dedup import DedupIndex
from .models import (
    CandidateStatus,
    DatasetExclusion,
    Difficulty,
    GroundingClaim,
    KnowledgeExample,
    LessonType,
    SourceRef,
    canonical_json,
)
from .writer import validate_generation, write_generation


@dataclass(frozen=True, slots=True)
class BuildReport:
    generation: Any = None
    accepted: int = 0
    excluded: int = 0
    teacher_calls: int = 0
    network_attempts: int = 0
    mt5_calls: int = 0
    tool_calls: int = 0
    errors: tuple[str, ...] = ()


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _packet_ref_map(packet: Any) -> dict[str, SourceRef]:
    result: dict[str, SourceRef] = {}
    for ref in (_get(packet, "refs", ()) or ()):
        ref_id = _get(ref, "ref_id", None) or _get(ref, "block_id", None) or _get(ref, "chunk_id", None)
        if ref_id:
            result[str(ref_id)] = ref
    return result


def _coerce_ref(value: Any, packet_refs: Mapping[str, SourceRef]) -> SourceRef:
    if isinstance(value, SourceRef):
        return value
    if isinstance(value, str):
        ref = packet_refs.get(value)
        if ref is None:
            raise ValueError(f"unknown source ref: {value}")
        return ref
    if isinstance(value, Mapping):
        key = value.get("ref_id") or value.get("block_id") or value.get("chunk_id")
        if key in packet_refs and not value.get("document_id"):
            return packet_refs[str(key)]
        return SourceRef(document_id=str(value["document_id"]), source_filename=str(value["source_filename"]), source_hash=str(value["source_hash"]), chunk_id=str(value.get("chunk_id", key or "")), generation_id=str(value["generation_id"]), page=value.get("page"), page_start=value.get("page_start"), page_end=value.get("page_end"), chapter=value.get("chapter"), section=value.get("section"), content_type=value.get("content_type", "PROSE"))
    raise ValueError("source ref must be a string or object")


def _coerce_candidate(candidate: Any, packet: Any) -> KnowledgeExample:
    if isinstance(candidate, KnowledgeExample):
        return candidate
    if not isinstance(candidate, Mapping):
        raise ValueError("teacher candidate must be a mapping or KnowledgeExample")
    refs = _packet_ref_map(packet)
    source_values = candidate.get("source_refs", candidate.get("refs", ())) or ()
    source_refs = tuple(_coerce_ref(item, refs) for item in source_values)
    claims: list[GroundingClaim] = []
    for item in candidate.get("claims", candidate.get("grounding_claims", ())) or ():
        if isinstance(item, GroundingClaim):
            claims.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ValueError("claim must be an object")
        claim_refs = item.get("source_refs", item.get("refs", ())) or ()
        claims.append(GroundingClaim(str(item.get("claim", item.get("text", ""))), tuple(_coerce_ref(r, refs) for r in claim_refs)))
    content = dict(candidate)
    content["source_refs"] = source_refs
    content["claims"] = tuple(claims)
    stable = dict(content)
    stable.pop("example_id", None)
    content["example_id"] = str(content.get("example_id") or hashlib.sha256(canonical_json(stable).encode()).hexdigest())
    content["lesson_type"] = LessonType(content["lesson_type"])
    content["difficulty"] = Difficulty(content.get("difficulty", Difficulty.FOUNDATIONAL))
    content.setdefault("system", content.get("system_instruction", "Explain the supplied source evidence."))
    content.setdefault("user", content.get("user_instruction", content.get("question", "Explain the supplied source evidence.")))
    content.setdefault("assistant", content.get("assistant_target", content.get("answer", "")))
    content.setdefault("source_type", "BOOK_KNOWLEDGE")
    content.setdefault("status", CandidateStatus.ACCEPTED)
    names = ("example_id", "lesson_type", "topic", "difficulty", "system", "user", "assistant", "source_refs", "claims", "source_type", "status")
    return KnowledgeExample(**{name: content[name] for name in names if name in content})


def _reason(value: Any, default: str) -> str:
    reason = getattr(value, "reason", None) or default
    return str(getattr(reason, "value", reason))


class DistillationFactory:
    def __init__(self, *, grounding=None, quality=None, dedup=None, splitter=None):
        self.grounding = grounding
        self.quality = quality
        self.dedup = dedup or DedupIndex()
        self.splitter = splitter

    def plan(self, source, config):
        from .planning import PlannerConfig, SourcePacketPlanner
        topics = _get(config, "topics", ())
        planner_config = config if isinstance(config, PlannerConfig) else PlannerConfig(max_blocks=_get(config, "max_blocks", 8), max_chars=_get(config, "max_chars", 12000), max_estimated_tokens=_get(config, "max_estimated_tokens", 3000), min_score=_get(config, "min_score", 1))
        return SourcePacketPlanner.plan(source, topics, planner_config)

    def distill(self, plan, teacher, output_root, *, metadata=None):
        if teacher is None:
            raise RuntimeError("DISTILLATION_TEACHER_NOT_CONFIGURED")
        accepted: list[KnowledgeExample] = []
        excluded: list[DatasetExclusion] = []
        errors: list[str] = []
        calls = 0
        for packet in getattr(plan, "packets", plan):
            calls += 1
            packet_id = _get(packet, "packet_id")
            try:
                result = teacher.generate(packet, getattr(teacher, "config", None))
                error_code = _get(result, "error_code")
                if error_code:
                    excluded.append(DatasetExclusion("TEACHER_FAILED", packet_id=packet_id, diagnostic=str(error_code)))
                    continue
                candidate = _coerce_candidate(_get(result, "candidate", result), packet)
                if self.grounding is not None:
                    report = self.grounding.validate(candidate, packet)
                    if not getattr(report, "accepted", bool(report)):
                        excluded.append(DatasetExclusion(_reason(report, "GROUNDING_FAILED"), packet_id=packet_id, diagnostic=str(_get(report, "diagnostics", ""))))
                        continue
                if self.quality is not None:
                    report = self.quality.validate(candidate, packet)
                    if not getattr(report, "accepted", bool(report)):
                        excluded.append(DatasetExclusion(_reason(report, "SCHEMA_INVALID"), packet_id=packet_id, diagnostic=str(_get(report, "diagnostics", ""))))
                        continue
                duplicate_reason = self.dedup.check(candidate)
                if duplicate_reason:
                    excluded.append(DatasetExclusion(duplicate_reason, packet_id=packet_id, candidate_id=candidate.example_id))
                    continue
                accepted.append(candidate)
            except Exception as exc:
                code = "SCHEMA_INVALID" if isinstance(exc, (TypeError, ValueError, KeyError)) else "TEACHER_FAILED"
                errors.append(type(exc).__name__)
                excluded.append(DatasetExclusion(code, packet_id=packet_id, diagnostic=type(exc).__name__))
        metadata_out = dict(metadata or {})
        if self.splitter is None:
            from .splits import GroupedSplitter
            self.splitter = GroupedSplitter()
        try:
            assignments = self.splitter.assign(accepted)
            metadata_out.setdefault("split_status", "COMPLETE")
        except ValueError as exc:
            if str(exc) != "INSUFFICIENT_DATA":
                raise
            assignments = {}
            metadata_out["split_status"] = "INSUFFICIENT_DATA"
        generation = write_generation(output_root, accepted, excluded, assignments, metadata=metadata_out)
        return BuildReport(generation, len(accepted), len(excluded), calls, errors=tuple(errors))

    @staticmethod
    def validate(path):
        return validate_generation(path)


def plan(source, config):
    return DistillationFactory().plan(source, config)


def distill(plan_value, teacher, output_root, **kwargs):
    return DistillationFactory().distill(plan_value, teacher, output_root, **kwargs)


__all__ = ["DistillationFactory", "BuildReport", "plan", "distill"]
