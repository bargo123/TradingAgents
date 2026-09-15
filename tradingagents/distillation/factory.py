"""Deterministic Phase 7 knowledge distillation orchestration."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .writer import write_generation, validate_generation

@dataclass(frozen=True)
class BuildReport:
    generation: Any = None
    accepted: int = 0
    excluded: int = 0
    teacher_calls: int = 0
    network_attempts: int = 0
    mt5_calls: int = 0
    errors: tuple[str, ...] = ()

class DistillationFactory:
    def __init__(self, *, grounding=None, quality=None, dedup=None, splitter=None):
        self.grounding = grounding
        self.quality = quality
        self.dedup = dedup
        self.splitter = splitter

    def plan(self, source, config):
        from .planning import SourcePacketPlanner
        return SourcePacketPlanner().plan(source, getattr(config, "topics", ()), config)

    def distill(self, plan, teacher, output_root, *, metadata=None):
        if teacher is None:
            raise RuntimeError("DISTILLATION_TEACHER_NOT_CONFIGURED")
        accepted, excluded, errors = [], [], []
        calls = 0
        packets = getattr(plan, "packets", plan)
        for packet in packets:
            calls += 1
            try:
                result = teacher.generate(packet, getattr(teacher, "config", None))
                if getattr(result, "error_code", None):
                    excluded.append({"reason": getattr(result, "error_code"), "packet_id": getattr(packet, "packet_id", None)})
                    continue
                candidate = getattr(result, "candidate", result)
                if self.grounding:
                    report = self.grounding.validate(candidate, packet)
                    valid = report if isinstance(report, bool) else getattr(report, "valid", False)
                    if not valid:
                        excluded.append(candidate)
                        continue
                if self.quality:
                    report = self.quality.validate(candidate, packet)
                    valid = report if isinstance(report, bool) else getattr(report, "valid", False)
                    if not valid:
                        excluded.append(candidate)
                        continue
                if self.dedup:
                    duplicate_reason = self.dedup.check(candidate)
                    if duplicate_reason:
                        excluded.append({"reason": duplicate_reason, "candidate": candidate})
                        continue
                accepted.append(candidate)
            except Exception as exc:
                errors.append(type(exc).__name__)
                excluded.append({"reason": "TEACHER_FAILED", "diagnostic": type(exc).__name__})
        assignments = self.splitter.assign(accepted) if self.splitter else {}
        generation = write_generation(output_root, accepted, excluded, assignments, metadata=metadata)
        return BuildReport(generation, len(accepted), len(excluded), calls, errors=tuple(errors))

    @staticmethod
    def validate(path):
        return validate_generation(path)

def plan(source, config):
    return DistillationFactory().plan(source, config)

def distill(plan_value, teacher, output_root, **kwargs):
    return DistillationFactory().distill(plan_value, teacher, output_root, **kwargs)

__all__ = ["DistillationFactory", "BuildReport", "plan", "distill"]
