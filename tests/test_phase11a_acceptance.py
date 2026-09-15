"""Offline positive-path acceptance for the Phase 11A factory."""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.phase11a_knowledge import fixture_source, grounded_candidate


def test_offline_fixture_covers_closed_lesson_types(tmp_path: Path) -> None:
    from tradingagents.distillation.dedup import DedupIndex
    from tradingagents.distillation.factory import DistillationFactory
    from tradingagents.distillation.grounding import GroundingValidator
    from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner
    from tradingagents.distillation.quality import QualityPolicy
    from tradingagents.distillation.splits import GroupedSplitter
    from tradingagents.distillation.teacher import FakeTeacher

    lesson_types = (
        "DEFINITION",
        "MECHANISM",
        "SCENARIO_APPLICATION",
        "COMPARISON",
        "EQUATION_INTERPRETATION",
        "FAILURE_MODE",
    )
    plan = SourcePacketPlanner.plan(
        fixture_source(),
        (),
        PlannerConfig(max_blocks=1, max_chars=2_000, max_estimated_tokens=400),
    )

    def response(packet):
        index = len(teacher.calls)
        lesson_type = lesson_types[index % len(lesson_types)]
        return grounded_candidate(packet, lesson_type=lesson_type, topic=f"topic-{index}")

    teacher = FakeTeacher(response)
    report = DistillationFactory(
        grounding=GroundingValidator(),
        quality=QualityPolicy(max_lesson_chars=2_000),
        dedup=DedupIndex(near_threshold=1.0),
        splitter=GroupedSplitter(min_groups=1),
    ).distill(plan, teacher, tmp_path)
    manifest = json.loads((Path(report.generation) / "manifest.json").read_text(encoding="utf-8"))
    distribution = manifest["metadata"]["lesson_type_distribution"]
    assert report.accepted >= len(lesson_types)
    assert set(lesson_types).issubset(distribution)
    assert manifest["metadata"]["candidate_count"] == report.teacher_calls
    assert manifest["metadata"]["difficulty_distribution"]
    assert manifest["metadata"]["source_coverage"]["sections"] >= 1


def test_offline_distillation_publishes_grounded_grouped_generation(tmp_path: Path) -> None:
    from tradingagents.distillation.dedup import DedupIndex
    from tradingagents.distillation.factory import DistillationFactory
    from tradingagents.distillation.grounding import GroundingValidator
    from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner
    from tradingagents.distillation.quality import QualityPolicy
    from tradingagents.distillation.splits import GroupedSplitter
    from tradingagents.distillation.teacher import FakeTeacher

    source = fixture_source()
    plan = SourcePacketPlanner.plan(
        source,
        ("order flow imbalance", "liquidity", "market impact", "inventory risk"),
        PlannerConfig(max_blocks=1, max_chars=2_000, max_estimated_tokens=400),
    )
    teacher = FakeTeacher(lambda packet: grounded_candidate(packet))
    report = DistillationFactory(
        grounding=GroundingValidator(),
        quality=QualityPolicy(max_lesson_chars=2_000),
        dedup=DedupIndex(),
        splitter=GroupedSplitter(min_groups=1),
    ).distill(plan, teacher, tmp_path)

    assert report.teacher_calls == len(plan.packets)
    assert report.accepted >= 1
    generation = Path(report.generation)
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["safety"]["network_attempts"] == 0
    assert manifest["safety"]["mt5_calls"] == 0
    assert manifest["counts"]["accepted"] == report.accepted
    assert (generation / "train.jsonl").is_file()
    assert (generation / "validation.jsonl").is_file()
    assert (generation / "test.jsonl").is_file()
    assert DistillationFactory.validate(generation).valid


def test_no_action_or_future_labels_can_enter_fixture_lesson(tmp_path: Path) -> None:
    from tradingagents.distillation.factory import DistillationFactory
    from tradingagents.distillation.grounding import GroundingValidator
    from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner
    from tradingagents.distillation.quality import QualityPolicy
    from tradingagents.distillation.teacher import FakeTeacher

    source = fixture_source()
    plan = SourcePacketPlanner.plan(source, ("liquidity",), PlannerConfig(max_blocks=1))
    candidate = grounded_candidate(plan.packets[0])
    candidate["assistant_target"] = "BUY EURUSD now"
    report = DistillationFactory(
        grounding=GroundingValidator(),
        quality=QualityPolicy(),
    ).distill(plan, FakeTeacher(candidate), tmp_path)
    assert report.accepted == 0
    assert report.excluded >= 1
