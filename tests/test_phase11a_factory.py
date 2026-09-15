from pathlib import Path

from tests.fixtures.phase11a_knowledge import fixture_source, grounded_candidate
from tradingagents.distillation.factory import DistillationFactory
from tradingagents.distillation.grounding import GroundingValidator
from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner
from tradingagents.distillation.quality import QualityPolicy
from tradingagents.distillation.teacher import FakeTeacher


def test_factory_coerces_structured_candidate_and_records_failures(tmp_path: Path):
    source = fixture_source()
    plan = SourcePacketPlanner.plan(source, ("liquidity",), PlannerConfig(max_blocks=1))
    teacher = FakeTeacher(lambda packet: grounded_candidate(packet))
    report = DistillationFactory(grounding=GroundingValidator(), quality=QualityPolicy()).distill(
        plan, teacher, tmp_path
    )
    assert report.teacher_calls == len(plan.packets)
    assert report.accepted >= 1
    assert report.generation.is_dir()


def test_factory_does_not_fail_when_groups_are_insufficient(tmp_path: Path):
    source = fixture_source()
    plan = SourcePacketPlanner.plan(source, ("liquidity",), PlannerConfig(max_blocks=1))
    report = DistillationFactory(grounding=GroundingValidator(), quality=QualityPolicy()).distill(
        plan, FakeTeacher(lambda p: grounded_candidate(p)), tmp_path
    )
    assert report.generation.is_dir()


def test_factory_defaults_to_mandatory_safety_validation(tmp_path: Path):
    source = fixture_source()
    plan = SourcePacketPlanner.plan(source, ("liquidity",), PlannerConfig(max_blocks=1))
    candidate = grounded_candidate(plan.packets[0])
    candidate["source_refs"] = []
    candidate["claims"] = []
    candidate["assistant_target"] = "BUY now"
    report = DistillationFactory().distill(plan, FakeTeacher(candidate), tmp_path)
    assert report.accepted == 0


def test_factory_rejects_unknown_teacher_fields_instead_of_dropping_them(tmp_path: Path):
    source = fixture_source()
    plan = SourcePacketPlanner.plan(source, ("liquidity",), PlannerConfig(max_blocks=1))
    candidate = grounded_candidate(plan.packets[0])
    candidate["reasoning_trace"] = "hidden"
    report = DistillationFactory().distill(plan, FakeTeacher(candidate), tmp_path)
    assert report.accepted == 0
    assert report.excluded >= 1
