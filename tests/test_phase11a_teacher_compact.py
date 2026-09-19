import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from tests.fixtures.phase11a_knowledge import fixture_source
from tradingagents.distillation.grounding import GroundingValidator
from tradingagents.distillation.models import SourcePlan
from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner
from tradingagents.distillation.quality import QualityPolicy
from tradingagents.distillation.teacher import (
    TeacherAtom,
    _compact_packet_prompt,
    hydrate_teacher_atom,
    packet_aliases,
)


def _plan_two_blocks():
    plan = SourcePacketPlanner.plan(
        fixture_source(), (), PlannerConfig(max_blocks=2, max_chars=2_000, max_estimated_tokens=400)
    )
    packet = plan.packets[0]
    return replace(packet, blocks=(packet.blocks[0], packet.blocks[0]))


def _atom(*refs: str, claim: str = "The supplied source describes the concept."):
    return TeacherAtom(
        lesson_type="DEFINITION",
        difficulty="FOUNDATIONAL",
        topic="order flow imbalance",
        question="What is the concept?",
        answer="It describes a relationship in the supplied evidence.",
        claims=[{"claim": claim, "refs": list(refs)}],
    )


def test_compact_aliases_hydrate_to_exact_packet_refs_and_complete_provenance():
    packet = _plan_two_blocks()
    atom = _atom("B0", "B1")

    candidate = hydrate_teacher_atom(atom, packet)

    assert candidate["source_refs"][0] is packet.blocks[0].ref
    assert candidate["source_refs"][1] is packet.blocks[1].ref
    assert candidate["claims"][0].source_refs[0] is packet.blocks[0].ref
    assert candidate["claims"][0].source_refs[1] is packet.blocks[1].ref
    assert candidate["source_refs"][0].source_hash == packet.blocks[0].ref.source_hash
    assert candidate["source_refs"][0].generation_id == packet.blocks[0].ref.generation_id
    assert candidate["system"] == "Explain the supplied source evidence only."


def test_packet_aliases_are_deterministic_and_packet_local():
    packet = _plan_two_blocks()

    first = packet_aliases(packet)
    second = packet_aliases(packet)

    assert first == second
    assert list(first) == ["B0", "B1"]
    assert first["B0"] is packet.blocks[0].ref


def test_compact_schema_and_prompt_require_exact_packet_aliases():
    packet = _plan_two_blocks()
    schema = TeacherAtom.model_json_schema()
    prompt = _compact_packet_prompt(packet)

    claim_schema = schema["$defs"]["TeacherAtomClaim"]
    assert claim_schema["properties"]["refs"]["items"]["pattern"] == r"^B[0-9]+$"
    assert "ALIAS: B0" in prompt[1]["content"]
    assert "source_hash" not in prompt[1]["content"]
    assert "generation_id" not in prompt[1]["content"]


@pytest.mark.parametrize(
    "atom",
    [
        _atom("B9"),
        _atom("B0", "B0"),
        _atom("B0", "B9"),
    ],
)
def test_unknown_duplicate_or_out_of_packet_aliases_fail_closed(atom):
    packet = _plan_two_blocks()

    with pytest.raises(ValueError):
        hydrate_teacher_atom(atom, packet)


def test_empty_aliases_fail_schema_validation():
    with pytest.raises(ValidationError):
        TeacherAtom.model_validate(
            {
                "lesson_type": "DEFINITION",
                "difficulty": "FOUNDATIONAL",
                "topic": "topic",
                "question": "question",
                "answer": "answer",
                "claims": [{"claim": "claim", "refs": []}],
            }
        )


def test_malformed_or_truncated_json_fails_closed():
    with pytest.raises((ValidationError, ValueError, json.JSONDecodeError)):
        TeacherAtom.model_validate_json('{"lesson_type":"DEFINITION"')


def test_unsupported_claim_fails_after_alias_hydration():
    packet = _plan_two_blocks()
    candidate = hydrate_teacher_atom(_atom("B0", claim="The moon is made of cheese."), packet)

    report = GroundingValidator().validate(candidate, packet)

    assert not report.accepted
    assert report.reason == "UNSUPPORTED_CLAIM"


def test_quality_policy_still_runs_after_alias_hydration():
    packet = _plan_two_blocks()
    candidate = hydrate_teacher_atom(_atom("B0", claim="The evidence supports this."), packet)
    candidate["assistant"] = "BUY"

    report = QualityPolicy().validate(candidate, packet)

    assert not report.accepted
    assert report.reason == "UNSAFE_FUTURE_OUTCOME_INFERENCE"


def test_hydration_deterministic_metadata_is_identical_across_runs():
    packet = _plan_two_blocks()
    atom = _atom("B0")

    first = hydrate_teacher_atom(atom, packet)
    second = hydrate_teacher_atom(atom, packet)

    assert first["system"] == second["system"]
    assert first["user"] == second["user"]
    assert first["source_refs"] == second["source_refs"]
    assert first["claims"] == second["claims"]


def test_compact_bounds_keep_one_atomic_lesson():
    with pytest.raises(ValidationError):
        TeacherAtom(
            lesson_type="DEFINITION",
            difficulty="FOUNDATIONAL",
            topic="t" * 101,
            question="q",
            answer="a",
            claims=[{"claim": "c", "refs": ["B0"]}],
        )


def test_planning_source_type_is_read_only_and_does_not_require_teacher():
    source = fixture_source()

    plan = SourcePacketPlanner.plan(
        source, (), PlannerConfig(max_blocks=1, max_chars=500, max_estimated_tokens=100)
    )

    assert isinstance(plan, SourcePlan)
    assert plan.packets
