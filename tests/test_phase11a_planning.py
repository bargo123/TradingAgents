from types import SimpleNamespace

from tradingagents.distillation.models import (
    DISTILLATION_POLICY_VERSION,
    GROUNDING_POLICY_VERSION,
    SPLIT_POLICY_VERSION,
    SourceBlock,
    SourceRef,
)
from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner


def test_planning_is_deterministic_and_bounded():
    def b(i, t):
        return SourceBlock(
            SourceRef("d", "x.pdf", "h", f"c{i}", "g", section="s"), t, reading_order=i
        )

    source = SimpleNamespace(
        generation_id="g", blocks=lambda: (b(2, "OFI definition"), b(1, "OFI equation"))
    )
    a = SourcePacketPlanner.plan(source, ("OFI",), PlannerConfig(max_chars=1000))
    z = SourcePacketPlanner.plan(source, ("OFI",), PlannerConfig(max_chars=1000))
    assert [p.packet_id for p in a.packets] == [p.packet_id for p in z.packets]
    assert a.request_fingerprint == z.request_fingerprint and all(
        len(p.text) <= 1000 for p in a.packets
    )


def test_planner_binds_source_fingerprints_and_structural_companions():
    def b(i, t, typ):
        return SourceBlock(
            SourceRef("d", "x.pdf", "h", f"c{i}", "g", section=f"s{i}"),
            t,
            content_type=typ,
            reading_order=i,
        )

    source = SimpleNamespace(
        generation_id="g",
        source_fingerprints={"catalog": "cat"},
        blocks=lambda: (b(1, "table values", "TABLE"), b(2, "table caption", "FIGURE_CAPTION")),
    )
    plan = SourcePacketPlanner.plan(source, ("table",), PlannerConfig(max_blocks=2))
    assert plan.packets and len(plan.packets[0].blocks) == 2
    assert plan.to_dict().get("source_fingerprints") == {"catalog": "cat"}


def test_planner_keeps_preceding_and_following_structural_companions_and_policy_identity():
    def b(i, t, typ):
        return SourceBlock(
            SourceRef("d", "x.pdf", "h", f"c{i}", "g", section="s"),
            t,
            content_type=typ,
            reading_order=i,
        )

    source = SimpleNamespace(
        generation_id="g",
        source_fingerprints={"catalog": "cat"},
        blocks=lambda: (
            b(1, "definition of OFI", "DEFINITION"),
            b(2, "OFI = bid flow - ask flow", "EQUATION"),
            b(3, "OFI table values", "TABLE"),
            b(4, "caption for OFI table", "FIGURE_CAPTION"),
        ),
    )
    plan = SourcePacketPlanner.plan(source, ("OFI",), PlannerConfig(max_blocks=4))
    ids = [block.ref.chunk_id for packet in plan.packets for block in packet.blocks]
    assert ids == ["c1", "c2", "c3", "c4"]
    assert plan.policy_versions == {
        "distillation": DISTILLATION_POLICY_VERSION,
        "grounding": GROUNDING_POLICY_VERSION,
        "split": SPLIT_POLICY_VERSION,
    }
    changed = SimpleNamespace(
        generation_id="g",
        source_fingerprints={"catalog": "different"},
        blocks=source.blocks,
    )
    changed_plan = SourcePacketPlanner.plan(changed, ("OFI",), PlannerConfig(max_blocks=4))
    assert plan.request_fingerprint != changed_plan.request_fingerprint
