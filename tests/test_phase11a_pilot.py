from tests.fixtures.phase11a_knowledge import fixture_source
from tradingagents.distillation.models import SourcePlan
from tradingagents.distillation.pilot import select_representative_packets
from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner


def test_representative_pilot_selection_is_deterministic_and_bounded():
    plan = SourcePacketPlanner.plan(
        fixture_source(), (), PlannerConfig(max_blocks=1, max_chars=2_000, max_estimated_tokens=400)
    )

    subset = select_representative_packets(plan, 5)
    again = select_representative_packets(plan, 5)

    assert isinstance(subset, SourcePlan)
    assert 1 <= len(subset.packets) <= 5
    assert [packet.packet_id for packet in subset.packets] == [
        packet.packet_id for packet in again.packets
    ]
    assert subset.request_fingerprint != plan.request_fingerprint
    assert subset.source_fingerprints == plan.source_fingerprints


def test_representative_pilot_rejects_unbounded_counts():
    plan = SourcePacketPlanner.plan(fixture_source(), (), PlannerConfig(max_blocks=1))

    for count in (0, 11):
        try:
            select_representative_packets(plan, count)
        except ValueError:
            continue
        raise AssertionError("pilot count must be bounded to 1..10")
