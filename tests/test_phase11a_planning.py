from types import SimpleNamespace
from tradingagents.distillation.models import SourceBlock, SourceRef
from tradingagents.distillation.planning import SourcePacketPlanner, PlannerConfig
def test_planning_is_deterministic_and_bounded():
    def b(i,t): return SourceBlock(SourceRef("d","x.pdf","h",f"c{i}","g",section="s"),t,reading_order=i)
    source=SimpleNamespace(generation_id="g",blocks=lambda:(b(2,"OFI definition"),b(1,"OFI equation")))
    a=SourcePacketPlanner.plan(source,("OFI",),PlannerConfig(max_chars=1000)); z=SourcePacketPlanner.plan(source,("OFI",),PlannerConfig(max_chars=1000))
    assert [p.packet_id for p in a.packets]==[p.packet_id for p in z.packets]
    assert a.request_fingerprint==z.request_fingerprint and all(len(p.text)<=1000 for p in a.packets)

