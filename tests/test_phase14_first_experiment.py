from tests.test_phase14_orchestrator import _sources
from tradingagents.self_enhancement.orchestrator import SelfEnhancementOrchestrator


def test_first_experiment_is_fail_closed_on_empty_verified_experience(tmp_path):
    hft, demo = _sources(tmp_path)
    report = SelfEnhancementOrchestrator(tmp_path / "phase14").run_once(
        hft_path=hft,
        demo_path=demo,
        source_commit="abc",
        minimum_verified_trades=20,
        minimum_ticks=8,
    )
    assert report.status == "NO_EXPERIMENT"
    assert report.decisions == ()
