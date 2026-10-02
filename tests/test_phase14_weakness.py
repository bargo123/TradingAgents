from datetime import datetime, timezone

from tradingagents.self_enhancement.models import ExecutionMode, ExperienceTrade
from tradingagents.self_enhancement.weakness import detect_weaknesses


def _trade(index: int, pnl: float) -> ExperienceTrade:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ExperienceTrade(
        experience_id=f"e-{index}", source_database_id="demo", source_position_id=str(index),
        strategy_id="range_rejection", strategy_version="v1", config_version="v1",
        symbol="EURUSD", direction="LONG", entry_timestamp=start,
        exit_timestamp=start.replace(second=1), entry_bid=1.1, entry_ask=1.1001,
        entry_fill=1.1001, exit_bid=1.1, exit_ask=1.1001, exit_fill=1.1,
        spread_points=1, slippage_points=0, feature_snapshot={}, regime="NEUTRAL",
        expected_move_points=2, volume=.01, risk=.005, mfe_points=1, mae_points=-1,
        exit_reason="STOP_LOSS", broker_execution_latency_ms=1, gross_result=pnl,
        net_known_result=pnl, commission_known=False, profit_to_loss_flip=False,
        session="LONDON", volatility_state="NORMAL", data_quality_state="VALID",
        source_git_commit="abc", execution_mode=ExecutionMode.DEMO,
    )


def test_weakness_detection_is_minimum_sample_and_observation_only():
    assert detect_weaknesses([_trade(i, -1) for i in range(3)], minimum_samples=5) == ()
    findings = detect_weaknesses([_trade(i, -1) for i in range(5)], minimum_samples=5)
    assert {item.category for item in findings} >= {"NEGATIVE_EXPECTANCY", "POOR_PROFIT_FACTOR"}
    assert all(item.kind.value == "OBSERVATION" for item in findings)
