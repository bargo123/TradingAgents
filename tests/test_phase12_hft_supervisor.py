from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from tradingagents.forex.hft.models import Direction
from tradingagents.forex.hft.plan_store import AtomicPlanStore
from tradingagents.forex.hft.runtime import HftShadowConfig
from tradingagents.forex.hft.store import HftShadowStore
from tradingagents.forex.hft.supervisor import HftShadowWorker
from tradingagents.forex.shadow import ShadowTradeDecision

UTC = timezone.utc


def _decision(action="BUY"):
    completed = datetime.now(UTC) - timedelta(seconds=1)
    return ShadowTradeDecision(
        decision_id="decision-worker",
        created_at=completed,
        snapshot_timestamp=completed - timedelta(minutes=1),
        analysis_date=completed.date(),
        requested_symbol="EURUSD",
        resolved_symbol="EURUSD",
        action=action,
        raw_portfolio_manager_result={"rating": action},
        normalization_status="NORMALIZED",
        normalization_error=None,
        confidence=0.8,
        reference_bid=1.1,
        reference_ask=1.1001,
        reference_mid=1.10005,
        spread=0.0001,
        spread_points=10,
        analysis_timeframe="M15",
        trader_summary="summary",
        portfolio_manager_summary="summary",
        source_run_id="run-worker",
        valid_for_seconds=300,
        valid_until=completed + timedelta(seconds=240),
        decision_context_status="COMPLETE",
        decision_reference_status="UNAVAILABLE",
    )


class _Provider:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def initialize(self):
        return True

    def get_tick(self, symbol):
        self.calls += 1
        return SimpleNamespace(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            bid=1.1,
            ask=1.1001,
        )

    def shutdown(self):
        self.closed = True


def test_worker_accepts_only_valid_plan_and_hold_is_none(tmp_path: Path):
    artifact = tmp_path / "hft.sqlite3"
    plans = AtomicPlanStore()
    worker = HftShadowWorker(
        lambda **_: _Provider(),
        plans,
        config=HftShadowConfig(max_ticks=1, artifact_path=artifact),
        store=HftShadowStore(artifact),
        git_commit="abc123",
    )
    assert worker.handle_decision(_decision("BUY")) is True
    current = plans.current(datetime.now(UTC), "EURUSD", require_provenance=True)
    assert current is not None and current.primary_direction is Direction.LONG
    assert worker.handle_decision(_decision("HOLD")) is True
    current = plans.current(datetime.now(UTC), "EURUSD", require_provenance=True)
    assert current is not None and current.primary_direction is Direction.NONE


def test_worker_runs_one_tick_and_stops_without_execution(tmp_path: Path):
    provider = _Provider()
    worker = HftShadowWorker(
        lambda **_: provider,
        AtomicPlanStore(),
        config=HftShadowConfig(max_ticks=1, artifact_path=tmp_path / "hft.sqlite3"),
        store=HftShadowStore(tmp_path / "hft.sqlite3"),
        git_commit="abc123",
    )
    worker.start()
    assert worker.join(timeout=10) is True
    assert worker.result is not None
    assert worker.result["executed"] is False
    assert provider.calls == 1
    worker.stop()
