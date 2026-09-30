from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from tradingagents.forex.hft.models import Direction
from tradingagents.forex.hft.plan_store import AtomicPlanStore
from tradingagents.forex.hft.runtime import HftShadowConfig
from tradingagents.forex.hft.store import HftShadowStore
from tradingagents.forex.hft.supervisor import HftShadowWorker
from tradingagents.forex.shadow import ShadowTradeDecision
from tradingagents.forex.supervisor import _SharedReadOnlyMt5Provider

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
        decision_completed_timestamp=completed,
        decision_reference_timestamp=completed,
        decision_reference_bid=1.1,
        decision_reference_ask=1.1001,
        decision_reference_spread=0.0001,
        decision_reference_spread_points=10,
        decision_reference_status="AVAILABLE",
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


def test_worker_validator_rejects_source_decision_before_plan_creation(tmp_path: Path):
    artifact = tmp_path / "hft.sqlite3"
    plans = AtomicPlanStore()
    worker = HftShadowWorker(
        lambda **_: _Provider(),
        plans,
        config=HftShadowConfig(max_ticks=1, artifact_path=artifact),
        store=HftShadowStore(artifact),
        git_commit="abc123",
        decision_validator=lambda _decision: False,
    )

    assert worker.handle_decision(_decision("BUY")) is False
    assert plans.current(datetime.now(UTC), "EURUSD", require_provenance=True) is None


def test_worker_clears_directional_plan_when_temporal_update_is_invalid(tmp_path: Path):
    artifact = tmp_path / "hft.sqlite3"
    plans = AtomicPlanStore()
    worker = HftShadowWorker(
        lambda **_: _Provider(),
        plans,
        config=HftShadowConfig(max_ticks=1, artifact_path=artifact),
        store=HftShadowStore(artifact),
        git_commit="abc123",
    )
    valid = _decision("BUY")
    assert worker.handle_decision(valid) is True
    invalid = replace(
        valid,
        decision_reference_status="INVALID_TEMPORAL",
        decision_reference_timestamp=valid.decision_completed_timestamp - timedelta(seconds=1),
        decision_reference_delay_seconds=None,
    )

    assert worker.handle_decision(invalid) is False
    assert plans.current(datetime.now(UTC), "EURUSD", require_provenance=True) is None


def test_supervisor_shared_provider_has_one_lifecycle_and_read_only_close():
    class Provider:
        def __init__(self):
            self.initializations = 0
            self.shutdowns = 0

        def initialize(self):
            self.initializations += 1
            return True

        def shutdown(self):
            self.shutdowns += 1

        def get_tick(self, symbol):
            return symbol

    underlying = Provider()
    shared = _SharedReadOnlyMt5Provider(underlying)

    assert shared.initialize() is True
    assert shared.initialize() is True
    shared.shutdown()
    assert underlying.initializations == 1
    assert underlying.shutdowns == 0
    assert shared.get_tick("EURUSD") == "EURUSD"
    shared.close()
    shared.close()
    assert underlying.shutdowns == 1
    names = {name.casefold() for name in dir(shared)}
    assert not any(
        token in name
        for name in names
        for token in ("order_send", "buy", "sell", "close_position", "modify_position")
    )
