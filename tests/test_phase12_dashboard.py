import json
import sqlite3
from datetime import datetime, timedelta, timezone

from tradingagents.forex.hft.dashboard import read_hft_dashboard
from tradingagents.forex.hft.store import HftShadowStore

UTC = timezone.utc


def test_hft_dashboard_reads_scalar_shadow_metrics_read_only(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run-1", mode="REPLAY", source_fingerprint="abc")
    snapshot = read_hft_dashboard(path)
    assert snapshot.executed is False
    assert snapshot.runs == 1
    assert snapshot.ticks == 0


def test_hft_dashboard_missing_root_is_explicit(tmp_path):
    snapshot = read_hft_dashboard(tmp_path / "missing.sqlite3")
    assert snapshot.status == "NOT_INITIALIZED"
    assert snapshot.executed is False


def test_hft_dashboard_exposes_phase12c_latency_account_plan_and_lease_metrics(tmp_path):
    path = tmp_path / "hft.sqlite3"
    store = HftShadowStore(path)
    store.initialize()
    store.start_run("run-1", mode="SHADOW", source_fingerprint="abc")
    now = datetime.now(UTC)
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE hft_runs SET started_at=?, ended_at=?, status='COMPLETED' WHERE run_id='run-1'",
            (now.isoformat(), (now + timedelta(seconds=1)).isoformat()),
        )
        for index, timing in enumerate((0.2, 0.4, 0.8), start=1):
            db.execute(
                "INSERT INTO hft_actions(action_id,run_id,timestamp,action,reason,processing_ms) VALUES(?,?,?,?,?,?)",
                (f"a-{index}", "run-1", now.isoformat(), "NO_ACTION", "TEST", timing),
            )
        db.execute(
            "INSERT INTO hft_plans(plan_id,run_id,symbol,created_at,expires_at,payload_json) VALUES(?,?,?,?,?,?)",
            (
                "plan-1",
                "run-1",
                "EURUSD",
                now.isoformat(),
                (now + timedelta(seconds=1)).isoformat(),
                json.dumps({"plan_id": "plan-1", "direction": "NONE", "executed": False}),
            ),
        )
        db.execute(
            "INSERT INTO hft_account(run_id,timestamp,payload_json) VALUES(?,?,?)",
            (
                "run-1",
                now.isoformat(),
                json.dumps(
                    {
                        "balance": 100.0,
                        "equity": 100.0,
                        "max_drawdown": 0.0,
                        "compound_return": 0.0,
                        "trades": 0,
                        "win_rate": None,
                        "profit_factor": None,
                        "expectancy": None,
                        "benchmark_10pct_days": 0,
                    }
                ),
            ),
        )
        db.commit()

    snapshot = read_hft_dashboard(path)
    assert snapshot.p99_processing_ms == 0.8
    assert snapshot.ticks_per_second is not None
    assert snapshot.active_plan_id == "plan-1"
    assert snapshot.active_plan_direction == "NONE"
    assert snapshot.benchmark_target == 0.10
    assert snapshot.executed is False
