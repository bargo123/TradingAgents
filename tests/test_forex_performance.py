from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5Bar, Mt5SymbolInfo
from tradingagents.forex.context import build_forex_market_context, snapshot_to_dict
from tradingagents.forex.performance import (
    BenchmarkConfig,
    BenchmarkReport,
    ComparisonReport,
    ModelBenchmarkResult,
    ReplayCase,
    benchmark_models,
    compare_benchmarks,
    directional_distribution,
    load_replay_case,
    run_benchmark,
)
from tradingagents.forex.telemetry import critical_path_from_intervals


def _snapshot() -> ForexMarketSnapshot:
    timestamp = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    bar = Mt5Bar(
        timestamp=timestamp,
        open=1.1,
        high=1.101,
        low=1.099,
        close=1.1005,
        tick_volume=10,
        spread=1,
        real_volume=10,
    )
    return ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSD",
        bid=1.1000,
        ask=1.1002,
        spread=0.0002,
        spread_points=2,
        m1_candles=(bar,),
        m5_candles=(bar,),
        m15_candles=(bar,),
        h1_candles=(bar,),
        account=None,
        positions=(),
        symbol_info=Mt5SymbolInfo(name="EURUSD", digits=5, point=0.0001),
    )


def _source_db(path: Path, snapshot: ForexMarketSnapshot) -> bytes:
    encoded = json.dumps(snapshot_to_dict(snapshot), sort_keys=True)
    with sqlite3.connect(path) as db:
        db.execute(
            """
            CREATE TABLE forex_watch_runs (
                run_id TEXT PRIMARY KEY,
                run_status TEXT,
                requested_symbol TEXT,
                resolved_symbol TEXT,
                analysis_snapshot_timestamp TEXT,
                analysis_snapshot_bid REAL,
                analysis_snapshot_ask REAL,
                analysis_snapshot_spread REAL,
                analysis_snapshot_spread_points REAL,
                snapshot_json TEXT
            )
            """
        )
        db.execute(
            """
            INSERT INTO forex_watch_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-1",
                "SUCCEEDED",
                "EURUSD",
                "EURUSD",
                snapshot.timestamp.isoformat().replace("+00:00", "Z"),
                snapshot.bid,
                snapshot.ask,
                snapshot.spread,
                snapshot.spread_points,
                encoded,
            ),
        )
    return path.read_bytes()


def test_load_replay_case_decodes_one_causal_saved_snapshot(tmp_path: Path) -> None:
    snapshot = _snapshot()
    database = tmp_path / "source.sqlite3"
    before = _source_db(database, snapshot)

    case = load_replay_case(database, "run-1")

    assert isinstance(case, ReplayCase)
    assert case.source_run_id == "run-1"
    assert case.snapshot == snapshot
    assert case.snapshot_bytes == json.dumps(
        snapshot_to_dict(snapshot), sort_keys=True
    ).encode()
    assert case.case_fingerprint == hashlib.sha256(case.snapshot_bytes).hexdigest()
    assert database.read_bytes() == before


def test_run_benchmark_is_replay_only_and_does_not_write_source(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot()
    database = tmp_path / "source.sqlite3"
    before_bytes = _source_db(database, snapshot)
    before_rows = sqlite3.connect(database).execute(
        "SELECT run_id, run_status FROM forex_watch_runs"
    ).fetchall()

    class FakeRunner:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def analyze(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                normalized_action="HOLD",
                normalization_status="NORMALIZED",
                decision_context_status="COMPLETE",
                metrics={
                    "elapsed_seconds": 1.25,
                    "llm_calls": 0,
                    "tool_calls": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "stage_timings": {},
                },
            )

    fake = FakeRunner()
    output = tmp_path / "bench.json"
    report = run_benchmark(
        BenchmarkConfig(
            source_database_path=database,
            source_run_id="run-1",
            output_path=output,
            runner_factory=lambda **_: fake,
        )
    )

    assert report.run_kind == "REPLAY"
    assert report.source_run_id == "run-1"
    assert report.replay_status == "COMPLETE"
    assert output.exists()
    assert fake.calls[0]["persist"] is False
    assert fake.calls[0]["snapshot_bytes"] == _snapshot_bytes(snapshot)
    assert database.read_bytes() == before_bytes
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT run_id, run_status FROM forex_watch_runs").fetchall() == before_rows


def _snapshot_bytes(snapshot: ForexMarketSnapshot) -> bytes:
    return json.dumps(snapshot_to_dict(snapshot), sort_keys=True).encode()


def test_critical_path_accounts_for_overlap_and_dependency_wait() -> None:
    intervals = [
        {"node": "Market Analyst", "started": 0.0, "finished": 4.0},
        {"node": "News Analyst", "started": 0.0, "finished": 3.0},
        {"node": "Research Manager", "started": 5.0, "finished": 7.0},
        {"node": "Trader", "started": 7.0, "finished": 9.0},
    ]
    report = critical_path_from_intervals(
        intervals,
        (("Market Analyst", "Research Manager"), ("News Analyst", "Research Manager"), ("Research Manager", "Trader")),
    )

    assert report["wall_seconds"] == 9.0
    assert report["total_node_seconds"] == 11.0
    assert report["overlap_seconds"] == 2.0
    assert report["idle_wait_seconds"] == 1.0
    assert report["critical_path_seconds"] == 9.0
    assert report["critical_path_nodes"] == ["Market Analyst", "Research Manager", "Trader"]


def test_critical_path_ignores_malformed_intervals_without_inventing_timings() -> None:
    report = critical_path_from_intervals(
        [
            {"node": "valid", "started": 1.0, "finished": 2.0},
            {"node": "missing", "started": 2.0},
            {"node": "negative", "started": 4.0, "finished": 3.0},
        ],
        (),
    )

    assert report["wall_seconds"] == 1.0
    assert report["total_node_seconds"] == 1.0
    assert report["critical_path_seconds"] == 1.0
    assert report["unknown_intervals"] == 2


def test_forex_market_context_uses_causal_features_without_raw_candle_payload() -> None:
    snapshot = _snapshot()
    payload = snapshot_to_dict(snapshot, include_candles=False)
    context = build_forex_market_context(snapshot, "INTRADAY")

    assert "candles" not in payload
    assert set(payload["features"]) == {"M1", "M5", "M15", "H1"}
    assert "average_true_range" in context
    assert "range_pct" in context
    assert "Snapshot UTC:" in context


def test_model_benchmark_reports_candidates_without_auto_selecting(tmp_path: Path) -> None:
    snapshot = _snapshot()
    database = tmp_path / "source.sqlite3"
    _source_db(database, snapshot)
    seen_models: list[Mapping[str, object]] = []

    class FakeRunner:
        def analyze(self, **kwargs):
            config = kwargs["models"]
            seen_models.append(config)
            return SimpleNamespace(
                normalized_action="HOLD",
                normalization_status="NORMALIZED",
                decision_context_status="COMPLETE",
                metrics={"elapsed_seconds": 1.0, "llm_calls": 1, "tokens_in": 4, "tokens_out": 2},
            )

    config = BenchmarkConfig(
        source_database_path=database,
        source_run_id="run-1",
        output_path=tmp_path / "benchmark.json",
        runner_factory=lambda **_: FakeRunner(),
    )
    results = benchmark_models(
        config,
        {
            "quick-2b": {"quick": "qwen3.5:2b"},
            "quick-4b": {"quick": "qwen3.5:4b"},
        },
    )

    assert all(isinstance(result, ModelBenchmarkResult) for result in results)
    assert [result.candidate for result in results] == ["quick-2b", "quick-4b"]
    assert [result.model_config for result in results] == [
        {"quick": "qwen3.5:2b"},
        {"quick": "qwen3.5:4b"},
    ]
    assert seen_models == [
        {"quick": "qwen3.5:2b"},
        {"quick": "qwen3.5:4b"},
    ]
    assert all(result.case_fingerprint for result in results)
    assert results[0].configuration_fingerprint != results[1].configuration_fingerprint
    assert all(result.selected is False for result in results)


def _benchmark_report(
    *,
    case_fingerprint: str = "case",
    configuration_fingerprint: str = "config",
    action: str | None = "HOLD",
    signal_path: Mapping[str, object] | None = None,
    elapsed: float = 10.0,
) -> BenchmarkReport:
    return BenchmarkReport(
        report_version=1,
        run_kind="REPLAY",
        source_run_id="run-1",
        case_fingerprint=case_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        symbol="EURUSD",
        analysis_timestamp="2026-01-02T12:00:00+00:00",
        replay_status="COMPLETE",
        normalized_action=action,
        normalization_status="NORMALIZED" if action else "FAILED",
        decision_context_status="COMPLETE" if action else "INCOMPLETE",
        metrics={
            "elapsed_seconds": elapsed,
            "llm_calls": 12,
            "tokens_in": 100,
            "tokens_out": 20,
            "critical_path": {"critical_path_seconds": elapsed - 1},
            "signal_path": signal_path or {
                "research_manager_recommendation": "BUY",
                "trader_action": "HOLD",
                "portfolio_manager_rating": "Hold",
            },
        },
    )


def test_compare_benchmarks_accepts_same_case_and_reports_scalar_deltas() -> None:
    baseline = _benchmark_report(elapsed=10.0)
    candidate = _benchmark_report(elapsed=8.0)

    result = compare_benchmarks(baseline, candidate)

    assert isinstance(result, ComparisonReport)
    assert result.valid is True
    assert result.deltas["elapsed_seconds"] == pytest.approx(-2.0)
    assert result.deltas["critical_path_seconds"] == pytest.approx(-2.0)
    assert result.validity["baseline_context"] == "COMPLETE"
    assert result.validity["candidate_context"] == "COMPLETE"


def test_compare_benchmarks_rejects_mismatched_case_or_configuration() -> None:
    baseline = _benchmark_report()

    mismatched_case = compare_benchmarks(
        baseline, _benchmark_report(case_fingerprint="other")
    )
    mismatched_config = compare_benchmarks(
        baseline, _benchmark_report(configuration_fingerprint="other")
    )

    assert mismatched_case.valid is False
    assert mismatched_case.reason == "CASE_FINGERPRINT_MISMATCH"
    assert mismatched_config.valid is False
    assert mismatched_config.reason == "CONFIGURATION_FINGERPRINT_MISMATCH"


def test_directional_distribution_reports_transitions_without_holding_penalty() -> None:
    reports = [
        _benchmark_report(),
        _benchmark_report(
            action="BUY",
            signal_path={
                "research_manager_recommendation": "SELL",
                "trader_action": "SELL",
                "portfolio_manager_rating": "Underweight",
            },
        ),
    ]

    distribution = directional_distribution(reports)

    assert distribution["count"] == 2
    assert distribution["portfolio_manager_ratings"] == {"Hold": 1, "Underweight": 1}
    assert distribution["transitions"] == {"BUY->HOLD->Hold": 1, "SELL->SELL->Underweight": 1}
    assert distribution["hold_count"] == 1
