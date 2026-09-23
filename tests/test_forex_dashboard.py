from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.forex.evaluation import ShadowEvaluationStore
from tradingagents.forex.shadow import ShadowDecisionStore, ShadowTradeDecision
from tradingagents.forex.watch_store import WatcherStore

UTC = timezone.utc


def _ts(minutes: int = 0) -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC) + timedelta(minutes=minutes)


def _init_db(tmp_path: Path) -> Path:
    path = tmp_path / "dashboard.db"
    WatcherStore(path).initialize()
    ShadowDecisionStore(path).initialize()
    ShadowEvaluationStore(path).initialize()
    return path


def _decision(
    decision_id: str,
    *,
    context: str = "COMPLETE",
    normalization: str = "NORMALIZED",
    action: str | None = "HOLD",
    stale: int = 0,
    reference_status: str = "AVAILABLE",
) -> ShadowTradeDecision:
    completed = _ts()
    reference_timestamp = None
    reference_quote = (None, None, None, None)
    delay = None
    if reference_status == "AVAILABLE":
        reference_timestamp = completed + timedelta(seconds=1)
        reference_quote = (1.1000, 1.1001, 0.0001, 1.0)
        delay = 1.0
    elif reference_status == "INVALID_TEMPORAL":
        reference_timestamp = completed - timedelta(seconds=1)
        reference_quote = (1.1000, 1.1001, 0.0001, 1.0)
        delay = -1.0
    return ShadowTradeDecision(
        decision_id=decision_id,
        created_at=_ts(),
        analysis_date=date(2026, 1, 1),
        requested_symbol="EURUSD",
        resolved_symbol="EURUSD",
        snapshot_timestamp=_ts(-1),
        action=action,
        normalization_status=normalization,
        normalization_error=None if normalization == "NORMALIZED" else "failed",
        raw_portfolio_manager_result={"rating": "Hold"},
        confidence=0.5,
        reference_bid=1.1000,
        reference_ask=1.1001,
        reference_mid=1.10005,
        spread=0.0001,
        spread_points=1.0,
        analysis_timeframe="M15",
        trader_summary="plan",
        portfolio_manager_summary="summary",
        source_run_id=f"source-{decision_id}",
        decision_context_status=context,
        decision_completed_timestamp=completed,
        analysis_latency_seconds=60.0,
        decision_reference_timestamp=reference_timestamp,
        decision_reference_bid=reference_quote[0],
        decision_reference_ask=reference_quote[1],
        decision_reference_spread=reference_quote[2],
        decision_reference_spread_points=reference_quote[3],
        decision_reference_status=reference_status,
        decision_reference_delay_seconds=delay,
    )


def _insert_run(
    path: Path,
    decision_id: str,
    *,
    status: str = "SUCCEEDED",
    runtime: float = 10.0,
    llm_calls: int = 4,
    tokens_in: int = 100,
    tokens_out: int = 50,
    stale: int = 0,
    freshness_budget: int = 900,
) -> None:
    try:
        offset = int(decision_id.lstrip("d"))
    except ValueError:
        offset = sum(ord(char) for char in decision_id) % 30
    now = _ts(offset)
    opportunity = f"op-{decision_id}"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO forex_watch_opportunities (
                opportunity_key, requested_symbol, analysis_profile, analyst_set_json,
                schedule_timeframe, anchor_timestamp, bar_close_timestamp, eligible_after,
                config_fingerprint, status, run_id, decision_id, attempt_count,
                first_seen_at, updated_at
            ) VALUES (?, 'EURUSD', 'INTRADAY', '[\"market\",\"news\"]', 'M15',
                      ?, ?, ?, ?, 'DECISION_SAVED', ?, ?, 1, ?, ?)
            """,
            (opportunity, now.isoformat(), now.isoformat(), now.isoformat(), f"fp-{decision_id}",
             f"run-{decision_id}", decision_id, now.isoformat(), now.isoformat()),
        )
        conn.execute(
            """
            INSERT INTO forex_watch_runs (
                run_id, opportunity_key, attempt_number, owner_token, run_status,
                requested_symbol, resolved_symbol, started_at, completed_at, decision_id,
                source_run_id, decision_context_status, normalization_status,
                normalized_action, stale_by_completion, freshness_budget_seconds,
                analysis_profile, analyst_set_json, prompt_config_version,
                application_version, collector_contract_version, config_fingerprint,
                safe_config_json, runtime_seconds, llm_calls, tool_calls, tokens_in,
                tokens_out, reasoning_tokens
            ) VALUES (?, ?, 1, 'owner', ?, 'EURUSD', 'EURUSD', ?, ?, ?, ?,
                      'COMPLETE', 'NORMALIZED', 'HOLD', ?, ?, 'INTRADAY',
                      '[\"market\",\"news\"]', 'test', 'test', 'test', ?, '{}',
                      ?, ?, 0, ?, ?, 0)
            """,
            (f"run-{decision_id}", opportunity, status, now.isoformat(), now.isoformat(), decision_id,
             f"source-{decision_id}", stale, freshness_budget, f"fp-{decision_id}", runtime,
             llm_calls, tokens_in, tokens_out),
        )


def _insert_evaluations(
    path: Path,
    decision_id: str,
    statuses: dict[int, str],
    *,
    training_eligible: bool = False,
) -> None:
    with sqlite3.connect(path) as conn:
        for horizon, status in statuses.items():
            conn.execute(
                """
                INSERT INTO shadow_decision_evaluations (
                    decision_id, evaluation_basis, horizon_seconds, resolved_symbol,
                    evaluation_version, market_data_source, source_context_eligible,
                    training_eligible, training_eligibility_reason, evaluation_status,
                    created_at, selected_action, selected_action_net_points
                ) VALUES (?, 'DECISION_REFERENCE', ?, 'EURUSD', 'test', 'MT5', 1, ?,
                          'TEST', ?, ?, 'HOLD', ?)
                """,
                (decision_id, horizon, int(training_eligible), status, _ts().isoformat(),
                 1.0 if status == "COMPLETE" else None),
            )


def test_empty_database_snapshot_is_safe(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    from tradingagents.forex.dashboard import read_dashboard_snapshot

    snapshot = read_dashboard_snapshot(path)

    assert snapshot.total_runs == 0
    assert snapshot.total_decisions == 0
    assert snapshot.health == "WARNING"


def test_valid_collection_requires_all_runtime_gates(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    ShadowDecisionStore(path).record(_decision("valid"))
    _insert_run(path, "valid")
    from tradingagents.forex.dashboard import read_dashboard_snapshot

    snapshot = read_dashboard_snapshot(path)

    assert snapshot.valid_collection_decisions == 1
    assert snapshot.action_counts["HOLD"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context": "INCOMPLETE"},
        {"normalization": "FAILED", "action": None},
        {"stale": 1},
        {"reference_status": "INVALID_TEMPORAL"},
    ],
)
def test_invalid_decision_is_excluded_from_valid_collection(tmp_path: Path, kwargs: dict) -> None:
    path = _init_db(tmp_path)
    ShadowDecisionStore(path).record(_decision("invalid", **kwargs))
    _insert_run(path, "invalid", stale=kwargs.get("stale", 0))
    from tradingagents.forex.dashboard import read_dashboard_snapshot

    snapshot = read_dashboard_snapshot(path)

    assert snapshot.valid_collection_decisions == 0


def test_action_distribution_and_latency_percentiles_are_deterministic(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    for index, action in enumerate(("BUY", "SELL", "HOLD")):
        decision_id = f"d{index}"
        ShadowDecisionStore(path).record(_decision(decision_id, action=action))
        _insert_run(path, decision_id, runtime=float(index + 1), llm_calls=index + 1)
    from tradingagents.forex.dashboard import read_dashboard_snapshot

    snapshot = read_dashboard_snapshot(path)

    assert snapshot.action_counts == {"BUY": 1, "SELL": 1, "HOLD": 1}
    assert snapshot.latency["runtime_p50_seconds"] == 2.0
    assert snapshot.latency["runtime_p95_seconds"] == 3.0
    assert snapshot.latency["llm_calls_latest"] == 3


def test_evaluation_horizon_counts_and_training_progress(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    ShadowDecisionStore(path).record(_decision("evaluated"))
    _insert_run(path, "evaluated")
    _insert_evaluations(
        path,
        "evaluated",
        {300: "COMPLETE", 900: "COMPLETE", 1800: "COMPLETE", 3600: "COMPLETE"},
        training_eligible=True,
    )
    from tradingagents.forex.dashboard import read_dashboard_snapshot

    snapshot = read_dashboard_snapshot(path)

    assert snapshot.evaluation_horizons[300]["COMPLETE"] == 1
    assert snapshot.fully_evaluated_decisions == 1
    assert snapshot.fully_training_eligible_decisions == 1
    assert snapshot.outcome_performance[300]["sample_count"] == 1


def test_pending_and_unavailable_horizons_do_not_count_as_complete(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    ShadowDecisionStore(path).record(_decision("partial"))
    _insert_run(path, "partial")
    _insert_evaluations(path, "partial", {300: "COMPLETE", 900: "PENDING", 1800: "DATA_UNAVAILABLE", 3600: "INELIGIBLE"})
    from tradingagents.forex.dashboard import read_dashboard_snapshot

    snapshot = read_dashboard_snapshot(path)

    assert snapshot.evaluation_horizons[900]["PENDING"] == 1
    assert snapshot.evaluation_horizons[1800]["DATA_UNAVAILABLE"] == 1
    assert snapshot.fully_evaluated_decisions == 0


def test_reader_is_strictly_read_only(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    before = path.read_bytes()
    from tradingagents.forex.dashboard import read_dashboard_snapshot

    read_dashboard_snapshot(path)

    assert path.read_bytes() == before


def test_locked_database_raises_bounded_read_error(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    lock = sqlite3.connect(path)
    lock.execute("BEGIN EXCLUSIVE")
    try:
        from tradingagents.forex.dashboard import DashboardReadError, read_dashboard_snapshot

        with pytest.raises(DashboardReadError, match="busy|locked"):
            read_dashboard_snapshot(path, busy_timeout_seconds=0.01)
    finally:
        lock.rollback()
        lock.close()


def test_once_cli_renders_and_exits(tmp_path: Path, capsys) -> None:
    path = _init_db(tmp_path)
    from cli.forex_dashboard import main

    assert main(["--db-path", str(path), "--once"]) == 0
    assert "FOREX SHADOW COLLECTION DASHBOARD" in capsys.readouterr().out


def test_live_cli_keeps_last_snapshot_after_bounded_read_failure(tmp_path: Path) -> None:
    path = _init_db(tmp_path)
    from cli.forex_dashboard import main
    from tradingagents.forex.dashboard import DashboardReadError

    calls = 0

    def reader(_path: str) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            from tradingagents.forex.dashboard import read_dashboard_snapshot

            return read_dashboard_snapshot(path)
        raise DashboardReadError("database busy or locked")

    def stop(_seconds: float) -> None:
        raise KeyboardInterrupt

    assert main(
        ["--db-path", str(path)],
        snapshot_reader=reader,
        sleep=stop,
    ) == 0


def test_existing_untracked_dashboard_is_not_modified() -> None:
    path = Path("watch_dashboard.py")
    assert path.exists()
