"""Tiny Phase 5/6 SQLite source fixture used by Experience reader tests."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def create_source_db(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE shadow_decisions (
            decision_id TEXT PRIMARY KEY, created_at TEXT, analysis_date TEXT,
            source_run_id TEXT, requested_symbol TEXT, resolved_symbol TEXT,
            analysis_profile TEXT, analysis_timeframe TEXT, valid_for_seconds INTEGER,
            valid_until TEXT, action TEXT, normalization_status TEXT,
            normalization_error TEXT, decision_context_status TEXT,
            raw_portfolio_manager_result TEXT, trader_summary TEXT,
            portfolio_manager_summary TEXT, bull_summary TEXT, bear_summary TEXT,
            llm_provider TEXT, quick_model TEXT, deep_model TEXT,
            snapshot_timestamp TEXT, reference_bid REAL, reference_ask REAL,
            reference_mid REAL, spread REAL, spread_points REAL,
            analysis_snapshot_timestamp TEXT, analysis_snapshot_bid REAL,
            analysis_snapshot_ask REAL, analysis_snapshot_spread REAL,
            analysis_snapshot_spread_points REAL, decision_completed_timestamp TEXT,
            analysis_latency_seconds REAL, decision_reference_timestamp TEXT,
            decision_reference_bid REAL, decision_reference_ask REAL,
            decision_reference_spread REAL, decision_reference_spread_points REAL,
            decision_reference_status TEXT, decision_reference_delay_seconds REAL,
            decision_reference_error TEXT, snapshot_json TEXT, executed INTEGER
        );
        CREATE TABLE shadow_decision_evaluations (
            decision_id TEXT, evaluation_basis TEXT, horizon_seconds INTEGER,
            evaluation_version TEXT, market_data_source TEXT,
            source_context_eligible INTEGER, training_eligible INTEGER,
            training_eligibility_reason TEXT, target_timestamp TEXT,
            observation_timestamp TEXT, observation_lag_ms INTEGER, entry_timestamp TEXT,
            entry_bid REAL, entry_ask REAL, entry_spread REAL, entry_spread_points REAL,
            future_bid REAL, future_ask REAL, future_spread REAL, future_spread_points REAL,
            point REAL, digits INTEGER, buy_net_price REAL, buy_net_points REAL,
            sell_net_price REAL, sell_net_points REAL, selected_action TEXT,
            selected_action_net_price REAL, selected_action_net_points REAL,
            best_counterfactual_action TEXT, best_counterfactual_net_points REAL,
            hold_opportunity_cost_points REAL, buy_mfe_price REAL, buy_mfe_points REAL,
            buy_mae_price REAL, buy_mae_points REAL, sell_mfe_price REAL,
            sell_mfe_points REAL, sell_mae_price REAL, sell_mae_points REAL,
            evaluation_status TEXT, unavailable_reason TEXT, created_at TEXT,
            evaluated_at TEXT, recovered_from_unavailable_at TEXT,
            previous_unavailable_reason TEXT,
            PRIMARY KEY (decision_id, evaluation_basis, horizon_seconds)
        );
        CREATE TABLE forex_watch_runs (
            run_id TEXT PRIMARY KEY, opportunity_key TEXT, attempt_number INTEGER,
            source_run_id TEXT, requested_symbol TEXT, resolved_symbol TEXT,
            run_status TEXT, started_at TEXT, completed_at TEXT, failure_code TEXT,
            failure_message TEXT, context_status TEXT, normalization_status TEXT,
            action_status TEXT, analysis_snapshot_timestamp TEXT,
            decision_reference_timestamp TEXT, provider TEXT, model TEXT,
            analysis_profile TEXT, analysis_timeframe TEXT, metrics_json TEXT
        );
        CREATE TABLE forex_watch_opportunities (
            opportunity_key TEXT PRIMARY KEY, requested_symbol TEXT, analysis_profile TEXT,
            analyst_set TEXT, schedule_timeframe TEXT, anchor_timestamp TEXT,
            bar_close_timestamp TEXT, eligibility_timestamp TEXT, config_fingerprint TEXT,
            status TEXT, skip_reason TEXT, run_id TEXT, decision_id TEXT
        );
        CREATE TABLE forex_watcher_state (singleton INTEGER PRIMARY KEY, status TEXT);
        INSERT INTO shadow_decisions VALUES ('d1','2026-01-01T00:00:00+00:00','2026-01-01','run1','EURUSD','EURUSD','p','5m',300,NULL,'BUY','VALID',NULL,'COMPLETE',NULL,NULL,NULL,NULL,NULL,'luna','quick','deep','2026-01-01T00:00:00+00:00',1.1,1.2,1.15,0.1,1,'2026-01-01T00:00:00+00:00',1.1,1.2,0.1,1,'2026-01-01T00:00:01+00:00',1,'2026-01-01T00:00:02+00:00',1.1,1.2,0.1,1,'VALID',2,NULL,'{}',0);
        INSERT INTO shadow_decision_evaluations (decision_id,evaluation_basis,horizon_seconds,evaluation_status) VALUES ('d1','ANALYSIS_SNAPSHOT',300,'PENDING');
        INSERT INTO forex_watch_runs (run_id,source_run_id,run_status) VALUES ('wr1','run1','COMPLETE');
        INSERT INTO forex_watch_opportunities (opportunity_key,run_id,decision_id,status) VALUES ('op1','wr1','d1','ELIGIBLE');
        """
    )
    connection.commit()
    connection.close()
    return path
