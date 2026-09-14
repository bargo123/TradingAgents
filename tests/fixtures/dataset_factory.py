"""Tiny on-disk source fixtures for the Phase 10 public-factory tests."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tradingagents.datasets.sources import _AUDIT_FIELDS, ReadonlyPhase56Source

UTC = timezone.utc
_D = ["decision_id", "created_at", "analysis_date", "source_run_id", "requested_symbol", "resolved_symbol", "analysis_profile", "analysis_timeframe", "valid_for_seconds", "valid_until", "action", "normalization_status", "decision_context_status", "llm_provider", "quick_model", "deep_model", "snapshot_timestamp", "analysis_snapshot_timestamp", "analysis_snapshot_bid", "analysis_snapshot_ask", "analysis_snapshot_spread", "analysis_snapshot_spread_points", "decision_completed_timestamp", "decision_reference_timestamp", "decision_reference_status", "snapshot_json", "executed", "normalization_error", "raw_portfolio_manager_result", "trader_summary", "portfolio_manager_summary", "bull_summary", "bear_summary", "reference_bid", "reference_ask", "reference_mid", "spread", "spread_points"]
_E = ["decision_id", "resolved_symbol", "evaluation_basis", "horizon_seconds", "evaluation_version", "market_data_source", "source_context_eligible", "training_eligible", "training_eligibility_reason", "target_timestamp", "observation_timestamp", "observation_lag_ms", "entry_timestamp", "entry_bid", "entry_ask", "entry_spread", "entry_spread_points", "future_bid", "future_ask", "future_spread", "future_spread_points", "point", "digits", "buy_net_price", "buy_net_points", "sell_net_price", "sell_net_points", "selected_action", "selected_action_net_price", "selected_action_net_points", "best_counterfactual_action", "best_counterfactual_net_points", "hold_opportunity_cost_points", "buy_mfe_price", "buy_mfe_points", "buy_mae_price", "buy_mae_points", "sell_mfe_price", "sell_mfe_points", "sell_mae_price", "sell_mae_points", "evaluation_status", "unavailable_reason", "created_at", "evaluated_at", "recovered_from_unavailable_at", "previous_unavailable_reason"]


def _ts(day: int, second: int = 0) -> str:
    return (datetime(2026, 1, day, tzinfo=UTC) + timedelta(seconds=second)).isoformat().replace("+00:00", "Z")


def _table(db: sqlite3.Connection, name: str, columns: list[str], integer_columns: tuple[str, ...] = ()) -> None:
    integer_set = set(integer_columns)
    definitions = ", ".join(f'{c} {"INTEGER" if c in integer_set else "TEXT"}' for c in columns)
    db.execute(f'CREATE TABLE "{name}" ({definitions})')


def _features() -> dict[str, dict[str, Any]]:
    return {tf: {"return_over_bars": .01, "range_pct": .02, "close_position": .5,
                 "average_true_range": .001, "direction": "UP"}
            for tf in ("M1", "M5", "M15", "H1")}


def _snapshot(day: int, tier: str = "A") -> dict[str, Any]:
    features = _features()
    if tier == "B":
        features.pop("H1")
    quote = {"bid": 1.1, "ask": 1.1001, "spread_points": 10}
    if tier == "C":
        quote.pop("ask")
    return {"symbol": "EURUSD", "point": .00001, "digits": 5,
            "quote": quote, "features": features}


def _decision(did: str, day: int) -> dict[str, Any]:
    incomplete = did.startswith("incomplete")
    tier = "B" if did == "tier-b" else ("C" if did == "tier-c" else "A")
    return {
        "decision_id": did, "created_at": _ts(day, 1), "analysis_date": f"2026-01-{day:02d}",
        "source_run_id": f"run-{day}", "requested_symbol": "EURUSD", "resolved_symbol": "EURUSD",
        "analysis_profile": "INTRADAY", "analysis_timeframe": "M5", "valid_for_seconds": "60",
        "valid_until": _ts(day, 60), "action": "BUY", "normalization_status": "FAILED" if incomplete else "NORMALIZED",
        "decision_context_status": "INCOMPLETE" if incomplete else "COMPLETE", "llm_provider": "fixture",
        "quick_model": "fixture-quick", "deep_model": "fixture-deep", "snapshot_timestamp": _ts(day),
        "analysis_snapshot_timestamp": _ts(day), "analysis_snapshot_bid": "1.1", "analysis_snapshot_ask": "1.1001",
        "analysis_snapshot_spread": ".0001", "analysis_snapshot_spread_points": "10",
        "decision_completed_timestamp": _ts(day, 2), "decision_reference_timestamp": _ts(day, 3),
        "decision_reference_status": "AVAILABLE", "snapshot_json": json.dumps(_snapshot(day, tier), separators=(",", ":")),
        "executed": 0, "normalization_error": None, "raw_portfolio_manager_result": '{"action":"BUY"}',
        "trader_summary": "bounded trader", "portfolio_manager_summary": "bounded portfolio",
        "bull_summary": "bounded bull", "bear_summary": "bounded bear", "reference_bid": "1.1",
        "reference_ask": "1.1001", "reference_mid": "1.10005", "spread": ".0001", "spread_points": "10",
    }


def _evaluation(row: dict[str, Any], status: str = "COMPLETE") -> dict[str, Any]:
    day = int(row["analysis_date"][-2:])
    out = dict.fromkeys(_E)
    out.update({"decision_id": row["decision_id"], "resolved_symbol": "EURUSD",
        "evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": "300", "evaluation_version": "phase5.v1",
        "market_data_source": "fixture", "source_context_eligible": "1" if status == "COMPLETE" else "0",
        "training_eligible": "0", "training_eligibility_reason": "DEFERRED", "target_timestamp": _ts(day, 300),
        "observation_timestamp": _ts(day, 300), "observation_lag_ms": "0", "entry_timestamp": _ts(day),
        "entry_bid": "1.1", "entry_ask": "1.1001", "entry_spread": ".0001", "entry_spread_points": "10",
        "future_bid": "1.1002", "future_ask": "1.1003", "future_spread": ".0001", "future_spread_points": "10",
        "point": ".00001", "digits": "5", "buy_net_price": ".0001", "buy_net_points": "10",
        "sell_net_price": "-.0003", "sell_net_points": "-30", "selected_action": "BUY",
        "selected_action_net_price": ".0001", "selected_action_net_points": "10", "best_counterfactual_action": "BUY",
        "best_counterfactual_net_points": "10", "hold_opportunity_cost_points": "10", "buy_mfe_price": ".0002",
        "buy_mfe_points": "20", "buy_mae_price": "-.0001", "buy_mae_points": "-10", "sell_mfe_price": ".0001",
        "sell_mfe_points": "10", "sell_mae_price": "-.0002", "sell_mae_points": "-20", "evaluation_status": status,
        "unavailable_reason": "NO_QUOTE" if status == "DATA_UNAVAILABLE" else None, "created_at": _ts(day, 10),
        "evaluated_at": _ts(day, 301) if status == "COMPLETE" else None})
    return out


def _write_phase56(path: Path) -> tuple[Any, ...]:
    rows = tuple(_decision(did, day) for did, day in (
        ("valid-a", 1), ("valid-b", 2), ("valid-c", 3), ("valid-d", 10),
        ("valid-e", 11), ("valid-f", 12), ("valid-g", 13), ("valid-h", 14),
        ("tier-b", 4), ("tier-c", 5), ("incomplete-1", 6), ("unavailable-1", 7),
        ("future-1", 8), ("duplicate-1", 9), ("duplicate-1", 9)))
    evals = [_evaluation(row, "DATA_UNAVAILABLE" if row["decision_id"] == "unavailable-1" else "COMPLETE") for row in rows]
    future_index = next(index for index, row in enumerate(rows) if row["decision_id"] == "future-1")
    evals[future_index]["target_timestamp"] = _ts(30)  # future relative to the bounded as_of in the test
    db = sqlite3.connect(path)
    try:
        _table(db, "shadow_decisions", _D, ("executed",))
        _table(db, "shadow_decision_evaluations", _E, ("horizon_seconds", "source_context_eligible", "training_eligible"))
        db.executemany(f'INSERT INTO shadow_decisions VALUES ({",".join("?" for _ in _D)})', [tuple(r.get(k) for k in _D) for r in rows])
        db.executemany(f'INSERT INTO shadow_decision_evaluations VALUES ({",".join("?" for _ in _E)})', [tuple(r.get(k) for k in _E) for r in evals])
        db.commit()
    finally:
        db.close()
    read = ReadonlyPhase56Source(path).read()
    return tuple((d, next(e for e in read.evaluations if e.decision_id == d.decision_id)) for d in read.decisions)


def _write_phase8(root: Path, pairs: tuple[Any, ...]) -> None:
    root.mkdir(parents=True)
    db = sqlite3.connect(root / "catalog.sqlite3")
    try:
        _table(db, "experience_records", ["experience_id", "source_decision_id", "source_database_id", "source_decision_fingerprint", "source_run_id", "symbol", "requested_symbol", "analysis_profile", "analysis_timeframe", "analysis_snapshot_timestamp", "decision_completed_timestamp", "decision_reference_timestamp", "market_state_json", "decision_evidence_json", "provenance_json", "trust", "tombstoned", "source_evaluation_fingerprints_json", "experience_schema_version", "feature_schema_version", "feature_extractor_version", "trust_policy_version"])
        _table(db, "experience_source_aliases", ["source_database_id", "source_decision_id", "accepted_fingerprint", "experience_id", "state", "observed_at", "scan_status"])
        _table(db, "experience_outcome_snapshots", ["experience_id", "evaluation_basis", "horizon_seconds", "fingerprint", "evaluation_json", "provenance_json", "observed_at"])
        _table(db, "experience_feature_projections", ["experience_id", "feature_schema_version", "projection_json"])
        for d, e in pairs:
            fp, efp = d.fields["source_decision_fingerprint"], e.fields["source_evaluation_fingerprint"]
            eid = "exp-" + d.decision_id
            day = d.analysis_snapshot_timestamp.day
            trust = "TIER_C_DIAGNOSTIC_ONLY" if d.decision_id == "tier-c" else ("TIER_B_LIMITED" if d.decision_id == "tier-b" else "TIER_A_HIGH_TRUST")
            prov = {"source_decision_fingerprint": fp}
            db.execute("INSERT INTO experience_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (eid, d.decision_id, "fixture", fp, d.source_run_id, "EURUSD", "EURUSD", "INTRADAY", "M5", d.analysis_snapshot_timestamp.isoformat(), d.decision_completed_timestamp.isoformat(), d.fields["decision_reference_timestamp"], json.dumps(_snapshot(day)), "{}", json.dumps(prov), trust, 0, json.dumps({"ANALYSIS_SNAPSHOT:300": efp}), "phase8.experience.v1", "experience-features.v1", "phase8-feature-extractor.v1", "trust-policy.v1"))
            db.execute("INSERT INTO experience_source_aliases VALUES (?,?,?,?,?,?,?)", ("fixture", d.decision_id, fp, eid, "CURRENT", _ts(day), "OK"))
            ev = {**dict(e.fields), "evaluation_basis": e.evaluation_basis, "horizon_seconds": e.horizon_seconds, "evaluation_status": e.evaluation_status}
            db.execute("INSERT INTO experience_outcome_snapshots VALUES (?,?,?,?,?,?,?)", (eid, e.evaluation_basis, e.horizon_seconds, efp, json.dumps(ev), json.dumps({"evaluation_fingerprint": efp, "source_decision_id": d.decision_id}), _ts(day, 350)))
            db.execute(
                "INSERT INTO experience_feature_projections VALUES (?,?,?)",
                (
                    eid,
                    "experience-features.v1",
                    json.dumps(
                        {
                            "version": "phase8-feature-extractor.v1",
                            "features": _features(),
                        }
                    ),
                ),
            )
        db.commit()
    finally:
        db.close()


def _write_phase9(path: Path, pairs: tuple[Any, ...]) -> None:
    # Keep fixture audits aligned with the production Phase 9 graph-integrity
    # contract.  These are bounded metadata fingerprints, never report text.
    node_context_hashes = {
        node: f"fixture-{index:02d}-context-hash"
        for index, node in enumerate(
            (
                "Market Analyst",
                "News Analyst",
                "Bull Researcher",
                "Bear Researcher",
                "Research Manager",
                "Trader",
                "Aggressive Analyst",
                "Conservative Analyst",
                "Neutral Analyst",
                "Portfolio Manager",
            ),
            start=1,
        )
    }
    db = sqlite3.connect(path)
    try:
        _table(db, "evidence_usage_audit", list(_AUDIT_FIELDS))
        for d, _ in pairs:
            if d.decision_id == "future-1":
                continue
            v = dict.fromkeys(_AUDIT_FIELDS)
            v.update({"decision_id": d.decision_id, "source_run_id": d.source_run_id, "as_of": d.decision_completed_timestamp.isoformat(), "rendered_context": "bounded", "rendered_context_hash": "hash", "knowledge_generation_id": "kg", "experience_generation_id": "eg", "knowledge_query": "order flow imbalance", "integration_status": "INJECTED", "bundle_status": "COMPLETE", "evidence_use_status": "USED", "available_knowledge_ids": '["K1"]', "available_experience_ids": "[]", "available_statistics_ids": "[]", "evidence_refs_used": '["K1"]', "evidence_refs_rejected": "[]", "query_policy_version": "phase9.query.v1", "source_status": '{"knowledge":"COMPLETE","experience":"COMPLETE"}', "diagnostics": "{}", "source_errors": "{}", "retrieval_count": "1", "retrieval_latency_seconds": ".1", "builder_latency_seconds": ".1", "selected_counts": '{"knowledge":1,"experience":0,"statistics":0}', "dropped_counts": "{}", "telemetry_references": "[]", "node_context_hashes": json.dumps(node_context_hashes, sort_keys=True), "missing_nodes": "[]", "provider": "fixture", "model": "fixture", "audit_schema_version": "phase9.audit.v1", "evidence_audit_status": "VALID", "query_normalization_fingerprint": "query", "knowledge_query_fingerprint": "query-fp"})
            db.execute(f'INSERT INTO evidence_usage_audit VALUES ({",".join("?" for _ in _AUDIT_FIELDS)})', tuple(v[key] for key in _AUDIT_FIELDS))
        db.commit()
    finally:
        db.close()


def make_real_fixtures(root: Path) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = root / "phase56.sqlite3"
    pairs = _write_phase56(source)
    phase8 = root / "phase8"
    _write_phase8(phase8, pairs)
    audit = root / "phase9.sqlite3"
    _write_phase9(audit, pairs)
    return {"source": source, "phase8": phase8, "audit": audit, "pairs": pairs}


__all__ = ["make_real_fixtures"]
