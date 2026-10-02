"""Read-only scalar status for the Phase 14 research plane.

This module intentionally opens the Phase 14 catalog read-only.  It never
creates the artifact root, initializes a schema, imports source data, or
constructs MT5/LLM components.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _empty(root: Path) -> dict[str, Any]:
    return {
        "artifact_root": str(root),
        "status": "EMPTY",
        "incumbent_strategy_version": None,
        "active_challengers": 0,
        "verified_experience": 0,
        "new_evidence": 0,
        "experiments_run": 0,
        "rejected_candidates": 0,
        "validating_experiments": 0,
        "shadow_challengers": 0,
        "demo_challengers": 0,
        "latest_promotion": None,
        "latest_rollback_reason": None,
        "next_trigger": "explicit self-enhancement experiment command",
        "llm_calls": 0,
        "mt5_calls": 0,
    }


def _count(db: sqlite3.Connection, table: str, where: str = "", params: tuple[Any, ...] = ()) -> int:
    query = f'SELECT COUNT(*) FROM "{table}"' + (f" WHERE {where}" if where else "")
    return int(db.execute(query, params).fetchone()[0])


def read_dashboard_status(artifact_root: str | Path) -> dict[str, Any]:
    """Return a scalar-only status snapshot without creating or mutating files."""

    root = Path(artifact_root).expanduser().resolve()
    catalog = root / "phase14.sqlite3"
    if not catalog.is_file():
        return _empty(root)
    try:
        db = sqlite3.connect(f"file:{catalog.as_posix()}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise RuntimeError(f"phase14 catalog read failed: {exc}") from exc
    try:
        tables = {
            str(row[0])
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {"experience_trades", "experiments", "promotions", "deployments", "rollbacks", "candidates"}
        if not required.issubset(tables):
            return _empty(root)
        latest_promotion_row = db.execute(
            "SELECT decision, candidate_id, reason FROM promotions ORDER BY recorded_at DESC, promotion_id DESC LIMIT 1"
        ).fetchone()
        latest_rollback_row = db.execute(
            "SELECT reason FROM rollbacks ORDER BY recorded_at DESC, rollback_id DESC LIMIT 1"
        ).fetchone()
        latest_experiment = db.execute(
            "SELECT metadata_json FROM experiments ORDER BY updated_at DESC, experiment_id DESC LIMIT 1"
        ).fetchone()
        baseline_experience = 0
        if latest_experiment is not None:
            try:
                metadata = json.loads(latest_experiment["metadata_json"])
                baseline_experience = max(0, int(metadata.get("accepted_experience", 0)))
            except (TypeError, ValueError, json.JSONDecodeError):
                baseline_experience = 0
        experience_count = _count(db, "experience_trades")
        latest_version = None
        if latest_promotion_row is not None:
            candidate = db.execute(
                "SELECT payload_json FROM candidates WHERE candidate_id=? ORDER BY created_at DESC LIMIT 1",
                (latest_promotion_row["candidate_id"],),
            ).fetchone()
            if candidate is not None:
                try:
                    payload = json.loads(candidate["payload_json"])
                    parent = payload.get("parent", {}) if isinstance(payload, dict) else {}
                    latest_version = parent.get("strategy_version")
                except (TypeError, ValueError, json.JSONDecodeError):
                    latest_version = None
        shadow = _count(db, "deployments", "status='SHADOW_CHALLENGER'")
        demo = _count(db, "deployments", "status='DEMO_CHALLENGER'")
        experiments = _count(db, "experiments")
        return {
            "artifact_root": str(root),
            "status": "OK",
            "incumbent_strategy_version": latest_version,
            "active_challengers": shadow + demo,
            "verified_experience": experience_count,
            "new_evidence": max(0, experience_count - baseline_experience),
            "experiments_run": experiments,
            "rejected_candidates": _count(db, "promotions", "decision='REJECTED'"),
            "validating_experiments": _count(db, "experiments", "status='RUNNING'"),
            "shadow_challengers": shadow,
            "demo_challengers": demo,
            "latest_promotion": (
                None
                if latest_promotion_row is None
                else {
                    "decision": str(latest_promotion_row["decision"]),
                    "candidate_id": str(latest_promotion_row["candidate_id"]),
                    "reason": str(latest_promotion_row["reason"]),
                }
            ),
            "latest_rollback_reason": None if latest_rollback_row is None else str(latest_rollback_row["reason"]),
            "next_trigger": "explicit self-enhancement experiment command",
            "llm_calls": 0,
            "mt5_calls": 0,
        }
    finally:
        db.close()


__all__ = ["read_dashboard_status"]
