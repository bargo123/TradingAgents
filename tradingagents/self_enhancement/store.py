"""Transactional, isolated Phase 14 SQLite catalog."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import CandidateSpec, CandidateState, ExperienceTrade, LearningFinding, TriggerKind


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class SelfEnhancementStore:
    """Owns only the caller-provided Phase 14 artifact root."""

    SCHEMA_VERSION = "phase14.catalog.v1"

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS phase14_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experience_trades (
                    experience_id TEXT PRIMARY KEY,
                    source_fingerprint TEXT,
                    payload_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experience_quarantine (
                    quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    parent_version TEXT NOT NULL,
                    dataset_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidate_transitions (
                    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                    stage TEXT NOT NULL,
                    passed INTEGER NOT NULL CHECK(passed IN (0,1)),
                    metrics_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY(experiment_id, candidate_id, stage)
                );
                CREATE TABLE IF NOT EXISTS promotions (
                    promotion_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    rollback_package_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                    status TEXT NOT NULL,
                    previous_version TEXT NOT NULL,
                    config_snapshot_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rollbacks (
                    rollback_id TEXT PRIMARY KEY,
                    deployment_id TEXT NOT NULL REFERENCES deployments(deployment_id),
                    reason TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS triggers (
                    trigger_id TEXT PRIMARY KEY,
                    trigger_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experience_findings (
                    finding_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    experiment_id TEXT,
                    verified INTEGER NOT NULL CHECK(verified IN (0,1)),
                    recorded_at TEXT NOT NULL
                );
                """
            )
            db.execute(
                "INSERT OR IGNORE INTO phase14_meta(key,value) VALUES(?,?)",
                ("schema_version", self.SCHEMA_VERSION),
            )

    def record_experience(self, trade: ExperienceTrade) -> None:
        if not isinstance(trade, ExperienceTrade):
            raise TypeError("trade must be an ExperienceTrade")
        payload = trade.to_dict()
        fingerprint = trade.source_fingerprint or hashlib.sha256(_json(payload).encode()).hexdigest()
        with self._connect() as db:
            existing = db.execute(
                "SELECT payload_json, source_fingerprint FROM experience_trades WHERE experience_id=?",
                (trade.experience_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != _json(payload) or existing[1] != fingerprint:
                    raise ValueError("experience identity already exists with different provenance")
                return
            db.execute(
                "INSERT INTO experience_trades(experience_id,source_fingerprint,payload_json,recorded_at) VALUES(?,?,?,?)",
                (trade.experience_id, fingerprint, _json(payload), _now()),
            )

    def quarantine(self, source_id: str, reason: str, detail: Mapping[str, Any]) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO experience_quarantine(source_id,reason,detail_json,recorded_at) VALUES(?,?,?,?)",
                (str(source_id), str(reason), _json(dict(detail)), _now()),
            )

    def create_experiment(
        self,
        experiment_id: str,
        *,
        parent_version: str,
        dataset_fingerprint: str,
        metadata: Mapping[str, Any] | None = None,
        status: str = "RUNNING",
    ) -> str:
        now = _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO experiments(experiment_id,parent_version,dataset_fingerprint,status,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (str(experiment_id), str(parent_version), str(dataset_fingerprint), str(status), _json(dict(metadata or {})), now, now),
            )
        return str(experiment_id)

    def update_experiment_status(self, experiment_id: str, status: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        with self._connect() as db:
            row = db.execute("SELECT metadata_json FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
            if row is None:
                raise KeyError(experiment_id)
            current = json.loads(row[0])
            if metadata:
                current.update(dict(metadata))
            db.execute(
                "UPDATE experiments SET status=?,metadata_json=?,updated_at=? WHERE experiment_id=?",
                (str(status), _json(current), _now(), experiment_id),
            )

    def record_candidate(self, experiment_id: str, candidate: CandidateSpec) -> None:
        if not isinstance(candidate, CandidateSpec):
            raise TypeError("candidate must be a CandidateSpec")
        payload = {
            "candidate_id": candidate.candidate_id,
            "parent": {
                "strategy_id": candidate.parent.strategy_id,
                "strategy_version": candidate.parent.strategy_version,
                "config_version": candidate.parent.config_version,
                "parameters": dict(candidate.parent.parameters),
                "source_commit": candidate.parent.source_commit,
                "config_hash": candidate.parent.config_hash,
            },
            "strategy_id": candidate.strategy_id,
            "exit_policy": candidate.exit_policy.to_dict(),
            "hypothesis": candidate.hypothesis,
            "source_evidence": list(candidate.source_evidence),
            "state": candidate.state.value,
            "execution_mode": candidate.execution_mode.value,
            "real_money": False,
        }
        with self._connect() as db:
            db.execute(
                "INSERT INTO candidates(candidate_id,experiment_id,payload_json,state,created_at) VALUES(?,?,?,?,?)",
                (candidate.candidate_id, experiment_id, _json(payload), candidate.state.value, _now()),
            )
            db.execute(
                "INSERT INTO candidate_transitions(experiment_id,candidate_id,from_state,to_state,reason,recorded_at) VALUES(?,?,?,?,?,?)",
                (experiment_id, candidate.candidate_id, None, candidate.state.value, "INITIAL_REGISTRATION", _now()),
            )

    def transition_candidate(self, experiment_id: str, candidate_id: str, state: CandidateState | str, reason: str) -> None:
        target = CandidateState(state).value
        with self._connect() as db:
            row = db.execute(
                "SELECT state FROM candidates WHERE candidate_id=? AND experiment_id=?",
                (candidate_id, experiment_id),
            ).fetchone()
            if row is None:
                raise KeyError(candidate_id)
            current = str(row[0])
            if current == target:
                return
            db.execute("UPDATE candidates SET state=? WHERE candidate_id=?", (target, candidate_id))
            db.execute(
                "INSERT INTO candidate_transitions(experiment_id,candidate_id,from_state,to_state,reason,recorded_at) VALUES(?,?,?,?,?,?)",
                (experiment_id, candidate_id, current, target, str(reason), _now()),
            )

    def record_evaluation(
        self,
        experiment_id: str,
        candidate_id: str,
        stage: str,
        metrics: Mapping[str, Any],
        *,
        passed: bool,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO evaluations(experiment_id,candidate_id,stage,passed,metrics_json,recorded_at) VALUES(?,?,?,?,?,?)",
                (experiment_id, candidate_id, str(stage), int(bool(passed)), _json(dict(metrics)), _now()),
            )

    def record_promotion(
        self,
        experiment_id: str,
        candidate_id: str,
        *,
        decision: str,
        reason: str,
        rollback_package: Mapping[str, Any],
    ) -> str | None:
        decision = str(decision).upper()
        if decision not in {"PROMOTED", "SHADOW_CHALLENGER", "REJECTED", "NO_PROMOTION"}:
            raise ValueError("invalid promotion decision")
        if not isinstance(rollback_package, Mapping):
            raise TypeError("rollback_package must be a mapping")
        target_state = {
            "PROMOTED": CandidateState.APPROVED.value,
            "SHADOW_CHALLENGER": CandidateState.SHADOW_CHALLENGER.value,
            "REJECTED": CandidateState.REJECTED.value,
            "NO_PROMOTION": CandidateState.INSUFFICIENT_EVIDENCE.value,
        }[decision]
        with self._connect() as db:
            candidate = db.execute(
                "SELECT state FROM candidates WHERE candidate_id=? AND experiment_id=?",
                (candidate_id, experiment_id),
            ).fetchone()
            if candidate is None:
                raise KeyError(candidate_id)
            db.execute(
                "INSERT INTO promotions(promotion_id,experiment_id,candidate_id,decision,reason,rollback_package_json,recorded_at) VALUES(?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), experiment_id, candidate_id, decision, str(reason), _json(dict(rollback_package)), _now()),
            )
            if str(candidate[0]) != target_state:
                db.execute("UPDATE candidates SET state=? WHERE candidate_id=?", (target_state, candidate_id))
                db.execute(
                    "INSERT INTO candidate_transitions(experiment_id,candidate_id,from_state,to_state,reason,recorded_at) VALUES(?,?,?,?,?,?)",
                    (experiment_id, candidate_id, str(candidate[0]), target_state, str(reason), _now()),
                )
            if decision not in {"PROMOTED", "SHADOW_CHALLENGER"}:
                return None
            previous = str(rollback_package.get("previous_version", ""))
            if not previous:
                raise ValueError("promoted candidate requires previous_version")
            deployment_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO deployments(deployment_id,candidate_id,status,previous_version,config_snapshot_json,recorded_at) VALUES(?,?,?,?,?,?)",
                (deployment_id, candidate_id, decision, previous, _json(dict(rollback_package)), _now()),
            )
            return deployment_id

    def rollback(self, deployment_id: str, reason: str) -> None:
        with self._connect() as db:
            row = db.execute("SELECT status FROM deployments WHERE deployment_id=?", (deployment_id,)).fetchone()
            if row is None:
                raise KeyError(deployment_id)
            db.execute("UPDATE deployments SET status='ROLLED_BACK' WHERE deployment_id=?", (deployment_id,))
            db.execute(
                "INSERT INTO rollbacks(rollback_id,deployment_id,reason,recorded_at) VALUES(?,?,?,?)",
                (str(uuid.uuid4()), deployment_id, str(reason), _now()),
            )

    def record_trigger(self, trigger_type: TriggerKind | str, payload: Mapping[str, Any]) -> str:
        trigger = TriggerKind(trigger_type).value
        if not isinstance(payload, Mapping):
            raise TypeError("trigger payload must be a mapping")
        trigger_id = str(uuid.uuid4())
        with self._connect() as db:
            db.execute(
                "INSERT INTO triggers(trigger_id,trigger_type,payload_json,recorded_at) VALUES(?,?,?,?)",
                (trigger_id, trigger, _json(dict(payload)), _now()),
            )
        return trigger_id

    def record_finding(self, finding: LearningFinding) -> None:
        if not isinstance(finding, LearningFinding):
            raise TypeError("finding must be a LearningFinding")
        with self._connect() as db:
            existing = db.execute(
                "SELECT kind,statement,evidence_json,experiment_id,verified FROM experience_findings WHERE finding_id=?",
                (finding.finding_id,),
            ).fetchone()
            values = (
                finding.kind.value,
                finding.statement,
                _json(finding.evidence_ids),
                finding.experiment_id,
                int(finding.verified),
            )
            if existing is not None:
                if tuple(existing) != values:
                    raise ValueError("finding identity already exists with different evidence")
                return
            db.execute(
                "INSERT INTO experience_findings(finding_id,kind,statement,evidence_json,experiment_id,verified,recorded_at) VALUES(?,?,?,?,?,?,?)",
                (finding.finding_id, *values, _now()),
            )

    def recover_incomplete_experiments(self) -> tuple[str, ...]:
        with self._connect() as db:
            rows = db.execute("SELECT experiment_id FROM experiments WHERE status='RUNNING' ORDER BY experiment_id").fetchall()
            ids = tuple(str(row[0]) for row in rows)
            db.execute(
                "UPDATE experiments SET status='ABORTED_RECOVERABLE',updated_at=? WHERE status='RUNNING'",
                (_now(),),
            )
            return ids

    def experiment_status(self, experiment_id: str) -> str:
        with self._connect() as db:
            row = db.execute("SELECT status FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return str(row[0])

    def dataset_experiment_count(self, dataset_fingerprint: str) -> int:
        with self._connect() as db:
            return int(
                db.execute(
                    "SELECT COUNT(*) FROM experiments WHERE dataset_fingerprint=?",
                    (str(dataset_fingerprint),),
                ).fetchone()[0]
            )

    def snapshot(self) -> dict[str, int]:
        with self._connect() as db:
            return {
                name: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for name, table in {
                    "experience": "experience_trades",
                    "quarantine": "experience_quarantine",
                    "experiments": "experiments",
                    "candidates": "candidates",
                    "evaluations": "evaluations",
                    "promotions": "promotions",
                    "deployments": "deployments",
                    "rollbacks": "rollbacks",
                    "candidate_transitions": "candidate_transitions",
                    "findings": "experience_findings",
                    "triggers": "triggers",
                }.items()
            }


__all__ = ["SelfEnhancementStore"]
