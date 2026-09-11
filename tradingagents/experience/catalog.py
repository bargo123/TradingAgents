"""Authoritative, isolated Phase 8 Experience catalog."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .diagnostics import diagnostic_label
from .errors import SourceDecisionConflictError
from .models import ExperienceRecord, SourceAliasState, TrustTier
from .provenance import build_evaluation_provenance, validate_evaluation_provenance


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)


class ExperienceCatalog:
    """Owns only ``artifact_root/catalog.sqlite3`` and never opens source DBs."""

    def __init__(self, artifact_root: str | Path):
        self.artifact_root = Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.artifact_root / "catalog.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS experience_catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO experience_catalog_meta VALUES ('schema_version', 'phase8.catalog.v1');
                CREATE TABLE IF NOT EXISTS experience_sources (
                    source_database_id TEXT PRIMARY KEY, canonical_path TEXT, source_fingerprint TEXT,
                    source_schema_fingerprint TEXT, observed_at TEXT NOT NULL, metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experience_records (
                    experience_id TEXT PRIMARY KEY, source_decision_id TEXT NOT NULL UNIQUE,
                    source_database_id TEXT NOT NULL, source_decision_fingerprint TEXT NOT NULL,
                    source_run_id TEXT, symbol TEXT NOT NULL, requested_symbol TEXT,
                    analysis_profile TEXT, analysis_timeframe TEXT, analysis_snapshot_timestamp TEXT NOT NULL,
                    decision_completed_timestamp TEXT, decision_reference_timestamp TEXT,
                    market_state_json TEXT NOT NULL, decision_evidence_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL, trust TEXT NOT NULL, tombstoned INTEGER NOT NULL DEFAULT 0,
                    source_evaluation_fingerprints_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS experience_source_aliases (
                    source_database_id TEXT NOT NULL, source_decision_id TEXT NOT NULL,
                    accepted_fingerprint TEXT NOT NULL, experience_id TEXT NOT NULL,
                    state TEXT NOT NULL, observed_at TEXT NOT NULL, scan_status TEXT NOT NULL DEFAULT 'OK',
                    PRIMARY KEY(source_database_id, source_decision_id),
                    FOREIGN KEY(experience_id) REFERENCES experience_records(experience_id)
                );
                CREATE TABLE IF NOT EXISTS experience_decision_evidence (experience_id TEXT PRIMARY KEY, evidence_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS experience_market_provenance (experience_id TEXT PRIMARY KEY, provenance_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS experience_outcome_snapshots (
                    experience_id TEXT NOT NULL, evaluation_basis TEXT, horizon_seconds INTEGER,
                    fingerprint TEXT NOT NULL, evaluation_json TEXT NOT NULL, provenance_json TEXT NOT NULL,
                    observed_at TEXT NOT NULL, PRIMARY KEY(experience_id, fingerprint)
                );
                CREATE TABLE IF NOT EXISTS experience_feature_projections (experience_id TEXT, feature_schema_version TEXT, projection_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS experience_generations (
                    generation_id TEXT PRIMARY KEY, population_fingerprint TEXT NOT NULL,
                    metadata_json TEXT NOT NULL, published_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS experience_import_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, source_database_id TEXT, decision_id TEXT, experience_id TEXT, detail_json TEXT NOT NULL, observed_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS experience_quarantine (quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT, source_database_id TEXT, decision_id TEXT, fingerprint TEXT, reason TEXT NOT NULL, detail_json TEXT NOT NULL, observed_at TEXT NOT NULL);
                CREATE VIRTUAL TABLE IF NOT EXISTS experience_diagnostic_fts USING fts5(label, source_database_id UNINDEXED, decision_id UNINDEXED);
                """
            )

    @staticmethod
    def _experience_id(decision_id: str) -> str:
        digest = hashlib.sha256(("phase8.experience.v1\0" + decision_id).encode()).hexdigest()[:32]
        return "exp1-" + digest

    def upsert_source_alias(self, source_database_id: str, decision_id: str, fingerprint: str, **kwargs: Any) -> ExperienceRecord:
        now = _now()
        experience_id = self._experience_id(decision_id)
        with self._connect() as db:
            existing = db.execute("SELECT * FROM experience_records WHERE source_decision_id=?", (decision_id,)).fetchone()
            if existing and existing["source_decision_fingerprint"] != fingerprint:
                self._quarantine_db(db, source_database_id, decision_id, fingerprint, "SOURCE_DECISION_CONFLICT", {"accepted_fingerprint": existing["source_decision_fingerprint"]})
                db.commit()
                raise SourceDecisionConflictError(f"conflicting fingerprint for {decision_id}")
            if not existing:
                timestamp = kwargs.get("analysis_snapshot_timestamp") or datetime.now(timezone.utc)
                if isinstance(timestamp, str): timestamp = datetime.fromisoformat(timestamp)
                if timestamp.tzinfo is None: timestamp = timestamp.replace(tzinfo=timezone.utc)
                db.execute("INSERT INTO experience_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    experience_id, decision_id, source_database_id, fingerprint, kwargs.get("source_run_id"),
                    kwargs.get("symbol", "UNKNOWN"), kwargs.get("requested_symbol"), kwargs.get("analysis_profile"),
                    kwargs.get("analysis_timeframe"), timestamp.isoformat(), kwargs.get("decision_completed_timestamp"),
                    kwargs.get("decision_reference_timestamp"), _json(kwargs.get("market_state")), _json(kwargs.get("decision_evidence")),
                    _json(kwargs.get("provenance")), str(kwargs.get("trust", TrustTier.TIER_A_HIGH_TRUST.value)), 0, _json({})))
                db.execute("INSERT INTO experience_decision_evidence VALUES (?,?)", (experience_id, _json(kwargs.get("decision_evidence"))))
                db.execute("INSERT INTO experience_market_provenance VALUES (?,?)", (experience_id, _json(kwargs.get("provenance"))))
            db.execute("INSERT OR REPLACE INTO experience_source_aliases VALUES (?,?,?,?,?,?,?)", (source_database_id, decision_id, fingerprint, experience_id, "CURRENT", now, "OK"))
            db.execute("INSERT INTO experience_import_events(event_type,source_database_id,decision_id,experience_id,detail_json,observed_at) VALUES (?,?,?,?,?,?)", ("ALIAS_ACCEPTED", source_database_id, decision_id, experience_id, "{}", now))
            db.execute("UPDATE experience_records SET tombstoned=0 WHERE experience_id=?", (experience_id,))
            return self._record(db, experience_id)

    def _record(self, db: sqlite3.Connection, experience_id: str) -> ExperienceRecord:
        row = db.execute("SELECT * FROM experience_records WHERE experience_id=?", (experience_id,)).fetchone()
        aliases = {f"{r['source_database_id']}:{r['source_decision_id']}": r["state"] for r in db.execute("SELECT source_database_id,source_decision_id,state FROM experience_source_aliases WHERE experience_id=?", (experience_id,))}
        return ExperienceRecord(experience_id=row["experience_id"], source_database_id=row["source_database_id"], source_decision_id=row["source_decision_id"], symbol=row["symbol"], analysis_snapshot_timestamp=datetime.fromisoformat(row["analysis_snapshot_timestamp"]), source_aliases=aliases, source_run_id=row["source_run_id"], requested_symbol=row["requested_symbol"], analysis_profile=row["analysis_profile"], analysis_timeframe=row["analysis_timeframe"], decision_completed_timestamp=datetime.fromisoformat(row["decision_completed_timestamp"]) if row["decision_completed_timestamp"] else None, decision_reference_timestamp=datetime.fromisoformat(row["decision_reference_timestamp"]) if row["decision_reference_timestamp"] else None, market_state=json.loads(row["market_state_json"]), decision_evidence=json.loads(row["decision_evidence_json"]), provenance=json.loads(row["provenance_json"]), trust=row["trust"], source_decision_fingerprint=row["source_decision_fingerprint"], source_evaluation_fingerprints=json.loads(row["source_evaluation_fingerprints_json"]))

    def current_alias_count(self, experience_id: str) -> int:
        with self._connect() as db: return int(db.execute("SELECT COUNT(*) FROM experience_source_aliases WHERE experience_id=? AND state='CURRENT'", (experience_id,)).fetchone()[0])

    def active_records(self) -> tuple[ExperienceRecord, ...]:
        with self._connect() as db:
            ids = [r[0] for r in db.execute("SELECT DISTINCT experience_id FROM experience_source_aliases WHERE state='CURRENT' ORDER BY experience_id")]
            return tuple(self._record(db, i) for i in ids)

    def historical_records(self) -> tuple[ExperienceRecord, ...]:
        with self._connect() as db:
            ids = [r[0] for r in db.execute("SELECT experience_id FROM experience_records WHERE tombstoned=1 ORDER BY experience_id")]
            return tuple(self._record(db, i) for i in ids)

    def is_tombstoned(self, experience_id: str) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT tombstoned FROM experience_records WHERE experience_id=?", (experience_id,)).fetchone()
            return bool(row[0]) if row else False

    def mark_alias_removed(self, source_database_id: str, decision_id: str, fingerprint: str) -> None:
        with self._connect() as db:
            row = db.execute("SELECT experience_id FROM experience_source_aliases WHERE source_database_id=? AND source_decision_id=? AND accepted_fingerprint=?", (source_database_id, decision_id, fingerprint)).fetchone()
            if not row: return
            db.execute("UPDATE experience_source_aliases SET state='REMOVED', scan_status='SOURCE_MISSING', observed_at=? WHERE source_database_id=? AND source_decision_id=?", (_now(), source_database_id, decision_id))
            if not db.execute("SELECT 1 FROM experience_source_aliases WHERE experience_id=? AND state='CURRENT'", (row[0],)).fetchone(): db.execute("UPDATE experience_records SET tombstoned=1 WHERE experience_id=?", (row[0],))

    def mark_last_alias_removed(self, experience_id: str) -> None:
        with self._connect() as db:
            row = db.execute("SELECT source_database_id,source_decision_id,accepted_fingerprint FROM experience_source_aliases WHERE experience_id=? AND state='CURRENT' ORDER BY source_database_id LIMIT 1", (experience_id,)).fetchone()
        if row: self.mark_alias_removed(*row)

    def append_evaluation_snapshot(self, experience_id: str, evaluation: dict[str, Any], fingerprint: str, provenance: dict[str, Any] | None = None, **kwargs: Any) -> bool:
        prov = dict(provenance or {})
        if "evaluation_fingerprint" in prov and prov["evaluation_fingerprint"] != fingerprint:
            from .errors import ProvenanceViolationError
            raise ProvenanceViolationError("provenance evaluation fingerprint does not match snapshot fingerprint")
        prov.setdefault("evaluation_fingerprint", fingerprint)
        prov = validate_evaluation_provenance(prov)
        with self._connect() as db:
            before = db.total_changes
            db.execute("INSERT OR IGNORE INTO experience_outcome_snapshots VALUES (?,?,?,?,?,?,?)", (experience_id, evaluation.get("evaluation_basis"), evaluation.get("horizon_seconds"), fingerprint, _json(evaluation), _json(prov), _now()))
            return db.total_changes > before

    def evaluation_snapshots(self, experience_id: str) -> tuple[dict[str, Any], ...]:
        with self._connect() as db:
            return tuple({**json.loads(r["evaluation_json"]), "fingerprint": r["fingerprint"], "provenance": json.loads(r["provenance_json"])} for r in db.execute("SELECT * FROM experience_outcome_snapshots WHERE experience_id=? ORDER BY rowid", (experience_id,)))

    def evaluation_snapshot_fingerprints(self, experience_id: str) -> tuple[str, ...]: return tuple(s["fingerprint"] for s in self.evaluation_snapshots(experience_id))

    def record_failed_scan(self, source_database_id: str, reason: str, detail: Any = "") -> None:
        with self._connect() as db:
            label = diagnostic_label(reason, source_database_id=source_database_id)
            db.execute("INSERT INTO experience_import_events(event_type,source_database_id,detail_json,observed_at) VALUES (?,?,?,?)", ("FAILED_SCAN", source_database_id, _json({"reason": reason}), _now()))
            db.execute("INSERT INTO experience_diagnostic_fts(label,source_database_id) VALUES (?,?)", (label, source_database_id))

    def _quarantine_db(self, db, source_database_id, decision_id, fingerprint, reason, detail):
        db.execute("INSERT INTO experience_quarantine(source_database_id,decision_id,fingerprint,reason,detail_json,observed_at) VALUES (?,?,?,?,?,?)", (source_database_id, decision_id, fingerprint, reason, _json(detail), _now()))
        db.execute("INSERT INTO experience_diagnostic_fts(label,source_database_id,decision_id) VALUES (?,?,?)", (diagnostic_label(reason), source_database_id, decision_id))

    def record_quarantine(self, source_database_id: str, decision_id: str, fingerprint: str, reason: str, detail: Any = None) -> None:
        with self._connect() as db: self._quarantine_db(db, source_database_id, decision_id, fingerprint, reason, detail or {})

    def quarantine_count(self) -> int:
        with self._connect() as db: return int(db.execute("SELECT COUNT(*) FROM experience_quarantine").fetchone()[0])

    def publish_generation(self, generation_id: str, population_fingerprint: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {"generation_id": generation_id, "population_fingerprint": population_fingerprint, "metadata": metadata or {}, "published_at": _now()}
        with self._connect() as db:
            db.execute("UPDATE experience_generations SET active=0")
            db.execute("INSERT OR REPLACE INTO experience_generations VALUES (?,?,?,?,1)", (generation_id, population_fingerprint, _json(metadata), payload["published_at"]))
        return payload

    def active_generation(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM experience_generations WHERE active=1 ORDER BY published_at DESC LIMIT 1").fetchone()
            return {"generation_id": row["generation_id"], "population_fingerprint": row["population_fingerprint"], "metadata": json.loads(row["metadata_json"]), "published_at": row["published_at"]} if row else None


__all__ = ["ExperienceCatalog"]
