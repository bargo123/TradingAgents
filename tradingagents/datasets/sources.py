"""Read-only adapters for the Phase 5/6, 8, and 9 persistence boundaries."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from tradingagents.experience.identity import (
    source_decision_fingerprint,
    source_evaluation_fingerprint,
)
from tradingagents.experience.source_reader import (
    _DECISION_COLUMNS,
    _EVALUATION_COLUMNS,
)

from .errors import SourceIntegrityError, SourceReadError
from .models import EvaluationObservation, SourceFingerprint, SourceObservation

_DECISION_REQUIRED = set(_DECISION_COLUMNS)
_EVAL_REQUIRED = set(_EVALUATION_COLUMNS)
_AUDIT_TABLE = "evidence_usage_audit"
_AUDIT_FIELDS = (
    "decision_id",
    "source_run_id",
    "as_of",
    "rendered_context",
    "rendered_context_hash",
    "knowledge_generation_id",
    "experience_generation_id",
    "knowledge_query",
    "integration_status",
    "bundle_status",
    "evidence_use_status",
    "available_knowledge_ids",
    "available_experience_ids",
    "available_statistics_ids",
    "evidence_refs_used",
    "evidence_refs_rejected",
    "query_policy_version",
    "source_status",
    "diagnostics",
    "source_errors",
    "retrieval_count",
    "retrieval_latency_seconds",
    "builder_latency_seconds",
    "selected_counts",
    "dropped_counts",
    "telemetry_references",
    "node_context_hashes",
    "missing_nodes",
    "provider",
    "model",
    "audit_schema_version",
    "evidence_audit_status",
    "query_normalization_fingerprint",
    "knowledge_query_fingerprint",
)
_PHASE8_REQUIRED = {
    "experience_records": {
        "experience_id",
        "source_decision_id",
        "source_database_id",
        "source_decision_fingerprint",
        "symbol",
        "requested_symbol",
        "analysis_profile",
        "analysis_timeframe",
        "analysis_snapshot_timestamp",
        "decision_completed_timestamp",
        "decision_reference_timestamp",
        "market_state_json",
        "decision_evidence_json",
        "tombstoned",
        "trust",
        "provenance_json",
        "source_evaluation_fingerprints_json",
    },
    "experience_source_aliases": {
        "source_database_id",
        "source_decision_id",
        "accepted_fingerprint",
        "experience_id",
        "state",
        "observed_at",
        "scan_status",
    },
    "experience_outcome_snapshots": {
        "experience_id",
        "fingerprint",
        "evaluation_json",
        "provenance_json",
        "observed_at",
        "evaluation_basis",
        "horizon_seconds",
    },
    "experience_feature_projections": {
        "experience_id",
        "feature_schema_version",
        "projection_json",
    },
}


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _stamp(value: Any) -> Any:
    if not isinstance(value, str) or "T" not in value:
        return value
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if dt.utcoffset() != timezone.utc.utcoffset(dt):
        raise SourceReadError("timestamp is not UTC")
    return dt.astimezone(timezone.utc)


def _bounded(value: Any, limit: int = 2048) -> Any:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ")[:limit]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_bounded(x, limit) for x in value[:100]]
    if isinstance(value, dict):
        return {str(k)[:128]: _bounded(v, limit) for k, v in list(value.items())[:100]}
    return str(value)[:limit]


def _reject_oversized_collections(value: Any, limit: int = 100) -> None:
    """Reject collection values before the legacy diagnostic bounding pass."""
    if isinstance(value, (list, tuple)):
        if len(value) > limit:
            raise SourceReadError("Phase 9 array metadata exceeds bound")
        for item in value:
            _reject_oversized_collections(item, limit)
    elif isinstance(value, dict):
        if len(value) > limit:
            raise SourceReadError("Phase 9 object metadata exceeds bound")
        for item in value.values():
            _reject_oversized_collections(item, limit)


def _normalize_row(item: dict[str, Any]) -> dict[str, Any]:
    for key, value in tuple(item.items()):
        if value is not None and (
            key == "as_of" or key.endswith("_at") or key.endswith("_timestamp")
        ):
            item[key] = _stamp(value)
    return item


def _identity_row(item: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized source facts used to derive a row identity.

    Identity columns are adapter outputs, not source facts.  Excluding them
    prevents a stale or forged value persisted by an upstream source from
    changing the authoritative identity we derive here.
    """
    return {
        key: value
        for key, value in item.items()
        if key
        not in {
            "source_decision_fingerprint",
            "source_evaluation_fingerprint",
            "fingerprint",
        }
    }


def _decode_array(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        decoded = list(value)
    else:
        try:
            decoded = json.loads(value or "[]")
        except (TypeError, json.JSONDecodeError) as exc:
            raise SourceReadError("invalid Phase 9 array metadata") from exc
    if not isinstance(decoded, list):
        raise SourceReadError("Phase 9 array metadata must be a JSON array")
    _reject_oversized_collections(decoded)
    return _bounded(decoded, 500)


def _decode_mapping(value: Any) -> dict[str, Any]:
    """Decode JSON object columns without coercing them into empty arrays."""
    if isinstance(value, dict):
        decoded = value
    else:
        try:
            decoded = json.loads(value or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise SourceReadError("invalid Phase 9 object metadata") from exc
    if not isinstance(decoded, dict):
        raise SourceReadError("Phase 9 object metadata must be a JSON object")
    _reject_oversized_collections(decoded)
    return _bounded(decoded, 500)


def _decode_phase8_object(value: Any) -> dict[str, Any]:
    """Decode a bounded Phase 8 JSON object without silent truncation."""
    if isinstance(value, dict):
        decoded = value
    else:
        try:
            decoded = json.loads(value or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise SourceReadError("invalid Phase 8 JSON object") from exc
    if not isinstance(decoded, dict):
        raise SourceReadError("Phase 8 JSON object must be a JSON object")
    _validate_phase8_value(decoded)
    return decoded


def _validate_phase8_value(value: Any) -> None:
    """Validate JSON values before any adapter bounding can occur.

    Phase 8 JSON columns contain source facts.  Truncating a serialized object
    before decoding would turn an invalid/oversized source into a different,
    apparently valid object.  Reject the original value instead.
    """
    if isinstance(value, str):
        if len(value) > 500:
            raise SourceReadError("Phase 8 JSON object contains oversized value")
        return
    if value is None or isinstance(value, (int, float, bool)):
        return
    if isinstance(value, list):
        if len(value) > 100:
            raise SourceReadError("Phase 8 JSON object contains oversized array")
        for item in value:
            _validate_phase8_value(item)
        return
    if isinstance(value, dict):
        if len(value) > 100:
            raise SourceReadError("Phase 8 JSON object contains oversized mapping")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise SourceReadError("Phase 8 JSON object contains oversized key")
            _validate_phase8_value(item)
        return
    raise SourceReadError("Phase 8 JSON object contains unsupported value")


def _decode_phase8_fingerprints(value: Any) -> dict[str, str]:
    decoded = _decode_phase8_object(value)
    if len(decoded) > 64 or any(
        not isinstance(key, str)
        or not key
        or len(key) > 128
        or not isinstance(fingerprint, str)
        or not fingerprint
        or len(fingerprint) > 256
        for key, fingerprint in decoded.items()
    ):
        raise SourceReadError("invalid Phase 8 evaluation fingerprints")
    return decoded


def _related(
    db, experience_id: str, table: str, query: str, raw_json_keys: frozenset[str] = frozenset()
) -> list[dict[str, Any]]:
    names = [x[0] for x in db.execute(f'SELECT * FROM "{table}" LIMIT 0').description]
    return [
        {
            k: v if k in raw_json_keys else _bounded(v)
            for k, v in zip(names, row, strict=True)
        }
        for row in db.execute(query, (experience_id,))
    ]


@dataclass(frozen=True, slots=True)
class SourceReadResult:
    decisions: tuple[SourceObservation, ...] = ()
    evaluations: tuple[EvaluationObservation, ...] = ()
    records: tuple[dict[str, Any], ...] = ()
    audits: tuple[dict[str, Any], ...] = ()
    fingerprint: SourceFingerprint | None = None
    query_only: bool = True
    available: bool = True
    unavailable: UnavailableObservation | None = None

    @property
    def observations(self):
        return self.decisions


@dataclass(frozen=True, slots=True)
class UnavailableObservation:
    source: str
    status: str = "UNAVAILABLE"
    reason: str = "SOURCE_MISSING"
    available: bool = False


class SourceDatabaseUnavailableError(SourceReadError):
    """The explicitly supplied source path does not exist or cannot be opened."""


class SourceSchemaIncompatibleError(SourceReadError):
    """A source exists but does not satisfy its versioned read contract."""


class SourceSnapshotChangedError(SourceIntegrityError):
    """The database or WAL changed while its bounded snapshot was read."""


class _Readonly:
    kind = "source"

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._before_read_hook = None

    @property
    def sqlite_uri(self):
        return "file:" + quote(self.path.as_posix(), safe="/:\\") + "?mode=ro"

    def _marks(self):
        return (_sha(self.path), _sha(Path(str(self.path) + "-wal")))

    def _open(self):
        if not self.path.is_file():
            raise SourceDatabaseUnavailableError(f"source database is unavailable: {self.path}")
        try:
            db = sqlite3.connect(self.sqlite_uri, uri=True)
            db.execute("PRAGMA query_only=ON")
            return db
        except sqlite3.Error as exc:
            raise SourceDatabaseUnavailableError(str(exc)) from exc

    @staticmethod
    def _tables(db):
        tables = {}
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            cols = tuple(
                row[1]
                for row in db.execute(f'PRAGMA table_info("{name.replace(chr(34), chr(34) * 2)}")')
            )
            tables[name] = set(cols)
        return tables

    def connection_for_test(self):
        return self._open()

    def _fingerprint(self, schema: str, marks, snapshot: Any):
        payload = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
        snap = hashlib.sha256(payload.encode()).hexdigest()
        schema_fp = hashlib.sha256(schema.encode()).hexdigest()
        return SourceFingerprint(
            self.kind + "-" + (marks[0] or "missing")[:24],
            str(self.path),
            schema_fp,
            marks[0] or "missing",
            snap,
        )

    def _check_after(self, marks):
        if marks != self._marks():
            raise SourceSnapshotChangedError("source SQLite or WAL changed during read")


class ReadonlyPhase56Source(_Readonly):
    kind = "phase56"

    def read(self):
        marks = self._marks()
        db = self._open()
        try:
            if self._before_read_hook:
                self._before_read_hook()
            tables = self._tables(db)
            for table, required in (
                ("shadow_decisions", _DECISION_REQUIRED),
                ("shadow_decision_evaluations", _EVAL_REQUIRED),
            ):
                if table not in tables or not required <= tables[table]:
                    raise SourceSchemaIncompatibleError(f"schema drift: {table}")
            decisions = []
            cur = db.execute("SELECT * FROM shadow_decisions ORDER BY decision_id")
            names = [x[0] for x in cur.description]
            for row in cur:
                raw_item = dict(zip(names, row, strict=True))
                _normalize_row(raw_item)
                # Phase 5/6 does not persist the Phase 8 identity columns.  Derive
                # them from the complete normalized source row before removing
                # adapter-only identity fields below, matching the Phase 8
                # importer contract exactly.
                decision_fp = source_decision_fingerprint(_identity_row(raw_item))
                item = {k: _bounded(v) for k, v in raw_item.items()}
                _normalize_row(item)
                item["source_decision_fingerprint"] = decision_fp
                ts = _stamp(item.pop("analysis_snapshot_timestamp"))
                completed = _stamp(item.pop("decision_completed_timestamp", None))
                decisions.append(
                    SourceObservation(
                        item.pop("decision_id"),
                        ts,
                        completed,
                        fields=item,
                        source_run_id=str(item.pop("source_run_id", "")),
                        requested_symbol=str(item.pop("requested_symbol", "")),
                        resolved_symbol=str(item.pop("resolved_symbol", "")),
                        analysis_profile=str(item.pop("analysis_profile", "")),
                        analysis_timeframe=str(item.pop("analysis_timeframe", "")),
                        action=str(item.pop("action", "HOLD")),
                    )
                )
            evaluations = []
            cur = db.execute(
                "SELECT * FROM shadow_decision_evaluations ORDER BY decision_id,evaluation_basis,horizon_seconds"
            )
            names = [x[0] for x in cur.description]
            for row in cur:
                raw_item = dict(zip(names, row, strict=True))
                _normalize_row(raw_item)
                evaluation_fp = source_evaluation_fingerprint(_identity_row(raw_item))
                item = {k: _bounded(v) for k, v in raw_item.items()}
                _normalize_row(item)
                item["source_evaluation_fingerprint"] = evaluation_fp
                did, basis, horizon, status = (
                    item.pop("decision_id"),
                    item.pop("evaluation_basis"),
                    item.pop("horizon_seconds"),
                    item.pop("evaluation_status"),
                )
                evaluations.append(
                    EvaluationObservation(
                        did,
                        str(basis),
                        int(horizon),
                        str(status),
                        bool(item.pop("source_context_eligible", True)),
                        item,
                    )
                )
            self._check_after(marks)
            fp = self._fingerprint(
                "phase56:" + repr(sorted((k, sorted(v)) for k, v in tables.items())),
                marks,
                [(x.decision_id, x.action) for x in decisions],
            )
            return SourceReadResult(tuple(decisions), tuple(evaluations), fingerprint=fp)
        finally:
            db.close()


class ReadonlyExperienceSource(_Readonly):
    kind = "phase8"

    def read(self):
        marks = self._marks()
        db = self._open()
        try:
            if self._before_read_hook:
                self._before_read_hook()
            tables = self._tables(db)
            required = {
                "experience_records",
                "experience_source_aliases",
                "experience_outcome_snapshots",
                "experience_feature_projections",
            }
            if not required <= tables.keys():
                raise SourceSchemaIncompatibleError("schema drift: phase8 catalog table")
            for table, columns in _PHASE8_REQUIRED.items():
                if not columns <= tables[table]:
                    raise SourceSchemaIncompatibleError(f"schema drift: phase8 {table}")
            rows = []
            for row in db.execute(
                "SELECT * FROM experience_records WHERE tombstoned=0 ORDER BY experience_id"
            ):
                names = [
                    x[0] for x in db.execute("SELECT * FROM experience_records LIMIT 0").description
                ]
                json_keys = (
                    "market_state_json",
                    "decision_evidence_json",
                    "provenance_json",
                    "source_evaluation_fingerprints_json",
                )
                item = {
                    k: v if k in json_keys else _bounded(v)
                    for k, v in zip(names, row, strict=True)
                }
                for key in json_keys:
                    raw_json = item.pop(key)
                    name = key[:-5] if key.endswith("_json") else key
                    item[name] = (
                        _decode_phase8_fingerprints(raw_json)
                        if key == "source_evaluation_fingerprints_json"
                        else _decode_phase8_object(raw_json)
                    )
                experience_id = item["experience_id"]

                aliases = _related(
                    db,
                    experience_id,
                    "experience_source_aliases",
                    "SELECT * FROM experience_source_aliases WHERE experience_id=? AND state='CURRENT'",
                )
                projections = _related(
                    db,
                    experience_id,
                    "experience_feature_projections",
                    "SELECT * FROM experience_feature_projections WHERE experience_id=?",
                    frozenset(("projection_json",)),
                )
                snapshots = _related(
                    db,
                    experience_id,
                    "experience_outcome_snapshots",
                    "SELECT * FROM experience_outcome_snapshots WHERE experience_id=? ORDER BY observed_at,fingerprint",
                    frozenset(("evaluation_json", "provenance_json")),
                )
                for snapshot in snapshots:
                    snapshot["evaluation"] = _decode_phase8_object(snapshot.pop("evaluation_json"))
                    snapshot["provenance"] = _decode_phase8_object(snapshot.pop("provenance_json"))
                for projection in projections:
                    projection["projection"] = _decode_phase8_object(projection.pop("projection_json"))
                item["accepted_aliases"] = aliases
                item["feature_projections"] = projections
                item["evaluation_snapshots"] = snapshots
                _normalize_row(item)
                for alias in aliases:
                    _normalize_row(alias)
                for snapshot in snapshots:
                    _normalize_row(snapshot)
                rows.append(item)
            self._check_after(marks)
            return SourceReadResult(
                records=tuple(rows),
                fingerprint=self._fingerprint(
                    "phase8:" + repr(sorted(tables)), marks, [x.get("experience_id") for x in rows]
                ),
            )
        finally:
            db.close()


class ReadonlyPhase9AuditSource(_Readonly):
    kind = "phase9"

    def read(self):
        if not self.path.is_file():
            unavailable = UnavailableObservation("phase9", reason="SOURCE_MISSING")
            return SourceReadResult(audits=(), available=False, unavailable=unavailable)
        marks = self._marks()
        db = self._open()
        try:
            if self._before_read_hook:
                self._before_read_hook()
            tables = self._tables(db)
            if _AUDIT_TABLE not in tables:
                self._check_after(marks)
                unavailable = UnavailableObservation("phase9", reason="SCHEMA_UNAVAILABLE")
                return SourceReadResult(audits=(), available=False, unavailable=unavailable)
            missing = set(_AUDIT_FIELDS) - tables[_AUDIT_TABLE]
            if missing:
                raise SourceSchemaIncompatibleError("schema drift: phase9 audit")
            names = [
                x[0] for x in db.execute(f'SELECT * FROM "{_AUDIT_TABLE}" LIMIT 0').description
            ]
            rows = []
            for row in db.execute(
                f'SELECT * FROM "{_AUDIT_TABLE}" ORDER BY decision_id,source_run_id'
            ):
                item = {
                    k: _bounded(v, 500)
                    for k, v in zip(names, row, strict=True)
                    if k in _AUDIT_FIELDS
                }
                array_keys = (
                    "available_knowledge_ids",
                    "available_experience_ids",
                    "available_statistics_ids",
                    "evidence_refs_used",
                    "evidence_refs_rejected",
                    "telemetry_references",
                    "missing_nodes",
                    "source_errors",
                )
                for key in array_keys:
                    if key in item:
                        item[key] = _decode_array(item[key])
                mapping_keys = (
                    "selected_counts",
                    "dropped_counts",
                    "source_status",
                    "diagnostics",
                    "node_context_hashes",
                )
                for key in mapping_keys:
                    if key in item:
                        item[key] = _decode_mapping(item[key])
                _normalize_row(item)
                rows.append(item)
            self._check_after(marks)
            return SourceReadResult(
                audits=tuple(rows),
                fingerprint=self._fingerprint("phase9:" + repr(sorted(tables)), marks, rows),
            )
        finally:
            db.close()


__all__ = [
    "ReadonlyPhase56Source",
    "ReadonlyExperienceSource",
    "ReadonlyPhase9AuditSource",
    "SourceReadResult",
    "UnavailableObservation",
    "SourceDatabaseUnavailableError",
    "SourceSchemaIncompatibleError",
    "SourceSnapshotChangedError",
    "_decode_phase8_object",
]
