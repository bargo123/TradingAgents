"""Isolated, metadata-only persistence for Phase 9 evidence usage."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

_FORBIDDEN = re.compile(r"(?:prompt|completion|reasoning|chain[_ ]of[_ ]thought|api[_ ]key|password|token|credential)", re.I)
_DIAGNOSTIC_KEYS = {"diagnostics", "source_errors"}


class AuditWriteError(RuntimeError):
    """Raised when the isolated Phase 9 audit cannot be persisted."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(v) for v in value]
    return value


def _reject_forbidden(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _FORBIDDEN.search(str(key)):
                raise ValueError(f"forbidden audit field: {path}.{key}")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list, set, frozenset)):
        for i, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{i}]")
    elif isinstance(value, str) and _FORBIDDEN.search(value):
        raise ValueError(f"forbidden audit value: {path}")


def _bounded(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _bounded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_bounded(v) for v in value]
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ")[:500]
    return value


@dataclass(frozen=True, slots=True)
class EvidenceUsageAudit:
    decision_id: str
    source_run_id: str
    integration_status: str
    bundle_status: str
    as_of: datetime
    knowledge_generation_id: str | None = None
    experience_generation_id: str | None = None
    query_normalization_fingerprint: str | None = None
    knowledge_query: Any = None
    knowledge_query_fingerprint: str | None = None
    query_policy_version: str | None = None
    rendered_context: str = ""
    rendered_context_hash: str = ""
    available_knowledge_ids: tuple[str, ...] = ()
    available_experience_ids: tuple[str, ...] = ()
    available_statistics_ids: tuple[str, ...] = ()
    evidence_use_status: str = "UNAVAILABLE"
    evidence_refs_used: tuple[str, ...] = ()
    evidence_refs_rejected: tuple[Any, ...] = ()
    evidence_audit_status: str = "VALID"
    source_status: Mapping[str, Any] = None
    diagnostics: Mapping[str, Any] = None
    source_errors: Mapping[str, Any] = None
    retrieval_count: int = 0
    retrieval_latency_seconds: float | None = None
    builder_latency_seconds: float | None = None
    selected_counts: Mapping[str, int] = None
    dropped_counts: Mapping[str, int] = None
    telemetry_references: tuple[str, ...] = ()
    node_context_hashes: Mapping[str, str] = None
    missing_nodes: tuple[str, ...] = ()
    provider: str | None = None
    model: str | None = None
    audit_schema_version: str = "phase9.v1"

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        expected = hashlib.sha256(self.rendered_context.encode("utf-8")).hexdigest()
        if self.rendered_context_hash != expected:
            raise ValueError("rendered_context_hash does not match rendered_context")
        available = set(self.available_knowledge_ids) | set(self.available_experience_ids) | set(self.available_statistics_ids)
        if any(ref not in available for ref in self.evidence_refs_used):
            raise ValueError("evidence_refs_used contains an unknown reference")
        for rejection in self.evidence_refs_rejected:
            ref = rejection.get("ref") if isinstance(rejection, Mapping) else getattr(rejection, "ref", None)
            if ref not in available:
                raise ValueError("evidence_refs_rejected contains an unknown reference")
        _reject_forbidden(_jsonable(self))


class EvidenceAuditStore:
    def __init__(self, path: str | Path = "data_cache/evidence_runtime/evidence_audit.sqlite3") -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path) as db:
                db.execute("""CREATE TABLE IF NOT EXISTS evidence_usage_audit (
                    decision_id TEXT NOT NULL, source_run_id TEXT NOT NULL, integration_status TEXT NOT NULL,
                    bundle_status TEXT NOT NULL, as_of TEXT NOT NULL, knowledge_generation_id TEXT,
                    experience_generation_id TEXT, query_normalization_fingerprint TEXT, knowledge_query TEXT,
                    knowledge_query_fingerprint TEXT, query_policy_version TEXT, rendered_context TEXT NOT NULL,
                    rendered_context_hash TEXT NOT NULL, available_knowledge_ids TEXT, available_experience_ids TEXT,
                    available_statistics_ids TEXT, evidence_use_status TEXT, evidence_refs_used TEXT,
                    evidence_refs_rejected TEXT, evidence_audit_status TEXT, source_status TEXT, diagnostics TEXT,
                    source_errors TEXT, retrieval_count INTEGER, retrieval_latency_seconds REAL,
                    builder_latency_seconds REAL, selected_counts TEXT, dropped_counts TEXT, telemetry_references TEXT,
                    node_context_hashes TEXT, missing_nodes TEXT, provider TEXT, model TEXT,
                    audit_schema_version TEXT NOT NULL, PRIMARY KEY (decision_id, rendered_context_hash))""")
        except (OSError, sqlite3.Error) as exc:
            raise AuditWriteError(f"unable to initialize evidence audit store: {exc}") from exc

    def append(self, audit: EvidenceUsageAudit) -> None:
        try:
            self.initialize()
            payload = _jsonable(audit)
            for key in _DIAGNOSTIC_KEYS:
                payload[key] = _bounded(payload.get(key) or {})
            columns = [f.name for f in fields(audit)]
            values = [json.dumps(payload[c], sort_keys=True, separators=(",", ":")) if isinstance(payload[c], (dict, list)) else payload[c] for c in columns]
            placeholders = ",".join("?" for _ in columns)
            with sqlite3.connect(self.path) as db:
                db.execute(f"INSERT OR IGNORE INTO evidence_usage_audit ({','.join(columns)}) VALUES ({placeholders})", values)
        except ValueError:
            raise
        except (OSError, sqlite3.Error, TypeError) as exc:
            raise AuditWriteError(f"unable to append evidence audit: {exc}") from exc

    def _rows(self, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        try:
            self.initialize()
            with sqlite3.connect(self.path) as db:
                db.row_factory = sqlite3.Row
                rows = [dict(row) for row in db.execute(query, params)]
            return rows
        except (OSError, sqlite3.Error) as exc:
            raise AuditWriteError(f"unable to read evidence audit: {exc}") from exc

    def get(self, decision_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM evidence_usage_audit WHERE decision_id=? ORDER BY rowid LIMIT 1", (decision_id,))
        return rows[0] if rows else None

    def list_for_source_run(self, source_run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM evidence_usage_audit WHERE source_run_id=? ORDER BY rowid", (source_run_id,))
