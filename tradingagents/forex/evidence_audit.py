"""Isolated, metadata-only persistence for Phase 9 evidence usage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .evidence_context import (
    EvidenceAuditStatus,
    EvidenceBundleStatus,
    EvidenceIntegrationStatus,
    EvidenceReferenceRejectionReason,
    EvidenceUseStatus,
)

_FORBIDDEN_WORDS = (
    "prompt",
    "completion",
    "reasoning",
    "chain of thought",
    "api key",
    "apikey",
    "password",
    "token",
    "credential",
    "authorization",
    "secret",
    "bearer",
)
_DIAGNOSTIC_KEYS = {"diagnostics", "source_errors"}


def _contains_forbidden(value: str) -> bool:
    normalized = " ".join("".join(char if char.isalnum() else " " for char in value.lower()).split())
    return any(word in normalized for word in _FORBIDDEN_WORDS)


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
            if _contains_forbidden(str(key)):
                raise ValueError(f"forbidden audit field: {path}.{key}")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list, set, frozenset)):
        for i, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{i}]")
    elif isinstance(value, str) and _contains_forbidden(value):
        raise ValueError(f"forbidden audit value: {path}")


def _bounded(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _bounded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_bounded(v) for v in value]
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ")[:500]
    return value


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    """Normalize audit identifier lists without accepting scalar coercions."""
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be a sequence of strings")
    try:
        values = tuple(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a sequence of strings") from exc
    if any(not isinstance(item, str) or not item.strip() or item != item.strip() for item in values):
        raise ValueError(f"{name} entries must be non-empty trimmed strings")
    return values


def _rejection_sequence(value: Any) -> tuple[Any, ...]:
    """Normalize rejection records while preserving their structured reason."""
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("evidence_refs_rejected must be a sequence")
    try:
        values = tuple(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("evidence_refs_rejected must be a sequence") from exc
    for item in values:
        ref = item.get("ref") if isinstance(item, Mapping) else getattr(item, "ref", None)
        reason = item.get("reason") if isinstance(item, Mapping) else getattr(item, "reason", None)
        if not isinstance(ref, str) or not ref.strip() or ref != ref.strip():
            raise ValueError("evidence_refs_rejected entries must have a valid ref")
        try:
            EvidenceReferenceRejectionReason(reason)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "evidence_refs_rejected entries must have an allowed reason"
            ) from exc
    return values


def _closed_status(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an allowed status") from exc


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
        for name, enum_type in (
            ("integration_status", EvidenceIntegrationStatus),
            ("bundle_status", EvidenceBundleStatus),
            ("evidence_use_status", EvidenceUseStatus),
            ("evidence_audit_status", EvidenceAuditStatus),
        ):
            object.__setattr__(self, name, _closed_status(getattr(self, name), enum_type, name))
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        for field in (
            "source_status",
            "diagnostics",
            "source_errors",
            "selected_counts",
            "dropped_counts",
            "node_context_hashes",
        ):
            value = getattr(self, field)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{field} must be a mapping")
        for field in (
            "available_knowledge_ids",
            "available_experience_ids",
            "available_statistics_ids",
            "evidence_refs_used",
            "telemetry_references",
            "missing_nodes",
        ):
            object.__setattr__(self, field, _string_sequence(getattr(self, field), field))
        object.__setattr__(self, "evidence_refs_rejected", _rejection_sequence(self.evidence_refs_rejected))
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
            with closing(sqlite3.connect(self.path)) as db, db:
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
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute(f"INSERT OR IGNORE INTO evidence_usage_audit ({','.join(columns)}) VALUES ({placeholders})", values)
        except ValueError:
            raise
        except (OSError, sqlite3.Error, TypeError) as exc:
            raise AuditWriteError(f"unable to append evidence audit: {exc}") from exc

    def _rows(self, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        try:
            self.initialize()
            with closing(sqlite3.connect(self.path)) as db, db:
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
