"""Immutable, JSON-safe contracts for the Phase 10 dataset factory."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import DatasetConfigError

DATASET_SCHEMA_VERSION = "phase10.dataset.v1"
EXAMPLE_SCHEMA_VERSION = "phase10.dataset.example.v1"
ELIGIBILITY_POLICY_VERSION = "phase10.eligibility.v1"
CANONICALIZATION_VERSION = "phase10.canonicalization.v1"
SPLIT_POLICY_VERSION = "phase10.split.v1"
SOURCE_ADAPTER_VERSION = "phase10.source-adapter.v1"


class DatasetExclusionReason(str, Enum):
    UNTRUSTED_TIER = "UNTRUSTED_TIER"
    DECISION_CONTEXT_INCOMPLETE = "DECISION_CONTEXT_INCOMPLETE"
    TEMPORAL_INVALID = "TEMPORAL_INVALID"
    OUTCOME_INELIGIBLE = "OUTCOME_INELIGIBLE"
    OUTCOME_UNAVAILABLE = "OUTCOME_UNAVAILABLE"
    NORMALIZATION_FAILED = "NORMALIZATION_FAILED"
    CITATION_INVALID = "CITATION_INVALID"
    EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"
    SCHEMA_UNSUPPORTED = "SCHEMA_UNSUPPORTED"
    PROVENANCE_INCOMPLETE = "PROVENANCE_INCOMPLETE"
    DUPLICATE = "DUPLICATE"
    SOURCE_INTEGRITY_FAILED = "SOURCE_INTEGRITY_FAILED"


_FORBIDDEN = {
    "prompt",
    "completion",
    "reasoning",
    "private_reasoning",
    "tokens",
    "credentials",
    "api_key",
}


def _freeze(v):
    if isinstance(v, Mapping):
        return MappingProxyType(
            {str(k): _freeze(x) for k, x in sorted(v.items(), key=lambda i: str(i[0]))}
        )
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("non-finite numeric value")
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    raise TypeError(f"value is not JSON-safe: {type(v).__name__}")


def _validate_id(v, name="id"):
    if not isinstance(v, str) or not v.strip() or len(v) > 256:
        raise ValueError(f"{name} must be a bounded non-empty string")


def _validate_dt(v, name):
    if (
        not isinstance(v, datetime)
        or v.tzinfo is None
        or v.utcoffset() is None
        or v.utcoffset() != timezone.utc.utcoffset(v)
    ):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _clean(data):
    out = {}
    for k, v in data.items():
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden field: {k}")
        out[k] = v
    return out


class Contract:
    def to_dict(self):
        return _clean(_plain(self))

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plain(v):
    if dataclasses.is_dataclass(v):
        return {
            f.name: _plain(getattr(v, f.name))
            for f in dataclasses.fields(v)
            if getattr(v, f.name) is not None
        }
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in sorted(v.items(), key=lambda i: str(i[0]))}
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if isinstance(v, (Enum, Path)):
        return v.value if isinstance(v, Enum) else str(v)
    if isinstance(v, datetime):
        return v.isoformat()
    return v


@dataclass(frozen=True, slots=True)
class DatasetConfig(Contract):
    source_db_paths: tuple[Path, ...]
    phase8_root: Path
    phase9_audit_path: Path | None
    output_root: Path
    filters: Mapping[str, Any] = field(default_factory=dict)
    evaluation_basis: str = "ANALYSIS_SNAPSHOT"
    horizon_seconds: int = 300
    allow_empty: bool = True

    def __post_init__(self):
        paths = tuple(Path(p).resolve() for p in self.source_db_paths)
        object.__setattr__(self, "source_db_paths", paths)
        object.__setattr__(self, "phase8_root", Path(self.phase8_root).resolve())
        object.__setattr__(
            self,
            "phase9_audit_path",
            Path(self.phase9_audit_path).resolve() if self.phase9_audit_path else None,
        )
        out = Path(self.output_root).resolve()
        object.__setattr__(self, "output_root", out)
        object.__setattr__(self, "filters", _freeze(self.filters))
        if not paths:
            raise DatasetConfigError("at least one source database is required")
        if (
            not self.evaluation_basis
            or len(self.evaluation_basis) > 64
            or self.horizon_seconds <= 0
        ):
            raise DatasetConfigError("invalid basis or horizon")
        roots = (
            paths
            + (self.phase8_root,)
            + ((self.phase9_audit_path,) if self.phase9_audit_path else ())
        )
        if any(out == r or r in out.parents for r in roots):
            raise DatasetConfigError("output root overlaps a source root")


@dataclass(frozen=True, slots=True)
class SourceFingerprint(Contract):
    source_id: str
    canonical_path: str
    schema_fingerprint: str
    file_sha256: str
    snapshot_fingerprint: str
    contract_version: str = SOURCE_ADAPTER_VERSION

    def __post_init__(self):
        for n in (
            "source_id",
            "canonical_path",
            "schema_fingerprint",
            "file_sha256",
            "snapshot_fingerprint",
        ):
            _validate_id(getattr(self, n), n)


@dataclass(frozen=True, slots=True)
class SourceObservation(Contract):
    decision_id: str
    analysis_snapshot_timestamp: datetime
    decision_completed_timestamp: datetime | None = None
    source_run_id: str = ""
    requested_symbol: str = ""
    resolved_symbol: str = ""
    analysis_profile: str = ""
    analysis_timeframe: str = ""
    action: str = "HOLD"
    fields: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: SourceFingerprint | None = None

    def __post_init__(self):
        _validate_id(self.decision_id, "decision_id")
        _validate_dt(self.analysis_snapshot_timestamp, "analysis_snapshot_timestamp")
        if self.decision_completed_timestamp is not None:
            _validate_dt(self.decision_completed_timestamp, "decision_completed_timestamp")
        if self.action not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("unsupported action")
        object.__setattr__(self, "fields", _freeze(self.fields))


@dataclass(frozen=True, slots=True)
class EvaluationObservation(Contract):
    decision_id: str
    evaluation_basis: str = "ANALYSIS_SNAPSHOT"
    horizon_seconds: int = 300
    evaluation_status: str = "COMPLETE"
    source_context_eligible: bool = True
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_id(self.decision_id, "decision_id")
        object.__setattr__(self, "fields", _freeze(self.fields))


@dataclass(frozen=True, slots=True)
class EvidenceObservation(Contract):
    context_integrity: str = "COMPLETE"
    evidence_use_status: str = "NONE_RELEVANT"
    refs_used: tuple[str, ...] = ()
    refs_rejected: tuple[str, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "fields", _freeze(self.fields))


@dataclass(frozen=True, slots=True)
class JoinedObservation(Contract):
    decision: SourceObservation
    evaluation: EvaluationObservation | None = None
    evidence: EvidenceObservation | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "fields", _freeze(self.fields))


@dataclass(frozen=True, slots=True)
class CanonicalExampleV1(Contract):
    example_id: str
    decision: Mapping[str, Any] = field(default_factory=dict)
    market: Mapping[str, Any] = field(default_factory=dict)
    research: Mapping[str, Any] = field(default_factory=dict)
    outcome: Mapping[str, Any] = field(default_factory=dict)
    trust: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EXAMPLE_SCHEMA_VERSION

    def __post_init__(self):
        _validate_id(self.example_id, "example_id")
        [
            object.__setattr__(self, n, _freeze(getattr(self, n)))
            for n in ("decision", "market", "research", "outcome", "trust", "provenance")
        ]


@dataclass(frozen=True, slots=True)
class DatasetExclusion(Contract):
    decision_id: str
    reasons: tuple[DatasetExclusionReason, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_id(self.decision_id, "decision_id")
        object.__setattr__(self, "details", _freeze(self.details))


@dataclass(frozen=True, slots=True)
class SplitAssignment(Contract):
    example_id: str
    split: str
    group_id: str

    def __post_init__(self):
        _validate_id(self.example_id, "example_id")
        _validate_id(self.group_id, "group_id")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("unsupported split")


@dataclass(frozen=True, slots=True)
class DatasetManifest(Contract):
    dataset_id: str
    examples: int = 0
    exclusions: int = 0
    split_status: str = "INSUFFICIENT_DATA"
    safety: Mapping[str, int] = field(
        default_factory=lambda: {
            "network_attempts": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "mt5_calls": 0,
        }
    )

    def __post_init__(self):
        _validate_id(self.dataset_id, "dataset_id")
        object.__setattr__(self, "safety", _freeze(self.safety))


@dataclass(frozen=True, slots=True)
class BuildReport(Contract):
    status: str
    manifest: DatasetManifest | None = None
    exclusions: tuple[DatasetExclusion, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationReport(Contract):
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
