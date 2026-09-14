"""Immutable, JSON-safe contracts for the Phase 10 dataset factory."""

from __future__ import annotations

import dataclasses
import json
import math
import re
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


_SENSITIVE = re.compile(
    r"(?:prompt|completion|reasoning|chain[ _-]?of[ _-]?thought|\bcot\b|private[ _-]?reasoning|secret|password|credential|api[ _-]?key|token)",
    re.I,
)


def _freeze(v):
    _assert_safe(v)
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


def _assert_safe(v, key=""):
    if key and _SENSITIVE.search(str(key)):
        raise ValueError(f"forbidden field: {key}")
    if isinstance(v, Mapping):
        for k, child in v.items():
            _assert_safe(child, str(k))
    elif isinstance(v, (list, tuple)):
        for child in v:
            _assert_safe(child)
    elif isinstance(v, str) and _SENSITIVE.search(v):
        raise ValueError("forbidden sensitive value")


def _validate_id(v, name="id"):
    if not isinstance(v, str) or not v.strip() or len(v) > 256:
        raise ValueError(f"{name} must be a bounded non-empty string")


def _validate_optional_text(v, name):
    if not isinstance(v, str) or len(v) > 256:
        raise ValueError(f"{name} must be a bounded string")


def _require_mapping(v, name):
    if not isinstance(v, Mapping):
        raise ValueError(f"{name} must be a mapping")


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
        if _SENSITIVE.search(str(k)):
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
        if not isinstance(self.filters, Mapping):
            raise DatasetConfigError("filters must be a mapping")
        object.__setattr__(self, "filters", _freeze(self.filters))
        if not isinstance(self.allow_empty, bool):
            raise DatasetConfigError("allow_empty must be bool")
        if not paths:
            raise DatasetConfigError("at least one source database is required")
        if (
            not isinstance(self.evaluation_basis, str)
            or not self.evaluation_basis.strip()
            or len(self.evaluation_basis) > 64
            or not isinstance(self.horizon_seconds, int)
            or isinstance(self.horizon_seconds, bool)
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
        _validate_id(self.contract_version, "contract_version")


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
        if not isinstance(self.action, str) or self.action not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("unsupported action")
        _require_mapping(self.fields, "fields")
        object.__setattr__(self, "fields", _freeze(self.fields))
        for n in (
            "source_run_id",
            "requested_symbol",
            "resolved_symbol",
            "analysis_profile",
            "analysis_timeframe",
        ):
            _validate_optional_text(getattr(self, n), n)


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
        if (
            not isinstance(self.evaluation_basis, str)
            or not self.evaluation_basis.strip()
            or len(self.evaluation_basis) > 64
        ):
            raise ValueError("invalid evaluation basis")
        if (
            not isinstance(self.horizon_seconds, int)
            or isinstance(self.horizon_seconds, bool)
            or self.horizon_seconds <= 0
        ):
            raise ValueError("invalid horizon")
        if not isinstance(self.source_context_eligible, bool):
            raise ValueError("source_context_eligible must be bool")
        if not isinstance(self.evaluation_status, str) or self.evaluation_status not in {
            "PENDING",
            "COMPLETE",
            "DATA_UNAVAILABLE",
            "INELIGIBLE",
        }:
            raise ValueError("unsupported evaluation status")
        _require_mapping(self.fields, "fields")
        object.__setattr__(self, "fields", _freeze(self.fields))


@dataclass(frozen=True, slots=True)
class EvidenceObservation(Contract):
    context_integrity: str = "COMPLETE"
    evidence_use_status: str = "NONE_RELEVANT"
    refs_used: tuple[str, ...] = ()
    refs_rejected: tuple[str, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.context_integrity, str) or self.context_integrity not in {
            "COMPLETE",
            "INCOMPLETE",
            "UNAVAILABLE",
            "INVALID",
        }:
            raise ValueError("unsupported context integrity")
        if not isinstance(self.evidence_use_status, str) or self.evidence_use_status not in {
            "USED",
            "NONE_RELEVANT",
            "INCOMPLETE",
            "INVALID",
        }:
            raise ValueError("unsupported evidence status")
        for name, refs in (("refs_used", self.refs_used), ("refs_rejected", self.refs_rejected)):
            if not isinstance(refs, (list, tuple)):
                raise ValueError(f"{name} must be a sequence")
            for ref in refs:
                _validate_id(ref, name)
            object.__setattr__(self, name, tuple(refs))
        _require_mapping(self.fields, "fields")
        object.__setattr__(self, "fields", _freeze(self.fields))


@dataclass(frozen=True, slots=True)
class JoinedObservation(Contract):
    decision: SourceObservation
    evaluation: EvaluationObservation | None = None
    evidence: EvidenceObservation | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.decision, SourceObservation):
            raise ValueError("decision must be SourceObservation")
        if self.evaluation is not None and not isinstance(self.evaluation, EvaluationObservation):
            raise ValueError("evaluation must be EvaluationObservation")
        if self.evidence is not None and not isinstance(self.evidence, EvidenceObservation):
            raise ValueError("evidence must be EvidenceObservation")
        _require_mapping(self.fields, "fields")
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
        for name in ("decision", "market", "research", "outcome", "trust", "provenance"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be a mapping")
        if (
            not isinstance(self.schema_version, str)
            or not self.schema_version.strip()
            or len(self.schema_version) > 64
        ):
            raise ValueError("invalid schema version")
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
        if not isinstance(self.reasons, (list, tuple)) or any(
            not isinstance(r, DatasetExclusionReason) for r in self.reasons
        ):
            raise ValueError("reasons must contain DatasetExclusionReason values")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        _require_mapping(self.details, "details")
        object.__setattr__(self, "details", _freeze(self.details))


@dataclass(frozen=True, slots=True)
class SplitAssignment(Contract):
    example_id: str
    split: str
    group_id: str

    def __post_init__(self):
        _validate_id(self.example_id, "example_id")
        _validate_id(self.group_id, "group_id")
        if not isinstance(self.split, str) or self.split not in {"train", "validation", "test"}:
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
    status: str = "EMPTY_ELIGIBLE_SET"

    def __post_init__(self):
        _validate_id(self.dataset_id, "dataset_id")
        if not isinstance(self.status, str) or self.status not in {
            "PUBLISHED",
            "EMPTY_ELIGIBLE_SET",
            "FAILED",
        }:
            raise ValueError("unsupported manifest status")
        if not isinstance(self.split_status, str) or self.split_status not in {
            "COMPLETE",
            "INSUFFICIENT_DATA",
            "FAILED",
        }:
            raise ValueError("unsupported split status")
        for name, value in (("examples", self.examples), ("exclusions", self.exclusions)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "safety", _freeze(self.safety))
        if not isinstance(self.safety, Mapping):
            raise ValueError("safety must be a mapping")
        for name, value in self.safety.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"safety.{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class BuildReport(Contract):
    status: str
    manifest: DatasetManifest | None = None
    exclusions: tuple[DatasetExclusion, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.status, str) or self.status not in {
            "PUBLISHED",
            "EMPTY_ELIGIBLE_SET",
            "FAILED",
        }:
            raise ValueError("unsupported build status")
        if self.manifest is not None and not isinstance(self.manifest, DatasetManifest):
            raise ValueError("manifest must be DatasetManifest")
        if not isinstance(self.exclusions, (list, tuple)) or not isinstance(
            self.errors, (list, tuple)
        ):
            raise ValueError("report collections must be sequences")
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "errors", tuple(self.errors))
        if any(not isinstance(x, DatasetExclusion) for x in self.exclusions):
            raise ValueError("exclusions must contain DatasetExclusion values")
        if any(not isinstance(x, str) or len(x) > 256 for x in self.errors):
            raise ValueError("errors must contain bounded strings")


@dataclass(frozen=True, slots=True)
class ValidationReport(Contract):
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be bool")
        if not isinstance(self.errors, (list, tuple)) or not isinstance(
            self.warnings, (list, tuple)
        ):
            raise ValueError("report collections must be sequences")
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if any(not isinstance(x, str) or len(x) > 256 for x in self.errors + self.warnings):
            raise ValueError("report messages must contain bounded strings")
