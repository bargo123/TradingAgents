"""Versioned, JSON-safe Phase 11A value contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "phase11a-knowledge-example.v1"
DISTILLATION_POLICY_VERSION = "phase11a-distillation-policy.v1"
GROUNDING_POLICY_VERSION = "phase11a-grounding-policy.v1"
SPLIT_POLICY_VERSION = "phase11a-split-policy.v1"
MANIFEST_VERSION = "phase11a-manifest.v1"
MAX_TEXT_CHARS = 20_000
CONTENT_TYPES = frozenset({"PROSE", "EQUATION", "TABLE", "FIGURE_CAPTION", "DEFINITION", "REFERENCE", "LIST"})
_SENSITIVE = re.compile(
    r"(prompt|completion|reasoning|chain.of.thought|secret|api[_ -]?key|password)", re.I
)
_UNSAFE = re.compile(r"(buy|sell|hold|pnl|profit|reward|entry|exit|future|outcome)", re.I)


class SerializableModel:
    def to_dict(self) -> dict[str, Any]:
        return _json(self)

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return canonical_json(self)


class LessonType(str, Enum):
    DEFINITION = "DEFINITION"
    CONCEPT_EXPLANATION = "CONCEPT_EXPLANATION"
    MECHANISM = "MECHANISM"
    COMPARISON = "COMPARISON"
    SCENARIO_APPLICATION = "SCENARIO_APPLICATION"
    ASSUMPTION_AND_LIMITATION = "ASSUMPTION_AND_LIMITATION"
    EQUATION_INTERPRETATION = "EQUATION_INTERPRETATION"
    FAILURE_MODE = "FAILURE_MODE"
    MULTI_SOURCE_SYNTHESIS = "MULTI_SOURCE_SYNTHESIS"


class Difficulty(str, Enum):
    FOUNDATIONAL = "FOUNDATIONAL"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"


class CandidateStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    EXCLUDED = "EXCLUDED"


class ExclusionReason(str, Enum):
    SOURCE_PROVENANCE_INCOMPLETE = "SOURCE_PROVENANCE_INCOMPLETE"
    GROUNDING_FAILED = "GROUNDING_FAILED"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    CONTRADICTION = "CONTRADICTION"
    TARGET_INCOMPLETE = "TARGET_INCOMPLETE"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    DUPLICATE = "DUPLICATE"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    SOURCE_PACKET_TOO_LARGE = "SOURCE_PACKET_TOO_LARGE"
    LESSON_TOO_LONG = "LESSON_TOO_LONG"
    VERBATIM_OVERLAP_EXCESSIVE = "VERBATIM_OVERLAP_EXCESSIVE"
    UNSAFE_FUTURE_OUTCOME_INFERENCE = "UNSAFE_FUTURE_OUTCOME_INFERENCE"
    TEACHER_FAILED = "TEACHER_FAILED"
    JUDGE_REJECTED = "JUDGE_REJECTED"


def _json(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (Path,)):
        return str(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if is_dataclass(v):
        return {f.name: _json(getattr(v, f.name)) for f in fields(v)}
    if isinstance(v, Mapping):
        return {str(k): _json(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [_json(x) for x in v]
    if isinstance(v, (set, frozenset)):
        # A set has no stable iteration order.  Sort its canonical values so
        # hashes are identical across processes and Python hash seeds.
        return [_json(x) for x in sorted(v, key=canonical_json)]
    return v


def canonical_json(value: Any) -> str:
    return json.dumps(_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _text(value: str, name: str = "text", maximum: int = MAX_TEXT_CHARS) -> str:
    value = str(value)
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return value


def _tuple(v: Sequence[str] | str | None) -> tuple[str, ...]:
    if v is None:
        return ()
    if isinstance(v, str):
        v = (v,)
    return tuple(_text(x, "value", 1000) for x in v)


def _safe_mapping(v: Mapping[str, Any]) -> None:
    _validate_safe(v)


def _validate_safe(value: Any, path: str = "") -> None:
    """Reject sensitive/action/future fields at every nested boundary."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            field_path = f"{path}.{key_text}" if path else key_text
            if _SENSITIVE.search(key_text) or _UNSAFE.search(key_text):
                raise ValueError(f"unsafe field: {field_path}")
            _validate_safe(item, field_path)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            _validate_safe(item, f"{path}[{index}]")
    elif is_dataclass(value):
        for item in fields(value):
            _validate_safe(getattr(value, item.name), f"{path}.{item.name}" if path else item.name)


@dataclass(frozen=True, slots=True)
class SourceRef(SerializableModel):
    document_id: str
    source_filename: str
    source_hash: str
    chunk_id: str
    generation_id: str
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    chapter: str | None = None
    section: str | None = None
    content_type: str = "PROSE"

    def __post_init__(self):
        for n in ("document_id", "source_filename", "source_hash", "chunk_id", "generation_id"):
            _text(getattr(self, n), n, 500)
        if self.page is not None and int(self.page) < 1:
            raise ValueError("page must be positive")
        content_type = getattr(self.content_type, "value", self.content_type)
        content_type = str(content_type).upper()
        if content_type not in CONTENT_TYPES:
            raise ValueError(f"unsupported content_type: {content_type}")
        object.__setattr__(self, "content_type", content_type)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceRef:
        payload = dict(value)
        payload.setdefault("generation_id", payload.pop("phase7_generation_id", None))
        payload.setdefault("chunk_id", payload.pop("block_id", None))
        if "section" not in payload and payload.get("section_path") is not None:
            path = payload.pop("section_path")
            payload["section"] = " / ".join(path) if not isinstance(path, str) else path
        return cls(
            **{
                key: payload[key]
                for key in (
                    "document_id",
                    "source_filename",
                    "source_hash",
                    "chunk_id",
                    "generation_id",
                    "page",
                    "page_start",
                    "page_end",
                    "chapter",
                    "section",
                    "content_type",
                )
                if key in payload
            }
        )

    @property
    def phase7_generation_id(self) -> str:
        return self.generation_id

    @property
    def block_id(self) -> str:
        return self.chunk_id

    @property
    def section_path(self) -> tuple[str, ...]:
        return (
            ()
            if self.section is None
            else tuple(part.strip() for part in self.section.split(" / ") if part.strip())
        )


@dataclass(frozen=True, slots=True)
class GroundingClaim(SerializableModel):
    claim: str
    source_refs: tuple[SourceRef, ...]

    def __post_init__(self):
        _text(self.claim, "claim", 4000)
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        if not self.source_refs:
            raise ValueError("claim requires source_refs")


@dataclass(frozen=True, slots=True)
class SourceBlock(SerializableModel):
    ref: SourceRef
    text: str
    content_type: str = "PROSE"
    reading_order: int = 0
    section_path: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = None

    def __post_init__(self):
        _text(self.text)
        content_type = getattr(self.content_type, "value", self.content_type)
        content_type = str(content_type).upper()
        if content_type not in CONTENT_TYPES:
            raise ValueError(f"unsupported content_type: {content_type}")
        object.__setattr__(self, "content_type", content_type)
        object.__setattr__(self, "section_path", _tuple(self.section_path))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        _safe_mapping(self.metadata)


@dataclass(frozen=True, slots=True)
class SourcePacket(SerializableModel):
    packet_id: str
    blocks: tuple[SourceBlock, ...]
    request_fingerprint: str = ""

    def __post_init__(self):
        _text(self.packet_id, "packet_id", 500)
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if not self.blocks:
            raise ValueError("packet requires blocks")

    @property
    def refs(self):
        return tuple(b.ref for b in self.blocks)

    @property
    def text(self):
        return "\n\n".join(b.text for b in self.blocks)


@dataclass(frozen=True, slots=True)
class SourcePlan(SerializableModel):
    generation_id: str
    packets: tuple[SourcePacket, ...]
    request_fingerprint: str
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    source_fingerprints: Mapping[str, str] = None
    policy_versions: Mapping[str, str] = None
    source_root: str = ""

    def __post_init__(self):
        object.__setattr__(self, "packets", tuple(self.packets))
        object.__setattr__(self, "diagnostics", tuple(dict(x) for x in self.diagnostics))
        object.__setattr__(self, "source_fingerprints", dict(self.source_fingerprints or {}))
        object.__setattr__(self, "policy_versions", dict(self.policy_versions or {}))
        object.__setattr__(self, "source_root", str(self.source_root or ""))


@dataclass(frozen=True, slots=True)
class TeacherConfig(SerializableModel):
    provider: str = ""
    model: str = ""
    version: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    schema_version: str = CONTRACT_VERSION
    policy_version: str = DISTILLATION_POLICY_VERSION

    def __post_init__(self):
        if self.max_tokens < 1 or self.timeout_seconds <= 0:
            raise ValueError("invalid teacher bounds")


@dataclass(frozen=True, slots=True)
class KnowledgeExample(SerializableModel):
    example_id: str
    lesson_type: LessonType
    topic: str
    difficulty: Difficulty
    system: str
    user: str
    assistant: str
    source_refs: tuple[SourceRef, ...]
    claims: tuple[GroundingClaim, ...]
    source_type: str = "BOOK_KNOWLEDGE"
    status: CandidateStatus = CandidateStatus.ACCEPTED
    contract_version: str = CONTRACT_VERSION
    policy_versions: Mapping[str, str] = None
    grounding_status: str = "GROUNDED"
    quality_status: str = "ACCEPTED"
    quality_reasons: tuple[str, ...] = ()
    source_fingerprints: Mapping[str, str] = None
    split: str | None = None

    def __post_init__(self):
        _text(self.example_id, "example_id", 500)
        _text(self.topic, "topic", 500)
        _text(self.system, "system", MAX_TEXT_CHARS)
        _text(self.user, "user", MAX_TEXT_CHARS)
        _text(self.assistant, "assistant", MAX_TEXT_CHARS)
        object.__setattr__(self, "lesson_type", LessonType(self.lesson_type))
        object.__setattr__(self, "difficulty", Difficulty(self.difficulty))
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "claims", tuple(self.claims))
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(f"unsupported contract_version: {self.contract_version}")
        object.__setattr__(self, "policy_versions", dict(self.policy_versions or {}))
        object.__setattr__(self, "source_fingerprints", dict(self.source_fingerprints or {}))
        grounding_status = str(self.grounding_status).upper()
        if grounding_status not in {"GROUNDED", "UNVERIFIED", "FAILED"}:
            raise ValueError("unsupported grounding_status")
        object.__setattr__(self, "grounding_status", grounding_status)
        quality_status = str(self.quality_status).upper()
        if quality_status not in {"ACCEPTED", "UNVERIFIED", "EXCLUDED"}:
            raise ValueError("unsupported quality_status")
        object.__setattr__(self, "quality_status", quality_status)
        quality_reasons = tuple(_text(reason, "quality_reason", 200) for reason in self.quality_reasons)
        object.__setattr__(self, "quality_reasons", quality_reasons)
        if self.split is not None:
            split = str(self.split).lower()
            if split not in {"train", "validation", "test"}:
                raise ValueError("unsupported split")
            object.__setattr__(self, "split", split)
        if not all(isinstance(ref, SourceRef) for ref in self.source_refs):
            raise ValueError("source_refs must contain SourceRef values")
        if not all(isinstance(claim, GroundingClaim) for claim in self.claims):
            raise ValueError("claims must contain GroundingClaim values")
        _validate_safe(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeExample:
        payload = dict(value)
        payload.setdefault("system", payload.pop("system_instruction", None))
        payload.setdefault("user", payload.pop("user_instruction", None))
        payload.setdefault("assistant", payload.pop("assistant_target", None))
        payload["source_refs"] = tuple(
            ref if isinstance(ref, SourceRef) else SourceRef.from_dict(ref)
            for ref in payload.get("source_refs", ())
        )
        raw_claims = payload.pop("claims", payload.pop("grounding_claims", ()))
        payload["claims"] = tuple(
            claim
            if isinstance(claim, GroundingClaim)
            else GroundingClaim(
                claim=claim["claim"],
                source_refs=tuple(
                    SourceRef.from_dict(ref) if not isinstance(ref, SourceRef) else ref
                    for ref in claim.get("source_refs", ())
                ),
            )
            for claim in raw_claims
        )
        payload.setdefault("grounding_status", "GROUNDED")
        payload.setdefault("quality_status", "ACCEPTED")
        payload.setdefault("quality_reasons", ())
        return cls(**payload)

    @property
    def system_instruction(self) -> str:
        return self.system

    @property
    def user_instruction(self) -> str:
        return self.user

    @property
    def assistant_target(self) -> str:
        return self.assistant


@dataclass(frozen=True, slots=True)
class DatasetExclusion(SerializableModel):
    reason: ExclusionReason
    diagnostic: str = ""
    packet_id: str | None = None
    candidate_id: str | None = None
    source_refs: tuple[SourceRef, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "reason", ExclusionReason(self.reason))
        object.__setattr__(self, "diagnostic", str(self.diagnostic)[:1000])
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        if not all(isinstance(ref, SourceRef) for ref in self.source_refs):
            raise ValueError("source_refs must contain SourceRef values")


@dataclass(frozen=True, slots=True)
class DistillationManifest(SerializableModel):
    generation_id: str
    phase7_generation_id: str
    counts: Mapping[str, int]
    file_hashes: Mapping[str, str] = None
    contract_version: str = MANIFEST_VERSION

    def __post_init__(self):
        object.__setattr__(self, "counts", dict(self.counts))
        object.__setattr__(self, "file_hashes", dict(self.file_hashes or {}))


@dataclass(frozen=True, slots=True)
class ValidationReport(SerializableModel):
    valid: bool
    errors: tuple[str, ...] = ()
    counts: Mapping[str, int] = None

    def __post_init__(self):
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "counts", dict(self.counts or {}))
