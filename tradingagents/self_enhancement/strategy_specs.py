"""Strict, provenance-bound contracts for Phase 14B book-derived rules.

These value objects deliberately contain no model prompts, completions, or
reasoning.  A drafted rule is only a claim; it does not become executable
until every required stage is complete and independently validated.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar


class _ValueEnum(str, Enum):
    """Enum that can be read from strict JSON without accepting aliases."""


class RuleStage(_ValueEnum):
    DIRECTION = "DIRECTION"
    ENTRY = "ENTRY"
    CONFIRMATION = "CONFIRMATION"
    INVALIDATION = "INVALIDATION"
    EXPECTED_MOVE = "EXPECTED_MOVE"
    EXIT = "EXIT"
    PROFIT_PROTECTION = "PROFIT_PROTECTION"
    RISK = "RISK"
    STOP_BEHAVIOR = "STOP_BEHAVIOR"
    HORIZON = "HORIZON"
    SESSION_FILTER = "SESSION_FILTER"
    VOLATILITY_FILTER = "VOLATILITY_FILTER"
    SPREAD_FILTER = "SPREAD_FILTER"


class RuleOperator(_ValueEnum):
    UNSPECIFIED = "UNSPECIFIED"
    GREATER_THAN = "GREATER_THAN"
    GREATER_OR_EQUAL = "GREATER_OR_EQUAL"
    LESS_THAN = "LESS_THAN"
    LESS_OR_EQUAL = "LESS_OR_EQUAL"
    EQUALS = "EQUALS"
    CROSSES_ABOVE = "CROSSES_ABOVE"
    CROSSES_BELOW = "CROSSES_BELOW"
    BETWEEN = "BETWEEN"
    IN = "IN"
    NOT_IN = "NOT_IN"


class RuleDirection(_ValueEnum):
    UNSPECIFIED = "UNSPECIFIED"
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


class RuleOrigin(_ValueEnum):
    SOURCE_SUPPORTED_CONCEPT = "SOURCE_SUPPORTED_CONCEPT"
    RESEARCH_HYPOTHESIS_PARAMETER = "RESEARCH_HYPOTHESIS_PARAMETER"


class RuleValidationStatus(_ValueEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_SPECIFICATION = "INSUFFICIENT_SPECIFICATION"
    UNAVAILABLE_DATA = "UNAVAILABLE_DATA"


class StrategySuitability(_ValueEnum):
    HFT_SUITABLE = "HFT_SUITABLE"
    SHORT_TERM_SUITABLE = "SHORT_TERM_SUITABLE"
    INTRADAY_ONLY = "INTRADAY_ONLY"
    SWING_ONLY = "SWING_ONLY"
    UNIMPLEMENTABLE = "UNIMPLEMENTABLE"
    UNIMPLEMENTABLE_AUTONOMOUSLY = "UNIMPLEMENTABLE_AUTONOMOUSLY"
    UNAVAILABLE_DATA = "UNAVAILABLE_DATA"
    INSUFFICIENT_SPECIFICATION = "INSUFFICIENT_SPECIFICATION"


# The minimum source-supported contract needed by the deterministic V1 HFT
# primitive. Other filters (session, volatility, spread, risk) remain explicit
# when absent; global runtime risk/cost gates remain authoritative.
EXECUTABLE_REQUIRED_STAGES = frozenset(
    {
        RuleStage.ENTRY,
        RuleStage.CONFIRMATION,
        RuleStage.INVALIDATION,
        RuleStage.EXPECTED_MOVE,
        RuleStage.EXIT,
        RuleStage.PROFIT_PROTECTION,
        RuleStage.STOP_BEHAVIOR,
        RuleStage.HORIZON,
    }
)


_E = TypeVar("_E", bound=_ValueEnum)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REASON_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_MAX_TEXT = 4096
_MAX_QUOTE = 2048


def _enum(value: Any, enum_type: type[_E], field_name: str) -> _E:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be {enum_type.__name__}")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc


def _text(value: Any, field_name: str, *, maximum: int = _MAX_TEXT) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field_name} must be non-empty and at most {maximum} characters")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in value):
        raise ValueError(f"{field_name} contains a control character")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _knowledge_fingerprint(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("knowledge_fingerprint must be a SHA-256 identity")
    digest = value.removeprefix("sha256:")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError("knowledge_fingerprint must be a SHA-256 identity")
    return value


def _positive_int(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < (0 if allow_zero else 1):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be {comparator}")
    return value


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """Exact quote and source identity supplied by the trusted retrieval layer."""

    generation_id: str
    document_id: str
    chunk_id: str
    source_hash: str
    start_offset: int
    end_offset: int
    quote: str

    def __post_init__(self) -> None:
        for name in ("generation_id", "document_id", "chunk_id"):
            _text(getattr(self, name), name, maximum=256)
        _sha256(self.source_hash, "source_hash")
        _positive_int(self.start_offset, "start_offset", allow_zero=True)
        _positive_int(self.end_offset, "end_offset", allow_zero=True)
        if self.end_offset <= self.start_offset:
            raise ValueError("offset range must have end_offset greater than start_offset")
        _text(self.quote, "quote", maximum=_MAX_QUOTE)
        if self.end_offset - self.start_offset != len(self.quote):
            raise ValueError("quote length must exactly match the offset range")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source_hash": self.source_hash,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "quote": self.quote,
        }


@dataclass(frozen=True, slots=True)
class StrategyRuleClaim:
    """One atomic, typed rule claim; source support and hypotheses never mix."""

    stage: RuleStage
    operator: RuleOperator
    direction: RuleDirection
    value: str | int | float | None
    unit: str | None
    condition: str | None
    horizon_seconds: int | None
    origin: RuleOrigin
    evidence: EvidenceSpan | None

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("stage", RuleStage),
            ("operator", RuleOperator),
            ("direction", RuleDirection),
            ("origin", RuleOrigin),
        ):
            object.__setattr__(self, name, _enum(getattr(self, name), enum_type, name))
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (str, int, float)):
                raise TypeError("value must be a finite string/number or None")
            if isinstance(self.value, str):
                _text(self.value, "value", maximum=512)
            elif isinstance(self.value, float) and not math.isfinite(self.value):
                raise ValueError("value must be finite")
        if self.unit is not None:
            _text(self.unit, "unit", maximum=128)
        if self.condition is not None:
            _text(self.condition, "condition", maximum=1024)
        if self.horizon_seconds is not None:
            _positive_int(self.horizon_seconds, "horizon_seconds")
            if self.stage not in {RuleStage.EXPECTED_MOVE, RuleStage.HORIZON}:
                raise ValueError("horizon_seconds is only valid for expected-move or horizon rules")
        if self.evidence is not None and not isinstance(self.evidence, EvidenceSpan):
            raise TypeError("evidence must be EvidenceSpan or None")
        if self.origin is RuleOrigin.SOURCE_SUPPORTED_CONCEPT and self.evidence is None:
            raise ValueError("source-supported rule requires evidence")
        if self.origin is RuleOrigin.RESEARCH_HYPOTHESIS_PARAMETER and self.evidence is not None:
            raise ValueError("research hypothesis must not be represented as source evidence")

    @property
    def is_complete(self) -> bool:
        required = (
            self.operator is not RuleOperator.UNSPECIFIED,
            self.direction is not RuleDirection.UNSPECIFIED,
            self.value is not None,
            self.unit is not None,
            self.condition is not None,
            self.stage not in {RuleStage.EXPECTED_MOVE, RuleStage.HORIZON}
            or self.horizon_seconds is not None,
        )
        return all(required)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "operator": self.operator.value,
            "direction": self.direction.value,
            "value": self.value,
            "unit": self.unit,
            "condition": self.condition,
            "horizon_seconds": self.horizon_seconds,
            "origin": self.origin.value,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }


@dataclass(frozen=True, slots=True)
class RuleValidationResult:
    rule_fingerprint: str
    status: RuleValidationStatus
    reason_code: str

    def __post_init__(self) -> None:
        _sha256(self.rule_fingerprint, "rule_fingerprint")
        object.__setattr__(
            self,
            "status",
            _enum(self.status, RuleValidationStatus, "status"),
        )
        if not isinstance(self.reason_code, str) or not _REASON_CODE_RE.fullmatch(self.reason_code):
            raise ValueError("reason_code must be a closed uppercase machine code")

    def to_dict(self) -> dict[str, str]:
        return {
            "rule_fingerprint": self.rule_fingerprint,
            "status": self.status.value,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class StrategySpec:
    spec_id: str
    name: str
    family: str
    required_data: tuple[str, ...]
    rule_claims: tuple[StrategyRuleClaim, ...]
    generation_id: str
    knowledge_fingerprint: str
    model_provider: str
    model_id: str
    prompt_version: str
    schema_version: str
    created_at: datetime
    implementation_confidence: float
    suitability: StrategySuitability
    validation_results: tuple[RuleValidationResult, ...] = ()
    _content_hash: str = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        for name, maximum in (
            ("spec_id", 256),
            ("name", 256),
            ("family", 128),
            ("generation_id", 256),
            ("model_provider", 64),
            ("model_id", 256),
            ("prompt_version", 128),
            ("schema_version", 128),
        ):
            _text(getattr(self, name), name, maximum=maximum)
        if self.model_provider != "ollama-local":
            raise ValueError("model_provider must be ollama-local")
        _knowledge_fingerprint(self.knowledge_fingerprint)
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        try:
            utc_created_at = self.created_at.astimezone(UTC)
        except (OverflowError, ValueError) as exc:
            raise ValueError("created_at must have a valid timezone") from exc
        object.__setattr__(self, "created_at", utc_created_at)
        if isinstance(self.implementation_confidence, bool) or not isinstance(
            self.implementation_confidence, (int, float)
        ):
            raise TypeError("implementation_confidence must be a finite number")
        if not math.isfinite(float(self.implementation_confidence)) or not 0 <= self.implementation_confidence <= 1:
            raise ValueError("implementation_confidence must be between 0 and 1")
        object.__setattr__(
            self,
            "suitability",
            _enum(self.suitability, StrategySuitability, "suitability"),
        )
        if isinstance(self.required_data, (str, bytes)) or not isinstance(
            self.required_data, (tuple, list)
        ):
            raise TypeError("required_data must be a sequence of feature names")
        required_data = tuple(_text(item, "required_data item", maximum=128) for item in self.required_data)
        if len(required_data) != len(set(required_data)):
            raise ValueError("required_data must not contain duplicates")
        object.__setattr__(self, "required_data", tuple(sorted(required_data)))
        if isinstance(self.rule_claims, (str, bytes)) or not isinstance(self.rule_claims, (tuple, list)):
            raise TypeError("rule_claims must be a sequence of StrategyRuleClaim")
        claims = tuple(self.rule_claims)
        if any(not isinstance(claim, StrategyRuleClaim) for claim in claims):
            raise ValueError("rule_claims must contain StrategyRuleClaim values")
        if not claims:
            raise ValueError("rule_claims must contain at least one explicit claim")
        object.__setattr__(self, "rule_claims", claims)
        if isinstance(self.validation_results, (str, bytes)) or not isinstance(
            self.validation_results, (tuple, list)
        ):
            raise TypeError("validation_results must be a sequence of RuleValidationResult")
        validations = tuple(self.validation_results)
        if any(not isinstance(result, RuleValidationResult) for result in validations):
            raise ValueError("validation_results must contain RuleValidationResult values")
        if len({result.rule_fingerprint for result in validations}) != len(validations):
            raise ValueError("validation_results must not duplicate rule fingerprints")
        object.__setattr__(
            self,
            "validation_results",
            tuple(sorted(validations, key=lambda result: result.rule_fingerprint)),
        )
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        object.__setattr__(self, "_content_hash", digest)

    @property
    def content_hash(self) -> str:
        return self._content_hash

    @property
    def source_supported_rules(self) -> tuple[StrategyRuleClaim, ...]:
        return tuple(
            claim
            for claim in self.rule_claims
            if claim.origin is RuleOrigin.SOURCE_SUPPORTED_CONCEPT
        )

    @property
    def research_hypothesis_rules(self) -> tuple[StrategyRuleClaim, ...]:
        return tuple(
            claim
            for claim in self.rule_claims
            if claim.origin is RuleOrigin.RESEARCH_HYPOTHESIS_PARAMETER
        )

    @property
    def unspecified_stages(self) -> tuple[RuleStage, ...]:
        complete_source_stages = {
            claim.stage for claim in self.source_supported_rules if claim.is_complete
        }
        return tuple(stage for stage in RuleStage if stage not in complete_source_stages)

    @property
    def missing_executable_stages(self) -> tuple[RuleStage, ...]:
        complete_source_stages = {
            claim.stage for claim in self.source_supported_rules if claim.is_complete
        }
        return tuple(
            stage
            for stage in RuleStage
            if stage in EXECUTABLE_REQUIRED_STAGES and stage not in complete_source_stages
        )

    @property
    def is_executable(self) -> bool:
        if self.suitability is not StrategySuitability.HFT_SUITABLE:
            return False
        if self.research_hypothesis_rules or self.missing_executable_stages:
            return False
        validations = {result.rule_fingerprint: result.status for result in self.validation_results}
        return all(
            validations.get(claim.fingerprint) is RuleValidationStatus.SUPPORTED
            for claim in self.source_supported_rules
        )

    def _content(self) -> dict[str, Any]:
        claim_dicts = [claim.to_dict() for claim in self.rule_claims]
        claim_dicts.sort(key=_canonical_json)
        return {
            "spec_id": self.spec_id,
            "name": self.name,
            "family": self.family,
            "required_data": list(self.required_data),
            "rule_claims": claim_dicts,
            "generation_id": self.generation_id,
            "knowledge_fingerprint": self.knowledge_fingerprint,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "implementation_confidence": self.implementation_confidence,
            "suitability": self.suitability,
            "validation_results": [result.to_dict() for result in self.validation_results],
        }

    def canonical_json(self) -> str:
        return _canonical_json(self._content())

    def to_dict(self) -> dict[str, Any]:
        result = _canonical(self._content())
        result["content_hash"] = self.content_hash
        return result
