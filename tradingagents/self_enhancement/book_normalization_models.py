"""Immutable, strict research-only V2 rule proofs; no trading authority."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from decimal import Decimal
from enum import Enum
from typing import Any

from .strategy_specs import EvidenceSpan, RuleDirection, RuleOperator, RuleStage

SCHEMA_VERSION = "book-normalization-v2"
GRAMMAR_VERSION = "book-natural-language-v1"
FEATURE_CONTRACT_VERSION = "book-feature-binding-v1"
FAMILIES = frozenset({"MOMENTUM_CONTINUATION", "RANGE_REJECTION", "BREAKOUT", "FAILED_BREAKOUT", "MEAN_REVERSION", "PULLBACK", "TREND_CONTINUATION", "VOLATILITY_EXPANSION", "VOLATILITY_CONTRACTION", "OTHER_SUPPORTED"})


class NormalizationStatus(str, Enum):
    REJECTED = "REJECTED"
    SUPPORTED_NONEXECUTABLE = "SUPPORTED_NONEXECUTABLE"
    EXECUTABLE_ELIGIBLE = "EXECUTABLE_ELIGIBLE"


def _text(value: Any) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError("bounded nonempty text required")


def _int(value: Any, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError("bounded non-boolean integer required")


def _keys(cls: type, data: Any) -> None:
    if type(data) is not dict or set(data) != {f.name for f in fields(cls)}:
        raise ValueError("exact serialized fields required")


def _encode(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


class _Record:
    def to_dict(self) -> dict[str, Any]:
        return {f.name: _encode(getattr(self, f.name)) for f in fields(self)}


def _list(value: Any) -> list:
    if type(value) is not list:
        raise ValueError("JSON array required")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceBundle(_Record):
    spans: tuple[EvidenceSpan, ...]

    def __post_init__(self):
        if type(self.spans) is not tuple or not 1 <= len(self.spans) <= 8:
            raise ValueError("evidence bundle requires one to eight spans")
        if any(not isinstance(s, EvidenceSpan) or len(s.quote) > 600 for s in self.spans):
            raise ValueError("invalid evidence span or character bound")
        if len({(s.generation_id, s.document_id, s.chunk_id, s.source_hash) for s in self.spans}) != 1:
            raise ValueError("evidence must share exact source identity")
        if any(a.end_offset > b.start_offset for a, b in zip(self.spans, self.spans[1:], strict=False)):
            raise ValueError("evidence must be ordered and disjoint")

    @classmethod
    def from_dict(cls, data):
        _keys(cls, data)
        spans = []
        for item in _list(data["spans"]):
            _keys(EvidenceSpan, item)
            spans.append(EvidenceSpan(**item))
        return cls(tuple(spans))


@dataclass(frozen=True, slots=True)
class FieldProof(_Record):
    field: str
    evidence_index: int
    start_offset: int
    end_offset: int

    def __post_init__(self):
        _text(self.field)
        for v in (self.evidence_index, self.start_offset, self.end_offset):
            _int(v)
        if self.end_offset <= self.start_offset:
            raise ValueError("proof must address nonempty source text")

    @classmethod
    def from_dict(cls, data):
        _keys(cls, data)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class NormalizedRuleV2(_Record):
    rule_id: str
    stage: RuleStage
    family: str
    operation: str
    operands: tuple[str, ...]
    direction: RuleDirection
    operator: RuleOperator
    value: Decimal | None
    unit: str | None
    condition: str | None
    horizon_seconds: int | None
    evidence: EvidenceBundle
    field_proofs: tuple[FieldProof, ...]
    pattern_id: str
    schema_version: str
    grammar_version: str
    feature_contract_version: str
    semantic_fingerprint: str

    def __post_init__(self):
        for value in (self.rule_id, self.operation, self.pattern_id):
            _text(value)
        if self.family not in FAMILIES:
            raise ValueError("unknown concept family")
        for value, enum in ((self.stage, RuleStage), (self.direction, RuleDirection), (self.operator, RuleOperator)):
            if not isinstance(value, enum):
                raise TypeError("typed rule enum required")
        if type(self.operands) is not tuple or not 1 <= len(self.operands) <= 8:
            raise ValueError("one to eight typed operands required")
        for operand in self.operands:
            _text(operand)
        if self.value is not None and (not isinstance(self.value, Decimal) or not self.value.is_finite()):
            raise ValueError("finite Decimal required")
        for value in (self.unit, self.condition):
            if value is not None:
                _text(value)
        if self.horizon_seconds is not None:
            _int(self.horizon_seconds, minimum=1)
        if (self.schema_version, self.grammar_version, self.feature_contract_version) != (SCHEMA_VERSION, GRAMMAR_VERSION, FEATURE_CONTRACT_VERSION):
            raise ValueError("unsupported normalization version")
        if not isinstance(self.semantic_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", self.semantic_fingerprint):
            raise ValueError("SHA-256 fingerprint required")
        if not isinstance(self.evidence, EvidenceBundle) or type(self.field_proofs) is not tuple or not self.field_proofs:
            raise ValueError("evidence and field proofs required")
        for proof in self.field_proofs:
            if not isinstance(proof, FieldProof) or proof.evidence_index >= len(self.evidence.spans):
                raise ValueError("unknown proof span")
            span = self.evidence.spans[proof.evidence_index]
            if not span.start_offset <= proof.start_offset < proof.end_offset <= span.end_offset:
                raise ValueError("field proof outside evidence")
        if len({p.field for p in self.field_proofs}) != len(self.field_proofs):
            raise ValueError("duplicate proof fields")

    @classmethod
    def from_dict(cls, data):
        _keys(cls, data)
        d = dict(data)
        for name, enum in (("stage", RuleStage), ("direction", RuleDirection), ("operator", RuleOperator)):
            if type(d[name]) is not str:
                raise ValueError("serialized enum string required")
            d[name] = enum(d[name])
        if d["value"] is not None:
            if type(d["value"]) is not str or len(d["value"]) > 128:
                raise ValueError("Decimal string required")
            d["value"] = Decimal(d["value"])
        d["operands"] = tuple(_list(d["operands"]))
        d["field_proofs"] = tuple(FieldProof.from_dict(p) for p in _list(d["field_proofs"]))
        d["evidence"] = EvidenceBundle.from_dict(d["evidence"])
        return cls(**d)


def normalization_identity(rule: NormalizedRuleV2) -> str:
    d = rule.to_dict()
    d.pop("semantic_fingerprint")
    d.pop("rule_id")
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _result(status, reasons):
    if not isinstance(status, NormalizationStatus) or type(reasons) is not tuple:
        raise ValueError("typed status and tuple reasons required")
    if any(not isinstance(r, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", r) for r in reasons):
        raise ValueError("invalid reason code")


@dataclass(frozen=True, slots=True)
class NormalizationResult(_Record):
    status: NormalizationStatus
    rule: NormalizedRuleV2 | None
    reason_codes: tuple[str, ...]

    def __post_init__(self):
        _result(self.status, self.reason_codes)
        if self.rule is not None and not isinstance(self.rule, NormalizedRuleV2):
            raise ValueError("typed normalized rule required")
        if self.status is not NormalizationStatus.REJECTED and self.rule is None:
            raise ValueError("supported result requires a rule")

    @classmethod
    def from_dict(cls, data):
        _keys(cls, data)
        return cls(NormalizationStatus(data["status"]), NormalizedRuleV2.from_dict(data["rule"]) if data["rule"] is not None else None, tuple(_list(data["reason_codes"])))


@dataclass(frozen=True, slots=True)
class StrategyEnvelopeV2(_Record):
    strategy_id: str
    family: str
    rules: tuple[NormalizedRuleV2, ...]
    schema_version: str

    def __post_init__(self):
        _text(self.strategy_id)
        if self.family not in FAMILIES or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported envelope family or schema")
        if type(self.rules) is not tuple or not 1 <= len(self.rules) <= 64:
            raise ValueError("bounded nonempty rule tuple required")
        if any(not isinstance(r, NormalizedRuleV2) or r.family != self.family for r in self.rules):
            raise ValueError("envelope rule family mismatch")
        if len({r.rule_id for r in self.rules}) != len(self.rules):
            raise ValueError("duplicate envelope rule")

    @classmethod
    def from_dict(cls, data):
        _keys(cls, data)
        return cls(data["strategy_id"], data["family"], tuple(NormalizedRuleV2.from_dict(r) for r in _list(data["rules"])), data["schema_version"])


@dataclass(frozen=True, slots=True)
class CandidateValidationV2(_Record):
    envelope: StrategyEnvelopeV2
    status: NormalizationStatus
    reason_codes: tuple[str, ...]
    missing_stages: tuple[RuleStage, ...]

    def __post_init__(self):
        _result(self.status, self.reason_codes)
        if not isinstance(self.envelope, StrategyEnvelopeV2) or type(self.missing_stages) is not tuple or any(not isinstance(s, RuleStage) for s in self.missing_stages):
            raise ValueError("typed envelope and missing stages required")
        if self.status is NormalizationStatus.EXECUTABLE_ELIGIBLE and self.missing_stages:
            raise ValueError("incomplete candidate cannot be eligible")

    @classmethod
    def from_dict(cls, data):
        _keys(cls, data)
        return cls(StrategyEnvelopeV2.from_dict(data["envelope"]), NormalizationStatus(data["status"]), tuple(_list(data["reason_codes"])), tuple(RuleStage(s) for s in _list(data["missing_stages"])))
