"""Explicit, local-only Ollama boundary for Phase 14B StrategySpec drafts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from tradingagents.self_enhancement.book_atomic_extraction import (
    ATOMIC_PROMPT_VERSION,
    ATOMIC_SCHEMA_VERSION,
    AtomicRuleBatch,
    AtomicStrategyExtractor,
    ConceptGrouping,
    ModelCallTelemetry,
    PresenceDecision,
    prepare_evidence_sentences,
    resolve_evidence_id,
)
from tradingagents.self_enhancement.strategy_specs import (
    EXECUTABLE_REQUIRED_STAGES,
    RuleDirection,
    RuleOperator,
    RuleOrigin,
    RuleStage,
)

DRAFT_SCHEMA_VERSION = ATOMIC_SCHEMA_VERSION
DRAFT_PROMPT_VERSION = ATOMIC_PROMPT_VERSION
_MAX_TIMEOUT_SECONDS = 600.0
_MAX_OUTPUT_TOKENS = 8192
_MAX_CONTEXT_TOKENS = 32768
_MIN_CONTEXT_TOKENS = 512
_MAX_SPECS = 10
_MAX_HITS = 20
_MAX_HIT_TEXT_CHARS = 8000
_MAX_TOTAL_SOURCE_CHARS = 80000
_EVIDENCE_ALIAS_RE = re.compile(r"^E[1-9][0-9]{0,2}$")


def _draft_numeric_value(value: Decimal) -> int | float | None:
    """Return an exactly representable JSON number for a parsed source decimal."""

    if not isinstance(value, Decimal) or not value.is_finite():
        return None
    if value == value.to_integral_value():
        integer = int(value)
        try:
            return integer if math.isfinite(float(integer)) else None
        except OverflowError:
            return None
    number = float(value)
    if not math.isfinite(number) or Decimal(str(number)) != value:
        return None
    return number


class _StrictDraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class StrategyDraftClaim(_StrictDraftModel):
    """Untrusted atomic claim with a packet-local evidence alias only."""

    stage: RuleStage
    operator: RuleOperator
    direction: RuleDirection
    value: StrictStr | StrictInt | StrictFloat | None
    unit: StrictStr | None
    condition: StrictStr | None
    horizon_seconds: StrictInt | None = Field(default=None, ge=1)
    origin: RuleOrigin
    evidence_ref: StrictStr | None
    start_offset: StrictInt | None
    end_offset: StrictInt | None
    quote: StrictStr | None

    @model_validator(mode="after")
    def validate_evidence_binding(self) -> StrategyDraftClaim:
        if self.horizon_seconds is not None and self.stage not in {
            RuleStage.EXPECTED_MOVE,
            RuleStage.HORIZON,
        }:
            raise ValueError("horizon_seconds is only valid for expected-move or horizon rules")
        evidence_fields = (self.evidence_ref, self.start_offset, self.end_offset, self.quote)
        if self.origin is RuleOrigin.SOURCE_SUPPORTED_CONCEPT:
            if any(value is None for value in evidence_fields):
                raise ValueError("source-supported claims require an exact evidence span")
            assert self.evidence_ref is not None
            assert self.start_offset is not None
            assert self.end_offset is not None
            assert self.quote is not None
            if not _EVIDENCE_ALIAS_RE.fullmatch(self.evidence_ref):
                raise ValueError("evidence_ref must be a supplied evidence alias")
            if self.start_offset < 0 or self.end_offset <= self.start_offset:
                raise ValueError("evidence offsets must be non-negative and ordered")
            if self.end_offset - self.start_offset != len(self.quote):
                raise ValueError("quote length must exactly match the character offsets")
            if not self.quote.strip():
                raise ValueError("quote must not be blank")
        elif any(value is not None for value in evidence_fields):
            raise ValueError("research hypotheses cannot claim source evidence")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("value must be finite")
        return self


class StrategyDraft(_StrictDraftModel):
    """Model-proposed content only; runtime provenance is attached later."""

    name: StrictStr = Field(min_length=1, max_length=256)
    family: StrictStr = Field(min_length=1, max_length=128)
    required_data: list[StrictStr] = Field(max_length=32)
    implementation_confidence: StrictFloat = Field(ge=0.0, le=1.0)
    rule_claims: list[StrategyDraftClaim] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_names(self) -> StrategyDraft:
        if not self.name.strip() or not self.family.strip():
            raise ValueError("name and family must not be blank")
        if any(not value.strip() for value in self.required_data):
            raise ValueError("required_data entries must not be blank")
        if len(set(self.required_data)) != len(self.required_data):
            raise ValueError("required_data must not contain duplicates")
        return self


class StrategyDraftBatch(_StrictDraftModel):
    specs: list[StrategyDraft] = Field(max_length=_MAX_SPECS)


@dataclass(frozen=True, slots=True)
class DraftResult:
    """Persist-safe output: validated draft and scalar diagnostics only."""

    specs: tuple[StrategyDraft, ...]
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    timeout_seconds: float
    max_output_tokens: int
    context_tokens: int
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    draft_digest: str | None
    error_code: str | None = None
    error_type: str | None = None
    llm_calls: int = 0
    cache_hits: int = 0
    call_telemetry: tuple[ModelCallTelemetry, ...] = ()
    evidence_sentence_count: int = 0
    evidence_groups_processed: int = 0
    actionable_evidence_groups: int = 0
    actionable_concept_count: int = 0
    atomic_rule_count: int = 0
    unsupported_rule_count: int = 0
    provenance_failure_count: int = 0
    schema_failure_count: int = 0
    truncation_count: int = 0
    provider_failure_count: int = 0
    retry_count: int = 0
    insufficient_specification_count: int = 0
    cache_write_failures: int = 0
    model_version: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "specs": [spec.model_dump(mode="json") for spec in self.specs],
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": self.max_output_tokens,
            "context_tokens": self.context_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "draft_digest": self.draft_digest,
            "error_code": self.error_code,
            "error_type": self.error_type,
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "call_telemetry": [call.to_dict() for call in self.call_telemetry],
            "evidence_sentence_count": self.evidence_sentence_count,
            "evidence_groups_processed": self.evidence_groups_processed,
            "actionable_evidence_groups": self.actionable_evidence_groups,
            "actionable_concept_count": self.actionable_concept_count,
            "atomic_rule_count": self.atomic_rule_count,
            "unsupported_rule_count": self.unsupported_rule_count,
            "provenance_failure_count": self.provenance_failure_count,
            "schema_failure_count": self.schema_failure_count,
            "truncation_count": self.truncation_count,
            "provider_failure_count": self.provider_failure_count,
            "retry_count": self.retry_count,
            "insufficient_specification_count": self.insufficient_specification_count,
            "cache_write_failures": self.cache_write_failures,
        }


def _loopback_endpoint(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("endpoint must be a loopback Ollama URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint must be a loopback Ollama URL") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("endpoint must be a loopback Ollama URL")
    return f"http://{parsed.netloc}"


class OllamaStrategyDrafter:
    """Call only the local native Ollama endpoint, when explicitly invoked."""

    provider = "ollama-local"

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout_seconds: float = 120.0,
        max_output_tokens: int = 2048,
        context_tokens: int = 8192,
        model_version: str | None = None,
        cache_path: str | Path | None = None,
        transport: Any = None,
    ) -> None:
        self.endpoint = _loopback_endpoint(endpoint)
        if not isinstance(model, str) or not model or model.strip() != model:
            raise ValueError("model must be an explicit local Ollama model identifier")
        if len(model) > 256:
            raise ValueError("model identifier is too long")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise TypeError("timeout_seconds must be numeric")
        if not math.isfinite(float(timeout_seconds)) or not 0 < timeout_seconds <= _MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout_seconds must be between 0 and {_MAX_TIMEOUT_SECONDS}")
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= _MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_output_tokens must be between 1 and {_MAX_OUTPUT_TOKENS}")
        if type(context_tokens) is not int or not _MIN_CONTEXT_TOKENS <= context_tokens <= _MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"context_tokens must be between {_MIN_CONTEXT_TOKENS} and {_MAX_CONTEXT_TOKENS}"
            )
        self.model = model
        if model_version is not None and (
            not isinstance(model_version, str)
            or not model_version
            or model_version.strip() != model_version
            or len(model_version) > 512
        ):
            raise ValueError("model_version must be a bounded non-empty identifier")
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_tokens = max_output_tokens
        self.context_tokens = context_tokens
        self.model_version = model_version or model
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._transport = transport

    def _extract_atomic(self, hits: Sequence[Any], transport: Any) -> Any:
        extractor = AtomicStrategyExtractor(
            self.endpoint,
            self.model,
            timeout_seconds=self.timeout_seconds,
            max_output_tokens=self.max_output_tokens,
            context_tokens=self.context_tokens,
            model_version=self.model_version,
            cache_path=self.cache_path,
            transport=transport,
        )
        return extractor.extract(hits)

    def draft(self, hits: Sequence[Any], max_specs: int = 5) -> DraftResult:
        if type(max_specs) is not int or not 1 <= max_specs <= _MAX_SPECS:
            raise ValueError(f"max_specs must be between 1 and {_MAX_SPECS}")
        if isinstance(hits, (str, bytes)) or not isinstance(hits, (tuple, list)):
            raise TypeError("hits must be a sequence of retrieved knowledge hits")
        if not 1 <= len(hits) <= _MAX_HITS:
            raise ValueError(f"hits must contain between 1 and {_MAX_HITS} items")

        started = time.perf_counter()
        try:
            evidence = prepare_evidence_sentences(hits)
            if self._transport is None:
                # Ignore HTTP_PROXY/HTTPS_PROXY and related environment values:
                # all evidence must stay on the explicitly approved loopback.
                with httpx.Client(trust_env=False) as local_transport:
                    extracted = self._extract_atomic(hits, local_transport)
            else:
                extracted = self._extract_atomic(hits, self._transport)
            hit_order: dict[tuple[str, str, str], int] = {}
            for index, hit in enumerate(hits, start=1):
                hit_order.setdefault(
                    (
                        str(getattr(hit, "document_id", "")),
                        str(getattr(hit, "chunk_id", "")),
                        str(getattr(hit, "source_hash", "")),
                    ),
                    index,
                )
            evidence_by_id = {item.evidence_id: item for item in evidence}
            drafts: list[StrategyDraft] = []
            numeric_conversion_failures = 0
            for concept in extracted.concepts:
                if len(drafts) >= max_specs:
                    break
                claims: list[StrategyDraftClaim] = []
                required_data: set[str] = set()
                seen_rules: set[tuple[RuleStage, str]] = set()
                for rule in concept.rules:
                    source = resolve_evidence_id(rule.evidence.evidence_id, evidence_by_id)
                    if source is None:
                        continue
                    key = (rule.stage, source.evidence_id)
                    if key in seen_rules:
                        continue
                    seen_rules.add(key)
                    parsed = rule.parsed
                    source_key = (
                        str(getattr(source.hit, "document_id", "")),
                        str(getattr(source.hit, "chunk_id", "")),
                        str(getattr(source.hit, "source_hash", "")),
                    )
                    alias_index = hit_order.get(source_key)
                    if alias_index is None:
                        continue
                    feature = parsed.get("feature")
                    if isinstance(feature, str):
                        required_data.add(feature)
                    value = _draft_numeric_value(parsed["value"])
                    if value is None:
                        numeric_conversion_failures += 1
                        continue
                    claims.append(
                        StrategyDraftClaim(
                            stage=rule.stage,
                            operator=parsed["operator"],
                            direction=parsed["direction"],
                            value=value,
                            unit=parsed["unit"],
                            condition=parsed["condition"],
                            horizon_seconds=parsed["horizon_seconds"],
                            origin=RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
                            evidence_ref=f"E{alias_index}",
                            start_offset=source.start_offset,
                            end_offset=source.end_offset,
                            quote=source.text,
                        )
                    )
                if not claims:
                    continue
                family = concept.family.value
                drafts.append(
                    StrategyDraft(
                        name=f"Book-derived {family.replace('_', ' ').title()}",
                        family=family,
                        required_data=sorted(required_data),
                        # Confidence is deliberately conservative and does not
                        # turn model confidence into a trading-quality claim.
                        implementation_confidence=0.0,
                        rule_claims=claims,
                    )
                )

            canonical = json.dumps(
                [draft.model_dump(mode="json") for draft in drafts],
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            actual_calls = tuple(call for call in extracted.call_telemetry if not call.cache_hit)
            input_tokens = sum(call.input_tokens or 0 for call in actual_calls)
            output_tokens = sum(call.output_tokens or 0 for call in actual_calls)
            finish_reasons = [call.finish_reason for call in actual_calls if call.finish_reason]
            error_code = None
            error_type = None
            if actual_calls and not any(call.outcome == "SUCCESS" for call in actual_calls):
                outcomes = {call.outcome for call in actual_calls}
                if outcomes <= {"SCHEMA_INVALID", "TRUNCATED", "PROVENANCE_INVALID"}:
                    error_code = "SCHEMA_INVALID"
                elif outcomes == {"PROVIDER_ERROR"}:
                    error_code = "PROVIDER_ERROR"
                else:
                    error_code = "EXTRACTION_FAILED"
                error_type = next((call.error_type for call in actual_calls if call.error_type), None)
            insufficient_count = sum(
                not EXECUTABLE_REQUIRED_STAGES.issubset({claim.stage for claim in draft.rule_claims})
                for draft in drafts
            )
            return DraftResult(
                specs=tuple(drafts),
                provider=self.provider,
                model=self.model,
                prompt_version=DRAFT_PROMPT_VERSION,
                schema_version=DRAFT_SCHEMA_VERSION,
                timeout_seconds=self.timeout_seconds,
                max_output_tokens=self.max_output_tokens,
                context_tokens=self.context_tokens,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reasons[-1] if finish_reasons else None,
                draft_digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                error_code=error_code,
                error_type=error_type,
                llm_calls=extracted.llm_calls,
                cache_hits=extracted.cache_hits,
                call_telemetry=extracted.call_telemetry,
                evidence_sentence_count=extracted.evidence_sentence_count,
                evidence_groups_processed=extracted.evidence_groups_processed,
                actionable_evidence_groups=extracted.actionable_group_count,
                actionable_concept_count=extracted.concept_count,
                atomic_rule_count=extracted.atomic_rule_count,
                unsupported_rule_count=(
                    extracted.unsupported_rule_count + numeric_conversion_failures
                ),
                provenance_failure_count=extracted.provenance_failure_count,
                schema_failure_count=extracted.schema_failure_count,
                truncation_count=extracted.truncation_count,
                provider_failure_count=extracted.provider_failure_count,
                retry_count=extracted.retry_count,
                insufficient_specification_count=insufficient_count,
                cache_write_failures=extracted.cache_write_failures,
                model_version=self.model_version,
            )
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return DraftResult(
                specs=(),
                provider=self.provider,
                model=self.model,
                prompt_version=DRAFT_PROMPT_VERSION,
                schema_version=DRAFT_SCHEMA_VERSION,
                timeout_seconds=self.timeout_seconds,
                max_output_tokens=self.max_output_tokens,
                context_tokens=self.context_tokens,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                input_tokens=None,
                output_tokens=None,
                finish_reason=None,
                draft_digest=None,
                error_code="EXTRACTION_FAILED",
                error_type=type(exc).__name__,
                model_version=self.model_version,
            )
        except Exception as exc:  # Provider messages and model text remain private.
            return DraftResult(
                specs=(),
                provider=self.provider,
                model=self.model,
                prompt_version=DRAFT_PROMPT_VERSION,
                schema_version=DRAFT_SCHEMA_VERSION,
                timeout_seconds=self.timeout_seconds,
                max_output_tokens=self.max_output_tokens,
                context_tokens=self.context_tokens,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                input_tokens=None,
                output_tokens=None,
                finish_reason=None,
                draft_digest=None,
                error_code="PROVIDER_ERROR",
                error_type=type(exc).__name__,
                model_version=self.model_version,
            )


__all__ = [
    "ATOMIC_PROMPT_VERSION",
    "ATOMIC_SCHEMA_VERSION",
    "AtomicRuleBatch",
    "ConceptGrouping",
    "DRAFT_PROMPT_VERSION",
    "DRAFT_SCHEMA_VERSION",
    "DraftResult",
    "OllamaStrategyDrafter",
    "PresenceDecision",
    "StrategyDraft",
    "StrategyDraftBatch",
    "StrategyDraftClaim",
    "prepare_evidence_sentences",
    "resolve_evidence_id",
]
