"""Injectable, offline-safe teacher boundary for knowledge distillation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Annotated, Any, Protocol

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, ConfigDict, Field, StrictStr, StringConstraints, ValidationError

from tradingagents.llm_clients.openai_client import OllamaChatOpenAI

from .models import Difficulty, GroundingClaim, LessonType, TeacherConfig


@dataclass(frozen=True)
class TeacherResult:
    candidate: Any = None
    provider: str = ""
    model: str = ""
    error_code: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error_code is None and self.candidate is not None


class Teacher(Protocol):
    def generate(self, packet: Any, config: Any) -> TeacherResult: ...


class TeacherClaim(BaseModel):
    """Strict claim/ref payload accepted from the local teacher."""

    model_config = ConfigDict(extra="forbid")

    claim: StrictStr = Field(min_length=1, max_length=4000)
    source_refs: list[StrictStr] = Field(min_length=1, max_length=16)


class TeacherLesson(BaseModel):
    """The only JSON shape an Ollama teacher may return.

    The adapter converts this transport model into the existing factory
    candidate contract.  It intentionally has no free-form metadata fields,
    hidden reasoning fields, trading labels, or outcome fields.
    """

    model_config = ConfigDict(extra="forbid")

    lesson_type: LessonType
    topic: StrictStr = Field(min_length=1, max_length=500)
    difficulty: Difficulty
    system_instruction: StrictStr = Field(min_length=1, max_length=4000)
    user_instruction: StrictStr = Field(min_length=1, max_length=4000)
    assistant_target: StrictStr = Field(min_length=1, max_length=12000)
    source_refs: list[StrictStr] = Field(min_length=1, max_length=16)
    grounding_claims: list[TeacherClaim] = Field(min_length=1, max_length=32)


TEACHER_TRANSPORT_SCHEMA_VERSION = "phase11a-teacher-atom.v1"
_DEFAULT_SYSTEM_INSTRUCTION = "Explain the supplied source evidence only."
_PACKET_ALIAS_RE = re.compile(r"^B[0-9]+$")
_PacketAlias = Annotated[str, StringConstraints(strict=True, pattern=r"^B[0-9]+$")]


class TeacherAtomClaim(BaseModel):
    """Compact claim payload using packet-local aliases only."""

    model_config = ConfigDict(extra="forbid")

    claim: StrictStr = Field(min_length=1, max_length=300)
    refs: list[_PacketAlias] = Field(min_length=1, max_length=4)


class TeacherAtom(BaseModel):
    """Bounded semantic response exchanged with the local teacher.

    This is transport-only.  ``hydrate_teacher_atom`` restores immutable
    Phase 7 references before the normal grounding and quality gates run.
    """

    model_config = ConfigDict(extra="forbid")

    lesson_type: LessonType
    difficulty: Difficulty
    topic: StrictStr = Field(min_length=1, max_length=100)
    question: StrictStr = Field(min_length=1, max_length=250)
    answer: StrictStr = Field(min_length=1, max_length=800)
    claims: list[TeacherAtomClaim] = Field(min_length=1, max_length=3)


@dataclass
class TeacherMetrics:
    calls: int = 0
    schema_failures: int = 0
    grounding_failures: int = 0
    provider_failures: int = 0
    latencies: list[float] = field(default_factory=list)
    call_diagnostics: list[Mapping[str, Any]] = field(default_factory=list)

    @property
    def average_latency_seconds(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def max_latency_seconds(self) -> float:
        return max(self.latencies, default=0.0)

    def to_dict(self) -> dict[str, Any]:
        input_tokens = [
            item.get("input_tokens")
            for item in self.call_diagnostics
            if isinstance(item.get("input_tokens"), int)
        ]
        output_tokens = [
            item.get("output_tokens")
            for item in self.call_diagnostics
            if isinstance(item.get("output_tokens"), int)
        ]
        return {
            "calls": self.calls,
            "schema_failures": self.schema_failures,
            "grounding_failures": self.grounding_failures,
            "provider_failures": self.provider_failures,
            "average_latency_seconds": round(self.average_latency_seconds, 3),
            "max_latency_seconds": round(self.max_latency_seconds, 3),
            "input_tokens": sum(input_tokens) if input_tokens else None,
            "output_tokens": sum(output_tokens) if output_tokens else None,
            "call_diagnostics": [dict(item) for item in self.call_diagnostics],
        }


class _GroundingReferenceError(ValueError):
    pass


class _TeacherAliasError(_GroundingReferenceError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _failure_class(exc: BaseException) -> str:
    name = type(exc).__name__.casefold()
    module = type(exc).__module__.casefold()
    if isinstance(exc, TimeoutError) or "timeout" in name or "timeout" in module:
        return "PROVIDER_TIMEOUT"
    if "jsondecode" in name or ("json" in name and "decode" in name):
        return "JSON_DECODE"
    if "outputparser" in name or "parse" in name:
        return "STRUCTURED_OUTPUT_ERROR"
    if "connection" in name or "connect" in name:
        return "CONNECTION_ERROR"
    if getattr(exc, "status_code", None) is not None or "status" in name or "http" in name:
        return "HTTP_ERROR"
    return "PROVIDER_ERROR"


@dataclass
class _CallObservation:
    provider_status: str = "UNKNOWN"
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    failure_class: str | None = None

    def _update_mapping(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        finish_reason = value.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            self.finish_reason = finish_reason
        usage = value.get("token_usage") or value.get("usage") or value
        if isinstance(usage, Mapping):
            self.input_tokens = self.input_tokens or _safe_int(
                usage.get("prompt_tokens", usage.get("input_tokens"))
            )
            self.output_tokens = self.output_tokens or _safe_int(
                usage.get("completion_tokens", usage.get("output_tokens"))
            )
            self.total_tokens = self.total_tokens or _safe_int(usage.get("total_tokens"))

    def update_from_response(self, response: Any) -> None:
        self.provider_status = "SUCCESS"
        self._update_mapping(getattr(response, "llm_output", None))
        self._update_mapping(getattr(response, "response_metadata", None))
        for generation_group in getattr(response, "generations", ()) or ():
            for generation in generation_group or ():
                self._update_mapping(getattr(generation, "generation_info", None))
                message = getattr(generation, "message", None)
                self._update_mapping(getattr(message, "response_metadata", None))
                self._update_mapping(getattr(message, "usage_metadata", None))

    def as_dict(self) -> dict[str, Any]:
        if self.finish_reason in {"length", "max_tokens"} and self.failure_class is None:
            failure_class = "OUTPUT_TRUNCATED"
        else:
            failure_class = self.failure_class
        return {
            "provider_status": self.provider_status,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "failure_class": failure_class,
        }


class _CallMetadataHandler(BaseCallbackHandler):
    def __init__(self, observation: _CallObservation):
        super().__init__()
        self.observation = observation

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.observation.update_from_response(response)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.observation.provider_status = "ERROR"
        self.observation.failure_class = _failure_class(error)


def _ref_id(ref: Any) -> str:
    if isinstance(ref, str):
        return ref
    if isinstance(ref, Mapping):
        return str(ref.get("chunk_id") or ref.get("block_id") or ref.get("ref_id") or "")
    return str(getattr(ref, "chunk_id", "") or getattr(ref, "block_id", ""))


def _packet_prompt(packet: Any) -> list[dict[str, str]]:
    blocks = list(getattr(packet, "blocks", ()) or ())
    if not blocks:
        raise ValueError("teacher packet has no source blocks")
    evidence: list[str] = []
    for block in blocks:
        ref = getattr(block, "ref", block)
        ref_id = _ref_id(ref)
        if not ref_id:
            raise ValueError("teacher packet has an unidentifiable source reference")
        page = getattr(ref, "page", None)
        section = getattr(ref, "section", None)
        location = ", ".join(
            part for part in (f"page={page}" if page else "", f"section={section}" if section else "") if part
        )
        evidence.append(
            f"REF_ID: {ref_id}\n"
            f"SOURCE: {getattr(ref, 'source_filename', '')}"
            f"{(' (' + location + ')') if location else ''}\n"
            f"CONTENT_TYPE: {getattr(block, 'content_type', 'PROSE')}\n"
            f"TEXT:\n{getattr(block, 'text', '')}"
        )
    system = (
        "You are a local, source-grounded knowledge teacher. Return only the requested "
        "JSON object. Use only the supplied source blocks. Every factual claim must cite "
        "one or more exact REF_ID values from this packet. Do not add outside knowledge, "
        "trading labels, market actions, future outcomes, hidden reasoning, or metadata "
        "fields."
    )
    user = (
        "Create one concise educational lesson from the supplied evidence. The assistant "
        "target must explain the concept or mechanism and state only limitations supported "
        "by the packet. Each grounding_claims item must cite exact packet REF_ID values.\n\n"
        + "\n\n---\n\n".join(evidence)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def packet_aliases(packet: Any) -> dict[str, Any]:
    """Return deterministic packet-local aliases for immutable source refs."""

    blocks_value = getattr(packet, "blocks", None)
    if blocks_value is None and isinstance(packet, Mapping):
        blocks_value = packet.get("blocks", ())
    blocks = list(blocks_value or ())
    if not blocks:
        raise ValueError("teacher packet has no source blocks")
    aliases: dict[str, Any] = {}
    for index, block in enumerate(blocks):
        ref = block.get("ref", block) if isinstance(block, Mapping) else getattr(block, "ref", block)
        if ref is None:
            raise ValueError("teacher packet block has no source reference")
        aliases[f"B{index}"] = ref
    return aliases


def _ref_value(ref: Any, name: str, default: Any = None) -> Any:
    if isinstance(ref, Mapping):
        return ref.get(name, default)
    return getattr(ref, name, default)


def _compact_packet_prompt(packet: Any) -> list[dict[str, str]]:
    """Build a source-grounded prompt whose response cites only ``B*`` aliases."""

    aliases = packet_aliases(packet)
    blocks_value = getattr(packet, "blocks", None)
    if blocks_value is None and isinstance(packet, Mapping):
        blocks_value = packet.get("blocks", ())
    blocks = list(blocks_value or ())
    evidence: list[str] = []
    for index, block in enumerate(blocks):
        ref = aliases[f"B{index}"]
        page = _ref_value(ref, "page")
        page_start = _ref_value(ref, "page_start")
        page_end = _ref_value(ref, "page_end")
        section = _ref_value(ref, "section")
        location_parts = []
        if page is not None:
            location_parts.append(f"page={page}")
        elif page_start is not None or page_end is not None:
            location_parts.append(f"pages={page_start}-{page_end}")
        if section:
            location_parts.append(f"section={section}")
        location = f" ({', '.join(location_parts)})" if location_parts else ""
        evidence.append(
            f"ALIAS: B{index}\n"
            f"SOURCE: {_ref_value(ref, 'source_filename', '')}{location}\n"
            f"CONTENT_TYPE: {_ref_value(block, 'content_type', 'PROSE')}\n"
            f"TEXT:\n{_ref_value(block, 'text', '')}"
        )
    system = (
        "You are a local, source-grounded knowledge teacher. Return only the requested "
        "JSON object matching the schema. Produce exactly one concise educational lesson "
        "with exactly one grounding claim. "
        "Use only the supplied source blocks. Cite exact packet-local aliases such as B0 "
        "in every claim. Use each alias at most once in the whole response; combine "
        "supporting details into one claim instead of repeating an alias. Do not emit "
        "full source references, metadata, trading labels, future outcomes, reasoning, "
        "or essays. Keep the answer concise."
    )
    user = (
        "Create one useful lesson from the supplied evidence. The answer must explain one "
        "concept or mechanism and only limitations supported by the packet. Return exactly "
        "one grounding claim with one or more exact aliases, without repeating any alias.\n\n"
        + "\n\n---\n\n".join(evidence)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def hydrate_teacher_atom(atom: TeacherAtom | Mapping[str, Any], packet: Any) -> dict[str, Any]:
    """Hydrate compact aliases into exact packet refs and final candidate fields."""

    if not isinstance(atom, TeacherAtom):
        atom = TeacherAtom.model_validate(atom)
    aliases = packet_aliases(packet)
    used_aliases: set[str] = set()
    source_refs: list[Any] = []
    claims: list[GroundingClaim] = []
    for atom_claim in atom.claims:
        claim_refs: list[Any] = []
        for alias in atom_claim.refs:
            if not _PACKET_ALIAS_RE.fullmatch(alias):
                raise _TeacherAliasError("ALIAS_MALFORMED")
            if alias in used_aliases:
                raise _TeacherAliasError("ALIAS_DUPLICATE")
            ref = aliases.get(alias)
            if ref is None:
                raise _TeacherAliasError("ALIAS_UNKNOWN")
            used_aliases.add(alias)
            claim_refs.append(ref)
            source_refs.append(ref)
        claims.append(GroundingClaim(atom_claim.claim, tuple(claim_refs)))
    if not source_refs or not claims:
        raise _TeacherAliasError("ALIAS_EMPTY")
    return {
        "lesson_type": atom.lesson_type,
        "topic": atom.topic,
        "difficulty": atom.difficulty,
        "system": _DEFAULT_SYSTEM_INSTRUCTION,
        "user": atom.question,
        "assistant": atom.answer,
        "source_refs": tuple(source_refs),
        "claims": tuple(claims),
    }


def _validate_packet_refs(lesson: TeacherLesson, packet: Any) -> None:
    available = {_ref_id(ref) for ref in (getattr(packet, "refs", ()) or ())}
    if not available or not set(lesson.source_refs).issubset(available):
        raise _GroundingReferenceError("teacher returned a source reference outside the packet")
    for claim in lesson.grounding_claims:
        if not set(claim.source_refs).issubset(available):
            raise _GroundingReferenceError("teacher claim cited a source outside the packet")


def _normalise_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    if not endpoint:
        return "http://localhost:11434/v1"
    return endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"


class OllamaTeacher:
    """Explicit local Ollama teacher using strict JSON-schema output.

    Construction is never performed by planning or validation.  Callers must
    explicitly configure ``PHASE11A_TEACHER_PROVIDER=ollama`` or inject this
    adapter.  Only typed lesson fields and exact packet reference IDs cross the
    boundary; prompts, completions, and private reasoning are never returned in
    diagnostics or persisted by the adapter.
    """

    provider = "ollama"

    def __init__(
        self,
        config: TeacherConfig,
        *,
        endpoint: str = "http://localhost:11434/v1",
        client_factory=None,
        max_retries: int = 0,
    ):
        if config.provider.lower() != "ollama":
            raise ValueError("OllamaTeacher requires provider=ollama")
        if not config.model.strip():
            raise ValueError("OllamaTeacher requires an explicit model")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 1:
            raise ValueError("OllamaTeacher max_retries must be 0 or 1")
        self.config = config
        self.model = config.model
        self.endpoint = _normalise_endpoint(endpoint)
        self.max_retries = max_retries
        self.metrics = TeacherMetrics()
        if client_factory is None:
            self._llm = OllamaChatOpenAI(
                model=config.model,
                api_key="ollama",
                base_url=self.endpoint,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds,
                max_retries=max_retries,
                reasoning_effort="none",
                extra_body={"think": False},
            )
        else:
            self._llm = client_factory(config, self.endpoint)
        self._structured = self._llm.with_structured_output(
            TeacherAtom,
            method="json_schema",
            reasoning_effort="none",
            extra_body={"think": False},
        )

    def generate(self, packet: Any, config: Any) -> TeacherResult:
        started = perf_counter()
        self.metrics.calls += 1
        prompt: list[dict[str, str]] | None = None
        diagnostics: dict[str, Any] = {
            "schema": TeacherAtom.__name__,
            "transport_schema_version": TEACHER_TRANSPORT_SCHEMA_VERSION,
            "response_format": "json_schema",
            "reasoning_effort": "none",
            "retry_count": 0,
        }
        observation = _CallObservation()
        callback = _CallMetadataHandler(observation)
        try:
            prompt = _compact_packet_prompt(packet)
            raw = self._structured.invoke(prompt, config={"callbacks": [callback]})
            if observation.provider_status == "UNKNOWN":
                observation.provider_status = "SUCCESS"
            atom = raw if isinstance(raw, TeacherAtom) else TeacherAtom.model_validate(raw)
            candidate = hydrate_teacher_atom(atom, packet)
            return TeacherResult(
                candidate=candidate,
                provider=self.provider,
                model=self.model,
                diagnostics=diagnostics,
            )
        except _GroundingReferenceError as exc:
            if observation.provider_status == "UNKNOWN":
                observation.provider_status = "SUCCESS"
            observation.failure_class = "GROUNDING_FAILED"
            self.metrics.grounding_failures += 1
            diagnostics["error_type"] = type(exc).__name__
            return TeacherResult(
                provider=self.provider,
                model=self.model,
                error_code="GROUNDING_FAILED",
                diagnostics=diagnostics,
            )
        except (ValidationError, TypeError, ValueError) as exc:
            if observation.provider_status == "UNKNOWN":
                observation.provider_status = "SUCCESS"
            observation.failure_class = observation.failure_class or "SCHEMA_INVALID"
            self.metrics.schema_failures += 1
            diagnostics["error_type"] = type(exc).__name__
            return TeacherResult(
                provider=self.provider,
                model=self.model,
                error_code="SCHEMA_INVALID",
                diagnostics=diagnostics,
            )
        except Exception as exc:  # provider errors are intentionally type-only
            observation.provider_status = "ERROR"
            observation.failure_class = observation.failure_class or _failure_class(exc)
            self.metrics.provider_failures += 1
            diagnostics["error_type"] = type(exc).__name__
            return TeacherResult(
                provider=self.provider,
                model=self.model,
                error_code="TEACHER_FAILED",
                diagnostics=diagnostics,
            )
        finally:
            elapsed = perf_counter() - started
            self.metrics.latencies.append(elapsed)
            diagnostics.update(observation.as_dict())
            diagnostics["elapsed_seconds"] = round(elapsed, 3)
            if prompt is not None:
                diagnostics["input_chars"] = sum(len(item["content"]) for item in prompt)
            self.metrics.call_diagnostics.append(dict(diagnostics))


class FakeTeacher:
    """Deterministic teacher used by tests and offline acceptance."""

    def __init__(
        self,
        responses: Any = None,
        *,
        provider: str = "fake",
        model: str = "fixture",
        error: str | None = None,
        config: TeacherConfig | None = None,
    ):
        self.responses = responses
        self.provider = provider
        self.model = model
        self.error = error
        self.config = config or TeacherConfig(provider=provider, model=model)
        self.calls: list[Any] = []

    def generate(self, packet: Any, config: Any) -> TeacherResult:
        self.calls.append(packet)
        if self.error:
            return TeacherResult(
                provider=self.provider,
                model=self.model,
                error_code="TEACHER_FAILED",
                diagnostics={"error": self.error},
            )
        response = self.responses(packet) if callable(self.responses) else self.responses
        if response is None:
            return TeacherResult(
                provider=self.provider,
                model=self.model,
                error_code="TEACHER_FAILED",
                diagnostics={"reason": "empty_response"},
            )
        return TeacherResult(candidate=response, provider=self.provider, model=self.model)


class UnconfiguredTeacher:
    def __init__(self, config: TeacherConfig | None = None):
        self.config = config or TeacherConfig()

    def generate(self, packet: Any, config: Any) -> TeacherResult:
        return TeacherResult(error_code="DISTILLATION_TEACHER_NOT_CONFIGURED")


def teacher_from_environment(environ: Mapping[str, str] | None = None) -> Teacher:
    env = os.environ if environ is None else environ
    provider = env.get("PHASE11A_TEACHER_PROVIDER", "").strip()
    if not provider:
        return UnconfiguredTeacher()
    try:
        config = TeacherConfig(
            provider=provider,
            model=env.get("PHASE11A_TEACHER_MODEL", "").strip(),
            version=env.get("PHASE11A_TEACHER_VERSION", "").strip(),
            temperature=float(env.get("PHASE11A_TEACHER_TEMPERATURE", "0")),
            max_tokens=int(env.get("PHASE11A_TEACHER_MAX_TOKENS", "1024")),
            timeout_seconds=float(env.get("PHASE11A_TEACHER_TIMEOUT_SECONDS", "60")),
        )
        max_retries = int(env.get("PHASE11A_TEACHER_MAX_RETRIES", "0"))
    except (TypeError, ValueError):
        return UnconfiguredTeacher()
    if provider.lower() != "ollama" or not config.model:
        return UnconfiguredTeacher(config)
    try:
        return OllamaTeacher(
            config,
            endpoint=env.get("PHASE11A_TEACHER_ENDPOINT", "http://localhost:11434/v1"),
            max_retries=max_retries,
        )
    except (TypeError, ValueError):
        return UnconfiguredTeacher(config)


__all__ = [
    "FakeTeacher",
    "OllamaTeacher",
    "TEACHER_TRANSPORT_SCHEMA_VERSION",
    "Teacher",
    "TeacherAtom",
    "TeacherAtomClaim",
    "TeacherClaim",
    "TeacherLesson",
    "TeacherMetrics",
    "TeacherResult",
    "UnconfiguredTeacher",
    "hydrate_teacher_atom",
    "packet_aliases",
    "teacher_from_environment",
]
