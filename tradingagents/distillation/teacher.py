"""Injectable, offline-safe teacher boundary for knowledge distillation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from tradingagents.llm_clients.openai_client import OllamaChatOpenAI

from .models import Difficulty, LessonType, TeacherConfig


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


@dataclass
class TeacherMetrics:
    calls: int = 0
    schema_failures: int = 0
    grounding_failures: int = 0
    provider_failures: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def average_latency_seconds(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def max_latency_seconds(self) -> float:
        return max(self.latencies, default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "schema_failures": self.schema_failures,
            "grounding_failures": self.grounding_failures,
            "provider_failures": self.provider_failures,
            "average_latency_seconds": round(self.average_latency_seconds, 3),
            "max_latency_seconds": round(self.max_latency_seconds, 3),
        }


class _GroundingReferenceError(ValueError):
    pass


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

    def __init__(self, config: TeacherConfig, *, endpoint: str = "http://localhost:11434/v1", client_factory=None):
        if config.provider.lower() != "ollama":
            raise ValueError("OllamaTeacher requires provider=ollama")
        if not config.model.strip():
            raise ValueError("OllamaTeacher requires an explicit model")
        self.config = config
        self.model = config.model
        self.endpoint = _normalise_endpoint(endpoint)
        self.metrics = TeacherMetrics()
        if client_factory is None:
            self._llm = OllamaChatOpenAI(
                model=config.model,
                api_key="ollama",
                base_url=self.endpoint,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds,
                reasoning_effort="none",
                extra_body={"think": False},
            )
        else:
            self._llm = client_factory(config, self.endpoint)
        self._structured = self._llm.with_structured_output(
            TeacherLesson,
            method="json_schema",
            reasoning_effort="none",
            extra_body={"think": False},
        )

    def generate(self, packet: Any, config: Any) -> TeacherResult:
        started = perf_counter()
        self.metrics.calls += 1
        prompt: list[dict[str, str]] | None = None
        diagnostics: dict[str, Any] = {
            "schema": TeacherLesson.__name__,
            "response_format": "json_schema",
            "reasoning_effort": "none",
        }
        try:
            prompt = _packet_prompt(packet)
            raw = self._structured.invoke(prompt)
            lesson = raw if isinstance(raw, TeacherLesson) else TeacherLesson.model_validate(raw)
            _validate_packet_refs(lesson, packet)
            candidate = lesson.model_dump(mode="json")
            candidate["source_refs"] = list(lesson.source_refs)
            candidate["grounding_claims"] = [claim.model_dump(mode="json") for claim in lesson.grounding_claims]
            return TeacherResult(
                candidate=candidate,
                provider=self.provider,
                model=self.model,
                diagnostics=diagnostics,
            )
        except _GroundingReferenceError as exc:
            self.metrics.grounding_failures += 1
            return TeacherResult(
                provider=self.provider,
                model=self.model,
                error_code="GROUNDING_FAILED",
                diagnostics={**diagnostics, "error_type": type(exc).__name__},
            )
        except (ValidationError, TypeError, ValueError) as exc:
            self.metrics.schema_failures += 1
            return TeacherResult(
                provider=self.provider,
                model=self.model,
                error_code="SCHEMA_INVALID",
                diagnostics={**diagnostics, "error_type": type(exc).__name__},
            )
        except Exception as exc:  # provider errors are intentionally type-only
            self.metrics.provider_failures += 1
            return TeacherResult(
                provider=self.provider,
                model=self.model,
                error_code="TEACHER_FAILED",
                diagnostics={**diagnostics, "error_type": type(exc).__name__},
            )
        finally:
            elapsed = perf_counter() - started
            self.metrics.latencies.append(elapsed)
            diagnostics["elapsed_seconds"] = round(elapsed, 3)
            if prompt is not None:
                diagnostics["input_chars"] = sum(len(item["content"]) for item in prompt)


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
    except (TypeError, ValueError):
        return UnconfiguredTeacher()
    if provider.lower() != "ollama" or not config.model:
        return UnconfiguredTeacher(config)
    try:
        return OllamaTeacher(
            config,
            endpoint=env.get("PHASE11A_TEACHER_ENDPOINT", "http://localhost:11434/v1"),
        )
    except (TypeError, ValueError):
        return UnconfiguredTeacher(config)


__all__ = [
    "FakeTeacher",
    "OllamaTeacher",
    "Teacher",
    "TeacherClaim",
    "TeacherLesson",
    "TeacherMetrics",
    "TeacherResult",
    "UnconfiguredTeacher",
    "teacher_from_environment",
]
