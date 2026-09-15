"""Injectable, offline-safe teacher boundary for knowledge distillation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import TeacherConfig


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
    # Provider adapters are deliberately not instantiated implicitly. A caller
    # must inject one, preventing accidental network/model use during indexing.
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
    return UnconfiguredTeacher(config)
