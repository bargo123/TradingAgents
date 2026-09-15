"""Injectable, offline-safe teacher boundary for knowledge distillation."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping, Protocol


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
    def __init__(self, responses: Any = None, *, provider: str = "fake", model: str = "fixture", error: str | None = None):
        self.responses = responses
        self.provider = provider
        self.model = model
        self.error = error
        self.calls: list[Any] = []

    def generate(self, packet: Any, config: Any) -> TeacherResult:
        self.calls.append(packet)
        if self.error:
            return TeacherResult(provider=self.provider, model=self.model, error_code="TEACHER_FAILED", diagnostics={"error": self.error})
        response = self.responses(packet) if callable(self.responses) else self.responses
        if response is None:
            return TeacherResult(provider=self.provider, model=self.model, error_code="TEACHER_FAILED", diagnostics={"reason": "empty_response"})
        return TeacherResult(candidate=response, provider=self.provider, model=self.model)


class UnconfiguredTeacher:
    def generate(self, packet: Any, config: Any) -> TeacherResult:
        return TeacherResult(error_code="DISTILLATION_TEACHER_NOT_CONFIGURED")


def teacher_from_environment(environ: Mapping[str, str] | None = None) -> Teacher:
    env = os.environ if environ is None else environ
    provider = env.get("PHASE11A_TEACHER_PROVIDER", "").strip()
    if not provider:
        return UnconfiguredTeacher()
    # Provider adapters are deliberately not instantiated implicitly. A caller
    # must inject one, preventing accidental network/model use during indexing.
    return UnconfiguredTeacher()
