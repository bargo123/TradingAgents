"""Typed, bounded failures for Phase 11A."""

from __future__ import annotations


class DistillationError(RuntimeError):
    """Base error for deterministic distillation failures."""


class SourceUnavailableError(DistillationError):
    pass


class SourceGenerationInvalidError(DistillationError):
    pass


class SourcePacketTooLargeError(DistillationError):
    pass
