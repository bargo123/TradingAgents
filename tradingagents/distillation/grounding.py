"""Deterministic source-reference and claim grounding checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationDecision:
    accepted: bool
    reason: str | None = None
    diagnostics: dict[str, Any] | None = None

    @property
    def valid(self) -> bool:
        """Compatibility name used by the factory/other validators."""
        return self.accepted


def _value(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _refs(packet: Any) -> dict[str, Any]:
    result = {}
    for r in _value(packet, "refs", []) or []:
        rid = (
            _value(r, "ref_id", None) or _value(r, "block_id", None) or _value(r, "chunk_id", None)
        )
        if rid:
            result[str(rid)] = r
    return result


def _ref_id(ref: Any) -> str:
    if isinstance(ref, str):
        return ref
    return str(
        _value(ref, "ref_id", None) or _value(ref, "block_id", None) or _value(ref, "chunk_id", "")
    )


def _check_ref(ref: Any, available: dict[str, Any]) -> ValidationDecision | None:
    rid = _ref_id(ref)
    if not rid or rid not in available:
        return ValidationDecision(False, "GROUNDING_FAILED", {"ref_id": rid})
    expected = available[rid]
    # String references are IDs only; structured references must carry the
    # provenance fields that identify the source generation unambiguously.
    if not isinstance(ref, str):
        for field in ("document_id", "source_hash", "generation_id"):
            expected_value = _value(expected, field, None)
            if expected_value is None:
                continue
            ref_value = _value(ref, field, None)
            if ref_value is None:
                return ValidationDecision(
                    False, "SOURCE_PROVENANCE_INCOMPLETE", {"ref_id": rid, "field": field}
                )
            if ref_value != expected_value:
                return ValidationDecision(
                    False, "SOURCE_PROVENANCE_INCOMPLETE", {"ref_id": rid, "field": field}
                )
    return None


class GroundingValidator:
    def validate(self, candidate: Any, packet: Any) -> ValidationDecision:
        available = _refs(packet)
        claims = _value(candidate, "claims", []) or []
        refs = _value(candidate, "source_refs", []) or []
        if not refs:
            refs = [
                ref
                for claim in claims
                for ref in (_value(claim, "refs", _value(claim, "source_refs", [])) or [])
            ]
        if not refs or not claims:
            return ValidationDecision(False, "GROUNDING_FAILED", {"reason": "missing_provenance"})
        for ref in refs:
            failure = _check_ref(ref, available)
            if failure:
                return failure
        for claim in claims:
            refs = _value(claim, "refs", _value(claim, "source_refs", [])) or []
            if not refs:
                return ValidationDecision(False, "GROUNDING_FAILED", {"claim": "missing_ref"})
            for ref in refs:
                failure = _check_ref(ref, available)
                if failure:
                    return failure
        return ValidationDecision(True, diagnostics={"refs_checked": len(available)})
