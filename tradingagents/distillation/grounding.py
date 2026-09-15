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


class GroundingValidator:
    def validate(self, candidate: Any, packet: Any) -> ValidationDecision:
        available = _refs(packet)
        claims = _value(candidate, "claims", []) or []
        for claim in claims:
            refs = _value(claim, "refs", _value(claim, "source_refs", [])) or []
            if not refs:
                return ValidationDecision(False, "GROUNDING_FAILED", {"claim": "missing_ref"})
            for ref in refs:
                rid = _ref_id(ref)
                if rid not in available:
                    return ValidationDecision(False, "GROUNDING_FAILED", {"ref_id": rid})
        refs = _value(candidate, "source_refs", []) or []
        if refs and any(_ref_id(r) not in available for r in refs):
            return ValidationDecision(False, "GROUNDING_FAILED", {"reason": "unknown_source_ref"})
        return ValidationDecision(True, diagnostics={"refs_checked": len(available)})
