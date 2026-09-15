"""Deterministic source-reference and claim grounding checks."""

from __future__ import annotations

import re
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
    packet_refs = _value(packet, "refs", None)
    if packet_refs is None:
        packet_refs = [
            _value(block, "ref", None)
            for block in (_value(packet, "blocks", []) or [])
            if _value(block, "ref", None) is not None
        ]
    for r in packet_refs or []:
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


def _block_texts(packet: Any, available: dict[str, Any]) -> dict[str, str]:
    """Return packet block text keyed by the same IDs used for references."""
    result: dict[str, str] = {}
    for block in _value(packet, "blocks", ()) or ():
        ref = _value(block, "ref", block)
        rid = _ref_id(ref)
        text = _value(block, "text", "")
        if rid and isinstance(text, str):
            result[rid] = text
    # Lightweight packets may expose text directly on refs.
    for rid, ref in available.items():
        text = _value(ref, "text", "")
        if isinstance(text, str) and text:
            result.setdefault(rid, text)
    return result


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {"a", "an", "and", "by", "for", "from", "in", "is", "of", "on", "the", "to", "with"}
_META_WORDS = {"claim", "evidence", "scope", "source", "supplied", "concept", "defines", "defined"}
_OPPOSITES = (
    ({"increase", "increases", "increased", "increasing", "higher", "more"}, {"decrease", "decreases", "decreased", "decreasing", "lower", "less"}),
    ({"available", "present", "exists"}, {"unavailable", "absent", "missing"}),
)


def _words(value: str) -> set[str]:
    return {word for word in _WORD_RE.findall(value.lower()) if word not in _STOP_WORDS}


def _contradicts(left: str, right: str) -> bool:
    negations = {"not", "never", "no", "without", "cannot", "can't", "doesn't", "isn't"}
    l_words, r_words = _words(left), _words(right)
    l_neg = bool(l_words & negations)
    r_neg = bool(r_words & negations)
    l_words -= negations
    r_words -= negations
    shared = l_words & r_words
    negated = l_neg != r_neg and len(shared) >= 2 and len(shared) / max(1, min(len(l_words), len(r_words))) >= 0.7
    directional = any(
        bool(l_words & left) and bool(r_words & right) or bool(l_words & right) and bool(r_words & left)
        for left, right in _OPPOSITES
    ) and len(shared) >= 1
    return negated or directional


def _check_ref(ref: Any, available: dict[str, Any]) -> ValidationDecision | None:
    rid = _ref_id(ref)
    if not rid or rid not in available:
        return ValidationDecision(False, "GROUNDING_FAILED", {"ref_id": rid})
    expected = available[rid]
    # String references are IDs only; structured references must carry the
    # provenance fields that identify the source generation unambiguously.
    if not isinstance(ref, str):
        for field in (
            "document_id",
            "source_filename",
            "source_hash",
            "chunk_id",
            "generation_id",
            "page",
            "page_start",
            "page_end",
            "chapter",
            "section",
            "content_type",
        ):
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
        block_texts = _block_texts(packet, available)
        claims = _value(candidate, "claims", []) or []
        declared_refs = _value(candidate, "source_refs", None)
        refs = declared_refs or []
        # Older lightweight validator fixtures use a claims-only namespace.
        # Canonical Phase 11A examples always carry source_refs explicitly.
        if not refs and (declared_refs is not None or isinstance(candidate, dict)):
            return ValidationDecision(False, "GROUNDING_FAILED", {"reason": "missing_provenance"})
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
        claim_texts: list[str] = []
        for claim in claims:
            refs = _value(claim, "refs", _value(claim, "source_refs", [])) or []
            if not refs:
                return ValidationDecision(False, "GROUNDING_FAILED", {"claim": "missing_ref"})
            for ref in refs:
                failure = _check_ref(ref, available)
                if failure:
                    return failure
            claim_text = _value(claim, "claim", _value(claim, "text", ""))
            if not isinstance(claim_text, str) or not claim_text.strip():
                if block_texts:
                    return ValidationDecision(False, "UNSUPPORTED_CLAIM", {"claim": "missing_text"})
                continue
            claim_texts.append(claim_text)
            if block_texts:
                claim_words = _words(claim_text)
                supported = any(
                    len(claim_words & _words(block_texts.get(_ref_id(ref), ""))) >= 2
                    for ref in refs
                )
                if not supported and not (claim_words & _META_WORDS):
                    return ValidationDecision(False, "UNSUPPORTED_CLAIM", {"claim": claim_text[:120]})
        for index, left in enumerate(claim_texts):
            for right in claim_texts[index + 1 :]:
                if _contradicts(left, right):
                    return ValidationDecision(False, "CONTRADICTION", {"claims": [left[:120], right[:120]]})
        return ValidationDecision(True, diagnostics={"refs_checked": len(available)})
