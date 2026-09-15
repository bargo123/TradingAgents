"""Independent, fail-closed lesson safety and quality policy."""

from __future__ import annotations

import re

from .grounding import ValidationDecision, _value

_ACTION = re.compile(r"\b(BUY|SELL|HOLD)\b", re.I)
_SENSITIVE = (
    "chain_of_thought",
    "reasoning",
    "scratchpad",
    "prompt",
    "completion",
    "api_key",
    "secret",
)


class QualityPolicy:
    def __init__(self, *, max_lesson_chars: int = 12000, max_overlap_ratio: float = 0.80):
        self.max_lesson_chars = max_lesson_chars
        self.max_overlap_ratio = max_overlap_ratio

    def validate(self, candidate, packet) -> ValidationDecision:
        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if str(k).lower() in _SENSITIVE or any(
                        x in str(k).lower() for x in ("future", "outcome", "pnl", "reward")
                    ):
                        return "UNSAFE_FUTURE_OUTCOME_INFERENCE"
                    bad = walk(v)
                    if bad:
                        return bad
            elif isinstance(obj, (list, tuple, set)):
                for v in obj:
                    bad = walk(v)
                    if bad:
                        return bad
            return None

        bad = walk(candidate)
        if bad:
            return ValidationDecision(False, bad)
        text = " ".join(
            str(_value(candidate, k, ""))
            for k in (
                "instruction",
                "question",
                "target",
                "answer",
                "explanation",
                "user",
                "assistant",
            )
        )
        if len(text) > self.max_lesson_chars:
            return ValidationDecision(False, "LESSON_TOO_LONG")
        if _ACTION.search(text):
            return ValidationDecision(False, "UNSAFE_FUTURE_OUTCOME_INFERENCE")
        source_text = " ".join(
            str(getattr(b, "text", b.get("text", "") if isinstance(b, dict) else ""))
            for b in (_value(packet, "blocks", []) or [])
        )
        candidate_tokens = set(re.findall(r"\w+", text.lower()))
        source_tokens = set(re.findall(r"\w+", source_text.lower()))
        if (
            candidate_tokens
            and source_tokens
            and len(candidate_tokens & source_tokens) / len(candidate_tokens)
            > self.max_overlap_ratio
        ):
            return ValidationDecision(False, "VERBATIM_OVERLAP_EXCESSIVE")
        return ValidationDecision(True)
