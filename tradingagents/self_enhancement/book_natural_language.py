"""Finite source-grounded normalization. No model-generated semantics or orders."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal

from .book_normalization_models import (
    FEATURE_CONTRACT_VERSION,
    GRAMMAR_VERSION,
    SCHEMA_VERSION,
    EvidenceBundle,
    FieldProof,
    NormalizationResult,
    NormalizationStatus,
    NormalizedRuleV2,
    normalization_identity,
)
from .strategy_specs import EvidenceSpan, RuleDirection, RuleOperator, RuleStage

# The exact constructions have genuine corpus cases, not invented DSL examples.
# Comparison and holding-time families stay disabled without positive source cases.
DISABLED_PATTERNS = {"numeric-comparison-v1": "PATTERN_NO_GENUINE_CASE", "holding-limit-v1": "PATTERN_NO_GENUINE_CASE"}
_VALUE = r"(?P<value>[1-9][0-9]*(?:\.[0-9]+)?)"
_TARGET = re.compile(
    r"The (?P<stage>target) on each trade is a non-adjustable one and "
    r"(?P<operator>set to) " + _VALUE + r" (?P<unit>pip|pips) of (?P<operand>profit)\.", re.I,
)
_STOP = re.compile(
    r"Place an initial protective (?P<stage>stop) (?P<operator>no more than) "
    + _VALUE + r" (?P<unit>pip|pips) (?P<side>below) the (?P<operand>entry)\.?", re.I,
)


def _view(text: str) -> tuple[str, tuple[int, ...]]:
    """Whitespace-only normalization with original character addresses."""
    chars, offsets = [], []
    for i, ch in enumerate(text):
        if ch.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                offsets.append(i)
        else:
            chars.append(ch)
            offsets.append(i)
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), tuple(offsets)


def _rejected(reason: str) -> NormalizationResult:
    return NormalizationResult(NormalizationStatus.REJECTED, None, (reason,))


def normalize_rule(bundle: EvidenceBundle, *, stage: RuleStage, family: str,
                   source_lookup: Callable[[EvidenceSpan], str]) -> NormalizationResult:
    if not isinstance(bundle, EvidenceBundle) or not isinstance(stage, RuleStage):
        return _rejected("INVALID_RULE_INPUT")
    texts = []
    for span in bundle.spans:
        try:
            text = source_lookup(span)
        except (ValueError, KeyError, OSError):
            return _rejected("PROVENANCE_INVALID")
        if not isinstance(text, str) or text[span.start_offset:span.end_offset] != span.quote:
            return _rejected("SPAN_MISMATCH")
        texts.append(text)
    if any(t != texts[0] for t in texts):
        return _rejected("SOURCE_TEXT_CONFLICT")
    if any(a.end_offset < b.start_offset and texts[0][a.end_offset:b.start_offset].strip() for a, b in zip(bundle.spans, bundle.spans[1:], strict=False)):
        return _rejected("NONCONTIGUOUS_CONTEXT")
    # No context-defining pattern has been reviewed yet. Never ignore extra clauses.
    if len(bundle.spans) != 1:
        return _rejected("CONTEXT_PATTERN_UNSUPPORTED")
    source = bundle.spans[0]
    if any(ord(c) < 32 and c not in "\t\n\r" for c in source.quote):
        return _rejected("DAMAGED_SOURCE_TEXT")
    quote, offsets = _view(source.quote)
    match = _TARGET.fullmatch(quote)
    expected_stage, operator = RuleStage.EXPECTED_MOVE, RuleOperator.EQUALS
    reasons = ["UNSUPPORTED_PRIMITIVE", "SOURCE_UNIT_MISMATCH", "MISSING_DIRECTION", "MISSING_HORIZON"]
    if match is None:
        match = _STOP.fullmatch(quote)
        expected_stage, operator = RuleStage.STOP_BEHAVIOR, RuleOperator.LESS_OR_EQUAL
        reasons = ["UNSUPPORTED_PRIMITIVE", "SOURCE_UNIT_MISMATCH", "MISSING_DIRECTION"]
    if match is None:
        return _rejected("GRAMMAR_UNSUPPORTED")
    if stage is not expected_stage:
        return _rejected("STAGE_MISMATCH")
    if family != "OTHER_SUPPORTED":
        return _rejected("FAMILY_NOT_ESTABLISHED")
    # Nearby text can disqualify an isolated quoted example, never supply fields.
    prefix = texts[0][max(0, source.start_offset - 160):source.start_offset]
    if re.search(r"\b(example|hypothetical|criticiz\w*|incorrect|mistaken|do not use)\b[^.!?]*$", prefix, re.I):
        return _rejected("CONTEXT_NOT_INSTRUCTION")
    proofs = []
    for field, group in (("stage", "stage"), ("operation", "stage"), ("operator", "operator"), ("value", "value"), ("unit", "unit"), ("operands", "operand")):
        a, b = match.span(group)
        proofs.append(FieldProof(field, 0, source.start_offset + offsets[a], source.start_offset + offsets[b - 1] + 1))
    if expected_stage is RuleStage.STOP_BEHAVIOR:
        a, b = match.span("side")
        proofs.append(FieldProof("anchor_side", 0, source.start_offset + offsets[a], source.start_offset + offsets[b - 1] + 1))
    operands = (match["operand"].lower(),) if expected_stage is RuleStage.EXPECTED_MOVE else ("entry", "below")
    provisional = NormalizedRuleV2(
        "pending", stage, family, "price_distance", operands,
        RuleDirection.UNSPECIFIED, operator, Decimal(match["value"]),
        match["unit"].lower(), None, None, bundle, tuple(proofs),
        "numeric-distance-v1", SCHEMA_VERSION, GRAMMAR_VERSION, FEATURE_CONTRACT_VERSION, "0" * 64,
    )
    identity = normalization_identity(provisional)
    rule = replace(provisional, rule_id="rule-" + identity, semantic_fingerprint=identity)
    return NormalizationResult(NormalizationStatus.SUPPORTED_NONEXECUTABLE, rule, tuple(reasons))


def verify_rule(rule: NormalizedRuleV2, *, source_lookup: Callable[[EvidenceSpan], str]) -> NormalizationResult:
    if not isinstance(rule, NormalizedRuleV2):
        return _rejected("INVALID_RULE_INPUT")
    recomputed = normalize_rule(rule.evidence, stage=rule.stage, family=rule.family, source_lookup=source_lookup)
    if recomputed.rule != rule:
        return _rejected("RULE_PROOF_MISMATCH")
    return recomputed
