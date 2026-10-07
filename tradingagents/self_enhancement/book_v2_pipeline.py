"""Proof revalidation and conservative assembly for offline V2 book research."""
import hashlib
from collections import defaultdict
from collections.abc import Callable, Sequence

from .book_natural_language import verify_rule
from .book_normalization_models import (
    SCHEMA_VERSION,
    CandidateValidationV2,
    NormalizationResult,
    NormalizationStatus,
    StrategyEnvelopeV2,
)
from .strategy_specs import EXECUTABLE_REQUIRED_STAGES, EvidenceSpan


def validate_envelope(envelope: StrategyEnvelopeV2, *, source_lookup: Callable[[EvidenceSpan], str]) -> CandidateValidationV2:
    if not isinstance(envelope, StrategyEnvelopeV2):
        raise TypeError("V2 envelope required")
    outcomes = tuple(verify_rule(rule, source_lookup=source_lookup) for rule in envelope.rules)
    missing = tuple(sorted(EXECUTABLE_REQUIRED_STAGES - {r.stage for r in envelope.rules}, key=lambda s: s.value))
    reasons = {code for result in outcomes for code in result.reason_codes}
    if any(r.status is NormalizationStatus.REJECTED for r in outcomes):
        return CandidateValidationV2(envelope, NormalizationStatus.REJECTED, tuple(sorted(reasons)), missing)
    if missing:
        reasons.add("MISSING_REQUIRED_STAGES")
    signatures = defaultdict(set)
    for r in envelope.rules:
        signatures[r.stage].add((r.operation, r.operands, r.direction, r.operator, r.value, r.unit, r.condition, r.horizon_seconds))
    if any(len(values) > 1 for values in signatures.values()):
        reasons.add("CONFLICTING_STAGE_RULES")
        return CandidateValidationV2(envelope, NormalizationStatus.REJECTED, tuple(sorted(reasons)), missing)
    # No reviewed semantic-anchor pattern permits cross-book assembly yet.
    if len({r.evidence.spans[0].source_hash for r in envelope.rules}) > 1:
        reasons.add("MULTIBOOK_ANCHOR_UNSUPPORTED")
    if len({r.evidence.spans[0].source_hash for r in envelope.rules}) < 2:
        reasons.add("INSUFFICIENT_INDEPENDENT_SOURCES")
    status = NormalizationStatus.SUPPORTED_NONEXECUTABLE
    if not reasons and all(r.status is NormalizationStatus.EXECUTABLE_ELIGIBLE for r in outcomes):
        status = NormalizationStatus.EXECUTABLE_ELIGIBLE
    return CandidateValidationV2(envelope, status, tuple(sorted(reasons)), missing)


def assemble_v2(results: Sequence[NormalizationResult], *, source_lookup: Callable[[EvidenceSpan], str]) -> tuple[CandidateValidationV2, ...]:
    groups = defaultdict(dict)
    for result in results:
        if not isinstance(result, NormalizationResult):
            raise TypeError("normalization results required")
        if result.rule is not None:
            r = result.rule
            # No hidden assumptions linking separately retrieved charts/books.
            e = r.evidence.spans[0]
            groups[(r.family, e.document_id, e.chunk_id)][r.rule_id] = r
    candidates = []
    for key, rules in sorted(groups.items()):
        values = tuple(sorted(rules.values(), key=lambda r: r.rule_id))
        identity = hashlib.sha256("|".join(r.rule_id for r in values).encode()).hexdigest()
        envelope = StrategyEnvelopeV2("book-v2-" + identity, key[0], values, SCHEMA_VERSION)
        candidates.append(validate_envelope(envelope, source_lookup=source_lookup))
    return tuple(candidates)
