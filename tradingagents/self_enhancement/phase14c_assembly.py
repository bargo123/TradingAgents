"""Exact-source, conflict-aware assembly for offline Phase 14C concepts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from tradingagents.knowledge.models import KnowledgeHit
from tradingagents.self_enhancement.book_atomic_extraction import (
    ConceptFamily,
    ResolvedAtomicRule,
    StrategyConcept,
    UnresolvedAtomicRule,
)
from tradingagents.self_enhancement.book_pipeline import classify_suitability
from tradingagents.self_enhancement.book_rule_grammar import (
    FEATURE_UNITS,
    parse_supported_rule_quote,
)
from tradingagents.self_enhancement.phase14c_models import (
    AssembledConcept,
    AssemblyMode,
    AssemblyRejection,
    AssemblyReport,
    AssemblyStatus,
    ChunkIdentity,
    DuplicateClusterIndex,
    ResearchParameterKind,
    ResearchParameterRequest,
    SemanticAnchor,
    SourceConflict,
)
from tradingagents.self_enhancement.strategy_specs import (
    EXECUTABLE_REQUIRED_STAGES,
    EvidenceSpan,
    RuleOrigin,
    RuleStage,
    RuleValidationResult,
    RuleValidationStatus,
    StrategyRuleClaim,
    StrategySpec,
    StrategySuitability,
)

PHASE14C_ASSEMBLY_VERSION = "phase14c-exact-source-assembly-v1"
PHASE14C_ASSEMBLY_SCHEMA_VERSION = "strategy-spec-v1"

_ANCHOR_STAGES = (RuleStage.ENTRY, RuleStage.CONFIRMATION, RuleStage.INVALIDATION)
_NUMERIC_UNIT_RE = r"points|pips|ticks|bps|fraction|hz|ratio"
_INCOMPLETE_NUMERIC_RE = re.compile(
    rf"^(?:(?P<stage>ENTRY|CONFIRMATION|INVALIDATION|EXIT|PROFIT_PROTECTION|STOP_BEHAVIOR): )?"
    rf"(?P<direction>LONG|SHORT) when (?P<feature>[a-z][a-z0-9_]*) "
    rf"(?P<operator>>=|>|<=|<) (?P<unit>{_NUMERIC_UNIT_RE}) "
    r"(?P<condition>after [a-z][a-z0-9_-]*(?: [a-z][a-z0-9_-]*)*)\.?$"
)


@dataclass(frozen=True, slots=True)
class _ValidatedRule:
    claim: StrategyRuleClaim
    feature: str | None
    evidence_id: str

    @property
    def signature(self) -> tuple[Any, ...]:
        claim = self.claim
        return (
            claim.stage.value,
            claim.direction.value,
            claim.operator.value,
            self.feature,
            _decimal_text(claim.value),
            claim.unit,
            claim.condition,
            claim.horizon_seconds,
        )


@dataclass(frozen=True, slots=True)
class _ConceptInput:
    family: str
    rules: tuple[_ValidatedRule, ...]
    parameter_requests: tuple[tuple[RuleStage, str, EvidenceSpan], ...]
    rejections: tuple[AssemblyRejection, ...]
    anchor: SemanticAnchor | None
    ambiguous_anchor: bool


def _decimal_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception:
        return str(value)
    if not decimal_value.is_finite():
        return str(value)
    text = format(decimal_value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _source_content(hit: KnowledgeHit) -> dict[str, Any]:
    values = hit.to_dict()
    for field in ("score", "semantic_score", "lexical_score", "fused_score", "rerank_score"):
        values.pop(field, None)
    return values


def _population_hash_from_index(
    duplicate_clusters: DuplicateClusterIndex,
    generation_id: str,
) -> str | None:
    """Derive the selected pool's population identity, distinct from its fingerprint."""

    identities: set[tuple[str, str]] = set()
    for cluster in duplicate_clusters.clusters:
        for record in cluster.records:
            extra = record.hit.extra if isinstance(record.hit.extra, Mapping) else {}
            generation = extra.get("projection_generation")
            population_hash = extra.get("projection_population_hash")
            if not isinstance(generation, str) or not isinstance(population_hash, str):
                return None
            if not generation.strip() or not population_hash.strip():
                return None
            identities.add((generation, population_hash))
    if len(identities) != 1:
        return None
    indexed_generation, population_hash = next(iter(identities))
    if indexed_generation != generation_id:
        return None
    return population_hash


def _rejection(family: str, evidence: Any, reason_code: str) -> AssemblyRejection:
    hit = getattr(evidence, "hit", None)
    return AssemblyRejection(
        family=family,
        evidence_id=str(getattr(evidence, "evidence_id", "unknown-evidence")),
        document_id=str(getattr(hit, "document_id", "unknown-document")),
        chunk_id=str(getattr(hit, "chunk_id", "unknown-chunk")),
        reason_code=reason_code,
    )


def _resolve_evidence(
    rule: ResolvedAtomicRule,
    *,
    family: str,
    generation_id: str,
    population_hash: str | None,
    duplicate_clusters: DuplicateClusterIndex,
) -> tuple[KnowledgeHit | None, EvidenceSpan | None, AssemblyRejection | None]:
    evidence = getattr(rule, "evidence", None)
    hit = getattr(evidence, "hit", None)
    if not isinstance(hit, KnowledgeHit):
        return None, None, _rejection(family, evidence, "SOURCE_HIT_INVALID")
    try:
        identity = ChunkIdentity.from_hit(hit)
        indexed = duplicate_clusters.record_for_identity(identity).hit
    except (TypeError, ValueError, KeyError):
        return hit, None, _rejection(family, evidence, "SOURCE_IDENTITY_NOT_INDEXED")
    if _source_content(hit) != _source_content(indexed):
        return hit, None, _rejection(family, evidence, "SOURCE_CONTENT_MISMATCH")
    extra = hit.extra if isinstance(hit.extra, Mapping) else {}
    indexed_extra = indexed.extra if isinstance(indexed.extra, Mapping) else {}
    if (
        extra.get("projection_generation") != generation_id
        or extra.get("projection_population_hash") != population_hash
        or indexed_extra.get("projection_generation") != generation_id
        or indexed_extra.get("projection_population_hash") != population_hash
    ):
        return hit, None, _rejection(family, evidence, "PINNED_GENERATION_MISMATCH")
    start = getattr(evidence, "start_offset", None)
    end = getattr(evidence, "end_offset", None)
    quote = getattr(evidence, "text", None)
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or end > len(hit.text)
        or not isinstance(quote, str)
        or hit.text[start:end] != quote
    ):
        return hit, None, _rejection(family, evidence, "SOURCE_SPAN_MISMATCH")
    try:
        span = EvidenceSpan(
            generation_id,
            hit.document_id,
            hit.chunk_id,
            str(hit.source_hash),
            start,
            end,
            quote,
        )
    except (TypeError, ValueError):
        return hit, None, _rejection(family, evidence, "SOURCE_PROVENANCE_INVALID")
    return hit, span, None


def _incomplete_numeric_request(
    rule: ResolvedAtomicRule | UnresolvedAtomicRule,
    span: EvidenceSpan,
) -> tuple[RuleStage, str] | None:
    if not isinstance(rule.stage, RuleStage):
        return None
    match = _INCOMPLETE_NUMERIC_RE.fullmatch(span.quote)
    if match is None:
        return None
    explicit_stage = match.group("stage")
    source_stage = RuleStage(explicit_stage) if explicit_stage else RuleStage.ENTRY
    feature = match.group("feature")
    unit = match.group("unit")
    if source_stage is not rule.stage or FEATURE_UNITS.get(feature) != unit:
        return None
    return rule.stage, unit


def _anchor_for(rule: _ValidatedRule) -> SemanticAnchor | None:
    claim = rule.claim
    if claim.stage not in _ANCHOR_STAGES:
        return None
    return SemanticAnchor(
        stage=claim.stage,
        direction=claim.direction,
        operator=claim.operator,
        feature=rule.feature,
        value=_decimal_text(claim.value) or "",
        unit=claim.unit or "",
        condition=claim.condition or "",
        horizon_seconds=claim.horizon_seconds,
    )


def _anchor_priority(anchor: SemanticAnchor) -> int:
    return _ANCHOR_STAGES.index(anchor.stage)


def _stable_rule_key(rule: _ValidatedRule) -> tuple[Any, ...]:
    evidence = rule.claim.evidence
    return (
        rule.claim.stage.value,
        rule.signature,
        evidence.document_id if evidence else "",
        evidence.chunk_id if evidence else "",
        evidence.start_offset if evidence else -1,
    )


def _prepare_concept(
    concept: StrategyConcept,
    *,
    generation_id: str,
    population_hash: str | None,
    duplicate_clusters: DuplicateClusterIndex,
) -> _ConceptInput:
    raw_family = getattr(concept, "family", None)
    if not isinstance(raw_family, ConceptFamily):
        raise TypeError("concept family must be a closed ConceptFamily value")
    family = raw_family.value
    validated: list[_ValidatedRule] = []
    requests: list[tuple[RuleStage, str, EvidenceSpan]] = []
    rejections: list[AssemblyRejection] = []
    evidence_ids: set[str] = set()
    for rule in tuple(getattr(concept, "rules", ())):
        if not isinstance(rule, ResolvedAtomicRule):
            rejections.append(_rejection(family, getattr(rule, "evidence", None), "ATOMIC_RULE_INVALID"))
            continue
        evidence_id = str(getattr(rule.evidence, "evidence_id", ""))
        if not evidence_id or evidence_id in evidence_ids:
            rejections.append(_rejection(family, rule.evidence, "EVIDENCE_ID_DUPLICATE_OR_EMPTY"))
            continue
        evidence_ids.add(evidence_id)
        hit, span, failure = _resolve_evidence(
            rule,
            family=family,
            generation_id=generation_id,
            population_hash=population_hash,
            duplicate_clusters=duplicate_clusters,
        )
        if failure is not None:
            rejections.append(failure)
            continue
        assert hit is not None and span is not None
        if not isinstance(rule.stage, RuleStage):
            rejections.append(_rejection(family, rule.evidence, "RULE_STAGE_INVALID"))
            continue
        parsed = parse_supported_rule_quote(span.quote, rule.stage)
        if parsed is None:
            parameter = _incomplete_numeric_request(rule, span)
            if parameter is not None:
                requests.append((parameter[0], parameter[1], span))
            else:
                rejections.append(_rejection(family, rule.evidence, "UNSUPPORTED_RULE_GRAMMAR"))
            continue
        if not isinstance(rule.parsed, Mapping) or dict(rule.parsed) != dict(parsed):
            rejections.append(_rejection(family, rule.evidence, "PARSED_RULE_MISMATCH"))
            continue
        feature = parsed["feature"]
        if feature is not None and FEATURE_UNITS.get(feature) != parsed["unit"]:
            rejections.append(_rejection(family, rule.evidence, "FEATURE_UNIT_MISMATCH"))
            continue
        value = parsed["value"]
        numeric_value: int | float | str
        if value == value.to_integral_value():
            numeric_value = int(value)
        else:
            numeric = float(value)
            numeric_value = numeric if Decimal(str(numeric)) == value else str(value)
        claim = StrategyRuleClaim(
            stage=parsed["stage"],
            operator=parsed["operator"],
            direction=parsed["direction"],
            value=numeric_value,
            unit=parsed["unit"],
            condition=parsed["condition"],
            horizon_seconds=parsed["horizon_seconds"],
            origin=RuleOrigin.SOURCE_SUPPORTED_CONCEPT,
            evidence=span,
        )
        validated.append(_ValidatedRule(claim, feature, evidence_id))

    for rule in tuple(getattr(concept, "unresolved_rules", ())):
        if not isinstance(rule, UnresolvedAtomicRule):
            rejections.append(
                _rejection(family, getattr(rule, "evidence", None), "UNRESOLVED_RULE_INVALID")
            )
            continue
        evidence_id = str(rule.evidence.evidence_id)
        if not evidence_id or evidence_id in evidence_ids:
            rejections.append(_rejection(family, rule.evidence, "EVIDENCE_ID_DUPLICATE_OR_EMPTY"))
            continue
        evidence_ids.add(evidence_id)
        _, span, failure = _resolve_evidence(
            rule,
            family=family,
            generation_id=generation_id,
            population_hash=population_hash,
            duplicate_clusters=duplicate_clusters,
        )
        if failure is not None:
            rejections.append(failure)
            continue
        assert span is not None
        parameter = (
            _incomplete_numeric_request(rule, span)
            if rule.reason_code == "UNSUPPORTED_RULE_GRAMMAR"
            else None
        )
        if parameter is not None:
            requests.append((parameter[0], parameter[1], span))
        else:
            rejections.append(_rejection(family, rule.evidence, rule.reason_code))

    anchor_rules = [item for item in validated if item.claim.stage in _ANCHOR_STAGES]
    if anchor_rules:
        priority = min(_anchor_priority(_anchor_for(item)) for item in anchor_rules if _anchor_for(item) is not None)
        preferred = [_anchor_for(item) for item in anchor_rules if _anchor_priority(_anchor_for(item)) == priority]
        unique_anchors = set(preferred)
        ambiguous = len(unique_anchors) != 1
        anchor = None if ambiguous else next(iter(unique_anchors))
    else:
        ambiguous = False
        anchor = None
    if ambiguous:
        for item in anchor_rules:
            anchor_stage = _ANCHOR_STAGES[priority]
            if item.claim.stage is anchor_stage:
                span = item.claim.evidence
                rejections.append(
                    AssemblyRejection(
                        family,
                        item.evidence_id,
                        span.document_id if span else "unknown-document",
                        span.chunk_id if span else "unknown-chunk",
                        "AMBIGUOUS_CONCEPT_ANCHOR",
                    )
                )
    return _ConceptInput(
        family,
        tuple(sorted(validated, key=_stable_rule_key)),
        tuple(requests),
        tuple(rejections),
        anchor,
        ambiguous,
    )


def _canonical_documents(
    claims: Sequence[StrategyRuleClaim],
    duplicate_clusters: DuplicateClusterIndex,
    additional_evidence: Sequence[EvidenceSpan] = (),
) -> tuple[str, ...]:
    identities = tuple(
        ChunkIdentity(
            claim.evidence.document_id,
            claim.evidence.chunk_id,
            claim.evidence.source_hash,
        )
        for claim in claims
        if claim.evidence is not None
    ) + tuple(
        ChunkIdentity(span.document_id, span.chunk_id, span.source_hash)
        for span in additional_evidence
    )
    return duplicate_clusters.independent_document_ids(identities) if identities else ()


def _make_spec(
    *,
    family: str,
    anchor: SemanticAnchor,
    claims: Sequence[StrategyRuleClaim],
    features: Sequence[str],
    generation_id: str,
    generation_fingerprint: str,
    model_id: str,
    available_features: tuple[str, ...],
    mode: AssemblyMode,
) -> StrategySpec:
    ordered_claims = tuple(sorted(claims, key=lambda claim: (claim.stage.value, claim.fingerprint)))
    identity_payload = {
        "assembly_version": PHASE14C_ASSEMBLY_VERSION,
        "family": family,
        "anchor": anchor.to_dict(),
        "rule_fingerprints": sorted(claim.fingerprint for claim in ordered_claims),
        "generation_id": generation_id,
        "generation_fingerprint": generation_fingerprint,
    }
    digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    validations = tuple(
        RuleValidationResult(claim.fingerprint, RuleValidationStatus.SUPPORTED, "EXACT_SOURCE_RULE")
        for claim in ordered_claims
    )
    base = StrategySpec(
        spec_id=f"phase14c-spec-{digest[:24]}",
        name=f"{family.replace('_', ' ').title()} source concept",
        family=family,
        required_data=tuple(sorted(set(features))),
        rule_claims=ordered_claims,
        generation_id=generation_id,
        knowledge_fingerprint=generation_fingerprint,
        model_provider="ollama-local",
        model_id=model_id,
        prompt_version=PHASE14C_ASSEMBLY_VERSION,
        schema_version=PHASE14C_ASSEMBLY_SCHEMA_VERSION,
        created_at=datetime.now(UTC),
        implementation_confidence=0.0,
        suitability=StrategySuitability.INSUFFICIENT_SPECIFICATION,
        validation_results=validations,
    )
    if mode is AssemblyMode.COMPOSITE_RESEARCH_HYPOTHESIS:
        suitability = StrategySuitability.INSUFFICIENT_SPECIFICATION
    else:
        suitability = classify_suitability(base, available_features).suitability
    return replace(base, suitability=suitability)


def _record_group(
    family: str,
    anchor: SemanticAnchor | None,
    concepts: Sequence[_ConceptInput],
    *,
    generation_id: str,
    generation_fingerprint: str,
    model_id: str,
    available_features: tuple[str, ...],
    duplicate_clusters: DuplicateClusterIndex,
    conflicts_out: list[SourceConflict],
    requests_out: list[ResearchParameterRequest],
    rejections_out: list[AssemblyRejection],
) -> AssembledConcept:
    rules = tuple(rule for concept in concepts for rule in concept.rules)
    claims = tuple(
        sorted(
            {rule.claim.fingerprint: rule.claim for rule in rules}.values(),
            key=lambda claim: (claim.stage.value, claim.fingerprint),
        )
    )
    signatures_by_stage: dict[RuleStage, dict[tuple[Any, ...], list[EvidenceSpan]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rule in rules:
        if rule.claim.evidence is not None:
            signatures_by_stage[rule.claim.stage][rule.signature].append(rule.claim.evidence)
    group_conflicts: list[SourceConflict] = []
    if anchor is not None:
        for stage, variants in signatures_by_stage.items():
            if len(variants) > 1:
                refs = tuple(span for spans in variants.values() for span in spans)
                group_conflicts.append(SourceConflict(family, anchor, stage, refs))
    conflicts_out.extend(group_conflicts)
    group_rejections = tuple(item for concept in concepts for item in concept.rejections)
    rejections_out.extend(group_rejections)

    parameter_groups: dict[tuple[RuleStage, str], list[EvidenceSpan]] = defaultdict(list)
    for concept in concepts:
        for stage, unit, span in concept.parameter_requests:
            parameter_groups[(stage, unit)].append(span)
    group_requests = tuple(
        ResearchParameterRequest(
            family,
            anchor,
            stage,
            ResearchParameterKind.NUMERIC_THRESHOLD,
            unit,
            tuple(spans),
        )
        for (stage, unit), spans in sorted(parameter_groups.items(), key=lambda item: (item[0][0].value, item[0][1]))
    )
    requests_out.extend(group_requests)

    features = tuple(sorted({rule.feature for rule in rules if rule.feature is not None}))
    request_spans = tuple(
        span
        for concept in concepts
        for _, _, span in concept.parameter_requests
    )
    independent_documents = _canonical_documents(claims, duplicate_clusters, request_spans)
    has_conflicts = bool(group_conflicts)
    has_rejections = bool(group_rejections)
    has_parameters = bool(group_requests)
    present_stages = {claim.stage for claim in claims if claim.is_complete}
    missing = tuple(sorted(EXECUTABLE_REQUIRED_STAGES - present_stages, key=lambda item: item.value))
    is_complete = not missing and not has_parameters

    if has_conflicts:
        status = AssemblyStatus.CONFLICTED
        mode = None
        spec = None
    elif has_rejections:
        status = AssemblyStatus.REJECTED
        mode = None
        spec = None
    elif not is_complete:
        status = AssemblyStatus.PARTIAL
        mode = None
        spec = None
    else:
        docs_to_stages: dict[str, set[RuleStage]] = defaultdict(set)
        for claim in claims:
            if claim.evidence is not None and claim.is_complete:
                docs_to_stages[claim.evidence.document_id].add(claim.stage)
        single_source_complete = any(
            stages >= EXECUTABLE_REQUIRED_STAGES for stages in docs_to_stages.values()
        )
        if not single_source_complete:
            mode = AssemblyMode.COMPOSITE_RESEARCH_HYPOTHESIS
        elif len(independent_documents) > 1:
            mode = AssemblyMode.MULTI_SOURCE_CONSISTENT
        else:
            mode = AssemblyMode.SINGLE_SOURCE_COMPLETE
        status = AssemblyStatus.COMPLETE
        spec = _make_spec(
            family=family,
            anchor=anchor,
            claims=claims,
            features=features,
            generation_id=generation_id,
            generation_fingerprint=generation_fingerprint,
            model_id=model_id,
            available_features=available_features,
            mode=mode,
        )

    eligible = bool(
        status is AssemblyStatus.COMPLETE
        and mode is not AssemblyMode.COMPOSITE_RESEARCH_HYPOTHESIS
        and spec is not None
        and spec.is_executable
    )
    return AssembledConcept(
        family=family,
        semantic_anchor=anchor,
        independent_document_ids=independent_documents,
        rule_claims=claims,
        assembly_mode=mode,
        status=status,
        executable_eligible=eligible,
        spec=spec,
        missing_stages=missing,
    )


def assemble_normalized_concepts(results, *, source_lookup):
    """Explicit V2 dispatch; legacy atomic assembly stays unchanged."""
    from .book_v2_pipeline import assemble_v2
    return assemble_v2(results, source_lookup=source_lookup)


def assemble_atomic_concepts(
    concepts: Sequence[StrategyConcept],
    *,
    generation_id: str,
    generation_fingerprint: str,
    model_id: str,
    available_features: Iterable[str],
    duplicate_clusters: DuplicateClusterIndex,
) -> AssemblyReport:
    """Assemble exact, independently evidenced rules; never infer a missing rule."""

    for name, value in (
        ("generation_id", generation_id),
        ("generation_fingerprint", generation_fingerprint),
        ("model_id", model_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty")
    if not isinstance(duplicate_clusters, DuplicateClusterIndex):
        raise TypeError("duplicate_clusters must be DuplicateClusterIndex")
    feature_values = tuple(available_features)
    if any(not isinstance(item, str) or not item.strip() for item in feature_values):
        raise ValueError("available_features must contain non-empty strings")
    if len(feature_values) != len(set(feature_values)):
        feature_values = tuple(sorted(set(feature_values)))
    concept_values = tuple(concepts)
    if any(not isinstance(item, StrategyConcept) for item in concept_values):
        raise TypeError("concepts must contain StrategyConcept values")

    population_hash = _population_hash_from_index(duplicate_clusters, generation_id)
    prepared = tuple(
        _prepare_concept(
            concept,
            generation_id=generation_id,
            population_hash=population_hash,
            duplicate_clusters=duplicate_clusters,
        )
        for concept in concept_values
    )
    buckets: dict[tuple[str, SemanticAnchor | None, str | None], list[_ConceptInput]] = defaultdict(list)
    for index, concept in enumerate(prepared):
        if concept.ambiguous_anchor:
            # Keep ambiguous source facts auditable but never merge them.
            buckets[(concept.family, None, f"ambiguous-{index}")].append(concept)
        elif concept.anchor is None:
            buckets[(concept.family, None, f"unanchored-{index}")].append(concept)
        else:
            buckets[(concept.family, concept.anchor, None)].append(concept)

    conflicts: list[SourceConflict] = []
    requests: list[ResearchParameterRequest] = []
    rejections: list[AssemblyRejection] = []
    records = []
    for (family, anchor, _), values in sorted(
        buckets.items(),
        key=lambda item: (item[0][0], item[0][1].stage.value if item[0][1] else "", item[0][1].value if item[0][1] else "", item[0][2] or ""),
    ):
        records.append(
            _record_group(
                family,
                anchor,
                values,
                generation_id=generation_id,
                generation_fingerprint=generation_fingerprint,
                model_id=model_id,
                available_features=feature_values,
                duplicate_clusters=duplicate_clusters,
                conflicts_out=conflicts,
                requests_out=requests,
                rejections_out=rejections,
            )
        )
    records.sort(
        key=lambda item: (
            item.family,
            item.semantic_anchor.stage.value if item.semantic_anchor else "",
            item.semantic_anchor.value if item.semantic_anchor else "",
            item.spec.spec_id if item.spec else "",
            item.independent_document_ids,
        )
    )
    conflicts.sort(
        key=lambda item: (item.family, item.anchor.stage.value, item.anchor.value, item.stage.value)
    )
    return AssemblyReport(tuple(records), tuple(conflicts), tuple(requests), tuple(rejections))


__all__ = [
    "PHASE14C_ASSEMBLY_SCHEMA_VERSION",
    "PHASE14C_ASSEMBLY_VERSION",
    "assemble_atomic_concepts",
]
