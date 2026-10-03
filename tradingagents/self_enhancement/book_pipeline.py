"""Deterministic provenance validation for local Phase 14B book drafts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from tradingagents.forex.hft.features import TickFeatures
from tradingagents.knowledge.models import KnowledgeHit, KnowledgeQuery
from tradingagents.self_enhancement.book_drafter import StrategyDraft, StrategyDraftClaim
from tradingagents.self_enhancement.book_rule_grammar import (
    FEATURE_UNITS,
    parse_supported_rule_quote,
)
from tradingagents.self_enhancement.strategy_specs import (
    EvidenceSpan,
    RuleOrigin,
    RuleStage,
    RuleValidationResult,
    RuleValidationStatus,
    StrategyRuleClaim,
    StrategySpec,
    StrategySuitability,
)


class PinnedGenerationMismatch(ValueError):
    """The active read-only knowledge projection is not the pinned source."""


class MappingStatus(str, Enum):
    MATCHED = "MATCHED"
    PARTIALLY_MATCHED = "PARTIALLY_MATCHED"
    NO_DIRECT_MATCH = "NO_DIRECT_MATCH"


@dataclass(frozen=True, slots=True)
class SuitabilityResult:
    suitability: StrategySuitability
    missing_features: tuple[str, ...] = ()
    unspecified_stages: tuple[RuleStage, ...] = ()
    unsupported_rule_fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StrategyBatchReport:
    specs: tuple[StrategySpec, ...]
    rejected_rules: tuple[RuleValidationResult, ...]
    query_count: int
    retrieved_count: int
    draft_count: int
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class ComponentMapping:
    component: str
    status: MappingStatus
    evidence: tuple[EvidenceSpan, ...] = ()


@dataclass(frozen=True, slots=True)
class MappingReport:
    strategy_id: str
    overall_status: MappingStatus
    components: tuple[ComponentMapping, ...]


_STRATEGY_COMPONENTS = ("entry", "filter", "expected_move", "exit", "risk", "horizon")
_HFT_HORIZON_SECONDS = 60
_SHORT_TERM_HORIZON_SECONDS = 3600
_INTRADAY_HORIZON_SECONDS = 86400


def _sha256_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _draft_fingerprint(claim: StrategyDraftClaim) -> str:
    payload = json.dumps(
        claim.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_digest(payload)


def _make_result(
    fingerprint: str,
    status: RuleValidationStatus,
    reason_code: str,
) -> RuleValidationResult:
    return RuleValidationResult(fingerprint, status, reason_code)


def _validate_claim(
    claim: StrategyDraftClaim,
    hit_by_alias: Mapping[str, KnowledgeHit],
    *,
    generation: str,
    required_data: Sequence[str],
) -> tuple[StrategyRuleClaim | None, RuleValidationResult]:
    draft_fingerprint = _draft_fingerprint(claim)
    if claim.origin is RuleOrigin.RESEARCH_HYPOTHESIS_PARAMETER:
        value = StrategyRuleClaim(
            stage=claim.stage,
            operator=claim.operator,
            direction=claim.direction,
            value=claim.value,
            unit=claim.unit,
            condition=claim.condition,
            horizon_seconds=claim.horizon_seconds,
            origin=claim.origin,
            evidence=None,
        )
        return value, _make_result(
            value.fingerprint,
            RuleValidationStatus.UNSUPPORTED,
            "RESEARCH_HYPOTHESIS_NOT_SOURCE_EVIDENCE",
        )

    hit = hit_by_alias.get(claim.evidence_ref or "")
    if hit is None:
        return None, _make_result(draft_fingerprint, RuleValidationStatus.UNSUPPORTED, "EVIDENCE_REF_UNKNOWN")
    quote = claim.quote or ""
    start = claim.start_offset
    end = claim.end_offset
    if start is None or end is None or start < 0 or end > len(hit.text) or hit.text[start:end] != quote:
        return None, _make_result(draft_fingerprint, RuleValidationStatus.UNSUPPORTED, "SPAN_MISMATCH")
    parsed = parse_supported_rule_quote(quote, claim.stage)
    if parsed is None:
        return None, _make_result(draft_fingerprint, RuleValidationStatus.UNSUPPORTED, "GRAMMAR_UNSUPPORTED")
    feature = parsed["feature"]
    if feature is not None and feature not in required_data:
        return None, _make_result(draft_fingerprint, RuleValidationStatus.UNSUPPORTED, "DATA_FEATURE_MISMATCH")
    if feature is not None and FEATURE_UNITS.get(feature) != parsed["unit"]:
        return None, _make_result(draft_fingerprint, RuleValidationStatus.UNSUPPORTED, "FEATURE_UNIT_MISMATCH")
    try:
        claim_value = Decimal(str(claim.value)) if claim.value is not None else None
    except InvalidOperation:
        claim_value = None
    if (
        parsed["direction"] is not claim.direction
        or parsed["operator"] is not claim.operator
        or claim_value != parsed["value"]
        or parsed["unit"] != claim.unit
        or parsed["condition"] != claim.condition
        or parsed["stage"] is not claim.stage
        or parsed["horizon_seconds"] != claim.horizon_seconds
    ):
        return None, _make_result(draft_fingerprint, RuleValidationStatus.UNSUPPORTED, "RULE_FIELD_MISMATCH")
    source_hash = hit.source_hash
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        return None, _make_result(draft_fingerprint, RuleValidationStatus.UNSUPPORTED, "SOURCE_HASH_INVALID")
    span = EvidenceSpan(
        generation,
        hit.document_id,
        hit.chunk_id,
        source_hash,
        start,
        end,
        quote,
    )
    validated = StrategyRuleClaim(
        stage=claim.stage,
        operator=claim.operator,
        direction=claim.direction,
        value=claim.value,
        unit=claim.unit,
        condition=claim.condition,
        horizon_seconds=claim.horizon_seconds,
        origin=claim.origin,
        evidence=span,
    )
    if not validated.is_complete:
        return validated, _make_result(
            validated.fingerprint,
            RuleValidationStatus.INSUFFICIENT_SPECIFICATION,
            "RULE_FIELDS_UNSPECIFIED",
        )
    return validated, _make_result(
        validated.fingerprint,
        RuleValidationStatus.SUPPORTED,
        "EXACT_SOURCE_RULE",
    )


def classify_suitability(
    spec: StrategySpec,
    available_features: Iterable[str],
) -> SuitabilityResult:
    if not isinstance(spec, StrategySpec):
        raise TypeError("spec must be StrategySpec")
    features = tuple(available_features)
    if any(not isinstance(item, str) or not item.strip() for item in features):
        raise ValueError("available_features must contain non-empty strings")
    available = set(features)
    missing_features = tuple(sorted(set(spec.required_data) - available))
    if missing_features:
        return SuitabilityResult(
            StrategySuitability.UNAVAILABLE_DATA,
            missing_features=missing_features,
            unspecified_stages=spec.unspecified_stages,
        )
    if spec.research_hypothesis_rules or spec.missing_executable_stages:
        return SuitabilityResult(
            StrategySuitability.INSUFFICIENT_SPECIFICATION,
            unspecified_stages=spec.unspecified_stages,
        )
    validation_by_rule = {result.rule_fingerprint: result.status for result in spec.validation_results}
    unsupported = tuple(
        sorted(
            claim.fingerprint
            for claim in spec.source_supported_rules
            if validation_by_rule.get(claim.fingerprint) is not RuleValidationStatus.SUPPORTED
        )
    )
    if unsupported:
        return SuitabilityResult(
            StrategySuitability.UNIMPLEMENTABLE_AUTONOMOUSLY,
            unsupported_rule_fingerprints=unsupported,
        )
    horizons = tuple(
        claim.horizon_seconds
        for claim in spec.source_supported_rules
        if claim.stage is RuleStage.HORIZON and claim.horizon_seconds is not None
    )
    if not horizons:
        return SuitabilityResult(
            StrategySuitability.INSUFFICIENT_SPECIFICATION,
            unspecified_stages=(RuleStage.HORIZON,),
        )
    horizon = max(horizons)
    if horizon <= _HFT_HORIZON_SECONDS:
        suitability = StrategySuitability.HFT_SUITABLE
    elif horizon <= _SHORT_TERM_HORIZON_SECONDS:
        suitability = StrategySuitability.SHORT_TERM_SUITABLE
    elif horizon <= _INTRADAY_HORIZON_SECONDS:
        suitability = StrategySuitability.INTRADAY_ONLY
    else:
        suitability = StrategySuitability.SWING_ONLY
    return SuitabilityResult(suitability)


def _claim_semantics(claim: StrategyRuleClaim) -> str:
    payload = {
        "stage": claim.stage.value,
        "operator": claim.operator.value,
        "direction": claim.direction.value,
        "value": claim.value,
        "unit": claim.unit,
        "condition": claim.condition,
        "horizon_seconds": claim.horizon_seconds,
        "origin": claim.origin.value,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _spec_semantics(spec: StrategySpec) -> str:
    rules = sorted(_claim_semantics(claim) for claim in spec.rule_claims)
    return json.dumps(
        {
            "family": spec.family,
            "required_data": sorted(spec.required_data),
            "rules": rules,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def deduplicate_specs(specs: Iterable[StrategySpec]) -> tuple[StrategySpec, ...]:
    groups: dict[str, list[StrategySpec]] = {}
    for spec in specs:
        if not isinstance(spec, StrategySpec):
            raise TypeError("specs must contain StrategySpec values")
        groups.setdefault(_spec_semantics(spec), []).append(spec)
    results: list[StrategySpec] = []
    for _semantic_key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda spec: (spec.spec_id, spec.content_hash))
        base = ordered[0]
        retained: dict[str, StrategyRuleClaim] = {}
        seen_documents: set[tuple[str, str]] = set()
        for spec in ordered:
            for claim in spec.rule_claims:
                semantic = _claim_semantics(claim)
                if claim.origin is RuleOrigin.SOURCE_SUPPORTED_CONCEPT:
                    assert claim.evidence is not None
                    document_key = (semantic, claim.evidence.document_id)
                    if document_key in seen_documents:
                        continue
                    seen_documents.add(document_key)
                    retained[f"{semantic}|{claim.evidence.document_id}"] = claim
                else:
                    retained.setdefault(semantic, claim)
        merged_claims = tuple(retained[key] for key in sorted(retained))
        merged_fingerprints = {claim.fingerprint for claim in merged_claims}
        merged_validation: dict[str, RuleValidationResult] = {}
        for spec in ordered:
            for result in spec.validation_results:
                if result.rule_fingerprint in merged_fingerprints:
                    merged_validation.setdefault(result.rule_fingerprint, result)
        results.append(
            replace(
                base,
                rule_claims=merged_claims,
                validation_results=tuple(merged_validation.values()),
            )
        )
    return tuple(sorted(results, key=lambda spec: (_spec_semantics(spec), spec.spec_id)))


def map_existing_strategy(strategy_id: str, hits: Iterable[KnowledgeHit]) -> MappingReport:
    if strategy_id not in {"range_rejection", "momentum_continuation"}:
        raise ValueError("strategy_id must identify a reviewed incumbent strategy")
    # An exact name mention is not component-level evidence. Until an incumbent
    # component has a reviewed deterministic rule mapping, report no direct
    # matches rather than attaching generic book prose as a citation.
    tuple(hits)  # consume a one-shot iterable without mutating or retaining it
    components = tuple(
        ComponentMapping(name, MappingStatus.NO_DIRECT_MATCH, ())
        for name in _STRATEGY_COMPONENTS
    )
    return MappingReport(strategy_id, MappingStatus.NO_DIRECT_MATCH, components)


class BookStrategyPipeline:
    """Retrieve from a pinned Phase 7 generation and validate local drafts."""

    def __init__(
        self,
        query_service: Any,
        drafter: Any,
        *,
        pinned_generation: str,
        pinned_fingerprint: str,
        available_features: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(pinned_generation, str) or not pinned_generation.strip():
            raise ValueError("pinned_generation is required")
        if not isinstance(pinned_fingerprint, str) or not pinned_fingerprint.strip():
            raise ValueError("pinned_fingerprint is required")
        self.query_service = query_service
        self.drafter = drafter
        self.pinned_generation = pinned_generation
        self.pinned_fingerprint = pinned_fingerprint
        self.available_features = frozenset(
            available_features if available_features is not None else TickFeatures.__dataclass_fields__
        )

    def _catalog_generation(self) -> Any:
        catalog = getattr(self.query_service, "catalog", None)
        active = getattr(catalog, "active_generation", None)
        generation = active() if callable(active) else None
        if (
            generation is None
            or getattr(generation, "generation_id", None) != self.pinned_generation
            or getattr(generation, "population_hash", None) != self.pinned_fingerprint
            or getattr(generation, "status", None) != "VALIDATED"
            or not getattr(generation, "vector_ready", False)
            or not getattr(generation, "lexical_ready", False)
        ):
            raise PinnedGenerationMismatch("active Phase 7 generation does not match the pinned identity")
        return generation

    def _verify_hit(self, hit: Any) -> bool:
        if not isinstance(hit, KnowledgeHit):
            return False
        extra = hit.extra if isinstance(hit.extra, Mapping) else {}
        if (
            extra.get("projection_generation") != self.pinned_generation
            or extra.get("projection_population_hash") != self.pinned_fingerprint
            or not hit.document_id
            or not hit.chunk_id
            or not hit.text
            or not hit.source_hash
        ):
            return False
        catalog = self.query_service.catalog
        document = catalog.get_document(hit.document_id)
        if (
            document is None
            or document.source_hash != hit.source_hash
            or document.active is not True
            or document.projection_generation != self.pinned_generation
            or document.projection_population_hash != self.pinned_fingerprint
        ):
            return False
        matches = tuple(
            chunk for chunk in catalog.chunks_for_document(hit.document_id)
            if chunk.chunk_id == hit.chunk_id
        )
        if len(matches) != 1:
            return False
        chunk = matches[0]
        chunk_extra = getattr(chunk, "extra", {})
        return bool(
            chunk.document_id == hit.document_id
            and chunk.source_hash == hit.source_hash
            and chunk.text == hit.text
            and chunk.active is True
            and chunk.vector_ready is True
            and chunk.lexical_ready is True
            and chunk.projection_generation == self.pinned_generation
            and isinstance(chunk_extra, Mapping)
            and chunk_extra.get("projection_population_hash") == self.pinned_fingerprint
        )

    def _spec_from_draft(
        self,
        draft: StrategyDraft,
        *,
        hit_by_alias: Mapping[str, KnowledgeHit],
        model_id: str,
    ) -> tuple[StrategySpec | None, tuple[RuleValidationResult, ...]]:
        claims: list[StrategyRuleClaim] = []
        validations: list[RuleValidationResult] = []
        rejected: list[RuleValidationResult] = []
        for draft_claim in draft.rule_claims:
            claim, result = _validate_claim(
                draft_claim,
                hit_by_alias,
                generation=self.pinned_generation,
                required_data=draft.required_data,
            )
            if result.status is RuleValidationStatus.SUPPORTED:
                assert claim is not None
                claims.append(claim)
                validations.append(result)
            elif draft_claim.origin is RuleOrigin.RESEARCH_HYPOTHESIS_PARAMETER and claim is not None:
                claims.append(claim)
                validations.append(result)
            else:
                rejected.append(result)
        if not claims:
            return None, tuple(rejected)
        payload = draft.model_dump(mode="json")
        draft_digest = _sha256_digest(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        spec = StrategySpec(
            spec_id=f"book-spec-{draft_digest[:24]}",
            name=draft.name,
            family=draft.family,
            required_data=tuple(draft.required_data),
            rule_claims=tuple(claims),
            generation_id=self.pinned_generation,
            knowledge_fingerprint=self.pinned_fingerprint,
            model_provider="ollama-local",
            model_id=model_id,
            prompt_version="phase14b-source-grounded-draft-v1",
            schema_version="phase14b-strategy-draft-v1",
            created_at=datetime.now(UTC),
            implementation_confidence=draft.implementation_confidence,
            suitability=StrategySuitability.INSUFFICIENT_SPECIFICATION,
            validation_results=tuple(validations),
        )
        suitability = classify_suitability(spec, self.available_features)
        return replace(spec, suitability=suitability.suitability), tuple(rejected)

    def extract(self, queries: Sequence[str] | str, *, max_specs: int = 10) -> StrategyBatchReport:
        if type(max_specs) is not int or not 1 <= max_specs <= 10:
            raise ValueError("max_specs must be between 1 and 10")
        query_values = (queries,) if isinstance(queries, str) else tuple(queries)
        if not query_values or len(query_values) > 20 or any(
            not isinstance(query, str) or not query.strip() or len(query) > 1000 for query in query_values
        ):
            raise ValueError("queries must contain 1 to 20 bounded non-empty strings")
        self._catalog_generation()
        retrieved: list[KnowledgeHit] = []
        seen: set[tuple[str, str, str]] = set()
        try:
            for query in query_values:
                hits = self.query_service.search(KnowledgeQuery(text=query, top_k=10))
                for hit in hits:
                    if not self._verify_hit(hit):
                        return StrategyBatchReport((), (), len(query_values), 0, 0, "KNOWLEDGE_PROVENANCE_INVALID")
                    identity = (hit.document_id, hit.chunk_id, hit.source_hash or "")
                    if identity not in seen:
                        seen.add(identity)
                        retrieved.append(hit)
        except PinnedGenerationMismatch:
            raise
        except Exception:
            return StrategyBatchReport((), (), len(query_values), 0, 0, "QUERY_FAILED")
        self._catalog_generation()
        if not retrieved:
            return StrategyBatchReport((), (), len(query_values), 0, 0, "NO_RETRIEVAL_HITS")
        try:
            result = self.drafter.draft(tuple(retrieved), max_specs=max_specs)
        except Exception:
            return StrategyBatchReport((), (), len(query_values), len(retrieved), 0, "DRAFTER_FAILED")
        if not getattr(result, "ok", False):
            return StrategyBatchReport(
                (), (), len(query_values), len(retrieved), 0,
                getattr(result, "error_code", None) or "DRAFTER_FAILED",
            )
        if getattr(result, "provider", None) != "ollama-local":
            return StrategyBatchReport((), (), len(query_values), len(retrieved), 0, "NONLOCAL_DRAFTER_REJECTED")
        hit_by_alias = {f"E{index}": hit for index, hit in enumerate(retrieved, start=1)}
        specs: list[StrategySpec] = []
        rejected: list[RuleValidationResult] = []
        for draft in result.specs:
            if not isinstance(draft, StrategyDraft):
                return StrategyBatchReport((), (), len(query_values), len(retrieved), 0, "DRAFT_SCHEMA_INVALID")
            spec, rejected_claims = self._spec_from_draft(
                draft,
                hit_by_alias=hit_by_alias,
                model_id=result.model,
            )
            if spec is not None:
                specs.append(spec)
            rejected.extend(rejected_claims)
        return StrategyBatchReport(
            specs=tuple(specs),
            rejected_rules=tuple(rejected),
            query_count=len(query_values),
            retrieved_count=len(retrieved),
            draft_count=len(result.specs),
        )


__all__ = [
    "BookStrategyPipeline",
    "ComponentMapping",
    "MappingReport",
    "MappingStatus",
    "PinnedGenerationMismatch",
    "StrategyBatchReport",
    "SuitabilityResult",
    "classify_suitability",
    "deduplicate_specs",
    "map_existing_strategy",
]
