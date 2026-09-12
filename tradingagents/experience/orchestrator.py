"""Read-only composition of Phase 7 knowledge and Phase 8 experience evidence."""

from __future__ import annotations

from typing import Any

from tradingagents.knowledge.models import KnowledgeQuery

from .errors import (
    ExperienceError,
    ExperienceMemoryUnavailableError,
    Phase7KnowledgeUnavailableError,
)
from .models import (
    EvidenceBundle,
    EvidenceRequest,
    EvidenceSourceError,
    ExperienceQuery,
    ExperienceSearchResult,
    OrchestrationProvenance,
    OutcomeStatsRequest,
    TrustTier,
)


def _error(source: str, exc: BaseException) -> EvidenceSourceError:
    """Convert an exception to bounded typed diagnostics without its traceback."""
    error_type = type(exc).__name__
    if source == "knowledge" and not isinstance(exc, Phase7KnowledgeUnavailableError):
        error_type = Phase7KnowledgeUnavailableError.__name__
    elif source in {"experience", "statistics"} and not isinstance(exc, ExperienceError):
        error_type = ExperienceMemoryUnavailableError.__name__
    message = str(exc).replace("\r", " ").replace("\n", " ")[:500]
    return EvidenceSourceError(source=source, error_type=error_type, message=message)


class EvidenceOrchestrator:
    """Compose independent read-only services without ranking or interpretation."""

    def __init__(
        self, knowledge_service: Any, experience_service: Any, statistics_calculator: Any
    ) -> None:
        self.knowledge_service = knowledge_service
        self.experience_service = experience_service
        self.statistics_calculator = statistics_calculator

    @staticmethod
    def _experience_result(value: Any) -> ExperienceSearchResult:
        if isinstance(value, ExperienceSearchResult):
            return value
        return ExperienceSearchResult(tuple(value or ()))

    def query(self, request: EvidenceRequest) -> EvidenceBundle:
        if not isinstance(request, EvidenceRequest):
            raise TypeError("query expects an EvidenceRequest")

        knowledge: tuple[Any, ...] = ()
        experience = ExperienceSearchResult()
        statistics = None
        errors: list[EvidenceSourceError] = []
        source_status: dict[str, str] = {}
        requested = []

        if request.research_question is not None and request.research_question.strip():
            requested.append("knowledge")
            try:
                # Keep the Phase 7 envelope limited to its public query fields.
                knowledge_query = KnowledgeQuery(
                    text=request.research_question,
                    top_k=request.knowledge_top_k,
                    content_types=request.content_types,
                    document_ids=request.document_ids,
                )
                knowledge = tuple(self.knowledge_service.search(knowledge_query) or ())
                source_status["knowledge"] = "COMPLETE"
            except Exception as exc:  # service boundary: return typed source status
                source_status["knowledge"] = "FAILED"
                errors.append(_error("knowledge", exc))

        if request.market_state is not None:
            requested.append("experience")
            try:
                # An empty request trust policy means the ExperienceQuery default.
                tiers = request.trust_tiers or (
                    TrustTier.TIER_A_HIGH_TRUST,
                    TrustTier.TIER_B_LIMITED,
                )
                experience_query = ExperienceQuery(
                    market_state=request.market_state,
                    top_k=request.experience_top_k,
                    symbol=request.symbol,
                    analysis_profile=request.analysis_profile,
                    analysis_timeframe=request.analysis_timeframe,
                    as_of=request.as_of,
                    trust_tiers=tiers,
                    action_filter=request.action_filter,
                )
                experience = self._experience_result(
                    self.experience_service.search(experience_query)
                )
                source_status["experience"] = "COMPLETE"
            except Exception as exc:  # service boundary: return typed source status
                source_status["experience"] = "FAILED"
                errors.append(_error("experience", exc))

        if (
            request.evaluation_basis is not None
            and experience.hits
            and "experience" not in {e.source for e in errors}
        ):
            try:
                tiers = request.trust_tiers or (TrustTier.TIER_A_HIGH_TRUST,)
                stats_request = OutcomeStatsRequest(
                    experience_ids=tuple(hit.experience_id for hit in experience.hits),
                    evaluation_basis=request.evaluation_basis,
                    horizon_seconds=request.horizon_seconds or 0,
                    trust_tiers=tiers,
                    as_of=request.as_of,
                )
                statistics = self.statistics_calculator.calculate(stats_request)
                source_status["statistics"] = "COMPLETE"
            except Exception as exc:
                source_status["statistics"] = "FAILED"
                errors.append(_error("statistics", exc))

        successful = sum(source_status.get(source) == "COMPLETE" for source in requested)
        if not requested:
            status = "EMPTY"
        elif errors and successful:
            status = "PARTIAL"
        elif errors:
            status = "FAILED"
        elif knowledge or experience.hits or statistics is not None:
            status = "COMPLETE"
        else:
            status = "EMPTY"

        provenance = OrchestrationProvenance(
            query_normalization_fingerprint=experience.query_normalization_fingerprint,
            knowledge_requested="knowledge" in requested,
            experience_requested="experience" in requested,
        )
        return EvidenceBundle(
            status=status,
            knowledge=knowledge,
            experience=experience.hits,
            statistics=statistics,
            source_status=source_status,
            errors=tuple(errors),
            provenance=provenance,
        )


__all__ = ["EvidenceOrchestrator"]
