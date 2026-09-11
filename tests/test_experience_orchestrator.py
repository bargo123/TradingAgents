from datetime import datetime, timezone

from tradingagents.experience.models import (
    EvidenceRequest,
    ExperienceHit,
    ExperienceSearchResult,
    OutcomeStatistics,
)
from tradingagents.experience.orchestrator import EvidenceOrchestrator


UTC = timezone.utc


class Knowledge:
    def __init__(self, hits=(), error=None):
        self.hits = tuple(hits)
        self.error = error
        self.last_query = None

    def search(self, query):
        self.last_query = query
        if self.error:
            raise self.error
        return self.hits


class Experience:
    def __init__(self, hits=(), error=None):
        self.result = ExperienceSearchResult(tuple(hits), query_normalization_fingerprint="norm-1")
        self.error = error
        self.last_query = None

    def search(self, query):
        self.last_query = query
        if self.error:
            raise self.error
        return self.result


class Stats:
    def __init__(self):
        self.last_request = None

    def calculate(self, request):
        self.last_request = request
        return OutcomeStatistics(evaluation_basis=request.evaluation_basis, horizon_seconds=request.horizon_seconds)


def test_as_of_is_forwarded_unchanged():
    cutoff = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=UTC)
    experience = Experience([ExperienceHit("exp-1")])
    stats = Stats()
    EvidenceOrchestrator(Knowledge(), experience, stats).query(
        EvidenceRequest(market_state={"x": 1}, evaluation_basis="ANALYSIS_SNAPSHOT", horizon_seconds=300, as_of=cutoff)
    )
    assert experience.last_query.as_of == cutoff
    assert stats.last_request.as_of == cutoff


def test_both_sources_keep_separate_scores():
    knowledge = Knowledge(["knowledge-hit"])
    experience = Experience([ExperienceHit("exp-1", similarity_score=0.9)])
    bundle = EvidenceOrchestrator(knowledge, experience, Stats()).query(
        EvidenceRequest(research_question="order flow", market_state={"x": 1})
    )
    assert bundle.knowledge == ("knowledge-hit",)
    assert bundle.experience[0].similarity_score == 0.9
    assert not hasattr(bundle, "combined_score")
    assert bundle.status == "COMPLETE"


def test_one_source_failure_is_partial():
    bundle = EvidenceOrchestrator(
        Knowledge(error=RuntimeError("knowledge unavailable")),
        Experience([ExperienceHit("exp-1")]),
        Stats(),
    ).query(EvidenceRequest(market_state={"x": 1}, research_question="OFI"))
    assert bundle.status == "PARTIAL"
    assert bundle.errors[0].source == "knowledge"


def test_statistics_use_only_returned_experience_ids():
    stats = Stats()
    experience = Experience([ExperienceHit("returned")])
    EvidenceOrchestrator(Knowledge(), experience, stats).query(
        EvidenceRequest(market_state={"x": 1}, evaluation_basis="ANALYSIS_SNAPSHOT", horizon_seconds=300)
    )
    assert stats.last_request.experience_ids == ("returned",)


def test_empty_request_is_empty_without_calling_services():
    knowledge, experience, stats = Knowledge(), Experience(), Stats()
    bundle = EvidenceOrchestrator(knowledge, experience, stats).query(EvidenceRequest())
    assert bundle.status == "EMPTY"
    assert knowledge.last_query is None
    assert experience.last_query is None
    assert stats.last_request is None
