"""Deterministic leakage gates for the Phase 8 read-only experience boundary."""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradingagents.experience.models import ExperienceQuery, TrustTier
from tradingagents.experience.normalization import NormalizationCohortV1, SimilarityProfileV1, build_profile
from tradingagents.experience.models import OutcomeStatsRequest
from tradingagents.experience.outcomes import OutcomeStatsCalculator
from tradingagents.experience.query import ExperienceQueryService

UTC = timezone.utc
NAMES = tuple(f"f{i}" for i in range(8))
COHORT = NormalizationCohortV1("EURUSD", "INTRADAY", "M15", "experience-features.v1", "extractor")


def _row(eid: str, value: float = 0.0, *, when: datetime | None = None, completed: datetime | None = None,
         action: str = "BUY", outcome: float = 1.0, cohort: NormalizationCohortV1 = COHORT,
         tier: TrustTier = TrustTier.TIER_A_HIGH_TRUST, current: bool = True) -> dict:
    when = when or datetime(2026, 1, 1, tzinfo=UTC)
    return {"experience_id": eid, "symbol": cohort.resolved_symbol, "analysis_profile": cohort.analysis_profile,
            "analysis_timeframe": cohort.analysis_timeframe, "analysis_snapshot_timestamp": when,
            "decision_completed_timestamp": completed or when + timedelta(minutes=1), "trust": tier,
            "trust_tier": tier, "source_aliases": {"db:d": "CURRENT" if current else "REMOVED"},
            "values": tuple(value + i * .01 for i in range(8)), "mask": (True,) * 8,
            "feature_names": NAMES, "cohort": cohort, "feature_schema_version": cohort.feature_schema_version,
            "feature_extractor_version": cohort.feature_extractor_version, "feature_fingerprint": eid,
            "accepted": True, "provenance_valid": True, "action": action, "net_points": outcome}


def _profile(rows):
    return SimilarityProfileV1({n: 0.0 for n in NAMES}, {n: 1.0 for n in NAMES}, {n: 1.0 for n in NAMES},
                               {n: 1.0 for n in NAMES}, {n: 1.0 for n in NAMES}, 1e-9,
                               {n: 1.0 for n in NAMES}, (-8, 8), NAMES, "exclude-missing.v1",
                               (TrustTier.TIER_A_HIGH_TRUST, TrustTier.TIER_B_LIMITED), COHORT)


def _service(rows):
    return ExperienceQueryService(rows, profile=_profile(rows))


def _query(**kwargs):
    return ExperienceQuery({"values": (0.0,) * 8, "mask": (True,) * 8, "feature_names": NAMES,
                            "cohort": COHORT}, **kwargs)


def _projection(result):
    return tuple((h.experience_id, h.distance, h.similarity_score) for h in result.hits)


def test_outcome_change_does_not_change_similarity():
    rows = [_row("exp1"), _row("exp2", 1.0)]
    service = _service(rows)
    before = _projection(service.search(_query()))
    rows[0]["net_points"] = 999999.0
    assert _projection(service.search(_query())) == before


def test_action_change_does_not_change_default_similarity():
    rows = [_row("exp1"), _row("exp2", 1.0)]
    service = _service(rows)
    before = _projection(service.search(_query()))
    rows[0]["action"] = "SELL"
    assert _projection(service.search(_query())) == before


def test_candidate_time_future_row_is_excluded_by_cutoff():
    service = _service([_row("past"), _row("future", when=datetime(2026, 1, 3, tzinfo=UTC))])
    result = service.search(_query(as_of=datetime(2026, 1, 2, tzinfo=UTC)))
    assert [hit.experience_id for hit in result.hits] == ["past"]
    assert result.excluded_counts["as_of"] == 1


def test_completion_time_future_row_is_excluded_by_cutoff():
    service = _service([_row("past"), _row("future", completed=datetime(2026, 1, 3, tzinfo=UTC))])
    result = service.search(_query(as_of=datetime(2026, 1, 2, tzinfo=UTC)))
    assert [hit.experience_id for hit in result.hits] == ["past"]
    assert result.excluded_counts["as_of"] == 1


def test_future_rows_do_not_change_historical_normalization():
    cutoff = datetime(2026, 1, 2, tzinfo=UTC)
    before = [_row("old", 1.0, when=datetime(2026, 1, 1, tzinfo=UTC))]
    future = _row("future", 9999.0, when=datetime(2026, 1, 3, tzinfo=UTC))
    assert build_profile(before, COHORT, (TrustTier.TIER_A_HIGH_TRUST,), cutoff).to_fingerprint() == build_profile(
        before + [future], COHORT, (TrustTier.TIER_A_HIGH_TRUST,), cutoff).to_fingerprint()


def test_evaluation_future_row_is_not_visible_before_observation():
    snapshot = {"evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300,
                "evaluation_status": "COMPLETE", "source_context_eligible": 1,
                "observation_timestamp": "2026-01-03T12:00:00Z", "evaluated_at": "2026-01-03T13:00:00Z",
                "selected_action": "BUY", "buy_net_points": 2.0, "sell_net_points": -1.0,
                "hold_opportunity_cost_points": 0.0, "fingerprint": "future"}
    result = OutcomeStatsCalculator([{"experience_id": "exp1", "trust": TrustTier.TIER_A_HIGH_TRUST,
                                      "outcome_snapshots": (snapshot,)}]).calculate(
        OutcomeStatsRequest(("exp1",), as_of=datetime(2026, 1, 2, tzinfo=UTC), horizon_seconds=300))
    assert result.eligible_count == 0
    assert result.excluded_counts["EVALUATION_NOT_YET_AVAILABLE"] == 1


def test_outcome_action_and_completion_fields_never_enter_similarity_projection():
    rows = [_row("exp1", action="BUY", outcome=1.0), _row("exp2", 1.0)]
    service = _service(rows)
    before = _projection(service.search(_query()))
    rows[0].update(action="SELL", net_points=999999.0,
                   decision_completed_timestamp=datetime(2099, 1, 1, tzinfo=UTC))
    assert _projection(service.search(_query())) == before


def test_no_forbidden_phase8_imports():
    forbidden = ("mt5", "forex", "graph", "agents", "qwen", "ollama", "execution", "training",
                 "ingestor", "ingestion", "writer", "phase7")
    package = Path(__file__).parents[1] / "tradingagents" / "experience"
    violations = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                violations.update(name for name in names if any(word in name.lower().split(".") for word in forbidden))
    assert violations == set()
