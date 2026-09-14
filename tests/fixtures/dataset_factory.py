"""Tiny deterministic source fixtures for Phase 10 integration tests."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tradingagents.datasets.models import (
    CanonicalExampleV1,
    EvaluationObservation,
    JoinedObservation,
    SourceFingerprint,
    SourceObservation,
)
from tradingagents.datasets.sources import SourceReadResult

UTC = timezone.utc


def fingerprint(kind: str, path: Path, tag: str = "fixture") -> SourceFingerprint:
    return SourceFingerprint(
        f"{kind}-{tag}", str(path.resolve()), f"schema-{kind}",
        f"sha-{kind}-{tag}", f"snapshot-{kind}-{tag}",
    )


def observations(path: Path) -> tuple[JoinedObservation, ...]:
    ids = ("valid-a", "valid-b", "tier-b", "tier-c", "incomplete", "unavailable", "future")
    rows = []
    for index, decision_id in enumerate(ids):
        decision = SourceObservation(
            decision_id, datetime(2026, 1, 1 + index, tzinfo=UTC),
            decision_completed_timestamp=datetime(2026, 1, 1 + index, 0, 1, tzinfo=UTC),
            source_run_id=f"run-{index}", requested_symbol="EURUSD", resolved_symbol="EURUSD",
            analysis_profile="INTRADAY", analysis_timeframe="M5", action="BUY",
            fields={"source_decision_fingerprint": f"decision-{decision_id}", "snapshot_json": {"point": 0.00001, "quote": {"bid": 1.1, "ask": 1.1001}}},
        )
        evaluation = EvaluationObservation(
            decision_id, "ANALYSIS_SNAPSHOT", 300,
            "DATA_UNAVAILABLE" if decision_id == "unavailable" else "COMPLETE",
            decision_id not in {"incomplete", "future"},
            fields={"source_evaluation_fingerprint": f"eval-{decision_id}", "entry_timestamp": decision.analysis_snapshot_timestamp, "target_timestamp": datetime(2026, 1, 1 + index, 0, 5, tzinfo=UTC), "entry_bid": 1.1, "entry_ask": 1.1001, "buy_net_points": 1.0, "sell_net_points": -1.0, "selected_action": "BUY", "hold_opportunity_cost_points": 1.0},
        )
        rows.append(JoinedObservation(decision, evaluation, fields={
            "experience": {"source_decision_fingerprint": f"decision-{decision_id}", "trust": "A" if decision_id.startswith("valid") else "B", "trust_policy_version": "phase8.trust.v1", "experience_schema_version": "phase8.experience.v1", "feature_schema_version": "phase8.features.v1", "feature_extractor_version": "phase8.extractor.v1", "provenance": {"source_decision_id": decision_id}},
        }))
    return tuple(rows)


def canonical(decision_id: str, fps: dict[str, SourceFingerprint], index: int) -> CanonicalExampleV1:
    stamp = datetime(2026, 1, 1 + index, tzinfo=UTC).isoformat()
    return CanonicalExampleV1(
        f"example-{decision_id}",
        {"decision_id": decision_id, "source_run_id": f"run-{index}", "requested_symbol": "EURUSD", "resolved_symbol": "EURUSD", "analysis_profile": "INTRADAY", "analysis_timeframe": "M5", "action": "BUY", "analysis_snapshot_timestamp": stamp, "normalization_status": "NORMALIZED", "decision_context_status": "COMPLETE", "raw_result_fingerprint": f"raw-{decision_id}"},
        {"snapshot": {"symbol": "EURUSD", "timestamp": stamp, "point": 0.00001, "quote": {"bid": 1.1, "ask": 1.1001}}},
        {"context_integrity": "COMPLETE", "evidence_use_status": "USED", "refs_used": ("K1",), "refs_rejected": (), "bundle_status": "COMPLETE", "integration_status": "INJECTED"},
        {"evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300, "evaluation_status": "COMPLETE", "selected_action": "BUY", "buy_net_points": 1.0, "sell_net_points": -1.0, "hold_opportunity_cost_points": 1.0},
        {"tier": "A", "policy_version": "phase8.trust.v1", "experience_schema_version": "phase8.experience.v1", "feature_schema_version": "phase8.features.v1", "feature_extractor_version": "phase8.extractor.v1"},
        {"phase56": fps["phase56"].to_dict(), "phase8": {"source_fingerprint": fps["phase8"].to_dict()}, "phase9": {"source_fingerprint": fps["phase9"].to_dict()}, "source_decision_fingerprint": f"decision-{decision_id}"},
    )


def make_source_map(root: Path) -> dict[Path, SourceReadResult]:
    source = root / "source.db"
    fp56 = fingerprint("phase56", source)
    return {source: SourceReadResult(decisions=tuple(x.decision for x in observations(source)), evaluations=tuple(x.evaluation for x in observations(source)), fingerprint=fp56)}


__all__ = ["UTC", "canonical", "fingerprint", "make_source_map", "observations"]
