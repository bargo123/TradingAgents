from datetime import datetime, timezone

import pytest

from tradingagents.datasets.eligibility import (
    EligibilityResult,
    classify_observation,
    join_observations,
)
from tradingagents.datasets.models import (
    DatasetConfig,
    DatasetExclusionReason,
    EvaluationObservation,
    SourceFingerprint,
    SourceObservation,
)
from tradingagents.datasets.sources import SourceReadResult

UTC = timezone.utc


@pytest.mark.parametrize("details", [[], False, 0, ""])
def test_eligibility_details_reject_falsey_malformed_mapping(details):
    with pytest.raises(TypeError, match="details"):
        EligibilityResult(False, details=details)


@pytest.mark.parametrize("value", ["OUTCOME_UNAVAILABLE", b"OUTCOME_UNAVAILABLE", {"reason": "OUTCOME_UNAVAILABLE"}, ["OUTCOME_UNAVAILABLE"]])
def test_eligibility_rejects_malformed_reason_collections(value):
    with pytest.raises((TypeError, ValueError), match="reasons"):
        EligibilityResult(False, reasons=value)


def _config(tmp_path, **filters):
    return DatasetConfig((tmp_path / "source.sqlite",), tmp_path / "experience", None, tmp_path / "out", filters=filters)


def _decision(**fields):
    return SourceObservation(
        "d1", datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
        source_run_id="run1", requested_symbol="EURUSD", resolved_symbol="EURUSD",
        action="BUY", fields=fields,
    )


def _joined(tmp_path, **overrides):
    decision = _decision(**overrides.pop("decision_fields", {
        "normalization_status": "NORMALIZED", "decision_context_status": "COMPLETE",
        "source_decision_fingerprint": "dfp", "source_run_id": "run1",
        "analysis_profile": "p", "analysis_timeframe": "1h",
    }))
    ident = {"source_run_id": "run1", "source_decision_fingerprint": "dfp", "fingerprint": "efp", "provenance": {"evaluation_fingerprint": "efp", "source_decision_id": "d1"}}
    result = SourceReadResult(decisions=(decision,), evaluations=(EvaluationObservation("d1", "ANALYSIS_SNAPSHOT", 300, "COMPLETE", True, ident),), fingerprint=SourceFingerprint("src", "source", "schema", "file", "snapshot"))
    features = {tf: dict.fromkeys(("return_over_bars", "range_pct", "close_position", "average_true_range"), 1.0) | {"direction": "UP"} for tf in ("M1", "M5", "M15", "H1")}
    snapshot = {"quote": {"bid": 1.0, "ask": 1.1, "spread_points": 1000.0}, "point": 0.0001, "digits": 5, "features": features}
    records = ({"experience_id": "e1", "source_decision_id": "d1", "source_run_id": "run1", "source_decision_fingerprint": "dfp", "trust": "TIER_A_HIGH_TRUST", "experience_schema_version": "phase8.experience.v1", "feature_schema_version": "v1", "feature_extractor_version": "v1", "trust_policy_version": "v1", "snapshot_json": snapshot, "source_evaluation_fingerprints": {"ANALYSIS_SNAPSHOT:300": "efp"}, "provenance": {"source_decision_fingerprint": "dfp"}},)
    audit = ({"decision_id": "d1", "source_run_id": "run1", "context_integrity": "COMPLETE", "bundle_status": "COMPLETE", "evidence_audit_status": "VALID", "evidence_use_status": "NONE_RELEVANT", "evidence_refs_used": [], "evidence_refs_rejected": [], "missing_nodes": [], "node_context_hashes": {node: "hash-" + node for node in ("Market Analyst", "News Analyst", "Bull Researcher", "Bear Researcher", "Research Manager", "Trader", "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager")}},)
    return join_observations(result, SourceReadResult(records=records), SourceReadResult(audits=audit))[0]


def test_valid_tier_a_complete_observation_is_eligible(tmp_path):
    result = classify_observation(_joined(tmp_path), _config(tmp_path))
    assert isinstance(result, EligibilityResult)
    assert result.eligible
    assert result.reasons == ()


def test_noncanonical_audit_status_is_ineligible(tmp_path):
    observation = _joined(tmp_path)
    audit = dict(observation.fields["audit"])
    audit["evidence_audit_status"] = "valid"
    observation = observation.__class__(
        observation.decision,
        observation.evaluation,
        observation.evidence,
        {**observation.fields, "audit": audit},
    )

    result = classify_observation(observation, _config(tmp_path))

    assert not result.eligible
    assert DatasetExclusionReason.CITATION_INVALID in result.reasons


def test_malformed_evaluation_mapping_cannot_coerce_false_to_true(tmp_path):
    observation = _joined(tmp_path)
    fields = dict(observation.fields)
    evaluation = dict(fields["evaluations"][0])
    evaluation["source_context_eligible"] = "false"
    fields["evaluations"] = (evaluation,)
    malformed = observation.__class__(
        observation.decision, observation.evaluation, observation.evidence, fields
    )

    result = classify_observation(malformed, _config(tmp_path))

    assert not result.eligible
    assert DatasetExclusionReason.OUTCOME_UNAVAILABLE in result.reasons


def test_malformed_evaluation_mapping_cannot_coerce_horizon(tmp_path):
    observation = _joined(tmp_path)
    fields = dict(observation.fields)
    evaluation = dict(fields["evaluations"][0])
    evaluation["horizon_seconds"] = "300"
    fields["evaluations"] = (evaluation,)
    malformed = observation.__class__(
        observation.decision, observation.evaluation, observation.evidence, fields
    )

    result = classify_observation(malformed, _config(tmp_path))

    assert not result.eligible
    assert DatasetExclusionReason.OUTCOME_UNAVAILABLE in result.reasons


@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_malformed_decision_fingerprint_is_not_recomputed(value):
    decision = _decision(source_decision_fingerprint=value)
    with pytest.raises(ValueError, match="source decision fingerprint"):
        join_observations(
            SourceReadResult(decisions=(decision,)),
            SourceReadResult(),
            SourceReadResult(),
        )


@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_malformed_evaluation_fingerprint_is_not_recomputed(value):
    evaluation = EvaluationObservation(
        "d1",
        "ANALYSIS_SNAPSHOT",
        300,
        "COMPLETE",
        True,
        {"source_evaluation_fingerprint": value},
    )
    with pytest.raises(ValueError, match="source evaluation fingerprint"):
        join_observations(
            SourceReadResult(decisions=(_decision(),), evaluations=(evaluation,)),
            SourceReadResult(),
            SourceReadResult(),
        )


def test_source_availability_does_not_truthiness_coerce_text(tmp_path):
    observation = _joined(tmp_path)
    source = SourceReadResult(
        decisions=(observation.decision,),
        evaluations=(observation.evaluation,),
        available="false",
    )
    joined = join_observations(source, SourceReadResult(), SourceReadResult())

    assert joined[0].fields["source_available"] is False


def test_missing_graph_artifact_is_incomplete_even_when_top_level_context_complete(tmp_path):
    observation = _joined(tmp_path)
    audit = dict(observation.fields["audit"])
    audit["missing_nodes"] = ["Portfolio Manager"]
    observation = observation.__class__(
        observation.decision, observation.evaluation, observation.evidence,
        {**observation.fields, "audit": audit},
    )
    result = classify_observation(observation, _config(tmp_path))
    assert DatasetExclusionReason.DECISION_CONTEXT_INCOMPLETE in result.reasons


def test_missing_node_context_hashes_is_incomplete(tmp_path):
    observation = _joined(tmp_path)
    audit = dict(observation.fields["audit"])
    audit["node_context_hashes"] = {"Market Analyst": "only-one"}
    observation = observation.__class__(
        observation.decision, observation.evaluation, observation.evidence,
        {**observation.fields, "audit": audit},
    )
    result = classify_observation(observation, _config(tmp_path))
    assert DatasetExclusionReason.DECISION_CONTEXT_INCOMPLETE in result.reasons


def test_all_applicable_reasons_are_deterministically_ordered(tmp_path):
    observation = _joined(tmp_path, decision_fields={"normalization_status": "BAD", "decision_context_status": "INCOMPLETE", "action": "BAD", "source_decision_fingerprint": "wrong"})
    result = classify_observation(observation, _config(tmp_path))
    assert result.reasons == tuple(sorted(result.reasons, key=lambda r: tuple(DatasetExclusionReason).index(r)))
    assert {DatasetExclusionReason.UNTRUSTED_TIER, DatasetExclusionReason.DECISION_CONTEXT_INCOMPLETE, DatasetExclusionReason.NORMALIZATION_FAILED, DatasetExclusionReason.PROVENANCE_INCOMPLETE} <= set(result.reasons)


def test_duplicate_decision_keys_are_excluded(tmp_path):
    d = _decision()
    source = SourceReadResult(decisions=(d, d), evaluations=())
    joined = join_observations(source, SourceReadResult(), SourceReadResult())
    assert all(DatasetExclusionReason.DUPLICATE in classify_observation(x, _config(tmp_path)).reasons for x in joined)


def test_future_as_of_is_temporally_invalid(tmp_path):
    result = classify_observation(_joined(tmp_path), _config(tmp_path, as_of=datetime(2024, 1, 1, tzinfo=UTC)))
    assert DatasetExclusionReason.TEMPORAL_INVALID in result.reasons


def test_broker_reference_between_analysis_and_completion_is_temporally_invalid(tmp_path):
    result = classify_observation(
        _joined(
            tmp_path,
            decision_fields={
                "normalization_status": "NORMALIZED",
                "decision_context_status": "COMPLETE",
                "source_decision_fingerprint": "dfp",
                "source_run_id": "run1",
                "analysis_profile": "p",
                "analysis_timeframe": "1h",
                "decision_reference_timestamp": datetime(2025, 1, 1, 0, 0, 30, tzinfo=UTC),
                "decision_reference_status": "INVALID_TEMPORAL",
            },
        ),
        _config(tmp_path),
    )
    assert DatasetExclusionReason.TEMPORAL_INVALID in result.reasons


def test_broker_reference_after_completion_is_allowed(tmp_path):
    result = classify_observation(
        _joined(
            tmp_path,
            decision_fields={
                "normalization_status": "NORMALIZED",
                "decision_context_status": "COMPLETE",
                "source_decision_fingerprint": "dfp",
                "source_run_id": "run1",
                "analysis_profile": "p",
                "analysis_timeframe": "1h",
                "decision_reference_timestamp": datetime(2025, 1, 1, 0, 1, 30, tzinfo=UTC),
                "decision_reference_status": "AVAILABLE",
            },
        ),
        _config(tmp_path),
    )
    assert result.eligible


def test_explicit_unavailable_broker_reference_remains_allowed(tmp_path):
    result = classify_observation(
        _joined(
            tmp_path,
            decision_fields={
                "normalization_status": "NORMALIZED",
                "decision_context_status": "COMPLETE",
                "source_decision_fingerprint": "dfp",
                "source_run_id": "run1",
                "analysis_profile": "p",
                "analysis_timeframe": "1h",
                "decision_reference_timestamp": None,
                "decision_reference_status": "UNAVAILABLE",
            },
        ),
        _config(tmp_path),
    )
    assert result.eligible
    assert DatasetExclusionReason.TEMPORAL_INVALID not in result.reasons


def test_normal_evaluation_lifecycle_timestamps_after_analysis_are_allowed(tmp_path):
    observation = _joined(tmp_path)
    evaluation = EvaluationObservation(
        "d1",
        "ANALYSIS_SNAPSHOT",
        300,
        "COMPLETE",
        True,
        {
            "fingerprint": "efp",
            "created_at": datetime(2025, 1, 1, 0, 2, tzinfo=UTC),
            "evaluated_at": datetime(2025, 1, 1, 0, 6, tzinfo=UTC),
            "recovered_from_unavailable_at": datetime(2025, 1, 1, 0, 7, tzinfo=UTC),
            "entry_timestamp": datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
            "observation_timestamp": datetime(2025, 1, 1, 0, 6, tzinfo=UTC),
            "target_timestamp": datetime(2025, 1, 1, 0, 6, tzinfo=UTC),
            "provenance": {"evaluation_fingerprint": "efp", "source_decision_id": "d1"},
        },
    )
    observation = observation.__class__(observation.decision, evaluation, observation.evidence, observation.fields)
    result = classify_observation(observation, _config(tmp_path))
    assert result.eligible
    assert DatasetExclusionReason.TEMPORAL_INVALID not in result.reasons


def test_different_basis_or_horizon_evaluations_are_not_duplicates(tmp_path):
    decision = _decision(
        normalization_status="NORMALIZED",
        decision_context_status="COMPLETE",
        source_decision_fingerprint="dfp",
        source_run_id="run1",
        analysis_profile="p",
        analysis_timeframe="1h",
    )
    eval_300 = EvaluationObservation(
        "d1", "ANALYSIS_SNAPSHOT", 300, "COMPLETE", True,
        {"source_evaluation_fingerprint": "efp300"},
    )
    eval_600 = EvaluationObservation(
        "d1", "DECISION_REFERENCE", 600, "COMPLETE", True,
        {"source_evaluation_fingerprint": "efp600"},
    )
    source = SourceReadResult(decisions=(decision,), evaluations=(eval_300, eval_600))
    records = SourceReadResult(
        records=(
            {
                "experience_id": "e1", "source_decision_id": "d1", "source_run_id": "run1",
                "source_decision_fingerprint": "dfp", "trust": "TIER_A_HIGH_TRUST",
                "source_evaluation_fingerprints": {
                    "ANALYSIS_SNAPSHOT:300": "efp300", "DECISION_REFERENCE:600": "efp600",
                },
            },
        )
    )
    joined = join_observations(source, records, SourceReadResult())
    assert len(joined) == 1
    assert not joined[0].fields["duplicate"]


def test_identical_evaluation_key_is_duplicate(tmp_path):
    decision = _decision(source_decision_fingerprint="dfp")
    evaluations = tuple(
        EvaluationObservation("d1", "ANALYSIS_SNAPSHOT", 300, "COMPLETE", True,
                              {"source_evaluation_fingerprint": fp})
        for fp in ("efp-a", "efp-b")
    )
    joined = join_observations(
        SourceReadResult(decisions=(decision,), evaluations=evaluations),
        SourceReadResult(), SourceReadResult(),
    )
    assert joined[0].fields["duplicate"]


@pytest.mark.parametrize("field", ["decisions", "evaluations", "records", "audits"])
@pytest.mark.parametrize("value", [False, 0, "malformed", {"row": "malformed"}])
def test_join_rejects_malformed_adapter_collections(field, value):
    phase56 = SourceReadResult()
    phase8 = SourceReadResult()
    phase9 = SourceReadResult()
    target = {"decisions": phase56, "evaluations": phase56, "records": phase8, "audits": phase9}[field]
    object.__setattr__(target, field, value)

    with pytest.raises((TypeError, ValueError)):
        join_observations(phase56, phase8, phase9)


def test_phase8_snapshot_provenance_is_carried_to_source_evaluation(tmp_path):
    decision = _decision(source_decision_fingerprint="dfp")
    evaluation = EvaluationObservation(
        "d1", "ANALYSIS_SNAPSHOT", 300, "COMPLETE", True,
        {"source_evaluation_fingerprint": "efp"},
    )
    record = {
        "source_decision_id": "d1",
        "source_run_id": "run1",
        "source_decision_fingerprint": "dfp",
        "evaluation_snapshots": ({
            "evaluation_basis": "ANALYSIS_SNAPSHOT",
            "horizon_seconds": 300,
            "fingerprint": "efp",
            "provenance": {"evaluation_fingerprint": "efp"},
        },),
    }
    joined = join_observations(
        SourceReadResult(decisions=(decision,), evaluations=(evaluation,)),
        SourceReadResult(records=(record,)),
        SourceReadResult(),
    )
    assert joined[0].evaluation.fields["phase8_evaluation_fingerprint"] == "efp"
    assert joined[0].evaluation.fields["provenance"]["source_decision_id"] == "d1"


def test_phase8_recovered_snapshot_pairs_with_complete_state_not_old_unavailable(tmp_path):
    decision = _decision(source_decision_fingerprint="dfp")
    evaluation = EvaluationObservation(
        "d1", "ANALYSIS_SNAPSHOT", 300, "COMPLETE", True,
        {"source_evaluation_fingerprint": "efp", "evaluated_at": datetime(2025, 1, 1, 0, 10, tzinfo=UTC)},
    )
    record = {
        "source_decision_id": "d1", "source_run_id": "run1", "source_decision_fingerprint": "dfp",
        "evaluation_snapshots": (
            {"evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300, "fingerprint": "unavailable", "observed_at": datetime(2025, 1, 1, 0, 3, tzinfo=UTC), "evaluation": {"evaluation_status": "DATA_UNAVAILABLE"}},
            {"evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300, "fingerprint": "complete", "observed_at": datetime(2025, 1, 1, 0, 9, tzinfo=UTC), "evaluation": {"evaluation_status": "COMPLETE"}},
        ),
    }
    joined = join_observations(
        SourceReadResult(decisions=(decision,), evaluations=(evaluation,)),
        SourceReadResult(records=(record,)), SourceReadResult(),
    )
    assert joined[0].evaluation.fields["phase8_evaluation_fingerprint"] == "complete"
