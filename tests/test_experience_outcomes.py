from datetime import datetime

from tradingagents.experience.models import OutcomeStatsRequest, TrustTier
from tradingagents.experience.outcomes import OutcomeStatsCalculator


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _record(experience_id="exp1", *, trust=TrustTier.TIER_A_HIGH_TRUST, snapshots=()):
    return {
        "experience_id": experience_id,
        "trust": trust,
        "outcome_snapshots": tuple(snapshots),
    }


def _complete(fp="fp1", *, basis="ANALYSIS_SNAPSHOT", horizon=300, **extra):
    return {
        "fingerprint": fp,
        "evaluation_basis": basis,
        "horizon_seconds": horizon,
        "evaluation_status": "COMPLETE",
        "source_context_eligible": 1,
        "observation_timestamp": "2026-01-01T12:00:00Z",
        "evaluated_at": "2026-01-01T13:00:00Z",
        "selected_action": "BUY",
        "selected_action_net_points": 10.0,
        "buy_net_points": 10.0,
        "sell_net_points": -5.0,
        "hold_opportunity_cost_points": 0.0,
        **extra,
    }


def calculator():
    rows = (
        _record(snapshots=(_complete(),)),
        _record(
            "hold-exp",
            snapshots=(
                _complete(
                    "hold",
                    selected_action="HOLD",
                    selected_action_net_points=0.0,
                    buy_net_points=4.0,
                    sell_net_points=-3.0,
                    hold_opportunity_cost_points=0.0,
                ),
            ),
        ),
        _record(
            "exp-recovered",
            snapshots=(
                _complete(
                    "recovered",
                    recovered_from_unavailable_at="2026-01-02T15:00:00Z",
                    evaluated_at="2026-01-02T15:00:00Z",
                ),
            ),
        ),
        _record(
            "exp-with-prior",
            snapshots=(
                {
                    **_complete(
                        "prior",
                        evaluation_status="DATA_UNAVAILABLE",
                        evaluated_at="2026-01-02T13:00:00Z",
                        unavailable_reason="NO_QUOTES",
                    )
                },
                _complete(
                    "later",
                    evaluated_at="2026-01-02T15:00:00Z",
                    recovered_from_unavailable_at="2026-01-02T15:00:00Z",
                ),
            ),
        ),
    )
    return OutcomeStatsCalculator(rows)


def test_training_null_does_not_exclude_complete_descriptive_row():
    result = calculator().calculate(OutcomeStatsRequest(("exp1",), "ANALYSIS_SNAPSHOT", 300))
    assert result.eligible_count == 1


def test_complete_snapshot_with_invalid_selected_action_is_excluded():
    row = _complete("bad-action", selected_action="WAIT")
    result = OutcomeStatsCalculator((_record("bad-action", snapshots=(row,)),)).calculate(
        OutcomeStatsRequest(("bad-action",), "ANALYSIS_SNAPSHOT", 300)
    )

    assert result.eligible_count == 0
    assert result.excluded_counts["SELECTED_ACTION_INVALID"] == 1


def test_complete_snapshot_with_nonfinite_selected_action_value_is_excluded():
    row = _complete("bad-selected-value", selected_action_net_points=float("nan"))
    result = OutcomeStatsCalculator(
        (_record("bad-selected-value", snapshots=(row,)),)
    ).calculate(OutcomeStatsRequest(("bad-selected-value",), "ANALYSIS_SNAPSHOT", 300))

    assert result.eligible_count == 0
    assert result.excluded_counts["REQUIRED_FIELD_MISSING_OR_NONFINITE"] == 1


def test_basis_and_horizon_are_exact():
    result = calculator().calculate(OutcomeStatsRequest(("exp1",), "DECISION_REFERENCE", 900))
    assert result.requested_basis == "DECISION_REFERENCE"
    assert result.requested_horizon_seconds == 900
    assert result.excluded_counts["BASIS_OR_HORIZON_MISMATCH"] == 1


def test_hold_uses_counterfactual_opportunity_cost():
    result = calculator().calculate(OutcomeStatsRequest(("hold-exp",), "ANALYSIS_SNAPSHOT", 300))
    assert result.hold.opportunity_cost_points == (0.0,)
    assert result.hold.normal_win_rate is None


def test_as_of_excludes_late_recovery_without_prior_snapshot():
    request = OutcomeStatsRequest(
        ("exp-recovered",), "ANALYSIS_SNAPSHOT", 300, as_of=utc("2026-01-02T14:00:00Z")
    )
    result = calculator().calculate(request)
    assert result.excluded_counts["EVALUATION_NOT_YET_AVAILABLE"] == 1
    assert result.excluded_counts.get("DATA_UNAVAILABLE", 0) == 0


def test_recovered_complete_participates_after_recovery():
    request = OutcomeStatsRequest(
        ("exp-recovered",), "ANALYSIS_SNAPSHOT", 300, as_of=utc("2026-01-02T16:00:00Z")
    )
    assert calculator().calculate(request).eligible_count == 1


def test_genuine_prior_unavailable_snapshot_is_used_before_recovery():
    request = OutcomeStatsRequest(
        ("exp-with-prior",), "ANALYSIS_SNAPSHOT", 300, as_of=utc("2026-01-02T14:00:00Z")
    )
    result = calculator().calculate(request)
    assert result.excluded_counts["DATA_UNAVAILABLE"] == 1
    assert result.excluded_counts.get("EVALUATION_NOT_YET_AVAILABLE", 0) == 0


def test_prior_unavailable_without_market_observation_timestamp_is_still_historical():
    """A no-quote snapshot is known by catalog observation, not a market tick."""

    record = _record(
        "exp-with-no-quote-timestamp",
        snapshots=(
            _complete(
                "prior-unavailable",
                evaluation_status="DATA_UNAVAILABLE",
                observation_timestamp=None,
                evaluated_at="2026-01-02T13:00:00Z",
                observed_at="2026-01-02T13:00:00Z",
                unavailable_reason="NO_QUOTES",
            ),
            _complete(
                "recovered",
                evaluated_at="2026-01-02T15:00:00Z",
                observed_at="2026-01-02T15:00:00Z",
                recovered_from_unavailable_at="2026-01-02T15:00:00Z",
            ),
        ),
    )
    result = OutcomeStatsCalculator((record,)).calculate(
        OutcomeStatsRequest(
            ("exp-with-no-quote-timestamp",),
            "ANALYSIS_SNAPSHOT",
            300,
            as_of=utc("2026-01-02T14:00:00Z"),
        )
    )

    assert result.excluded_counts["DATA_UNAVAILABLE"] == 1
    assert result.excluded_counts.get("EVALUATION_NOT_YET_AVAILABLE", 0) == 0


def test_malformed_mapping_snapshot_horizon_fails_closed_without_crashing():
    record = {
        "experience_id": "exp-with-malformed-horizon",
        "trust": TrustTier.TIER_A_HIGH_TRUST,
        "outcome_evidence_by_basis_horizon": {
            "ANALYSIS_SNAPSHOT/not-an-integer": {
                "evaluation_status": "COMPLETE",
            }
        },
    }

    result = OutcomeStatsCalculator((record,)).calculate(
        OutcomeStatsRequest(("exp-with-malformed-horizon",), "ANALYSIS_SNAPSHOT", 300)
    )

    assert result.excluded_counts["EVALUATION_NOT_AVAILABLE"] == 1


def test_malformed_snapshot_entries_fail_closed_without_crashing():
    record = _record(
        "exp-with-malformed-snapshot",
        snapshots=(None, "not-a-mapping", 42),
    )

    result = OutcomeStatsCalculator((record,)).calculate(
        OutcomeStatsRequest(("exp-with-malformed-snapshot",), "ANALYSIS_SNAPSHOT", 300)
    )

    assert result.eligible_count == 0
    assert result.excluded_counts["EVALUATION_NOT_AVAILABLE"] == 1


def test_malformed_record_trust_fails_closed_without_crashing():
    record = _record(
        "exp-with-malformed-trust",
        trust="NOT_A_TRUST_TIER",
        snapshots=(_complete(),),
    )

    result = OutcomeStatsCalculator((record,)).calculate(
        OutcomeStatsRequest(("exp-with-malformed-trust",), "ANALYSIS_SNAPSHOT", 300)
    )

    assert result.eligible_count == 0
    assert result.excluded_counts["TRUST_TIER_INVALID"] == 1


def test_historical_selection_uses_phase8_observation_time() -> None:
    """Source timestamps cannot make an unobserved catalog state visible."""

    record = _record(
        "observed-late",
        snapshots=(
            _complete(
                "prior",
                evaluation_status="DATA_UNAVAILABLE",
                evaluated_at="2026-01-02T10:01:00Z",
                observation_timestamp="2026-01-02T10:00:00Z",
                observed_at="2026-01-02T13:00:00Z",
                unavailable_reason="NO_QUOTES",
            ),
            _complete(
                "recovered",
                evaluated_at="2026-01-02T10:02:00Z",
                observation_timestamp="2026-01-02T10:00:00Z",
                observed_at="2026-01-02T15:00:00Z",
                recovered_from_unavailable_at="2026-01-02T15:00:00Z",
            ),
        ),
    )

    before_import = OutcomeStatsCalculator((record,)).calculate(
        OutcomeStatsRequest(
            ("observed-late",),
            "ANALYSIS_SNAPSHOT",
            300,
            as_of=utc("2026-01-02T12:00:00Z"),
        )
    )
    assert before_import.excluded_counts["EVALUATION_NOT_YET_AVAILABLE"] == 1
    assert before_import.excluded_counts.get("DATA_UNAVAILABLE", 0) == 0

    after_prior_import = OutcomeStatsCalculator((record,)).calculate(
        OutcomeStatsRequest(
            ("observed-late",),
            "ANALYSIS_SNAPSHOT",
            300,
            as_of=utc("2026-01-02T14:00:00Z"),
        )
    )
    assert after_prior_import.excluded_counts["DATA_UNAVAILABLE"] == 1
    assert after_prior_import.excluded_counts.get("EVALUATION_NOT_YET_AVAILABLE", 0) == 0


def test_directional_counterfactuals_include_hold_and_expose_distribution_summary():
    result = calculator().calculate(OutcomeStatsRequest(("hold-exp",), "ANALYSIS_SNAPSHOT", 300))
    assert result.buy.net_points == (4.0,)
    assert result.sell.net_points == (-3.0,)
    assert result.buy.positive_net_count == 1
    assert result.sell.negative_net_count == 1
    assert result.buy.mean_net_points == 4.0
    assert result.sell.median_net_points == -3.0


def test_mfe_mae_and_hold_missed_opportunity_fields_are_reported():
    row = _complete(
        "rich",
        selected_action="HOLD",
        buy_net_points=5.0,
        sell_net_points=-2.0,
        hold_opportunity_cost_points=5.0,
        buy_mfe_points=8.0,
        buy_mae_points=-3.0,
        sell_mfe_points=2.0,
        sell_mae_points=-4.0,
        best_counterfactual_action="BUY",
    )
    calc = OutcomeStatsCalculator((_record("rich", snapshots=(row,)),))
    result = calc.calculate(OutcomeStatsRequest(("rich",), "ANALYSIS_SNAPSHOT", 300))
    assert result.buy.mfe_points == (8.0,)
    assert result.sell.mae_points == (-4.0,)
    assert result.hold.missed_buy_opportunity_points == (5.0,)
    assert result.hold.best_counterfactual_counts == {"BUY": 1}


def test_eligible_complete_status_is_not_an_exclusion():
    result = calculator().calculate(OutcomeStatsRequest(("exp1",), "ANALYSIS_SNAPSHOT", 300))
    assert result.exclusions_by_status == {}


def test_hold_preserves_tie_best_counterfactual_and_directional_quantiles():
    rows = [
        _complete(
            "tie-1",
            selected_action="HOLD",
            buy_net_points=1.0,
            sell_net_points=1.0,
            hold_opportunity_cost_points=1.0,
            best_counterfactual_action="TIE",
            buy_mfe_points=2.0,
            buy_mae_points=-1.0,
            sell_mfe_points=3.0,
            sell_mae_points=-2.0,
        ),
        _complete(
            "tie-2",
            selected_action="HOLD",
            buy_net_points=2.0,
            sell_net_points=2.0,
            hold_opportunity_cost_points=2.0,
            best_counterfactual_action="TIE",
            buy_mfe_points=4.0,
            buy_mae_points=-3.0,
            sell_mfe_points=5.0,
            sell_mae_points=-4.0,
        ),
    ]
    records = tuple(_record(f"tie-{i}", snapshots=(row,)) for i, row in enumerate(rows, 1))
    result = OutcomeStatsCalculator(records).calculate(
        OutcomeStatsRequest(("tie-1", "tie-2"), "ANALYSIS_SNAPSHOT", 300)
    )
    assert result.hold.best_counterfactual_counts == {"TIE": 2}
    assert result.buy.mfe_quantiles["p50"] == 3.0
    assert result.sell.mae_quantiles["p95"] == -2.1
