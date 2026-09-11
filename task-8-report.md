# Phase 8 Task 8 implementation report

## Scope

Implemented the read-only `OutcomeStatsCalculator` in
`tradingagents/experience/outcomes.py` and focused contract tests. Statistics
use only requested retained evaluation snapshots and record trust. Exact
basis/horizon, trust, source-context, status, finite numeric-field, UTC
availability, and historical `as_of` gates are applied. Historical selection
chooses one observed snapshot per experience/basis/horizon; recovery without a
retained prior snapshot is `EVALUATION_NOT_YET_AVAILABLE`, while a retained
prior `DATA_UNAVAILABLE` state remains visible at earlier cutoffs.
`training_eligible` is not an inclusion gate. BUY/SELL counterfactuals and HOLD
opportunity cost are separate, with reason/status/tier exclusion maps.

The catalog snapshot read contract preserves its append-observation timestamp.
`OutcomeStatistics` gained immutable directional, HOLD, denominator, request,
and exclusion fields while preserving existing defaults.

## TDD and verification

- RED: `pytest tests/test_experience_outcomes.py -q` failed at collection
  because `tradingagents.experience.outcomes` did not exist.
- GREEN: `pytest tests/test_experience_outcomes.py -q` — 6 passed.
- Regression: all Phase 8 experience tests — 74 passed.
- `python -m compileall -q tradingagents/experience/outcomes.py tradingagents/experience/models.py` — passed.
- `git diff --check` — passed.
- Ruff was unavailable as a command; no Ruff result is claimed.

## Boundaries

No orchestrator, CLI, smoke harness, trading, MT5, model, execution, training,
or Phase 7 ingestion code was implemented or modified.
