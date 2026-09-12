# Phase 8 Task 5 report — cohort-scoped robust normalization

## Scope

Implemented only Task 5: deterministic, cohort-local `NormalizationCohortV1`,
`FeatureRow`, `SimilarityProfileV1`, profile construction, and query
normalization fingerprints. No similarity service, importer, outcome code,
orchestrator, CLI, or external integration was added.

## TDD evidence

- RED: `pytest tests/test_experience_normalization.py -q` failed during
  collection with `ModuleNotFoundError: tradingagents.experience.normalization`.
- GREEN: `pytest tests/test_experience_normalization.py -q` — 6 passed.
- Regression: all Phase 8 experience tests (`tests/test_experience_*.py`) — 52
  passed.
- `python -m compileall -q tradingagents/experience` — passed.
- `git diff --check` — passed (only normal Git line-ending warnings).

## Implementation notes

Profiles filter exact symbol/profile/timeframe/schema/extractor cohorts and
Tier A/B populations. Current populations require a CURRENT alias; historical
populations require strict analysis and completion timestamps before `as_of`
and retain accepted evidence regardless of later alias removal. Robust scales
use IQR, then converted MAD, then a persisted per-feature fallback. Missing
values are excluded by mask policy, feature weights and clipping are explicit,
and fingerprints include cohort, tiers, cutoff, policy, and population data.

## Commit

`75dc125a23568b639da9ff7a3ee1f1ea9122ea2d`

## Risks

The implementation accepts both mapping rows and the local `FeatureRow`
adapter. Catalog/importer projection wiring is intentionally deferred to the
later tasks. Persisted-profile lookup is also intentionally outside Task 5's
pure profile-builder API.
