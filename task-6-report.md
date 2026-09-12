# Phase 8 Task 6 implementation report

## Scope

Implemented the exact NumPy similarity index and read-only experience query
service only. It consumes caller-provided vectors, records, and
`SimilarityProfileV1`; it does not open source databases or MT5.

## TDD evidence

- RED: `pytest tests/test_experience_similarity.py tests/test_experience_query.py -q` failed during collection because the requested modules did not exist.
- GREEN: focused suite — `5 passed`.
- Regression: all `tests/test_experience_*.py` — `57 passed`.
- `python -m compileall -q tradingagents/experience` — passed.
- `git diff --check` — passed.

After implementation:

## Behavior delivered

- Masked float32 weighted RMS with profile normalization, clipping, minimum
  eight dimensions and 50% overlap, score `1 / (1 + distance)`.
- Deterministic ordering by distance, analysis snapshot timestamp, and ID.
- Query gates for symbol/profile/timeframe, cohort/schema compatibility, trust
  tiers, provenance/acceptance, strict as-of analysis/completion, and current
  versus historical tombstones. Tier C is rejected from numeric similarity.
- Result envelope includes normalization fingerprint, generation ID, candidate
  count, exclusion counts, and immutable `ExperienceHit` values.

## Files

- `tradingagents/experience/similarity.py`
- `tradingagents/experience/query.py`
- `tests/test_experience_similarity.py`
- `tests/test_experience_query.py`

## Commit

Recorded in the Git commit created for this task.

## Risks / follow-up

The service intentionally uses dependency seams for feature projections and
profiles. Import, outcome statistics, orchestrator, CLI, smoke, and forbidden
integrations remain outside this task.

## Reviewer-fix evidence

- Added RED tests for explicit feature schema/extractor mismatch, timezone-safe
  missing-timestamp ordering, and invalid naive timestamps under `as_of`.
- Focused suite after fixes: `8 passed`.
- Full experience regression after fixes: `60 passed`.
- `python -m compileall -q tradingagents/experience` and `git diff --check`
  passed.

## Reviewer-fix round 2

- Malformed timestamp strings are now treated as invalid and excluded by the
  strict `as_of` gate.
- Non-default trust-tier or cutoff policies rebuild a query-local profile from
  the gated projections and use its matching normalization fingerprint.
- Focused suite: `10 passed`; full experience regression: `62 passed`.

## Whole-branch review fix

- `action_filter` is applied after identity, trust, provenance, tombstone, and
  as-of gates; it only changes the candidate population, never distance/score.
- Query market-state cohort and ordered feature names are validated against the
  selected normalization profile before numeric search.
- Focused suite: `13 passed`; full experience regression: `119 passed`.
- `python -m compileall -q tradingagents/experience` and `git diff --check`
  passed.

