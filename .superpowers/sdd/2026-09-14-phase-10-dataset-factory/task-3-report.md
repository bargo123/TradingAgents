# Task 3 report — join and eligibility classifier

Implemented `tradingagents/datasets/eligibility.py` and focused tests in
`tests/test_dataset_eligibility.py`.

## Behavior

- Joins Phase 5/6 decisions and evaluations with Phase 8 records and Phase 9
  audit rows by decision identity, source run, and source decision fingerprint.
- Preserves ambiguous joins as duplicate metadata and classifies them closed,
  without inventing missing facts.
- Applies deterministic, closed `DatasetExclusionReason` ordering for trust,
  context, temporal/as-of/future leakage, normalization, citation/evidence,
  provenance/schema, outcome, duplicate, and source availability failures.
- Accepts only Tier A, normalized, complete-context observations with an exact
  complete/source-context-eligible basis+horizon outcome and complete source
  provenance.
- Performs no writes, network calls, LLM calls, MT5 calls, or prompt handling.

## Verification

- TDD RED: initial focused test collection failed because the production module
  did not exist (`ModuleNotFoundError`).
- TDD GREEN: `pytest -q tests/test_dataset_eligibility.py tests/test_dataset_models.py tests/test_dataset_sources.py` — 28 passed.
- Dataset suite: `pytest -q` over all `tests/test_dataset_*.py` — 28 passed.
- `ruff check tradingagents/datasets/eligibility.py tests/test_dataset_eligibility.py` — clean.
- `python -m compileall -q tradingagents/datasets` — clean.
- `git diff --check` — clean.

## Review round 1 fixes

- Recomputed trust via Phase 8 `extract_market_state` and `classify_trust`; stored
  Tier A is not trusted by itself.
- Enforced exact decision/run/fingerprint identity joins and strict audit field
  presence/reference partition checks.
- Required provenance, source evaluation fingerprints, feature extractor/version
  fields, and malformed/future outcome timestamps to fail closed.

## Review round 2 fixes

- Phase 5/6 evaluations are matched by authoritative decision ID and their
  exact basis/horizon; optional run/fingerprint fields are enforced only when
  present, with collisions excluded.
- Future horizon targets are permitted; observation/entry/evaluated/created,
  recovery, and other authoritative timestamps are validated for UTC shape,
  ordering, as-of, and leakage.
- Provenance fingerprints are checked for internal decision/evaluation
  consistency rather than mere non-empty values.

## Review round 3 fixes

- Evaluation timestamp validation now uses the authoritative `created_at`,
  `evaluated_at`, and `recovered_from_unavailable_at` names while preserving
  future `target_timestamp` horizon behavior.
- Evaluation fingerprints are compared across the actual evaluation row,
  Phase 8 source-evaluation map, and evaluation provenance/source decision ID.

## Review round 4 fixes

- The public Phase 5/6 adapter now derives and carries
  `source_decision_fingerprint` and `source_evaluation_fingerprint` with the
  existing Phase 8 identity helpers, computed from the complete normalized
  source rows before bounded fields are emitted.
- Joined evaluations carry the matching Phase 8 retained snapshot fingerprint
  and provenance (including the authoritative source decision ID), while
  retaining the Phase 5/6-derived source evaluation fingerprint for exact
  comparison; the legacy optional record map remains supported.
- Eligibility permits normal post-analysis evaluation lifecycle and
  observation timestamps (`created_at`, `evaluated_at`,
  `recovered_from_unavailable_at`, and future horizon observations), while
  retaining UTC shape, as-of, entry/target, and ordering guards.
- Join metadata retains all basis/horizon evaluations as JSON-safe facts and
  marks duplicates only for identical `(decision_id, evaluation_basis,
  horizon_seconds)` keys; a configured classifier selects its exact key.
- Added authoritative public-path regressions for adapter fingerprints,
  post-analysis evaluation timestamps, distinct basis/horizon rows, and
  identical evaluation-key duplicates, plus Phase 8 snapshot provenance
  enrichment.

## Review round 4 verification

- TDD RED: the new regression collection failed before implementation with a
  missing source fingerprint, `TEMPORAL_INVALID`, and false duplicate result.
- TDD GREEN and focused verification:
  `pytest -q tests/test_dataset_eligibility.py tests/test_dataset_sources.py tests/test_dataset_models.py`
  — 33 passed.
- `ruff check tradingagents/datasets/eligibility.py
  tradingagents/datasets/sources.py tests/test_dataset_eligibility.py
  tests/test_dataset_sources.py` — clean.
- `python -m compileall -q tradingagents cli scripts` — clean.
- `git diff --check` — clean (only expected Git LF/CRLF warnings).

## Review round 5 fixes

- Broker reference timestamps are now required to follow both analysis and
  decision completion; a reference captured between those events is classified
  as `TEMPORAL_INVALID`, matching Phase 6 `INVALID_TEMPORAL` semantics. An
  explicitly unavailable reference with no timestamp remains valid for this
  gate.
- The Phase 5/6 adapter now recomputes decision and evaluation fingerprints
  from normalized source facts even when stale or forged identity columns are
  present. Derived identity columns are excluded from the helper input, while
  Phase 8 identity joins continue to use the authoritative values.
- Added public regressions for reference ordering, unavailable references, and
  bad persisted identity columns, including a Phase 8 join-match check.

## Review round 5 verification

- TDD RED: the new regressions failed before implementation (missing temporal
  ordering gate and adapter fingerprints returned `BAD`).
- TDD GREEN: `pytest -q tests/test_dataset_eligibility.py
-  tests/test_dataset_sources.py tests/test_dataset_models.py` — 38 passed.
- Dataset-focused suite: `pytest -q tests -k dataset` — 38 passed, 1455
  deselected.
- `ruff check tradingagents/datasets/eligibility.py
  tradingagents/datasets/sources.py tests/test_dataset_eligibility.py
  tests/test_dataset_sources.py` — clean.
- `python -m compileall -q tradingagents cli scripts` — clean.
- `git diff --check` — clean (only expected Git LF/CRLF warnings).
