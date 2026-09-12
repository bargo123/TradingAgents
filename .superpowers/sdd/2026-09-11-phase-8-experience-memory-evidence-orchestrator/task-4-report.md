# Phase 8 Task 4 report — pre-decision features and trust

## RED evidence

Ran `pytest tests/test_experience_features.py tests/test_experience_trust.py -q` before implementation. Collection failed with the expected `ModuleNotFoundError` because `tradingagents.experience.features` and `trust` did not yet exist.

## GREEN evidence

Ran the focused feature/trust tests and adjacent Phase 8 contracts:

```text
28 passed in 1.01s
```

Also ran `python -m compileall -q tradingagents/experience` and `git diff --check`; both completed cleanly.

## Delivered files

- `tradingagents/experience/features.py` — immutable V1 market vector, exact ordered fields, masks, source paths/missing reasons, UTC encoding, cohort and deterministic fingerprint, typed validation.
- `tradingagents/experience/trust.py` — deterministic Tier A/B/C classification and policy reason codes.
- `tests/test_experience_features.py`
- `tests/test_experience_trust.py`
- `tradingagents/experience/__init__.py` exports the new public APIs.

The extractor consumes only decision metadata and persisted `snapshot_json`; action/outcome fields are not vector inputs. It does not open source databases, invoke Luna/LLMs, or modify stock, forex, MT5, or Phase 7 behavior.

## Commit

`4043a4582ebf76771547e5a4aed5c373fbfdb2f4` (`feat: add phase 8 features and trust policy`)

## Risks / follow-up boundaries

- Missing numeric features are represented by `NaN` plus a false mask and remain available for Tier B when the minimum usable dimensions are present; downstream normalization/query tasks must honor the mask.
- Feature extraction validates persisted point/digits and UTC timestamps, but importer/catalog wiring and richer provenance reconciliation belong to later tasks.

## Reviewer-fix RED/GREEN evidence

Added regression tests for malformed non-mapping `snapshot_json.features`/`quote`, unknown direction values, and non-finite required quote fields. The first run failed on all three gaps. After the fixes:

```text
pytest tests/test_experience_features.py tests/test_experience_trust.py -q
9 passed
pytest tests/test_experience_*.py -q
44 passed
```

`features.py` now emits typed schema/direction diagnostics (and raises the typed extraction error for malformed nested structures); `trust.py` treats quote diagnostics as Tier C. Compilation and `git diff --check` remain clean.

## Reviewer-fix round 2

Added RED tests for non-finite numeric feature values and malformed individual timeframe sections. Both failed before the fix. The extractor now emits typed diagnostics and fails closed for `NaN`/`Infinity` numeric inputs and non-mapping `M1`/`M5`/`M15`/`H1` sections.

Final verification:

```text
pytest tests/test_experience_features.py tests/test_experience_trust.py -q: 11 passed
pytest tests/test_experience_*.py -q: 46 passed
compileall: clean
git diff --check: clean
```
