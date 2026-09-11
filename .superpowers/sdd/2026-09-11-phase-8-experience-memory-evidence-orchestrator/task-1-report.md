# Phase 8 Task 1 implementation report

## Scope

Implemented only the Phase 8 public contracts/configuration/version boundary.
The package remains independent of MT5, execution, forex, TradingAgents, and
Phase 7 writers.

## TDD evidence

- RED: `pytest tests/test_experience_models.py tests/test_experience_config.py -q`
  failed during collection with `ModuleNotFoundError: No module named
  'tradingagents.experience'`.
- GREEN: `pytest tests/test_experience_models.py tests/test_experience_config.py -q`
  passed: 9 tests.
- Additional checks: `python -m compileall -q tradingagents/experience` passed;
  `git diff --check` passed.

## Files changed

- `tradingagents/experience/__init__.py`
- `tradingagents/experience/models.py`
- `tradingagents/experience/config.py`
- `tradingagents/experience/errors.py`
- `tests/test_experience_models.py`
- `tests/test_experience_config.py`
- `pyproject.toml` (direct `numpy>=1.26` dependency)

## Contract notes and risks

Models are frozen dataclasses with deterministic JSON conversion and UTC-aware
timestamp validation. `training_eligible` is accepted only within provenance;
`EvidenceBundle` contains no recommendation/order/portfolio fields. Historical
actions remain evidence on hits and are not similarity dimensions. Typed error
classes are defined for later task boundaries. No catalog, ingestion, feature,
query, outcome, orchestrator, or CLI behavior was implemented.

## Commit

Implementation commit SHA: `5905b863343f2efaf5218a821e0ca57f81246bca`.

## Review round 1 fixes

Added `EvaluationStatus.INELIGIBLE`; deep-froze nested mappings and copied
payloads; sorted set serialization deterministically; rejected reserved training
eligibility keys outside provenance; and validated hit timestamp values as
timezone-aware UTC.

- RED: four new reviewer-contract tests failed as expected.
- GREEN: focused suite passed: 13 tests.
- Compileall and `git diff --check` passed.
- Fix commit: recorded by the follow-up commit containing this report update.

## Review round 2 fixes

Deep-froze `ExperienceQuery.market_state`, including nested mappings, and
applied the provenance-only reserved-key validation at the query boundary.
The focused suite passed 14 tests, with compileall and `git diff --check` also
passing.
