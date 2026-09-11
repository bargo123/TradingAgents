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
