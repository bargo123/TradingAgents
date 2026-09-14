# Phase 10 Task 1 report

## Scope

Added the `tradingagents.datasets` package contracts and typed errors only.
The contracts are frozen dataclasses, normalize mappings to immutable
`MappingProxyType` values, validate bounded identifiers and UTC timestamps,
reject unsafe/non-finite values, and provide deterministic JSON serialization.
`DatasetConfig` canonicalizes paths and rejects output roots overlapping source
roots. The closed exclusion-reason enum matches the approved design.

## TDD evidence

The new tests were run before implementation and failed at collection because
`tradingagents.datasets.models` did not exist. After the smallest
implementation, the same tests passed.

## Verification

- `pytest -q tests/test_dataset_models.py`: 3 passed
- `ruff check tradingagents/datasets`: passed
- `pytest -q`: 1,446 passed, 6 skipped, 6 pre-existing Phase 9 integration
  failures (spawned evidence integration timed out/fell back); no failures
  referenced the new dataset package.
- No source readers, writers, network, LLM, MT5, or execution behavior was
  added or changed.

## Files

- `tradingagents/datasets/__init__.py`
- `tradingagents/datasets/models.py`
- `tradingagents/datasets/errors.py`
- `tests/test_dataset_models.py`

## Round 1 review fix

Added recursive sensitive/no-CoT key and value rejection, closed status and
reason validation, immutable nested references and report collections, and
bounded type checks for source/config/split/report fields.

Verification command/output:

```text
ruff check tradingagents/datasets tests/test_dataset_models.py
All checks passed!
pytest -q tests/test_dataset_models.py
7 passed
```
