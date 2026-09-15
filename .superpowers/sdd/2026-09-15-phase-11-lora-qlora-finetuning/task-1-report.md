# Task 1 report — contracts, statuses, and configuration

Implemented the Phase 11 dependency-free contract layer in
`tradingagents/finetuning`:

- lazy package exports and typed Phase 11 errors;
- version constants and closed `Phase11Status` values;
- frozen dataclass contracts for training, adapter, dataset, SFT, manifest,
  metrics, training, and validation data;
- deterministic compact JSON and SHA-256 helpers;
- recursive rejection of sensitive fields/values and non-finite numbers;
- bounds and enum/version validation for configuration and reports.

Verification:

- `pytest -q tests/test_phase11_models.py` — 4 passed
- `ruff check tradingagents/finetuning tests/test_phase11_models.py` — passed

No torch, Transformers, or other training dependencies are imported by the
contract modules. No other phase or documentation files were changed except
this task report.

Review follow-up: persisted metrics/reports now carry and validate the run
manifest version; mutable revisions (`latest`, `main`, `master`, `HEAD`, and
`default`) are rejected; all relevant numeric fields reject non-finite values;
invalid report message containers/types raise `ContractError`; nested sets
are frozen in canonical order; and `DatasetBinding` validates the Phase 10
dataset schema version.

Numeric validation follow-up: integer fields reject booleans and wrong types,
float fields reject non-finite values, and regression tests cover training
configuration, dataset/prepared-manifest counts, and metrics.

Final strict-integer follow-up: `TrainingConfig.seed` and `LoraConfig.r` /
`alpha` now reject booleans and non-integer values, with focused regression
tests.
