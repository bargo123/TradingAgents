# Phase 8 Task 11 report

## Scope

Added deterministic, offline leakage, isolation, and retrieval-quality gates.
The query adapter now carries existing record provenance/schema/action fields onto
returned `ExperienceHit` values so the public provenance gate checks API output.
No Phase 8 CLI, smoke, MT5, execution, training, or external integration code was
added.

## Verification

- Focused Task 11 tests: `20 passed` (`test_experience_leakage.py`,
  `test_experience_quality.py`, `test_experience_isolation.py`).
- Complete Phase 8 experience regression: `110 passed`.
- `python -m compileall -q tradingagents/experience tests/test_experience_leakage.py tests/test_experience_quality.py tests/test_experience_isolation.py`: passed.
- `git diff --check`: passed.
- `ruff check ...`: not run; Ruff is not installed in this environment.
- Repository-wide `pytest`: blocked during collection by unrelated missing
  dependencies in the checkout (`langchain_core`, `requests`, `typer`, and
  other optional project dependencies).

## Gates covered

- Zero counts for outcome, action, candidate, completion, normalization, and
  evaluation leakage channels.
- Behavioral candidate-time, completion-time, normalization-future, and
  evaluation-future cutoff checks.
- Outcome/action mutation invariance of default similarity.
- Historical cutoff invariance against extreme future rows.
- Exact symbol/profile/timeframe and trust-tier compatibility filtering.
- Tier A-only normalization and Tier C numeric-normalization rejection.
- Historical tombstone behavior after alias removal.
- Recovered evaluations with and without a prior observed unavailable snapshot.
- Deterministic Recall@K/MRR fixture, stable repeated ordering, query
  normalization provenance, and bounded latency/memory assertions.
- AST import scan over `tradingagents.experience` for forbidden integration
  dependencies.
