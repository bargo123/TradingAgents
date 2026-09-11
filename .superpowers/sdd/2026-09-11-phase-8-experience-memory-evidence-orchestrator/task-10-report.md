# Task 10 report — explicit Experience CLI

## Scope

Implemented only the Phase 8 Task 10 CLI surface. Added the `experience`
console entry point, argparse routing for import/rebuild/status/list/show/
similar/stats/quarantine/evidence, stable compact JSON serialization, exact
symbol/profile/timeframe/trust/as-of/basis/horizon options, and bounded typed
error output. Metadata commands avoid creating a missing artifact catalog.
Numeric similarity uses Phase 8 services only; the market-state evidence path
does not construct Phase 7 components.

## Verification

- RED: `pytest tests/test_experience_cli.py -q` — 4 failed with the expected
  missing `tradingagents.experience.cli` module.
- GREEN: `pytest tests/test_experience_cli.py -q` — 4 passed.
- Regression: all 17 existing `tests/test_experience_*.py` files — 87 passed.
- `python -m compileall -q tradingagents/experience` — passed.
- `git diff --check` — passed.
- Ruff was unavailable in the environment (`ruff` command not found).
- Full repository collection was not runnable because the environment lacks
  existing optional dependencies (including langchain, typer, requests, and
  yfinance); no repository files were changed for that condition.

## Changed files

- `tradingagents/experience/cli.py`
- `tests/test_experience_cli.py`
- `pyproject.toml` (added only `experience = "tradingagents.experience.cli:main"`)

## Reviewer follow-up

Added the missing read-only Phase 7 query path for
`evidence --question --knowledge-artifact-root PATH`. The service is built
lazily from the active generation using `KnowledgeCatalog`,
`FastEmbedProvider`, `VectorIndexReader`, `LexicalIndexReader`, and
`KnowledgeQueryService`; no parser, scanner, ingestor, or index writer is
constructed. Added a regression test proving the supplied root is used and
the question reaches the Phase 7 service.

- Follow-up RED: the new test failed because `_build_knowledge_service` was
  absent.
- Follow-up GREEN: focused test passed.
- Final Phase 8 regression: 88 passed.

## Reviewer follow-up 2

Fixed omitted trust defaults at the CLI boundary. `similar` now passes Tier A
and Tier B to `ExperienceQuery`; `stats` now passes Tier A to
`OutcomeStatsRequest`. Explicit repeated `--trust-tier` values remain
unchanged. Added regression tests for both defaults.

- Follow-up RED: both new tests failed with an empty trust tuple.
- Follow-up GREEN: CLI tests 7 passed; full Phase 8 regression 90 passed.
- `python -m compileall -q tradingagents/experience` — passed.
- `git diff --check` — passed.
