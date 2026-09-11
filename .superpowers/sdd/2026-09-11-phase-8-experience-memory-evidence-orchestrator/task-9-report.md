# Phase 8 Task 9 implementation report

## Scope

Implemented the read-only `EvidenceOrchestrator` composition boundary. It
constructs a Phase 7 `KnowledgeQuery` only for a research question and a Phase
8 `ExperienceQuery` only for market state, forwards `as_of` unchanged, and
requests statistics only for returned experience IDs. Knowledge scores and
experience similarity scores remain separate; no ranking, recommendation,
prompt generation, model, MT5, parser, writer, or source-reader path is used.

The bundle reports `COMPLETE`, `PARTIAL`, `EMPTY`, or `FAILED`, with bounded
typed source errors and orchestration provenance including the experience
normalization fingerprint. `EvidenceSourceError`, `EvidenceWarning`, and
`OrchestrationProvenance` are public immutable contract values.

## TDD and verification

- RED: `pytest tests/test_experience_orchestrator.py -q` failed during
  collection because `tradingagents.experience.orchestrator` did not exist.
- GREEN: `pytest tests/test_experience_orchestrator.py -q` — 5 passed.
- Regression: all Phase 8 experience tests — 83 passed.
- Compile: `python -m compileall -q tradingagents/experience/orchestrator.py tradingagents/experience/models.py` — passed.
- Diff check: `git diff --check` — passed.

## Boundaries

Only Task 9 files were changed: the orchestrator, public model contracts,
package export, focused tests, and this report. No CLI, smoke harness, source
reader, catalog writer, execution, MT5, LLM, or Phase 9 code was added.
