# Phase 8 Task 7 report

Implemented the incremental importer, alias reconciliation, staging/lock
events, generation rebuild, and failure recovery seams.

## Delivered

- Added `tradingagents/experience/importer.py` with `ExperienceImporter`,
  `ExperienceRebuilder`, `ImportReport`, and `GenerationManifest`.
- Added catalog seams for source observations, import events, source alias
  enumeration, and retained feature projections.
- Import reads through `ReadonlySourceReader`, fingerprints decisions and
  evaluations, preserves observed evaluation rows (including recovery fields),
  quarantines conflicting immutable decisions, and leaves failed scans intact.
- Imports use `.staging/<run-id>` and `locks/import.lock`; RUNNING/COMPLETED/
  INTERRUPTED events are recorded and generation publication remains atomic.
- Added focused importer and rebuild recovery tests.
- Subsequent imports now process observed evaluation rows even when the
  decision fingerprint is unchanged, while catalog fingerprint deduplication
  keeps repeated imports idempotent.
- Catalog writes for each complete source snapshot now run inside one
  commit/rollback transaction. The import lock is acquired before staging is
  created, so lock collisions cannot leave orphan staging directories.
- Import report counters are staged per source and merged only after a source
  transaction commits; rolled-back work is never reported as indexed.

## Verification

- `pytest tests/test_experience_importer.py tests/test_experience_recovery.py -q`
  — 6 passed.
- `pytest tests/test_experience_*.py -q` (expanded by PowerShell to all
  experience tests) — 68 passed.
- `python -m compileall -q tradingagents/experience` — passed.
- `git diff --check` — passed.

The implementation is limited to Task 7; no outcomes, orchestrator, CLI,
smoke, or external integrations were added.
