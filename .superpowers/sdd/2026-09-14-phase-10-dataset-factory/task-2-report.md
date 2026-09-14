# Phase 10 Task 2 report

## Scope

Added read-only adapters for Phase 5/6 SQLite decisions/evaluations, Phase 8
experience catalogs, and Phase 9 evidence audit databases. Readers use SQLite
`mode=ro` and `PRAGMA query_only=ON`, verify required schemas, convert UTC
timestamps exactly, bound retained scalar/JSON fields, derive deterministic
source fingerprints, and compare main-file/WAL hashes before and after reads.
Missing optional Phase 9 audit data returns a typed unavailable observation.

No source writer, network, LLM, MT5, or execution API is imported or called.

## TDD evidence

The focused source tests were written first and failed during collection with
`ModuleNotFoundError: tradingagents.datasets.sources`. After implementation:

```text
pytest -q tests/test_dataset_sources.py
4 passed
```

## Verification

```text
ruff check tradingagents/datasets/sources.py tests/test_dataset_sources.py
All checks passed!

pytest -q tests/test_dataset_sources.py
4 passed
```

## Files

- `tradingagents/datasets/sources.py`
- `tests/test_dataset_sources.py`

## Round 1 review fix

Adapters now validate the complete Phase 5/6 contract column sets from the
existing source reader, every required Phase 8 table's columns, and the full
structured Phase 9 audit fields. Phase 8 retained timestamps and observed-at
provenance are normalized to UTC, and audit query/policy fingerprints are
retained. Existing malformed schemas remain typed errors; missing optional
audit files/tables remain typed unavailable observations with integrity checks
on present files.

Verification command/output:

```text
ruff check tradingagents/datasets/sources.py tests/test_dataset_sources.py
All checks passed!

pytest -q tests/test_dataset_sources.py tests/test_dataset_models.py
21 passed

git diff --check
passed (warnings only for Git LF/CRLF conversion)
```

## Round 2 review fix

Aligned Phase 8 validation with the complete catalog contract, including
tombstone state, market/evidence JSON, requested and decision timestamps,
aliases, projections, and retained evaluation metadata. The Phase 9 adapter
now validates and retains every field written by `EvidenceAuditStore`, including
as-of/context data, query and policy fingerprints, source diagnostics, latency,
telemetry, node metadata, provider/model, and schema version. Structured JSON
fields remain bounded and timestamps are normalized to UTC.

Verification command/output:

```text
ruff check tradingagents/datasets/sources.py tests/test_dataset_sources.py
All checks passed!

pytest -q tests/test_dataset_sources.py tests/test_dataset_models.py
21 passed
```
