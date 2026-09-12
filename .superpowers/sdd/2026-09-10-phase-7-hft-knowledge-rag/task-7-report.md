# Task 7 — incremental knowledge ingestion and recovery evidence

## Scope

- Added `tradingagents/knowledge/ingestion.py`, the explicit local-only
  incremental/rebuild coordinator.
- Extended `tradingagents/knowledge/catalog.py` with additive SQLite
  migrations for component fingerprints, staged-run ownership, persisted chunk
  provenance, compatible-document lookup, resource state transitions, and
  durable `RUNNING`/`INTERRUPTED` recovery markers.
- Added focused coverage in `tests/test_knowledge_ingestion.py`.
- No query API, CLI, stock/forex/MT5/trading integration, source mutation,
  model download, or networked behavior was added.

## TDD evidence

The first focused command was intentionally red:

```powershell
pytest tests/test_knowledge_ingestion.py -q
```

It failed for the expected missing production interface:
`ModuleNotFoundError: No module named 'tradingagents.knowledge.ingestion'`.

The green implementation was then verified by the same focused command.

## Delivered behavior

- The constructor derives exactly
  `ChunkPolicy.for_embedding(embedder.spec)` and constructs
  `StructureAwareChunker(policy, embedder.tokenizer)`; it has no
  character/word-limit fallback.
- Incremental scans retain unsupported files visibly, recheck bytes before and
  after parsing, skip compatible unchanged documents, and attach exact-byte
  aliases without parsing or embedding.
- Alias removal is independent per resource.  A document remains active while
  any `CURRENT` alias references it; a failed changed alias becomes
  `RETAINED_PREVIOUS` and cannot deactivate another current duplicate.
- Parsing, scan classification, chunking, embedding, and projection failures
  are isolated per resource.  The run summary reports visible state counts and
  a `SUCCEEDED`, `PARTIAL_FAILURE`, or `FAILED` aggregate.
- Each publication builds and validates a matched vector/lexical generation
  before `activate_generation`; active documents reuse persisted chunk/vector
  artifacts and do not reinvoke the embedder for an unchanged run.
- A single `artifact_root/locks/index.lock` ownership file prevents concurrent
  writers.  A durable `RUNNING` run is converted to `INTERRUPTED` on recovery,
  and only that run's staged rows/files are removed, preserving the previous
  active matched pair.

## Fresh verification

```powershell
pytest tests/test_knowledge_ingestion.py -q
pytest tests/test_knowledge_models.py tests/test_knowledge_config.py tests/test_knowledge_identity.py tests/test_knowledge_discovery.py tests/test_knowledge_catalog.py tests/test_knowledge_parser.py tests/test_knowledge_scanned.py tests/test_knowledge_chunking.py tests/test_knowledge_embeddings.py tests/test_knowledge_indexes.py tests/test_knowledge_ingestion.py -m "not integration" -q
python -m compileall -q tradingagents/knowledge tests/test_knowledge_ingestion.py
git diff --check
```

Results: focused Task 7 `5 passed`; Tasks 1–7 `118 passed, 1 deselected`.
Compilation and whitespace checks exited zero. `ruff` was unavailable in the
current Python environment (`No module named ruff`), so no lint-success claim
is made here.

## Review-fix continuation

The follow-up publication/recovery corrections are retained in the worktree
for the Task 7 fix commit. They make SQLite publication cover the active
generation, aliases, full current-document readiness, chunks, events, and the
terminal run marker; failed run-owned staging is discarded; and startup
recovery holds a process-scoped advisory writer lock.

## Review-fix completion evidence — 2026-09-11

Addressed the remaining focused review failures:

- Source-mutation regression now mutates `second.pdf` to same-length bytes,
  proving the post-parser source check is hash-based rather than size-only.
  Resource staging uses a short deterministic per-resource leaf under the run
  directory so Windows temp paths do not turn parser staging writes into
  unrelated `PARSE_FAILED` results.
- `discard_staged_run()` clears ingestion-event references to unaliased
  run-owned documents before deleting those document rows, so SQLite foreign
  keys stay valid while failed staged catalog rows are removed.

Fresh commands:

```powershell
C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m pytest tests/test_knowledge_ingestion.py::test_source_mutation_discards_resource_staging_artifacts tests/test_knowledge_ingestion.py::test_index_failure_discards_unpublished_catalog_rows_and_artifacts -q
# 2 passed in 0.88s

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m pytest tests/test_knowledge_ingestion.py -q
# 10 passed in 4.71s

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m pytest tests/test_knowledge_models.py tests/test_knowledge_config.py tests/test_knowledge_identity.py tests/test_knowledge_discovery.py tests/test_knowledge_catalog.py tests/test_knowledge_parser.py tests/test_knowledge_scanned.py tests/test_knowledge_chunking.py tests/test_knowledge_embeddings.py tests/test_knowledge_indexes.py tests/test_knowledge_ingestion.py -m "not integration" -q
# 123 passed, 1 deselected in 10.95s

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m compileall -q tradingagents\knowledge tests\test_knowledge_ingestion.py
# exit 0

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m ruff check tradingagents\knowledge\catalog.py tradingagents\knowledge\ingestion.py tests\test_knowledge_ingestion.py
# All checks passed!

git diff --check
# exit 0
```

## Review-fix completion evidence — round 2 — 2026-09-11

Addressed the remaining Round 2 review findings:

- Failed `REBUILD` no longer makes already-current, same-hash aliases
  `RETAINED_PREVIOUS`. Parsed rebuild candidates are kept in memory until
  catalog publication succeeds, and same-hash projection failures leave prior
  aliases, document readiness, and the active generation untouched.
- Successful ingestion now writes `state/active-index.json` from the active
  catalog registry after the catalog publication transaction commits. Startup
  repair still reconciles a stale pointer through `resolve_active_generation()`.

RED tests added and observed:

```powershell
C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m pytest tests/test_knowledge_ingestion.py::test_failed_rebuild_preserves_prior_active_generation_and_current_aliases tests/test_knowledge_ingestion.py::test_successful_ingestion_writes_and_repairs_active_pointer -q
# initial RED: failed rebuild emptied current_document_ids; pointer file was missing
```

Fresh GREEN commands:

```powershell
C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m pytest tests/test_knowledge_ingestion.py::test_failed_rebuild_preserves_prior_active_generation_and_current_aliases tests/test_knowledge_ingestion.py::test_successful_ingestion_writes_and_repairs_active_pointer -q
# 2 passed in 1.05s

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m pytest tests/test_knowledge_ingestion.py -q
# 12 passed in 5.44s

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m pytest tests/test_knowledge_models.py tests/test_knowledge_config.py tests/test_knowledge_identity.py tests/test_knowledge_discovery.py tests/test_knowledge_catalog.py tests/test_knowledge_parser.py tests/test_knowledge_scanned.py tests/test_knowledge_chunking.py tests/test_knowledge_embeddings.py tests/test_knowledge_indexes.py tests/test_knowledge_ingestion.py -m "not integration" -q
# 125 passed, 1 deselected in 12.31s

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m compileall -q tradingagents\knowledge tests\test_knowledge_ingestion.py
# exit 0

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m compileall -q tradingagents cli
# exit 0

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m ruff check tradingagents\knowledge\catalog.py tradingagents\knowledge\ingestion.py tests\test_knowledge_ingestion.py
# All checks passed!

git diff --check
# exit 0
```
