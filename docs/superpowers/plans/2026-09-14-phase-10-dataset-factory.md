# Phase 10 — Dataset Factory Implementation Plan

> **Execution rule:** use `superpowers:test-driven-development` for every task
> and `superpowers:subagent-driven-development` for task dispatch/review. The
> Phase 10 design at
> `docs/superpowers/specs/2026-09-14-phase-10-dataset-factory-design.md` is the
> authority. Work only in the Phase 10 worktree based on
> `194c46edf805db98d62fd642db3de1d222b5b958`.

## Global constraints

- Read Phase 5/6 SQLite, Phase 8 catalog, and Phase 9 audit only through
  query-only/read-only adapters. Never call a writer, LLM, Ollama, MT5, or
  network API and never modify source artifacts.
- Preserve existing Phase 5 outcome, Phase 8 trust, and Phase 9 context/citation
  contracts. The factory classifies; it does not redefine those contracts or
  promote `training_eligible`.
- Emit no prompt, completion, private reasoning, reasoning-token text,
  credentials, or unbounded source prose. Persist bounded structured facts,
  IDs, hashes, and provenance only.
- Keep Knowledge and Experience provenance separate. Do not query, embed, or
  write Phase 7 artifacts.
- All generated output is outside input roots, uses UTF-8 compact sorted JSON
  with LF line endings, and is deterministic across repeated builds and input
  enumeration order. Existing published generations are never overwritten.
- Supported closed exclusion codes are exactly: `UNTRUSTED_TIER`,
  `DECISION_CONTEXT_INCOMPLETE`, `TEMPORAL_INVALID`, `OUTCOME_INELIGIBLE`,
  `OUTCOME_UNAVAILABLE`, `NORMALIZATION_FAILED`, `CITATION_INVALID`,
  `EVIDENCE_INCOMPLETE`, `SCHEMA_UNSUPPORTED`, `PROVENANCE_INCOMPLETE`,
  `DUPLICATE`, and `SOURCE_INTEGRITY_FAILED`.
- Eligible examples require a unique joined source identity, valid UTC temporal
  ordering/no future leakage, Phase 9 COMPLETE/NORMALIZED/valid structured
  decision and evidence partition, Phase 8 Tier A trust, and an exact Phase 5
  or 6 COMPLETE/source-context-eligible outcome for the requested basis/horizon.
  Tier B/C, pending/unavailable/ineligible outcomes, and missing evidence remain
  explicit exclusions.
- HOLD keeps the source directional counterfactuals and existing
  `hold_opportunity_cost_points`; no fabricated PnL or floored cost-aware MFE.
- Grouped chronological split is 70/15/15; group is `source_run_id` or
  `decision_id`; no group crosses train/validation/test; insufficient data is an
  explicit status, not fabricated samples.
- Manifest safety counters must report `network_attempts=0`, `llm_calls=0`,
  `tool_calls=0`, and `mt5_calls=0`.

## Task 1 — Package contracts, policy versions, and typed errors

**Files:** create `tradingagents/datasets/__init__.py`,
`tradingagents/datasets/models.py`, `tradingagents/datasets/errors.py`;
create `tests/test_dataset_models.py`.

Define frozen, JSON-serializable contracts:

- `DATASET_SCHEMA_VERSION`, `EXAMPLE_SCHEMA_VERSION`,
  `ELIGIBILITY_POLICY_VERSION`, `CANONICALIZATION_VERSION`,
  `SPLIT_POLICY_VERSION`, and `SOURCE_ADAPTER_VERSION` constants;
- `DatasetExclusionReason` with the exact closed values in Global Constraints;
- `DatasetConfig` with explicit source DB tuple, Phase 8 root, optional Phase 9
  audit path, output root, filters, basis/horizon, and `allow_empty=True`;
- `SourceFingerprint`, `SourceObservation`, `EvaluationObservation`,
  `EvidenceObservation`, `JoinedObservation`;
- `CanonicalExampleV1`, `DatasetExclusion`, `SplitAssignment`,
  `DatasetManifest`, `BuildReport`, and `ValidationReport`.

Validate non-empty IDs, timezone-aware UTC timestamps, closed actions/statuses,
bounded strings, source/output-root separation, and JSON-safe values. Add tests
for enum closure, UTC validation, immutable mappings, no forbidden fields, and
deterministic serialization. The tests must be RED before implementation and
GREEN after it. Commit this task as `feat: add phase10 dataset contracts`.

## Task 2 — Read-only Phase 5/6 and Phase 8/9 source adapters

**Files:** create `tradingagents/datasets/sources.py`;
create `tests/test_dataset_sources.py` and reuse only tiny local fixture helpers
from `tests/fixtures/experience_source_db.py` where compatible.

Implement:

- `ReadonlyPhase56Source.read()` using SQLite URI `mode=ro`,
  `PRAGMA query_only=ON`, schema verification against the existing
  `shadow_decisions`/`shadow_decision_evaluations` contracts, file/WAL hashes
  before/after, and bounded row conversion;
- `ReadonlyExperienceSource.read()` against an existing Phase 8
  `catalog.sqlite3`, including accepted record, trust, source decision
  fingerprint, projections, and all retained evaluation snapshots with their
  observed-at provenance;
- `ReadonlyPhase9AuditSource.read()` against the existing audit SQLite schema,
  joining by decision/source run and returning only bounded structured audit
  fields (context hash, generation IDs, statuses, selected/available/used/
  rejected IDs, policy fingerprints, counts);
- deterministic source IDs/fingerprints and a `SourceReadError` hierarchy.

Readers must never create missing source files/directories, never use a Phase 8
writer, and must detect a changed DB/WAL as `SOURCE_INTEGRITY_FAILED`. Missing
optional audit data is represented as a typed unavailable observation so the
eligibility layer can exclude it. Add tests for query-only behavior, schema
drift, unchanged reads, file/WAL change detection, exact datetime conversion,
bounded audit fields, and no writer/network/LLM/MT5 calls. Commit as
`feat: add phase10 read-only source adapters`.

## Task 3 — Join and deterministic eligibility/exclusion classifier

**Files:** create `tradingagents/datasets/eligibility.py`;
create `tests/test_dataset_eligibility.py`.

Implement pure functions/classes:

- `join_observations(phase56, experience, audit)` keyed by decision ID,
  source-run ID, and Phase 8 source decision fingerprint;
- `classify_observation(observation, config) -> EligibilityResult`;
- deterministic multi-reason ordering and duplicate-key detection.

Use existing `classify_trust`, `EvaluationStatus`, and Phase 9
`validate_evidence_references` semantics through adapters/immutable facts; do
not duplicate their policy. Check UTC analysis/completion/reference ordering,
as-of/future leakage, normalized action, context COMPLETE, citation VALID,
Tier A, evaluation basis/horizon/status/source-context eligibility, and complete
provenance/version fingerprints. Emit all applicable closed reasons and never
turn missing facts into defaults. Add RED→GREEN tests covering every reason,
Tier B/C filtering, duplicate decisions, valid Tier A/complete outcome, and
future/as-of leakage. Commit as `feat: add phase10 eligibility classifier`.

## Task 4 — Canonical examples and evidence/outcome provenance

**Files:** create `tradingagents/datasets/canonical.py`;
create `tests/test_dataset_canonical.py`.

Implement `canonicalize(result) -> CanonicalExampleV1` only for eligible
observations. Include the exact structured decision fields, snapshot quote/state
facts, objective evaluation fields (both directional counterfactuals, selected
action, HOLD opportunity cost, MFE/MAE, availability), trust policy/version,
Phase 5/6/8/9 fingerprints, and bounded evidence usage. Preserve K/E/S IDs,
closed rejection reasons, context hash, and distinct `provenance.knowledge` vs
`provenance.phase8`. Store a raw-result fingerprint, never raw result text.

Derive `example_id` deterministically from decision ID, basis, horizon, source
decision fingerprint, and policy version. Reject forbidden prompt/completion/
reasoning/token/credential keys recursively. Add tests for stable IDs and JSON,
all outcome fields, HOLD semantics, evidence partition, provenance, no-CoT
rejection, and duplicate canonical keys. Commit as `feat: add phase10 canonical examples`.

## Task 5 — Chronological grouped splitter

**Files:** create `tradingagents/datasets/splits.py`;
create `tests/test_dataset_splits.py`.

Implement `assign_splits(examples)` with stable sort by analysis timestamp,
decision ID, basis, horizon, and example ID; group by source run or decision;
chronological 70/15/15 largest-remainder boundaries; and
`SPLIT_STATUS=INSUFFICIENT_DATA` for zero/too-small groups. Implement
`validate_split_assignments` to reject cross-split groups, non-chronological
rows, duplicate example keys, and malformed statuses. Add tests for exact
boundary/tie behavior, same decision at multiple horizons, run grouping,
input-order independence, timestamp ties, and insufficient populations. Commit
as `feat: add phase10 chronological splits`.

## Task 6 — Deterministic generation writer and validator

**Files:** create `tradingagents/datasets/writer.py`;
create `tests/test_dataset_writer.py`.

Implement fresh generation staging under `output_root`, canonical JSONL
serialization (`manifest.json`, `examples.jsonl`, `excluded.jsonl`,
`train.jsonl`, `validation.jsonl`, `test.jsonl`), file SHA-256/size/count
collection, and atomic publish without overwriting an existing published
generation. Manifest must include all policy/source/version/fingerprint fields,
counts/reason counts, class/split/group/time-range summaries, and zero safety
counters. `validate_generation(path)` must re-read hashes/schema/IDs,
duplicates, chronology, split leakage, source fingerprints, forbidden fields,
and no-CoT constraints.

Handle `EMPTY_ELIGIBLE_SET` explicitly; interrupted staging remains unpublished
and is safely discoverable. Add tests for byte-identical repeated builds,
newline/key ordering, file hash validation, existing-generation refusal,
interrupted staging, empty population, tampering detection, and no source-path
writes. Commit as `feat: add phase10 deterministic generation writer`.

## Task 7 — Factory orchestration and safe incremental behavior

**Files:** create `tradingagents/datasets/factory.py`;
create `tests/test_dataset_factory.py`.

Implement `DatasetFactory.build(config)` to read each source once, join facts,
classify every candidate, canonicalize eligible rows, assign grouped splits,
and publish through Task 6. Sort source paths before reading; deduplicate
byte-identical sources/decision fingerprints; retain changed/removed source
diagnostics; isolate a broken source; and produce a `BuildReport` with
candidate/eligible/excluded counts and reason totals. The factory must never
instantiate Phase 7/8 writers, embedder, TradingAgents graph, MT5, LLM, or
network client.

Add deterministic fixture tests for first build, unchanged repeat, changed
decision, duplicate file, removed row, broken source isolation, source/output
root safety, and zero external-call counters. Commit as
`feat: add phase10 dataset factory`.

## Task 8 — Explicit dataset-factory CLI

**Files:** create `tradingagents/datasets/cli.py`; modify `pyproject.toml` to
register `dataset-factory = "tradingagents.datasets.cli:main"`; create
`tests/test_dataset_cli.py`.

Add `build`, `validate`, and `status` subcommands with required explicit paths.
`build` prints bounded JSON report and returns nonzero on typed source/schema/
output errors; `validate` and `status` never instantiate parser, embedder,
ingestor, writer for source artifacts, LLM, MT5, or network clients. Refuse an
output root inside any input root and refuse to overwrite published generations.
Add tests for parser/argument errors, JSON output, read-only boundary, explicit
empty status, generation validation failure, and unchanged existing stock
`tradingagents` CLI imports/entrypoint. Commit as `feat: add phase10 dataset CLI`.

## Task 9 — Contract/fixture integration and acceptance harness

**Files:** create `tests/fixtures/dataset_factory.py`,
`tests/test_dataset_factory_integration.py`, and
`scripts/phase10_dataset_smoke.py`.

Create tiny deterministic Phase 5/6 + Phase 8 + Phase 9 fixtures with valid,
Tier B/C, incomplete, unavailable, duplicate, and future-leakage rows. The
integration tests must run the public factory end-to-end, verify exact output
bytes across two builds and reversed input ordering, source fingerprints,
reason counts, grouped splits, no-CoT output, and no external calls. The smoke
harness accepts explicit real source DB, Phase 8 root, optional Phase 9 audit
path, and a fresh output root; it runs offline with no LLM/MT5/network and
reports counts/status/fingerprints only. A real empty eligible set is valid if
explicitly reported. Do not make the harness mutate or rebuild Phase 5–9.
Commit as `test: add phase10 integration and smoke harness`.

## Task 10 — Documentation, broad verification, and review

**Files:** update `docs/phase10-dataset-factory.md` (usage, schema, exclusion
reasons, provenance, safety, Phase 11 handoff); no production behavior outside
the new package/CLI.

Run focused Phase 10 tests, affected Phase 5–9 tests, then the full suite. Run
Ruff over `tradingagents cli scripts tests`, `compileall` over
`tradingagents cli scripts`, and `git diff --check`. Execute one real local
smoke using the approved Phase 5/6 DB, existing Phase 8 catalog, existing Phase
9 audit if available, and a new output root; record before/after source
fingerprints and zero external calls. Run the mandatory whole-branch code
review against the merge base; fix only review findings through the SDD review
loop. Commit docs/verification fixes separately as needed. Do not merge, push,
start Phase 11, call Ollama/MT5/network, or change Phase 5–9 artifacts.

## Task dependency and review gates

Tasks 1→2→3→4→5→6→7→8→9→10 are ordered because each later task consumes the
previous contracts. Every task requires RED tests before production code,
GREEN focused tests, an implementer report, a review package, and a clean
task-scoped reviewer verdict before the next task. The final review must include
all deferred-minor/parked ledger entries and the final verification commands.

## Self-review checklist

- Every file named above is either created or modified exactly once in its task
  (later tasks may add tests/docs only).
- No task writes to Phase 5/6/7/8/9 source paths or adds an execution/RAG/
  training path.
- No task silently changes existing trust, outcome, citation, or stock CLI
  semantics.
- Empty/insufficient real data is an explicit result, never fabricated labels.
- Source and output fingerprints, schema versions, and no-CoT validation are
  load-bearing requirements covered by tests and manifest.
