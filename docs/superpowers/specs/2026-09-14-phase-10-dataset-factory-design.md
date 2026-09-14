# Phase 10 — Deterministic Dataset Factory Design

**Status:** approved design for implementation
**Date:** 2026-09-14
**Baseline:** `194c46edf805db98d62fd642db3de1d222b5b958` (Phase 9 closure)

## 1. Goals and non-goals

Phase 10 creates a local, deterministic, read-only dataset factory. It reads
the already accepted Phase 5/6 objective records, Phase 8 experience records,
and Phase 9 structured decision/audit records and emits versioned supervised
examples plus chronological grouped splits.

The factory must:

- preserve the existing Phase 5/6 outcome, Phase 8 trust, and Phase 9
  completeness/citation semantics;
- fail closed when identity, provenance, temporal, schema, or outcome evidence
  is incomplete;
- keep every eligible or excluded row explainable by a closed reason code;
- produce byte-identical output for the same inputs/configuration, independent
  of filesystem enumeration order;
- emit only bounded structured facts, never prompts, completions, private
  reasoning, or reasoning-token text;
- keep Knowledge (Phase 7) and Experience (Phase 8) provenance distinct; and
- remain local and offline.

It does **not** train, fine-tune, balance, label profitability, call an LLM,
call MT5, modify a TradingAgents decision, change `forex-watch`, or combine
published knowledge with proprietary experience in one vector store. Phase 11
owns model training and evaluation policy.

## 2. Architectural options

### Option A — Reuse Phase 8 importer/catalog as the dataset source

This minimizes adapter code but the importer is a writer, records a new
observation time, and intentionally applies Phase 8 ownership and mutation
semantics. Reusing it would make a dataset build mutate or depend on an
artifact-generation side effect and would blur source versus generated
provenance. Rejected.

### Option B — Direct SQL/JSON parsing in one CLI script

This is quick initially, but contracts, eligibility, split logic, and output
format become coupled to the command. It is hard to test and encourages
silently accepting schema drift. Rejected.

### Option C — Pure dataset domain plus read-only source adapters (recommended)

Small adapters read existing SQLite/catalog/audit files through query-only
connections and return immutable source observations. A pure eligibility
classifier, canonicalizer, splitter, manifest builder, and JSONL writer then
operate on those observations. The CLI only assembles these components. This
costs a few explicit interfaces, but keeps mutation impossible by construction,
supports fixture fakes, and makes every policy/version fingerprintable.

## 3. Architecture and package boundaries

```
Phase 5/6 SQLite ─┐
Phase 8 catalog ───┼─> read-only source adapters -> joined observations
Phase 9 audit ────┘                         │
                                            v
                                  eligibility + exclusion reasons
                                            v
                                  canonical examples (pure)
                                            v
                                  chronological group splitter
                                            v
                                  manifest + JSONL writer
```

New package: `tradingagents.datasets`.

- `models.py`: frozen enums/dataclasses for source observations, canonical
  examples, exclusions, split assignments, manifests, and build reports.
- `errors.py`: typed configuration, source-integrity, schema, and output
  errors; messages are bounded and contain no prompt/completion text.
- `sources.py`: `ReadonlyPhase56Source`, `ReadonlyExperienceSource`, and
  `ReadonlyPhase9AuditSource`. Each opens files query-only, fingerprints before
  and after reading, and never imports a writer or calls a network/LLM/MT5 API.
- `eligibility.py`: pure gate evaluation and closed `DatasetExclusionReason`
  values. It delegates trust/outcome interpretation to the existing Phase 8
  contracts instead of redefining them.
- `canonical.py`: deterministic conversion into the v1 example contract,
  bounded evidence references/rejections, and no hidden text.
- `splits.py`: deterministic grouped chronological 70/15/15 assignment with
  explicit insufficient-data status.
- `writer.py`: fresh generation directory creation, canonical JSON/newline
  serialization, file hashes, and atomic publication of the generation.
- `factory.py`: orchestration of adapters and pure stages; no CLI parsing and
  no writes to source paths.
- `cli.py`: explicit `dataset-factory build`, `validate`, and `status` commands.

No module in this package imports `TradingAgentsGraph`, an execution API,
`MetaTrader5`, Ollama, a hosted provider, or a Phase 7/8 writer.

## 4. Inputs and immutable source boundary

`BuildConfig` requires explicit paths for:

- one or more Phase 5/6 source SQLite databases;
- an existing Phase 8 artifact root/catalog;
- an existing Phase 9 evidence-audit SQLite path (optional only when the source
  contract proves no Phase 9 audit is present; a missing required audit is an
  exclusion, not a fabricated audit);
- an output dataset root outside all source roots; and
- optional profile/symbol/basis/horizon filters.

Readers use SQLite URI `mode=ro` plus `PRAGMA query_only=ON`, and reject WAL or
main-file changes observed during a read. Source paths are canonicalized before
identity and are never opened with a writer. Output roots are checked not to be
inside a source root. Existing published generations are not overwritten;
generation IDs are content/config derived and an already identical generation
is reused only after hash verification.

## 5. Source identity and joining

The source decision identity is the Phase 5/6 `decision_id`; the run identity is
`source_run_id`. A Phase 8 `experience_id` must point to exactly one accepted
`source_decision_id` and its `source_decision_fingerprint`. Phase 9 audit rows
join by `decision_id` and `source_run_id`.

Every source adapter emits a `SourceFingerprint` containing canonical path,
schema fingerprint, file/WAL SHA-256, observed source snapshot fingerprint,
and contract version. A joined observation retains per-source fingerprints;
they are not collapsed into a single anonymous hash. Duplicate byte-identical
source files deduplicate by source SHA and decision fingerprint. Different
paths or changed bytes remain separately auditable and are never silently
merged.

## 6. Canonical example schema v1

Each eligible `ExampleV1` is a JSON object with stable keys:

```json
{
  "schema_version": "phase10.dataset.example.v1",
  "example_id": "ex1-…",
  "decision": {
    "decision_id": "…", "source_run_id": "…", "requested_symbol": "EURUSD",
    "resolved_symbol": "EURUSD", "analysis_profile": "INTRADAY",
    "analysis_timeframe": "M15", "analysis_snapshot_timestamp": "…",
    "decision_completed_timestamp": "…", "action": "BUY|SELL|HOLD",
    "normalization_status": "NORMALIZED", "decision_context_status": "COMPLETE",
    "raw_result_fingerprint": "…"
  },
  "market": {"snapshot": {}, "snapshot_fingerprint": "…"},
  "research": {"context_integrity": "COMPLETE", "evidence_use_status": "USED|NONE_RELEVANT", "refs_used": [], "refs_rejected": []},
  "outcome": {"evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300,
    "evaluation_status": "COMPLETE", "source_context_eligible": true,
    "training_eligible": false, "buy_net_points": 0.0, "sell_net_points": 0.0,
    "selected_action_net_points": 0.0, "hold_opportunity_cost_points": 0.0,
    "buy_mfe_points": 0.0, "buy_mae_points": 0.0,
    "sell_mfe_points": 0.0, "sell_mae_points": 0.0},
  "trust": {"tier": "TIER_A_HIGH_TRUST", "policy_version": "trust-policy.v1"},
  "provenance": {"phase56": {}, "phase8": {}, "phase9": {}, "knowledge": []}
}
```

The actual implementation serializes optional values only where the contract
allows them and validates the closed action/status enums. `raw_result` itself
is not copied into the corpus; only a fingerprint and the existing structured
fields needed for supervised input are retained. No prompt, completion,
reasoning, token telemetry, account credentials, or arbitrary prose is emitted.
`training_eligible` is copied as source fact only; the factory never promotes
it and emits a separate dataset eligibility decision.

The example ID is SHA-256 over the canonical decision ID, basis, horizon,
source decision fingerprint, and dataset policy version. One decision may yield
several examples at different horizons, but each `(decision_id, basis, horizon)`
is unique.

## 7. Eligibility and closed exclusions

Eligibility is the conjunction of:

1. unique joined source identity and matching source fingerprints;
2. valid UTC analysis/completion/reference ordering and no future leakage;
3. Phase 9 context `COMPLETE`, structured RM/Trader/PM results, normalized
   action, and valid evidence partition/citation audit;
4. Phase 8 accepted record with Tier A trust (Tier B/C are retained only in
   `excluded.jsonl`); and
5. a Phase 5/6 evaluation for the requested exact basis/horizon that is
   `COMPLETE`, source-context eligible, and temporally known at the decision
   cutoff. Pending, unavailable, ineligible, or invalid observations are never
   treated as labels.

`DatasetExclusionReason` is closed and includes:

`UNTRUSTED_TIER`, `DECISION_CONTEXT_INCOMPLETE`, `TEMPORAL_INVALID`,
`OUTCOME_INELIGIBLE`, `OUTCOME_UNAVAILABLE`, `NORMALIZATION_FAILED`,
`CITATION_INVALID`, `EVIDENCE_INCOMPLETE`, `SCHEMA_UNSUPPORTED`,
`PROVENANCE_INCOMPLETE`, `DUPLICATE`, `SOURCE_INTEGRITY_FAILED`.

Multiple failures are retained in deterministic severity/code order; the first
reason is never used to hide additional evidence. A failed source read is an
excluded source/error record and cannot remove already emitted generations.

## 8. Outcome semantics

The factory preserves objective Phase 5 values exactly: basis, horizon,
availability status, target/observation timestamps, entry/future quotes,
spread/point/digits, directional net points, selected-action net points,
cost-aware MFE/MAE, and HOLD opportunity cost. It never computes a new PnL,
floors cost-aware MFE, or turns a missing observation into zero.

BUY and SELL counterfactuals are retained for every eligible example. HOLD
retains `hold_opportunity_cost_points` using the existing Phase 5 rule and does
not receive a fabricated directional PnL.

## 9. Evidence and provenance

Phase 9 `EvidenceUsageAudit` is represented as bounded metadata: context hash,
bundle/integration/status, selected counts, available/used/rejected display IDs,
query policy/fingerprint, and generation IDs. Evidence refs are validated as a
complete partition under the existing Phase 9 contract; closed rejection
reasons are preserved. Knowledge fields, if present in the audit, retain Phase
7 document/chunk provenance and remain under `provenance.knowledge`; experience
fields retain Phase 8 source IDs under `provenance.phase8`. The factory never
embeds or re-renders text and never puts knowledge and experience into one
retrieval index.

## 10. Deterministic grouped chronological splits

Rows are sorted by UTC analysis snapshot timestamp, then decision ID, basis,
horizon, and example ID. The split group is `source_run_id` when present,
otherwise decision ID; all horizons for a decision/run stay together. Groups
are assigned chronologically to train/validation/test using target proportions
70%/15%/15% with deterministic largest-remainder boundaries and no balancing or
random seed. An empty or too-small population returns `SPLIT_STATUS=
INSUFFICIENT_DATA` and keeps all rows in the manifest/exclusion accounting;
the factory does not invent samples. A validator rejects cross-split group
leakage and any timestamp inversion.

## 11. Generation layout and manifest

Each build writes a new directory under the configured output root:

```
<output-root>/<generation-id>/
  manifest.json
  examples.jsonl
  excluded.jsonl
  train.jsonl
  validation.jsonl
  test.jsonl
```

JSON is UTF-8, sorted keys, compact separators, and one LF-terminated record
per line. Record ordering is canonical and independent of input enumeration.
`manifest.json` contains schema/policy/parser/source-adapter versions, factory
and Python version, generation/config/source fingerprints, input paths (with
canonical IDs, not credentials), candidate/eligible/excluded counts and reason
counts, symbol/profile/basis/horizon/time range, split/group/class counts,
per-file SHA-256/size/record counts, and safety telemetry:
`network_attempts=0`, `llm_calls=0`, `tool_calls=0`, `mt5_calls=0`.

Manifest `status` is `PUBLISHED`, `EMPTY_ELIGIBLE_SET`, or `FAILED`; failed
generations are never made active. A build is atomic within the output root and
never replaces an existing published generation.

## 12. CLI

The new `dataset-factory` executable has explicit commands:

```text
dataset-factory build --source-db … --experience-root … --audit-db … --output-root …
dataset-factory validate --generation …
dataset-factory status --output-root …
```

`build` performs the complete deterministic pipeline. `validate` is read-only
and checks manifest/file hashes, schema versions, IDs, duplicate keys, split
group leakage, chronology, source fingerprints, and no-CoT fields. `status`
only inspects manifests and never opens a source writer. No command starts
indexing, launches a graph, loads Ollama, touches MT5, or performs network I/O.

## 13. Incrementality and recovery

Source observations are fingerprinted per build. An unchanged source can be
reused by a future cache layer, but the v1 factory remains safe if it re-reads
it; a changed source produces a new generation ID and changed decisions are
re-evaluated. Byte-identical duplicate files are de-duplicated. Removed source
rows are absent from the next candidate set and reported in the manifest as
stale/removed observations; prior published generations remain immutable.

Each output file is staged before publication. An interruption leaves only an
unpublished staging directory, which `status` reports and a later build may
clean only when it can prove the staging generation is owned by this factory.
One broken source is isolated to its exclusion/error record and does not corrupt
other source results.

## 14. Versioning and reproducibility

The manifest records `dataset_schema_version`, `eligibility_policy_version`,
`canonicalization_version`, `split_policy_version`, `source_adapter_version`,
and each Phase 5/6/8/9 schema/policy version. Generation identity includes all
configuration and source fingerprints. A validator rejects a generation when
required version fields, canonical file hashes, or source fingerprints do not
match. No incompatible schema is silently mixed into a generation.

## 15. Testing strategy

All tests use tiny SQLite/JSON fixtures and fake adapters; CI never reads real
books, calls Ollama/MT5, downloads models, requires CUDA, or needs network.
Tests cover:

- first build and exact repeated byte identity;
- input enumeration/order independence;
- duplicate decisions and duplicate byte-identical sources;
- every closed eligibility/exclusion reason;
- Phase 8 trust filtering and Phase 9 COMPLETE/NORMALIZED/VALID partition;
- missing/changed/broken source and source fingerprint mismatch;
- future leakage and UTC ordering;
- grouped chronological split boundaries and insufficient-data status;
- exact objective outcome preservation including HOLD opportunity cost and MFE/MAE;
- canonical IDs, schema/version fields, provenance, and no-CoT rejection;
- stable JSONL/newline/hash output and validator failures;
- interrupted staging recovery and non-overwrite of published generations;
- CLI read-only boundary and zero network/LLM/MT5 calls.

## 16. Retrieval/knowledge separation

Knowledge is not queried or embedded by the dataset factory. If Phase 9 audit
metadata carries selected knowledge IDs, they are provenance-only and remain
distinct from Experience fields. The factory never writes Phase 7 or Phase 8
artifacts and never adds RAG context to a trading decision.

## 17. Windows and performance

The design uses Python stdlib, SQLite query-only reads, streaming JSONL writes,
and bounded in-memory canonical rows. It does not assume CUDA or a server.
Sorting is deterministic and may use a temporary local list for the expected
development corpus; a future external-sort seam can be added if the corpus
outgrows 16 GB RAM. Paths use `pathlib`, UTF-8, and explicit LF output so the
same bytes are produced on Windows and Linux.

## 18. Security and privacy

The process is offline by construction. No credentials are read or persisted;
source paths and hashes are the only identifying metadata. Forbidden-field
validation rejects prompt, completion, reasoning, token, credential, and secret
keys recursively before a record is written. Inputs are read-only and outputs
are placed outside the approved source roots.

## 19. Future extension seams

Interfaces are intentionally replaceable for a future corpus builder,
multi-symbol policies, richer structured state, alternative split policies,
and additional objective horizons. Phase 11 may consume the published JSONL
and manifest, but must not bypass the eligibility/exclusion evidence or relabel
rows in place.

## 20. Explicit Phase 11 handoff boundary

Phase 10 ends after deterministic dataset generation and validation. Phase 11
may train/evaluate models from a selected manifest generation. It owns any
sampling, balancing, tokenizer/model choices, and training labels. It must
explicitly exclude `excluded.jsonl`, incomplete outcomes, and unsupported
versions and must not modify the Phase 10 generation or any Phase 5–9 source.

## Self-review

- No TODO/TBD placeholders remain.
- No source writer, MT5, LLM, network, execution, strategy, RAG injection,
  training, or fine-tuning path is introduced.
- Eligibility delegates to existing trust/outcome/citation contracts and does
  not redefine them.
- Phase 7 Knowledge and Phase 8 Experience remain separate provenance domains.
- Empty/small real populations are explicit, not fabricated into a successful
  training corpus.
