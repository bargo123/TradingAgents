# Phase 8 — Experience Memory + Evidence Orchestrator

**Status:** Design only; implementation is not authorized by this document
**Date:** 2026-09-11
**Repository:** C:\AITrading\TradingAgents
**Design branch:** codex/phase-8-experience-memory-design
**Source baseline:** 12460b345ef0b38f71e2e93375bcf3248ffb77df (completed Phase 7)
**Main observed before isolation:** 6bf750590ab3b5eece4a36af11d6629fde2d0885

## 1. Goals and non-goals

### Goals

Phase 8 adds two local, deterministic, read-only research capabilities:

1. **Experience Memory** — a rebuildable catalog and leakage-safe similarity
   projection over persisted Phase 5/6 shadow decisions and evaluations.
2. **Evidence Orchestrator** — a composition boundary that independently
   queries Phase 7 published Knowledge RAG and Phase 8 Experience Memory and
   returns a provenance-complete evidence bundle.

The source rows in the Phase 5/6 SQLite database remain authoritative. Phase 8
never changes what a decision meant. It creates versioned projections that can
be discarded and rebuilt when feature, trust, or statistics rules change.

The intended Phase 8 flow is:

~~~
Phase 5/6 shadow database (read-only)
    -> schema-checked importer
    -> one ExperienceRecord per shadow decision
    -> pre-decision feature projection
    -> deterministic trust classification
    -> exact numeric similarity search
    -> explicit basis/horizon outcome statistics

Phase 7 KnowledgeQueryService (read-only)
    +
Phase 8 ExperienceQueryService (read-only)
    -> Evidence Orchestrator
    -> separate KnowledgeEvidence and ExperienceEvidence collections
~~~

### Hard non-goals and safety boundaries

Phase 8 must not:

- call MT5 or create an MT5 provider/session;
- place, modify, close, or simulate an order;
- change forex-watch, forex-shadow, ForexShadowRunner, or Phase 6 collection
  behavior;
- change TradingAgents, LangGraph, prompts, analyst routing, or decisions;
- inject evidence into Qwen, Ollama, hosted models, or any trading prompt;
- generate BUY, SELL, HOLD, PortfolioDecision, strategies, or execution
  instructions;
- change Phase 5 evaluation mathematics, statuses, bases, horizons, HOLD
  semantics, or training-eligibility meaning;
- build training datasets, assign training labels, fine-tune, or train Qwen;
- modify the Phase 7 Knowledge RAG catalog, LanceDB tables, FTS5 tables, or
  source books/papers;
- place books/papers and experiences in a shared vector table, FTS table, or
  catalog;
- copy hidden chain-of-thought, private reasoning, or unrestricted report
  prose into Experience Memory;
- reconstruct absent market features with future prices, MT5 backfills,
  external APIs, LLM output, or guesses; or
- begin Phase 9.

The output is evidence only. A caller remains responsible for any later
interpretation, and this phase makes no profitability or live-performance
claim.

## 2. Existing Phase 5/6/7 repository context

### 2.1 Branch and inspection basis

The requested Phase 7 final SHA is not the main checkout at the time of
design. The design work is therefore isolated in a linked worktree created
from the completed Phase 7 branch at
12460b345ef0b38f71e2e93375bcf3248ffb77df. Main is not modified.

The repository is a Python/setuptools project with pytest, Ruff, separate
forex-shadow, forex-evaluate, forex-watch, and knowledge entry points. The
stock tradingagents CLI remains a separate surface.

### 2.2 Phase 5 decision contract

tradingagents.forex.shadow.ShadowTradeDecision is the authoritative decision
record. The persisted shadow_decisions table contains:

- identity/time: decision_id, created_at, analysis_date, source_run_id;
- requested/resolved instrument and profile: requested_symbol,
  resolved_symbol, analysis_profile, analysis_timeframe, valid_for_seconds,
  valid_until;
- normalized decision evidence: action, normalization_status,
  normalization_error, decision_context_status,
  raw_portfolio_manager_result, trader_summary,
  portfolio_manager_summary, bull_summary, bear_summary;
- model provenance: llm_provider, quick_model, deep_model;
- analysis-snapshot quote: snapshot_timestamp, reference_bid,
  reference_ask, reference_mid, spread, spread_points;
- explicit temporal quote fields:
  analysis_snapshot_timestamp, analysis_snapshot_bid,
  analysis_snapshot_ask, analysis_snapshot_spread,
  analysis_snapshot_spread_points, decision_completed_timestamp,
  analysis_latency_seconds, decision_reference_timestamp,
  decision_reference_bid, decision_reference_ask, decision_reference_spread,
  decision_reference_spread_points, decision_reference_status,
  decision_reference_delay_seconds, and decision_reference_error;
- the serialized, normalized market context in snapshot_json; and
- the safety invariant executed, which is constrained to zero.

The legacy snapshot and reference columns are the analysis snapshot, not an
executable decision reference. Phase 8 preserves that distinction.

raw_portfolio_manager_result is imported only through a bounded structured
allow-list. Free-form summaries are not copied into similarity features and
are not treated as private reasoning.

### 2.3 Phase 5 evaluation contract

tradingagents.forex.evaluation.ShadowEvaluationStore persists
shadow_decision_evaluations, one row per
(decision_id, evaluation_basis, horizon_seconds). The actual columns include:

- basis: ANALYSIS_SNAPSHOT or DECISION_REFERENCE;
- horizon: the configured integer seconds, currently 300, 900, 1800, and 3600
  (5m, 15m, 30m, 60m);
- eligibility/provenance: evaluation_version, market_data_source,
  source_context_eligible, training_eligible (currently always NULL), and
  training_eligibility_reason;
- temporal evidence: target_timestamp, observation_timestamp,
  observation_lag_ms, entry_timestamp;
- entry/future quote and symbol metadata: entry_bid, entry_ask,
  entry_spread, entry_spread_points, future_bid, future_ask, future_spread,
  future_spread_points, point, digits;
- cost-aware counterfactuals: buy_net_price, buy_net_points,
  sell_net_price, sell_net_points, selected_action,
  selected_action_net_price, selected_action_net_points,
  best_counterfactual_action, best_counterfactual_net_points,
  hold_opportunity_cost_points;
- cost-aware excursions: BUY/SELL MFE and MAE in price and points; and
- status/audit fields: evaluation_status, unavailable_reason, created_at,
  evaluated_at, recovered_from_unavailable_at, and
  previous_unavailable_reason.

Phase 8 treats basis and horizon as mandatory dimensions. It never substitutes
one basis or horizon for another. A recovered DATA_UNAVAILABLE -> COMPLETE
row is a new source-evaluation fingerprint for the same logical decision and
basis/horizon, not a rewrite of the original source evidence.

### 2.4 Phase 6 watcher/run contract

tradingagents.forex.watch_store.WatcherStore uses the same SQLite path as the
shadow decision/evaluation stores. The Phase 6 tables are:

- forex_watcher_state: singleton lifecycle, lease, heartbeat, current run,
  evaluation-due, circuit, and bounded failure counters;
- forex_watch_opportunities: deterministic opportunity key, requested symbol,
  profile, analyst set, schedule timeframe, anchor/bar-close/eligibility
  timestamps, config fingerprint, status, skip reason, run and decision IDs;
  and
- forex_watch_runs: run_id, opportunity_key, attempt_number, source_run_id,
  requested/resolved symbols, run status, start/completion times, failure
  metadata, context/normalization/action statuses, analysis and
  decision-reference timestamps, staleness/freshness fields, provider/model
  identifiers, analyst/profile/config provenance, runtime, LLM/tool/token
  counters, and bounded metrics_json.

Watcher rows are operational provenance. They are imported for traceability
and diagnostics, not as additional decisions. A skipped opportunity without a
shadow_decisions row is not an ExperienceRecord.

### 2.5 Actual persisted market-state payload

The current runner persists snapshot_json produced by
tradingagents.forex.context.snapshot_to_dict. Its pre-decision portions are:

~~~
timestamp
symbol
quote: bid, ask, spread, spread_points
symbol_metadata: name, description, digits, point, visible, trade_mode,
                 currency_base, currency_profit
features:
  M1, M5, M15, H1:
    candle_count, latest, return_over_bars, recent_high, recent_low,
    range, direction, average_true_range, range_pct, close_position
broker_clock
account
positions
candles (bounded OHLC/tick-volume arrays)
~~~

Phase 8 uses only the deterministic quote/features/timestamp/profile fields
listed in the feature schema below. Account and position data are retained
only as bounded source provenance when the caller explicitly requests source
diagnostics; they are not similarity inputs.

The inspected real Phase 6 validation databases contain all five tables above
with these columns. Phase 8 does not assume that a real run is complete:
INCOMPLETE, normalization failure, PENDING, INELIGIBLE, and DATA_UNAVAILABLE
states are imported and labeled as such.

### 2.6 Phase 7 knowledge contract

The completed Phase 7 package exposes:

- tradingagents.knowledge.models.KnowledgeQuery, with query text, top-k,
  content-type and document filters;
- KnowledgeHit, with text, ranking streams, content type, document/source
  hash, page/section, parser/chunker/embedding/index provenance;
- KnowledgeQueryService.search(KnowledgeQuery), which reads one validated
  matched vector/FTS5 generation and may instantiate only the local query
  embedder; and
- KnowledgeCatalog.active_generation(), which enforces complete generation
  and embedding-spec compatibility.

The orchestrator consumes this public read-only contract. It does not import
Phase 7 parser, ingestor, writers, or source folder, and it does not change
KnowledgeHit values or their scores.

## 3. Design alternatives and recommendation

### Approach A — SQLite catalog plus exact numeric projection (recommended)

Keep an authoritative Phase 8 SQLite catalog for normalized source evidence,
provenance, trust, outcome rows, import events, and diagnostics. Extract a
versioned numeric feature schema from each pre-decision snapshot. Persist a
compact NumPy matrix, missing-value mask, ordered experience IDs, and
normalization metadata. Perform an exact CPU scan behind a SimilarityIndex
interface.

**Advantages:** source and audit state are transactional; numeric similarity
cannot accidentally depend on prose, action, or outcomes; exact search is
simple and reproducible at the expected corpus size; the design does not
introduce another embedding model or server; and a future ANN backend can
replace only the index implementation.

**Trade-offs:** a projection must be rebuilt when feature or normalization
versions change, and the importer must maintain catalog/projection generation
integrity.

### Approach B — separate experience text/vector index

Embed a textual serialization of each decision and outcome into a new LanceDB
table, physically separate from Phase 7, and retrieve by vector similarity.

**Advantages:** familiar dense retrieval machinery and easy retrieval of
diagnostic prose.

**Trade-offs:** a text blob would make outcome/action leakage likely, would
mix model/report style with market-state similarity, would require another
embedding-spec lifecycle, and would make missing numeric fields ambiguous.
It is rejected for v1 similarity. A small diagnostic FTS5 projection is
allowed, but it is not a market-state similarity index.

### Approach C — SQLite-only distance queries

Store feature values in normalized rows and calculate distances using SQL
joins for every query.

**Advantages:** no NumPy artifact and a small deployment footprint.

**Trade-offs:** repeated joins and missing-mask calculations are slower and
harder to make numerically identical across SQLite versions; vectorized exact
scans are clearer for robust scaling and deterministic tie handling. This
remains a possible implementation fallback behind the same interface, not the
v1 recommendation.

### Selected approach

Choose Approach A. It keeps published knowledge and proprietary experience
physically separate, makes the pre-decision boundary explicit, and is
practical on the six-core/16 GB Windows machine without cloud services,
CUDA, an ANN server, or a second semantic model.

## 4. Architecture and data flow

~~~
Phase 5/6 SQLite (mode=ro, query_only=ON)
       |
       v
SourceSchemaReader -> row fingerprints -> ExperienceImporter
       |                                      |
       |                                      +--> experience catalog.sqlite3
       |                                      +--> trust/outcome projections
       |                                      +--> feature matrix + mask
       |                                      +--> import/quarantine audit
       v
ExperienceQueryService -> compatibility gates -> exact SimilarityIndex
       |                                      |
       +--> ExperienceSearchResult           +--> StatisticsCalculator

Phase 7 KnowledgeQueryService --------------------+
                                                    v
                              EvidenceOrchestrator
                                  |
                                  +--> KnowledgeEvidence[] (KnowledgeHit)
                                  +--> ExperienceEvidence[] (ExperienceHit)
                                  +--> optional explicit statistics
                                  +--> source status/errors/warnings
~~~

The importer and projection builder are explicit maintenance operations. A
TradingAgents, forex, watcher, evaluator, or MT5 command never starts them.
Experience queries are read-only. The orchestrator is also read-only and
never calls an LLM.

The catalog is authoritative for which projection generation is active.
Feature matrices and diagnostic FTS5 tables are replaceable projections. A
writer publishes a complete generation only after all metadata and integrity
checks pass; readers continue using the previous generation if a rebuild
fails.

## 5. Package and module boundaries

The implementation should add a self-contained package with focused
responsibilities:

~~~
tradingagents/experience/
  __init__.py              # public contracts only
  config.py                # source DBs, artifact root, policies, versions
  models.py                # records, queries, hits, stats, statuses
  source_reader.py         # SQLite mode=ro schema/row reader; no store init
  identity.py              # source and experience fingerprints
  catalog.py               # transactional Phase 8 catalog authority
  features.py              # ExperienceFeatureSchemaV1 extraction/validation
  normalization.py         # SimilarityProfile and robust scaling
  trust.py                 # deterministic Tier A/B/C policy
  similarity.py             # SimilarityIndex protocol and exact NumPy scan
  query.py                  # read-only ExperienceQueryService
  outcomes.py              # basis/horizon statistics after retrieval
  provenance.py            # source/evaluation trace validation
  diagnostics.py           # bounded errors, quarantine, safe FTS text
  importer.py              # incremental import/rebuild coordinator
  orchestrator.py          # Knowledge + Experience evidence composition
  cli.py                   # explicit experience/evidence commands only
~~~

Dependency direction:

~~~
CLI -> importer/query/orchestrator -> catalog/projections
source_reader -> stdlib sqlite3 and immutable local contracts
features/trust/normalization/similarity/outcomes -> models only
orchestrator -> KnowledgeQueryService protocol + ExperienceQueryService protocol
~~~

The experience package must not import MetaTrader5, MT5 provider modules,
forex-watch, ForexShadowRunner, graph modules, agent modules, Qwen/Ollama
clients, hosted clients, or order/execution code. The source reader must not
call ShadowDecisionStore.initialize() or ShadowEvaluationStore.initialize()
because those methods can create/migrate tables. It reads their SQL schema
directly in read-only mode.

## 6. Generated storage layout

The default root is data_cache/experience, outside the Phase 7
data_cache/knowledge tree and outside the approved book source folder:

~~~
data_cache/experience/
  catalog.sqlite3                         # authoritative Phase 8 catalog
  manifests/
    sources.jsonl                          # source identities/fingerprints
    experiences.jsonl                      # safe record manifests
    feature-schema-v1.json
    similarity-profile-v1.json
    trust-policy-v1.json
    statistics-policy-v1.json
  features/
    experience-features-v1/
      rows.jsonl                           # deterministic feature audit
      matrix.f32.npy                       # rebuildable numeric matrix
      mask.u8.npy                          # missing-feature mask
      ids.jsonl                            # row order and provenance
  projections/
    similarity-profile-v1/
      generation.json
      matrix.f32.npy
      mask.u8.npy
      ids.jsonl
  state/
    import-runs.jsonl                      # append-only bounded summaries
    active-generation.json                 # repairable pointer
  diagnostics/
    fts5.sqlite3                           # safe status/error text only
  quarantine/
    <import-run-id>/<record-key>.json      # bounded diagnostics, no reports
  locks/
    import.lock                            # one writer
  .staging/
    <run-id>/                              # unpublished recoverable artifacts
~~~

The exact NumPy/JSON files are rebuildable and never authoritative without a
matching catalog generation row. Phase 8 never writes under
C:\Users\Zaid barghouthi\Downloads\new books or under data_cache/knowledge.

The catalog contains separate logical tables for source identities,
experience records, decision evidence, market-state evidence, outcome
evidence, projection generations, import events, and quarantine. An
experience_diagnostic_fts table, if used, belongs only to this experience
catalog or its dedicated local diagnostic database; it is not the Phase 7
FTS5 table.

## 7. Source database identity and read-only contract

### 7.1 Source identity

Each configured source database is represented by:

- source_database_id: SHA-256 of the versioned identity string, canonical
  absolute path, and compatible source-schema fingerprint;
- canonical_path: resolved path recorded for operator audit;
- source_file_hash: SHA-256 of the SQLite main file observed for the import;
- source_snapshot_fingerprint: SHA-256 of the ordered source table
  fingerprints observed in that import, including evaluation and watcher rows
  when present;
- source_schema_fingerprint: SHA-256 of sorted table names, column names,
  declared types, and primary-key/unique metadata; and
- observed_at plus file size/mtime metadata for diagnostics only.

The file hash is a snapshot identity, not a logical decision identity. A copy
of the same database at another path may have another source ID while rows
with the same decision ID and source fingerprint still deduplicate.

### 7.2 Read-only opening and stability

The source reader:

1. resolves and verifies the path is a regular local file;
2. opens file:<path>?mode=ro with SQLite URI mode;
3. executes PRAGMA query_only=ON and does not issue DDL/DML;
4. verifies required schema before reading rows;
5. reads all selected rows in deterministic primary-key order;
6. records main/WAL file stats before and after the read; and
7. fails with a typed SourceSnapshotChangedError if the source changes
   during the read rather than combining two snapshots.

The reader never creates a missing database, runs migrations, removes rows,
updates evaluation status, calls MT5, or reconstructs missing fields. A source
that is unavailable or schema-incompatible produces a visible import failure
while the last valid Phase 8 projection remains active.

### 7.3 Required and optional source tables

shadow_decisions is required and must contain the Phase 5 identity, normalized
and context status, action, quote, timestamp, snapshot_json, executed,
model/provider, profile, and source-run columns listed in Section 2.2. Older
rows may have NULL Phase 5 extension columns; the adapter records that
limitation rather than inventing values.

shadow_decision_evaluations is optional for importing decision context but
required for outcome statistics. If present, the adapter requires all
basis/horizon/status, eligibility, quote, counterfactual, excursion, and audit
columns in Section 2.3. A missing table means outcome availability is
explicitly MISSING, not COMPLETE.

forex_watch_runs and forex_watch_opportunities are optional operational
provenance. When present, they are joined by source_run_id, run_id,
decision_id, and opportunity key. A missing watcher table does not invalidate
a standalone forex-shadow decision; provenance is marked limited.

forex_watcher_state is read only for bounded diagnostics and is never treated
as a decision or market state.

### 7.4 Available versus unavailable features

The v1 importer may use only values persisted in the decision snapshot and
deterministic values derived from its analysis timestamp:

| Feature family | V1 source | Policy |
|---|---|---|
| bid/ask/spread | snapshot_json.quote and explicit analysis fields | spread points is a feature; quote sides are provenance |
| volatility/range | snapshot_json.features.<TF>.average_true_range, range_pct, range | use persisted values only |
| return/momentum | snapshot_json.features.<TF>.return_over_bars, direction | encode direction deterministically |
| price position | snapshot_json.features.<TF>.close_position | use persisted value |
| time | analysis_snapshot_timestamp or validated legacy timestamp | UTC hour encoding only |
| symbol/profile/timeframe | decision columns and snapshot symbol | exact compatibility gates |

No persisted Phase 4/5/6 data is assumed to contain order-book depth,
queue imbalance, OFI, VPIN, microprice, Kyle lambda, Hawkes intensity,
liquidity, or a reliable latency feature. Those fields are UNAVAILABLE in v1
and are never filled from future prices, books, or external services.

## 8. Experience identity and record

One logical shadow decision creates at most one experience_id:

~~~
experience_id = "exp1-" +
  SHA256("phase8.experience.v1\0" + source_decision_id).hexdigest()[:32]
~~~

source_decision_id is the Phase 5 primary key. The importer also stores
source_database_id, source_decision_fingerprint, and the complete
source/evaluation provenance. If an apparently identical decision ID arrives
with a different source fingerprint, the importer records
SOURCE_DECISION_CONFLICT and retains the prior immutable projection; it never
silently overwrites it.

`source_aliases` records every configured source database/row alias that has
presented the same source_decision_id and source_decision_fingerprint. An
ExperienceRecord is active iff at least one CURRENT, nonremoved source alias
still points to its accepted decision fingerprint. Removing one duplicate
alias therefore leaves the logical experience active. Removing the last
current alias makes it historical/tombstoned. A failed or unavailable source
scan is not evidence of removal and must not change alias currentness; alias
status transitions and scan provenance remain auditable.

The logical record contains:

~~~
ExperienceRecord
  experience_id
  source_database_id
  source_aliases
  source_decision_id
  source_run_id
  symbol
  requested_symbol
  analysis_profile
  analysis_timeframe
  analysis_snapshot_timestamp
  decision_completed_timestamp
  decision_reference_timestamp
  market_state
  decision_evidence
  outcome_evidence_by_basis_horizon
  trust
  provenance
  experience_schema_version
  feature_schema_version
  feature_extractor_version
  similarity_profile_version
  trust_policy_version
  statistics_policy_version
  source_decision_fingerprint
  source_evaluation_fingerprints
~~~

decision_evidence is a safe structured projection:

- exact normalized action when available;
- raw PM structured field names and scalar values from this allow-list:
  rating, analysis_profile, valid_for_seconds, time_horizon, price_target,
  confidence, and schema/version identifiers when present. String values are
  bounded and no narrative field (executive_summary, investment_thesis,
  rationale, reasoning, or strategic_actions) is copied;
- normalization/context status and bounded error code;
- provider/model identifiers;
- valid-for/profile/timeframe metadata;
- state artifact presence/size counters when present in metrics_json; and
- executed, which must be false for a normal record.

The projection does not copy trader_summary, bull_summary, bear_summary, or
unrestricted PM prose into similarity data. Diagnostic text, if enabled, is
separately bounded and status-only as described in Section 15.

outcome_evidence_by_basis_horizon references the exact source evaluation
fingerprint and stores only the existing Phase 5 fields required for explicit
statistics and provenance. It does not collapse rows into a single outcome.

## 9. Immutable source evidence and versioned projections

The source decision row and source evaluation rows are immutable inputs from
Phase 8's perspective. A source evaluation may legitimately move from
PENDING to COMPLETE, PENDING to DATA_UNAVAILABLE, or recovered
DATA_UNAVAILABLE to COMPLETE; that transition changes the evaluation
fingerprint while preserving decision_id, basis, horizon, timestamps, and the
prior unavailable reason.

The catalog therefore separates:

- experience_records: logical identity and latest observed source fingerprints;
- experience_source_aliases: one row per configured source database alias,
  with CURRENT/REMOVED state, accepted decision fingerprint, and scan
  provenance;
- experience_outcome_snapshots: append-only source-evaluation projections,
  keyed by experience, basis, horizon, and evaluation fingerprint;
- experience_feature_projections: versioned feature values/masks;
- experience_similarity_generations: active projection metadata; and
- experience_import_events: append-only skip/import/recovery/conflict events.

Only a new Phase 8 projection generation may become active. A failed import or
rebuild leaves the previous valid generation and all previous source
provenance queryable. Re-importing an unchanged decision/evaluation is an
idempotent skip event, not a new experience.

The following version identities are mandatory:

~~~
experience_schema_version = "phase8.experience.v1"
feature_schema_version = "experience-features.v1"
feature_extractor_version = "phase8-feature-extractor.v1"
similarity_profile_version = "similarity-profile.v1"
trust_policy_version = "trust-policy.v1"
statistics_policy_version = "statistics-policy.v1"
projection_version = "experience-projection.v1"
~~~

Every projection stores the source database/schema fingerprints and a
population fingerprint (ordered experience IDs plus feature fingerprints).
No vector or normalized row from an incompatible version is silently mixed.

## 10. ExperienceFeatureSchemaV1

### 10.1 Categorical compatibility fields

The default similarity query must match:

- exact resolved_symbol (requested symbol is retained for provenance);
- exact analysis_profile;
- exact analysis_timeframe;
- feature_schema_version; and
- a valid source market-state contract.

point and digits are validated symbol metadata and retained with the feature
row. They are not silently inferred and are not distance dimensions. V1 does
not define fuzzy or cross-timeframe compatibility: resolved_symbol,
analysis_profile, analysis_timeframe, feature_schema_version, and
feature_extractor_version must all match exactly. A future compatibility table
would require an explicit versioned design change.

### 10.2 Numeric fields

The v1 numeric vector is ordered and versioned as follows:

~~~
spread_points

M1.return_over_bars
M1.range_pct
M1.close_position
M1.average_true_range
M1.direction_code

M5.return_over_bars
M5.range_pct
M5.close_position
M5.average_true_range
M5.direction_code

M15.return_over_bars
M15.range_pct
M15.close_position
M15.average_true_range
M15.direction_code

H1.return_over_bars
H1.range_pct
H1.close_position
H1.average_true_range
H1.direction_code

utc_hour_sin
utc_hour_cos
~~~

direction_code is an explicit encoding of the persisted field:
UP = 1.0, FLAT = 0.0, DOWN = -1.0, and INSUFFICIENT_DATA = missing.
utc_hour_sin and utc_hour_cos are deterministic encodings of the validated
analysis-snapshot UTC hour and are not broker-clock estimates.

candle_count, latest OHLC, account, positions, broker-clock calibration, and
raw candle arrays are retained as provenance/diagnostic data but are not
distance dimensions in v1. Absolute bid/ask price levels are not dimensions,
so one old price level cannot dominate the market-state distance.

### 10.3 Missing values and feature extraction

Missing values are represented by a separate mask bit and are never replaced
with numeric zero. Non-finite values, malformed nested JSON, invalid point or
digits, or a missing symbol/profile/timestamp produce a typed extraction
diagnostic. A row may remain queryable as limited/diagnostic history when it
has enough valid fields; it is not silently repaired.

The extractor records each value's source JSON path, conversion rule, and
missing reason. It does not read any evaluation table while constructing the
market vector. The feature fingerprint covers the ordered names, values,
mask, source paths, and extractor version.

## 11. Trust policy

Trust is a deterministic record-level classification and is separate from
per-basis/per-horizon outcome eligibility.

### TIER_A_HIGH_TRUST

Assign Tier A only when all of the following hold:

1. executed == 0;
2. decision_context_status == COMPLETE;
3. normalization_status == NORMALIZED and action is exactly BUY, SELL, or
   HOLD;
4. the analysis-snapshot timestamp is timezone-aware UTC;
5. the pre-decision quote/point/digits and the complete v1 market feature
   vector are finite and valid;
6. decision completion, when present, is not before the snapshot and its
   latency is consistent;
7. no decision-reference field is marked INVALID_TEMPORAL; and
8. source decision, watcher linkage when available, and feature provenance
   fingerprints are internally consistent.

### TIER_B_LIMITED

Assign Tier B when the decision has a valid, usable pre-decision market state
and stable identity, but one or more non-critical limitations apply: a partial
feature mask with the minimum overlap still available, an old row without
explicit completion/reference fields, unavailable/stale decision reference,
incomplete or unavailable outcome evidence, missing watcher provenance, or
another explicitly recorded limitation. Tier B is never silently treated as
Tier A.

### TIER_C_DIAGNOSTIC_ONLY

Assign Tier C when the row is unsuitable for trading statistics or market
similarity: normalization/model/provider failure; materially incomplete
decision context; missing/invalid action; invalid/non-UTC analysis timestamp;
malformed required quote/point/digits; executed != 0; source conflict,
schema/provenance violation, or insufficient market features for the v1
minimum overlap. Tier C remains visible for reliability and debugging.

The exact reason codes are persisted (for example,
NORMALIZATION_FAILED, CONTEXT_INCOMPLETE, TEMPORAL_INVALID,
MARKET_FEATURES_INSUFFICIENT, SOURCE_CONFLICT, or PROVENANCE_INVALID). A new
trust-policy version is required for any rule change.

Tier C is diagnostic/history evidence only. It is excluded from default
numeric market-state similarity and from default outcome statistics. Tier C
remains queryable through structured status/trust/reason/provenance filters
and the optional bounded diagnostic FTS5 projection. V1 defines no opt-in Tier
C numeric nearest-neighbor mode; adding one later would require a separate
versioned policy and explicit caller intent.

## 12. Outcome eligibility and temporal semantics

### 12.1 Explicit basis and horizon

Every outcome request names exactly one evaluation_basis and one
horizon_seconds. Valid values are the Phase 5 values, including
ANALYSIS_SNAPSHOT versus DECISION_REFERENCE and 5m/15m/30m/60m. No
aggregation crosses either dimension.

For a requested row, descriptive-statistics eligibility requires:

1. the caller-selected trust tier is allowed by the statistics policy;
2. the source evaluation row exists with the exact requested basis/horizon;
3. evaluation_status == COMPLETE;
4. source_context_eligible == 1;
5. all fields required by the requested statistic are finite/present; and
6. when as_of is supplied, the evaluation evidence and snapshot version were
   available by that historical cutoff (Section 12.4).

PENDING, DATA_UNAVAILABLE, INELIGIBLE, and missing rows are explicit
exclusions. training_eligible and training_eligibility_reason are imported
only as source provenance; they do not gate Phase 8 descriptive statistics.
Phase 8 never sets or infers training eligibility.

The default StatisticsPolicyV1 allows Tier A only. A caller may explicitly
request Tier B, but results retain separate per-tier counts and a limitation
warning; Tier B is never silently merged with Tier A. Tier C is never allowed
for trading statistics.

### 12.2 Analysis versus actionable evidence

ANALYSIS_SNAPSHOT statistics describe signal/reasoning quality anchored at the
quote analyzed by the graph. They are not described as executable entry
performance.

DECISION_REFERENCE statistics describe the fresh broker quote captured after
completion and are actionable-counterfactual evidence only when the reference
status is AVAILABLE and temporal evidence is valid. An unavailable or invalid
reference is not substituted with the analysis snapshot.

The analysis timestamp, completion timestamp, reference broker timestamp,
reference delay, both quote sets, and all evaluation provenance remain
separate in every hit and statistic response. A slow stale decision may be
useful for offline signal research while being ineligible for executable
performance evidence.

### 12.3 Leakage-safe as_of

ExperienceQuery.as_of is optional. When present, a candidate is eligible only
when its completed decision was available strictly before the cutoff. Define:

~~~
experience_available_at = decision_completed_timestamp
~~~

For a normally completed decision, all of the following are required:

~~~
candidate.analysis_snapshot_timestamp < as_of
candidate.experience_available_at < as_of
~~~

The comparisons are strict and UTC-normalized. A candidate at exactly as_of
is excluded. A decision that began before as_of but completed at or after it
is not exposed, including its action and trust record. Legacy rows without a
trustworthy completion timestamp are excluded from historical similarity with
the explicit reason AVAILABILITY_TIMESTAMP_UNKNOWN rather than guessed.
Phase 8 does not obtain as_of from MT5; a future caller provides it. Outcome
rows are never used to decide whether a candidate passes the temporal
market-state filter.

### 12.4 Historical evaluation availability

When an outcome-statistics request supplies as_of, an evaluation snapshot is
eligible only if its observation/evidence occurred no later than as_of and the
evaluation state/version was known by as_of. The importer preserves
observation_timestamp, evaluated_at, created_at,
recovered_from_unavailable_at, previous_unavailable_reason, and evaluation
fingerprints so the calculator can select the latest snapshot that was
actually available by the cutoff.

For deterministic selection, `evaluation_available_at` is
`recovered_from_unavailable_at` for a recovered snapshot, otherwise
`evaluated_at` when present, otherwise `created_at`; the selected timestamp
must be trustworthy UTC and no later than as_of. The calculator requires
`observation_timestamp <= as_of`, filters snapshots by that availability time,
then selects the latest one (ties break by evaluation fingerprint). This
preserves the prior DATA_UNAVAILABLE snapshot for an earlier cutoff while
allowing the recovered COMPLETE snapshot after its recovery time.

For example, if an evaluation is DATA_UNAVAILABLE at 13:00 and recovered to
COMPLETE at 15:00, a query with as_of=14:00 must not see the 15:00 COMPLETE
state; a query with as_of=16:00 may use it when all other eligibility gates
pass. A recovery timestamp after as_of cannot retroactively make historical
statistics complete.

## 13. Similarity algorithm

### 13.1 Query contract

The public query boundary is conceptually:

~~~
ExperienceQuery(
    market_state: Mapping[str, object],
    top_k: int = 50,
    symbol: str | None = None,
    analysis_profile: str | None = None,
    analysis_timeframe: str | None = None,
    as_of: datetime | None = None,
    trust_tiers: tuple[str, ...] = (
        "TIER_A_HIGH_TRUST",
        "TIER_B_LIMITED",
    ),
    action_filter: str | None = None,
)
~~~

market_state is a structured v1 feature payload or a validated
MarketStateVector; arbitrary text is rejected. action_filter, when explicitly
supplied, filters candidates after identity/time/trust gates. It does not
enter the distance or score. The default similarity score is therefore
invariant under changing a candidate's BUY/SELL/HOLD action.

The service returns a read-only result envelope:

~~~
ExperienceSearchResult(
    hits: tuple[ExperienceHit, ...],
    query_normalization_fingerprint,
    active_generation_id,
    candidate_count,
    excluded_counts,
)
~~~

Each ExperienceHit retains the existing fields (experience_id,
source_decision_id, similarity_score, distance, comparable_feature_count,
trust_tier, market_state, action, timestamps, outcome_availability,
provenance, feature_schema_version, and similarity_profile_version).
The query-level result records the deterministic normalization fingerprint
used for this search; orchestrator provenance carries it alongside the hits.

Scores are experience-similarity scores only. They are not comparable with
Phase 7 semantic or lexical scores.

### 13.2 Compatibility and minimum overlap

Before distance calculation, apply:

1. exact symbol match when either query or configuration specifies a symbol;
2. exact analysis_profile and exact analysis_timeframe compatibility;
3. matching feature schema and extractor version;
4. valid as_of test; and
5. minimum overlap.

V1 requires at least 8 comparable numeric fields and at least 50% of both the
query's and candidate's non-missing dimensions. A candidate with two matching
values cannot appear highly similar to a full vector. If no candidates satisfy
the gates, return an explicit INSUFFICIENT_COMPARABLE_FEATURES result/error
rather than relaxing them.

### 13.3 Robust normalization and weights

SimilarityProfileV1 stores, for every numeric feature:

- median;
- IQR;
- fallback scale when IQR is zero;
- missing-mask policy;
- weight; and
- the population fingerprint from which the statistics were computed.

For an overlapping feature, the scale is selected deterministically:

~~~
if IQR > configured_epsilon:
    scale = IQR
elif MAD-derived scale is available and > configured_epsilon:
    scale = MAD-derived scale
else:
    scale = versioned_per_feature_fallback_scale
z = clip((value - median) / scale, -8.0, 8.0)
~~~

The configured epsilon, MAD conversion, per-feature fallback scales, and
fallback-selection policy are persisted in the profile metadata. A zero IQR
must never silently use 1e-12 as the effective scale, because that would
amplify insignificant differences. Any fallback-policy change requires a new
similarity-profile version.

V1 weights are explicit and versioned:

- spread_points: 2.0;
- each per-timeframe return/range/position/ATR/direction field: 1.0; and
- utc_hour_sin/utc_hour_cos: 0.5.

The distance is the weighted root-mean-square of the normalized differences
over the overlap only. Missing fields are excluded, never imputed as zero.
The score is 1 / (1 + distance), a finite monotonic presentation score.
Changing medians, IQRs, weights, clipping, or missing policy creates a new
similarity-profile version and generation.

Normalization statistics are computed only from valid pre-decision feature
rows, never from action, outcome, evaluation, or future fields. Tier C rows do
not define the default profile population. A normal query without as_of uses
the active persisted SimilarityProfile generation. A historical query with
as_of resolves an AS-OF-safe normalization view using only experiences whose
availability timestamps are strictly before as_of (the same temporal boundary
used for candidate eligibility). V1 computes this exact population at query
time when needed; no future row may influence its medians, IQRs, fallback
scales, normalized values, distances, or ranking.

Every search result persists/returns a deterministic
query_normalization_fingerprint containing at least:

- feature_schema_version;
- similarity_profile_version;
- normalization_cutoff/as_of and the eligible population fingerprint;
- median/IQR/fallback-policy fingerprint; and
- the ordered feature/missing-mask policy fingerprint.

### 13.4 Exact CPU index and ordering

SimilarityIndex is the replaceable interface:

~~~
search(
    query_vector,
    query_mask,
    candidate_ids,
    top_k,
    profile,
) -> tuple[SimilarityHit, ...]
~~~

The v1 implementation loads a float32 matrix and uint8 mask, applies
vectorized NumPy operations to gated rows, and returns exact distances. It
does not introduce an ANN service or GPU dependency.

Ordering is deterministic:

1. ascending distance;
2. ascending analysis_snapshot_timestamp; and
3. ascending experience_id.

The same corpus, query, and generation must produce byte-for-byte identical
ordered IDs and scores.

## 14. Experience outcome statistics

Statistics are calculated only after similar IDs have been retrieved. They
never affect retrieval ranking, profile normalization, or candidate selection.

The request is conceptually:

~~~
OutcomeStatsRequest(
    experience_ids,
    evaluation_basis,
    horizon_seconds,
    trust_tiers=("TIER_A_HIGH_TRUST",),
)
~~~

The result always exposes the requested basis/horizon, policy versions,
eligible sample denominator, excluded counts by reason/status/tier, and source
evaluation fingerprints.

For eligible COMPLETE rows, directional fields include, separately for BUY and
SELL:

- sample count;
- positive-net count/rate;
- negative-net count/rate;
- zero count;
- mean and median net points; and
- deterministic MFE/MAE summaries and selected quantiles when enough values
  exist.

BUY and SELL counterfactuals are retained for every eligible row regardless of
the selected action. The selected-action fields are reported separately and
are never substituted for the missing direction.

HOLD is not a winning trade and has no normal win rate. HOLD output contains:

- HOLD sample count;
- opportunity-cost distribution from hold_opportunity_cost_points;
- BUY-side and SELL-side missed-opportunity distributions/counts; and
- best-counterfactual action/counts when the source field is present.

The existing Phase 5 semantics are preserved:
hold_opportunity_cost_points is max(0, buy_net_points, sell_net_points), so it
is zero when both directions lose. Every statistic names its denominator. No
output is called a profitability claim.

## 15. Diagnostic history

Tier C records remain queryable by deterministic fields: status, trust reason,
normalization/context status, model/provider ID, profile, symbol, source run,
evaluation status, unavailable reason, and version/provenance identifiers.

A local SQLite FTS5 diagnostic projection may index only bounded, sanitized
status/error labels and metadata. It must not index raw PM reports, debate
history, hidden reasoning, prompts, account details, or future prices. V1 does
not add a second semantic embedding model for diagnostic prose.

## 16. Incremental import, removal, and rebuild

### 16.1 Explicit incremental import

experience import is the only operation that reads configured source
databases. For each source decision:

1. compute the deterministic source decision fingerprint;
2. join watcher/evaluation provenance when available;
3. extract only the safe decision and pre-decision market fields;
4. classify trust and record bounded reasons;
5. upsert a new logical record only when its source identity is new;
6. append a new evaluation snapshot only when a basis/horizon fingerprint
   changes; and
7. stage feature/projection updates for one atomic generation.

Unchanged decision and evaluation fingerprints produce UNCHANGED events.
Duplicate database copies with the same decision ID and fingerprint produce a
single logical experience plus a source alias event. Alias currentness is
computed across all configured source databases, not from whichever source was
scanned most recently. Different decision IDs remain different experiences
even when their bytes look similar.

### 16.2 Changes and recovery

A changed decision row is SOURCE_DECISION_CONFLICT; the old derived
projection remains active and the new row is quarantined for operator review.
Recoverable evaluation changes are allowed and retain
previous_unavailable_reason, recovery time, source timestamps, and both
fingerprints.

After a successful scan proves that a source row is absent, the importer marks
only that source alias REMOVED and records SOURCE_MISSING. The logical record
remains active while another CURRENT alias points to the same accepted
fingerprint; it becomes an inactive historical tombstone only when the last
current alias is removed. Tombstones are excluded from default current
similarity/statistics queries but remain auditable. An unavailable/failed scan
does not mark any alias removed, and no record is deleted merely because a
source database was copied or temporarily unavailable.

experience rebuild explicitly rebuilds feature matrices, normalization
statistics, diagnostic projections, and the active generation from retained
catalog source snapshots. It never silently refreshes source databases;
experience import precedes a source refresh.

### 16.3 Crash recovery and single writer

An OS/file lock at locks/import.lock permits one importer/rebuilder per
artifact root. Each run writes under .staging/<run-id>, records a RUNNING
event, validates every artifact, and publishes the generation and pointer
atomically. On restart, RUNNING runs are marked INTERRUPTED; complete prior
generations remain active. A failed record is quarantined without rolling back
unrelated valid records.

## 17. Evidence Orchestrator

### 17.1 Read-only API

The orchestrator accepts no action recommendation, order, prompt, or
model-generation parameter. An optional action_filter is only a caller-requested
filter over historical source actions; it never becomes a recommendation or a
similarity dimension:

~~~
EvidenceRequest(
    research_question: str | None = None,
    market_state: Mapping[str, object] | None = None,
    knowledge_top_k: int = 10,
    experience_top_k: int = 50,
    content_types: tuple[str, ...] = (),
    document_ids: tuple[str, ...] = (),
    symbol: str | None = None,
    analysis_profile: str | None = None,
    analysis_timeframe: str | None = None,
    evaluation_basis: str | None = None,
    horizon_seconds: int | None = None,
    as_of: datetime | None = None,
    trust_tiers: tuple[str, ...] = (),
    action_filter: str | None = None,
)
~~~

evaluation_basis and horizon_seconds are either both absent or both present.
If present, they request descriptive statistics only; they do not change
similarity.

The result is:

~~~
EvidenceBundle(
    status = COMPLETE | PARTIAL | EMPTY | FAILED,
    knowledge: tuple[KnowledgeHit, ...],
    experience: tuple[ExperienceHit, ...],
    statistics: OutcomeStatistics | None,
    source_status: {knowledge: ..., experience: ...},
    warnings: tuple[EvidenceWarning, ...],
    errors: tuple[EvidenceSourceError, ...],
    provenance: OrchestrationProvenance,
)
~~~

KnowledgeHit values and their semantic/lexical/rerank scores are returned
unchanged. ExperienceHit values contain experience similarity scores only and
may expose the historical source action as evidence. The bundle has no
combined score, top-level recommendation, confidence, order, or portfolio
field.

### 17.2 Independent source behavior

- With only research_question, query Phase 7 Knowledge RAG; the experience
  collection is empty.
- With only market_state, query Experience Memory; the knowledge collection
  is empty.
- With both, query both independently and preserve each source's scores and
  provenance.
- With an explicit basis/horizon, calculate statistics only over the returned
  experience IDs after retrieval, passing through as_of when supplied so
  evaluation availability is historically bounded as well.
- If one source fails, return PARTIAL with successful evidence plus a typed,
  source-specific error. If both fail, return FAILED; never fabricate a
  fallback.

The orchestrator uses the existing Phase 7 read-only query service. A
knowledge query may construct the local query embedder required by Phase 7,
but it must not construct a parser, ingestor, vector writer, lexical writer,
source scanner, MT5 provider, or LLM client. Experience queries construct
only read-only catalog/index readers.

### 17.3 Provenance

Every experience hit and statistic retains:

- Phase 8 source artifact/database identity and snapshot fingerprint;
- source_decision_id, source_run_id, and source decision fingerprint;
- exact basis/horizon and source evaluation fingerprint/status when used;
- analysis/reference/completion timestamps and their temporal basis;
- feature schema/extractor/similarity/trust/statistics versions; and
- the active generation/population fingerprint and, for historical queries,
  the query_normalization_fingerprint used for scaling/ranking.

Every knowledge hit retains Phase 7 document/chunk/source hash, page/section,
content type, parser/chunker/embedding/index versions. The orchestrator never
rewrites either provenance into anonymous text.

## 18. CLI

Phase 8 adds an explicit experience command surface. It never changes the
stock tradingagents command or the forex collector commands:

~~~
experience import   --source-db PATH [--source-db PATH ...] [--artifact-root PATH]
experience rebuild  [--artifact-root PATH]
experience status   [--artifact-root PATH] [--json]
experience list     [--state STATE] [--trust-tier TIER] [--artifact-root PATH]
experience show     EXPERIENCE_ID [--artifact-root PATH] [--json]
experience similar  --market-state-json PATH [filters] [--artifact-root PATH]
experience stats    --basis BASIS --horizon-seconds N --experience-id ID ...
experience quarantine [--artifact-root PATH] [--json]
experience evidence [--question TEXT] [--market-state-json PATH] [filters]
~~~

import and rebuild are the only writer operations, and they write only under
data_cache/experience. They do not modify source DBs. status, list, show, and
quarantine read only the Phase 8 catalog. similar and stats read only the
active Phase 8 generation. evidence may invoke the existing local Phase 7
query boundary for a question, but it never starts Phase 7 ingestion or
creates any writer.

The CLI accepts no credential, model, MT5, order, prompt, or execution
option. It emits typed errors and bounded JSON diagnostics, never source
reports or reasoning.

## 19. Failure and recovery semantics

The implementation must expose typed, deterministic states/errors at least
for:

~~~
SourceDatabaseUnavailableError
SourceSchemaIncompatibleError
SourceSnapshotChangedError
SourceDecisionConflictError
DecisionEvidenceMalformedError
FeatureExtractionIncompleteError
InsufficientComparableFeaturesError
SimilarityProjectionIncompatibleError
SimilarityProfileMismatchError
TrustPolicyMismatchError
EvaluationEvidenceUnavailableError
Phase7KnowledgeUnavailableError
ExperienceMemoryUnavailableError
PartialEvidenceError
ProvenanceViolationError
ExperienceImportLockedError
ExperienceImportInterrupted
~~~

One malformed source row is quarantined with its key, stage, bounded error
type/message/fingerprint, and source fingerprints; it does not destroy other
records or the last valid projection. A missing optional evaluation or watcher
table is represented as unavailable provenance, not silently treated as
complete.

No error handler may fall back to a cloud API, MT5 read, LLM summary,
future-price lookup, or fabricated value.

## 20. Versioning and reproducibility

An active similarity generation is compatible only when all of the following
match the query/index contract:

- experience schema;
- feature schema and ordered field list;
- feature extractor version;
- similarity profile version and normalization statistics;
- population fingerprint;
- trust policy version;
- statistics policy version;
- source Phase 5/6 schema fingerprints; and
- projection/index version.

The generation manifest records Python/package/runtime information useful for
reproduction, NumPy dtype, matrix shape, mask policy, feature weights,
clipping/fallback scales, and deterministic ordering policy. A mismatch
raises a typed error before distance or statistics calculation. Dimensions
alone are never sufficient compatibility evidence.

## 21. Security and privacy

All source databases, features, outcomes, books, and evidence bundles remain
local. Normal operation requires no network and stores no credentials.

The importer stores no hidden chain-of-thought, prompts, or unrestricted model
responses. Raw PM structured values are bounded and allow-listed. Diagnostic
messages are sanitized and length-limited. Account, position, and broker
details are not similarity dimensions and are not emitted by default.

The package has no hosted/cloud client dependency and no import path into
execution or MT5. A future caller may combine the returned evidence at a
higher layer, but Phase 8 itself does not send it to any model or trading
workflow.

## 22. Windows and 16 GB performance

The v1 matrix uses float32 values and a compact uint8 mask. Candidate filters
run before the vectorized scan. SQLite connections are short-lived for reads;
the single writer uses a bounded lock and staged files. No server, CUDA, GPU,
or background daemon is required.

The exact scan is the default until a measured corpus reaches at least
100,000 feature rows, the matrix exceeds 256 MB, or benchmarked p95
interactive similarity latency exceeds 250 ms on the target machine. Those
thresholds trigger a future design review only; they do not authorize ANN
implementation in Phase 8.

Import and rebuild latency are maintenance metrics. Query benchmarks record
candidate count, comparable dimensions, p50/p95 latency, and peak resident
memory without recording source text.

## 23. Deterministic testing strategy

All normal Phase 8 tests use tiny SQLite fixtures, synthetic snapshots, fake
knowledge readers, and fake similarity matrices. They do not call MT5,
Ollama, Qwen, hosted APIs, a network service, a real 30-minute analysis,
Docling, FastEmbed, LanceDB, CUDA, or the new books folder.

The implementation test matrix must cover:

- first import of decisions, evaluations, and watcher provenance;
- unchanged import skip;
- recoverable evaluation becoming COMPLETE;
- duplicate/idempotent import;
- source database read-only guarantee and no DDL/DML;
- unavailable source and schema incompatibility;
- Tier A/B/C classification and trust-policy versioning;
- missing market features and safe diagnostic quarantine;
- feature/extractor/profile/generation mismatch;
- minimum comparable-feature threshold;
- exact similarity ordering and stable equal-distance ties;
- symbol/profile/timeframe compatibility;
- exact (not fuzzy) symbol/profile/timeframe/schema/extractor compatibility;
- strict as_of candidate and completed-decision availability exclusion,
  including analysis-before-cutoff/completion-after-cutoff;
- historical evaluation availability: evaluation completion/recovery after
  as_of is excluded while the prior state is visible, and becomes eligible
  only after the recovery cutoff;
- historical normalization future-leakage test: add extreme experiences after
  as_of and prove IDs, order, scores, and query normalization fingerprint are
  unchanged;
- outcome-leakage test: change net_points, MFE, MAE, result/status, and prove
  similarity IDs/order/scores are identical;
- action-leakage test: change BUY/SELL/HOLD and prove default similarity is
  identical (explicit action filters remain filter-only);
- basis separation and horizon separation;
- Tier C and incomplete outcomes excluded from default statistics;
- BUY/SELL directional denominators and positive-rate math;
- HOLD statistics kept separate from normal win rate and opportunity-cost
  semantics;
- recovered evaluation provenance and previous-unavailable reason;
- source-alias currentness: remove one duplicate alias (experience stays
  active), remove the last alias (tombstone), failed scan (no removal), and
  conflicting fingerprint (quarantine without replacement);
- Tier C excluded from default numeric similarity while remaining available
  through diagnostic/status retrieval;
- zero-IQR normalization uses the configured MAD/per-feature fallback and
  never an implicit 1e-12 scale;
- a COMPLETE evaluation with source_context_eligible=1 and
  training_eligible=NULL contributes to descriptive statistics;
- knowledge-only, experience-only, and combined orchestrator queries;
- partial-source failure and typed error isolation;
- complete provenance on every hit/statistic;
- separate Phase 7/Phase 8 paths and tables;
- no top-level action/recommendation/order fields in the orchestrator result
  (a historical source action may remain on an ExperienceHit);
- no MT5/forex/execution imports;
- offline/no-network operation; and
- interrupted import/rebuild recovery with last-generation preservation.

The outcome-leakage and future-data tests are mandatory gates, not optional
quality checks.

## 24. Retrieval-quality benchmark

The benchmark is a deterministic experience-retrieval benchmark, not a
profitability test. Fixtures contain known market states, missing masks,
symbols/profiles, actions, and outcome variants.

Measure:

- nearest-neighbor correctness;
- expected-neighbor Recall@K;
- compatibility/filter correctness;
- strict future-leakage violations (must be zero);
- outcome-leakage violations (must be zero);
- provenance correctness (must be 100%);
- stable repeated-query ordering;
- per-query comparable-feature counts; and
- p50/p95 CPU retrieval latency and bounded memory.

The benchmark has six zero-violation leakage gates:

1. changing net/MFE/MAE/outcome leaves default similarity unchanged;
2. changing BUY/SELL/HOLD leaves default similarity unchanged;
3. a candidate unavailable at or after as_of is excluded;
4. analysis before as_of but completion at/after as_of is excluded;
5. adding extreme future rows leaves historical normalization, IDs, order, and
   scores unchanged; and
6. evaluation/recovery that became available after as_of cannot affect
   historical statistics.

A future candidate at or after as_of must never be returned. The benchmark must
report exclusions and denominators, not merely that a query returned
something.

## 25. Acceptance criteria

Phase 8 is complete only when the implementation can demonstrate:

~~~
read-only Phase 5/6 source rows
    -> versioned one-record-per-decision import
    -> explicit Tier A/B/C trust
    -> deterministic pre-decision feature projection
    -> leakage-safe exact similarity with strict compatibility/as_of gates
    -> top similar experiences with complete provenance
    -> explicit basis/horizon statistics, including separate HOLD semantics

Phase 7 Knowledge RAG
    +
Phase 8 Experience Memory
    -> read-only EvidenceBundle
    -> separate scores, stores, trust, and provenance
~~~

Acceptance also requires:

- source SQLite files are unchanged byte-for-byte and are opened read-only;
- all decisions, including incomplete/failed ones, are represented or
  explicitly quarantined;
- duplicate decisions do not create duplicate logical experiences;
- recoverable evaluations retain their audit history;
- source removals are tombstoned rather than silently deleted;
- incompatible generations fail closed;
- action, outcome, candidate/completion, normalization, and evaluation
  future-data leakage tests pass with zero violations;
- the bounded real local acceptance smoke imports an already-produced Phase
  5/6 SQLite database in read-only mode and demonstrates real decisions,
  evaluations, feature extraction, Tier A/B/C distribution, normalization,
  exact similarity, explicit basis/horizon statistics where available, and a
  combined Phase 7 Evidence Orchestrator query. It must not run a new
  30-40-minute Qwen analysis;
- the real smoke records the source database path, decision/evaluation/import
  counts, Tier A/B/C counts, quarantine and duplicate/alias counts, feature
  population and active generation IDs, normalization population and
  fingerprint, similarity examples with comparable-feature counts,
  basis/horizon selected, eligible/excluded statistic counts, Phase 7 result
  count, EvidenceBundle status, query latency, and network-attempt count; and
- before/after hashes, sizes, and timestamps/metadata (including WAL when
  present) prove that the source SQLite database was unchanged;
- no Phase 7 table/catalog is reused;
- no MT5, TradingAgents, execution, LLM, training, or Phase 9 path is
  reachable from the package; and
- no recommendation or profitability label is emitted.

## 26. Explicit Phase 9 handoff boundary

Phase 8 hands off only:

- a local, versioned ExperienceQuery/ExperienceHit read-only contract;
- explicit descriptive OutcomeStatistics for a caller-selected basis/horizon;
- the existing Phase 7 KnowledgeQuery/KnowledgeHit contract through a
  separate collection; and
- an EvidenceBundle with source status, warnings, and provenance.

Phase 9 may separately design how evidence is evaluated, selected, or
presented to a human or a future model. Such a phase would need a new
approval, safety review, temporal policy, and training/evaluation decision.
Phase 8 does not inject evidence into prompts, change a trading graph, assign
labels, train/fine-tune a model, generate strategies, or execute anything.

## 27. Design self-review

The specification was reviewed against the requested boundaries:

- no unfinished sections or unspecified owner of source/projection writes;
- source decisions/evaluations remain read-only and basis/horizon separated;
- training_eligible is provenance only and no longer gates descriptive stats;
- as_of is strict for both analysis and completed-decision availability, and
  evaluation snapshots are selected only when historically available;
- historical normalization is recomputed from the pre-as_of population, so
  future rows cannot affect medians, IQRs, fallback scales, distances, or
  ranking;
- similarity uses no outcomes, actions, or future fields;
- HOLD is not counted as a normal win and uses existing opportunity-cost
  semantics;
- Tier A/B/C trust is separate from per-outcome eligibility, and Tier C is not
  in default numeric similarity;
- source-alias currentness is derived across all configured aliases, with
  failed scans distinguished from confirmed removals;
- timeframe compatibility is exact in v1;
- zero-IQR scaling uses a persisted MAD/per-feature fallback policy rather than
  an arbitrary 1e-12 scale;
- the bounded real Phase 5/6 acceptance smoke is mandatory in addition to
  fixture-only CI;
- no raw chain-of-thought, prompts, or unrestricted reports are stored;
- Phase 7 Knowledge RAG remains in a physically separate artifact root and is
  queried only through its public read-only contract;
- no MT5, TradingAgents, LangGraph, Qwen, execution, training, or Phase 9
  behavior is modified;
- v1 uses exact CPU numeric search, not an ANN service or cloud/GPU stack;
- source schema, row, evaluation, feature, policy, and generation provenance
  are explicit; and
- one malformed source row or failed projection cannot destroy the last valid
  generation.

There are no blocking design questions. Implementation must validate the
required columns against the actual source database at runtime and fail
closed when a deployment contains an older incompatible schema.
