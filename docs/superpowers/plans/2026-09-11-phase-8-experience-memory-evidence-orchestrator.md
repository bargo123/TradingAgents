# Phase 8 Experience Memory + Evidence Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, deterministic, read-only Experience Memory over Phase 5/6 shadow evidence and a read-only Evidence Orchestrator that composes Phase 7 Knowledge RAG with similar historical experience without changing trading behavior.

**Architecture:** Phase 5/6 SQLite databases remain authoritative read-only inputs. Phase 8 builds a separate SQLite experience catalog plus versioned numeric feature projections and exact NumPy similarity, then composes those results with the existing Phase 7 KnowledgeQueryService through a read-only EvidenceBundle.

**Tech Stack:** Python >=3.10, stdlib sqlite3, NumPy, SQLite FTS5 where required for bounded diagnostics, argparse, pytest, Ruff.

**Spec:** docs/superpowers/specs/2026-09-11-phase-8-experience-memory-evidence-orchestrator-design.md

**Implementation baseline:** `a878fb39c8df22e4dfbe8f049a5e52861233bfaa` (corrected Phase 8 design commit; verify again before execution).

## Global Constraints

- Phase 5/6 source SQLite databases are read-only.
- Open source DBs using SQLite URI `mode=ro`.
- Set `PRAGMA query_only=ON`.
- Never initialize/migrate Phase 5/6 stores.
- Experience artifacts live only under `data_cache/experience`.
- Phase 7 Knowledge artifacts remain physically separate.
- No Phase 7 table/catalog/index is reused for Experience Memory.
- One logical ExperienceRecord per accepted shadow decision.
- All decisions are retained/imported or explicitly quarantined.
- Tier A/B/C trust is deterministic and versioned.
- Tier C is diagnostic-only and not part of V1 numeric similarity.
- Similarity uses pre-decision market state only.
- Outcomes/actions must not affect default similarity.
- No future data may affect historical retrieval.
- `as_of` applies to decision completion, normalization population and outcome availability.
- Normalization is scoped to exact resolved symbol, analysis profile, analysis timeframe, feature schema, and feature extractor.
- Historical normalization respects requested trust tiers.
- Missing values use masks, never zero imputation.
- V1 exact similarity uses NumPy CPU scanning.
- No ANN server.
- No GPU/CUDA requirement.
- Statistics require explicit basis + horizon.
- Do not mix ANALYSIS_SNAPSHOT and DECISION_REFERENCE.
- Do not mix 5m/15m/30m/60m.
- `training_eligible` is provenance only and does not gate descriptive statistics.
- HOLD does not have a normal win rate.
- Knowledge scores and Experience similarity scores are never merged.
- Evidence Orchestrator is read-only.
- No Qwen/Ollama/LLM invocation from Phase 8.
- No TradingAgents/LangGraph modification.
- No MT5.
- No execution.
- No order simulation.
- No training/fine-tuning.
- No strategy generation.
- No Phase 9.

## Implementation map

The new `tradingagents/experience/` package is intentionally independent of
`tradingagents.forex`, `tradingagents.graph`, `tradingagents.agents`, MT5, and
the Phase 7 parser/index writers. `models.py` contains only stable public
value objects; `source_reader.py` is the only module that reads Phase 5/6
SQLite; `catalog.py` owns the separate Phase 8 SQLite writer; `features.py`,
`trust.py`, `normalization.py`, and `similarity.py` operate on Phase 8 models;
`query.py`, `outcomes.py`, and `orchestrator.py` are read-only services; and
`cli.py` exposes an explicit `experience` command without modifying the stock
or forex command surfaces.

Every task below is TDD: write the named failing test, run the exact RED
command, implement only the contract needed by that test, run the exact GREEN
command, then commit the task. A task may not skip its RED evidence or combine
unrelated behavior into a broad implementation.

## Task 1 — Public contracts, configuration, versions, and package boundary

**Files:**

- Create: `tradingagents/experience/__init__.py`
- Create: `tradingagents/experience/models.py`
- Create: `tradingagents/experience/config.py`
- Create: `tradingagents/experience/errors.py`
- Modify: `pyproject.toml` (add direct `numpy>=1.26` dependency; do not alter existing CLI behavior)
- Test: `tests/test_experience_models.py`
- Test: `tests/test_experience_config.py`

**Interfaces:**

- Produces `TrustTier`, `SourceAliasState`, `EvaluationStatus`, `ImportState`, `ExperienceRecord`, `ExperienceQuery`, `ExperienceHit`, `ExperienceSearchResult`, `OutcomeStatsRequest`, `OutcomeStatistics`, `EvidenceRequest`, and `EvidenceBundle` as frozen, JSON-serializable dataclasses/enums.
- Produces typed exception classes named in the design, including `SourceDatabaseUnavailableError`, `SourceSchemaIncompatibleError`, `SourceSnapshotChangedError`, `SourceDecisionConflictError`, `DecisionEvidenceMalformedError`, `FeatureExtractionIncompleteError`, `InsufficientComparableFeaturesError`, `SimilarityProjectionIncompatibleError`, `SimilarityProfileMismatchError`, `TrustPolicyMismatchError`, `EvaluationEvidenceUnavailableError`, `Phase7KnowledgeUnavailableError`, `ExperienceMemoryUnavailableError`, `PartialEvidenceError`, `ProvenanceViolationError`, `ExperienceImportLockedError`, and `ExperienceImportInterrupted`.
- `OutcomeStatsRequest` must have the exact field `as_of: datetime | None = None`.

- [ ] **Step 1: Write the failing contract tests**

```python
def test_outcome_stats_request_carries_as_of() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    request = OutcomeStatsRequest(("exp-1",), "ANALYSIS_SNAPSHOT", 300, as_of=cutoff)
    assert request.as_of == cutoff
    assert request.evaluation_basis == "ANALYSIS_SNAPSHOT"

def test_experience_query_defaults_to_tier_a_and_b() -> None:
    assert ExperienceQuery({}).trust_tiers == (
        TrustTier.TIER_A_HIGH_TRUST,
        TrustTier.TIER_B_LIMITED,
    )

def test_evidence_bundle_has_no_recommendation_field() -> None:
    assert not hasattr(EvidenceBundle, "recommendation")
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_models.py tests/test_experience_config.py -q`

Expected: FAIL because `tradingagents.experience` and its contracts do not yet exist.

- [ ] **Step 3: Implement the minimal contracts**

Use frozen dataclasses and `str, Enum` values. Require timezone-aware UTC for
timestamps at construction/validation boundaries. Keep action only as
historical evidence on `ExperienceHit`/`decision_evidence`; do not add an
action, recommendation, order, or portfolio field to `EvidenceBundle`.

Define `ExperienceSearchResult` with `hits`,
`query_normalization_fingerprint`, `active_generation_id`, `candidate_count`,
and `excluded_counts`. Define `ExperienceRecord.source_aliases` and
`ExperienceHit.currently_tombstoned`. Store
`training_eligible`/`training_eligibility_reason` only in a provenance mapping.
Add direct NumPy dependency without adding any cloud, MT5, LLM, or Phase 7
writer dependency.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_models.py tests/test_experience_config.py -q`

Expected: PASS, including serialization round trips and validation of exact
default trust tiers and `OutcomeStatsRequest.as_of`.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/experience/__init__.py tradingagents/experience/models.py tradingagents/experience/config.py tradingagents/experience/errors.py pyproject.toml tests/test_experience_models.py tests/test_experience_config.py
git commit -m "feat: add phase 8 experience contracts"
```

## Task 2 — Read-only source reader, schema validation, and fingerprints

**Files:**

- Create: `tradingagents/experience/source_reader.py`
- Create: `tradingagents/experience/identity.py`
- Test: `tests/test_experience_source_reader.py`
- Test: `tests/test_experience_identity.py`
- Test fixture helper: `tests/fixtures/experience_source_db.py`

**Interfaces:**

- Produces `ReadonlySourceReader(path).read_snapshot()` and
  `ReadonlySourceSnapshot` containing source schema, file/WAL fingerprints,
  ordered decision/evaluation/watcher rows, and read provenance.
- Produces `source_decision_fingerprint(row)`,
  `source_evaluation_fingerprint(row)`, and `source_snapshot_fingerprint(metadata)`.
- Consumes the Phase 5/6 table/column contract from design Section 2 without
  importing or initializing `ShadowDecisionStore`, `ShadowEvaluationStore`,
  or `WatcherStore`.

- [ ] **Step 1: Write the failing read-only tests**

```python
def test_reader_uses_mode_ro_and_query_only(source_db: Path) -> None:
    snapshot = ReadonlySourceReader(source_db).read_snapshot()
    assert snapshot.query_only is True
    assert snapshot.sqlite_uri.endswith("mode=ro")
    with sqlite3.connect(source_db) as connection:
        before = connection.execute("SELECT COUNT(*) FROM shadow_decisions").fetchone()[0]
    assert snapshot.decisions
    with sqlite3.connect(source_db) as connection:
        after = connection.execute("SELECT COUNT(*) FROM shadow_decisions").fetchone()[0]
    assert before == after

def test_source_fingerprint_is_key-order_independent() -> None:
    assert source_decision_fingerprint({"b": 2, "a": 1}) == source_decision_fingerprint({"a": 1, "b": 2})

def test_mutation_sql_is_rejected(source_db: Path) -> None:
    with pytest.raises(sqlite3.OperationalError):
        ReadonlySourceReader(source_db).execute_for_test("CREATE TABLE forbidden(x INTEGER)")
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_source_reader.py tests/test_experience_identity.py -q`

Expected: FAIL because the reader, fixture adapter, and fingerprint functions
are not present.

- [ ] **Step 3: Implement the minimal reader**

Open the absolute path with a SQLite URI formed as
`file:<escaped-path>?mode=ro`, execute `PRAGMA query_only=ON`, validate the
required table/column sets, and read rows in deterministic primary-key order.
Capture SHA-256, size, mtime, and WAL SHA-256/size before and after the read;
raise `SourceSnapshotChangedError` if any observed value changes. Convert
timestamps only to validated UTC values. The fixture-only mutation seam must
execute against the same read-only connection so DDL/DML fails visibly.

Canonicalize fingerprints with sorted-key JSON, stable separators, and explicit
source schema/version fields. Never call a Phase 5/6 store initializer or
perform DDL/DML on the source path.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_source_reader.py tests/test_experience_identity.py -q`

Expected: PASS, including unchanged source bytes and WAL metadata, deterministic
row order, schema incompatibility errors, and mutation rejection.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/experience/source_reader.py tradingagents/experience/identity.py tests/test_experience_source_reader.py tests/test_experience_identity.py tests/fixtures/experience_source_db.py
git commit -m "feat: add read-only phase 5 source reader"
```

## Task 3 — Separate catalog, source aliases, provenance, and observed outcome snapshots

**Files:**

- Create: `tradingagents/experience/catalog.py`
- Create: `tradingagents/experience/provenance.py`
- Create: `tradingagents/experience/diagnostics.py`
- Test: `tests/test_experience_catalog.py`
- Test: `tests/test_experience_provenance.py`

**Interfaces:**

- Produces `ExperienceCatalog(artifact_root)` with its own `catalog.sqlite3`
  and tables for sources, experience records, source aliases, decision
  evidence, market-state provenance, append-only observed evaluation snapshots,
  feature projections, generations, import events, quarantine, and bounded
  diagnostic FTS5.
- Produces `upsert_source_alias`, `append_evaluation_snapshot`,
  `mark_alias_removed`, `active_records`, `historical_records`,
  `is_tombstoned`, `current_alias_count`, `mark_last_alias_removed`,
  `evaluation_snapshots`, `record_failed_scan`, `record_quarantine`, and
  `publish_generation` methods.
- Consumes `ReadonlySourceSnapshot` and identity fingerprints from Task 2;
  writes only inside the Phase 8 artifact root.

- [ ] **Step 1: Write the failing catalog tests**

```python
def test_duplicate_aliases_share_one_logical_record(catalog: ExperienceCatalog) -> None:
    first = catalog.upsert_source_alias("db-a", decision_id="d1", fingerprint="fp1")
    second = catalog.upsert_source_alias("db-b", decision_id="d1", fingerprint="fp1")
    assert first.experience_id == second.experience_id
    assert len(catalog.active_records()) == 1

def test_last_alias_removal_tombstones_but_failed_scan_does_not(catalog: ExperienceCatalog) -> None:
    catalog.upsert_source_alias("db-a", decision_id="d1", fingerprint="fp1")
    catalog.upsert_source_alias("db-b", decision_id="d1", fingerprint="fp1")
    catalog.mark_alias_removed("db-a", "d1", "fp1")
    assert catalog.active_records()[0].source_aliases["db-b"] == "CURRENT"
    catalog.record_failed_scan("db-b", "SOURCE_DATABASE_UNAVAILABLE")
    assert catalog.active_records()[0].source_aliases["db-b"] == "CURRENT"
    catalog.mark_alias_removed("db-b", "d1", "fp1")
    record = catalog.historical_records()[0]
    assert catalog.is_tombstoned(record.experience_id) is True

def test_conflicting_fingerprint_is_quarantined(catalog: ExperienceCatalog) -> None:
    catalog.upsert_source_alias("db-a", "d1", "fp1")
    with pytest.raises(SourceDecisionConflictError):
        catalog.upsert_source_alias("db-a", "d1", "fp2")
    assert catalog.quarantine_count() == 1

def test_recovery_snapshot_is_append_only_only_when_observed(catalog: ExperienceCatalog) -> None:
    catalog.append_evaluation_snapshot("exp1", {"evaluation_status": "DATA_UNAVAILABLE"}, "old-fp")
    catalog.append_evaluation_snapshot("exp1", {"evaluation_status": "COMPLETE"}, "new-fp")
    assert catalog.evaluation_snapshot_fingerprints("exp1") == ("old-fp", "new-fp")
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_catalog.py tests/test_experience_provenance.py -q`

Expected: FAIL because the Phase 8 catalog and alias/provenance APIs do not exist.

- [ ] **Step 3: Implement the catalog transactionally**

Use a catalog schema version owned by Phase 8. Store one logical
`experience_id` per accepted source decision ID, and one alias row per source
database/row identity. Compute active state as “at least one CURRENT,
nonremoved alias points to the accepted fingerprint.” A failed/unavailable
scan records diagnostics only. A confirmed absence marks only that alias
REMOVED. Conflicting fingerprints quarantine the new row and leave the
accepted record unchanged.

Append an evaluation snapshot only when the importer actually observed that
state. Store `training_eligible` as provenance, never as a statistics gate.
Use per-record transactions, a bounded sanitized diagnostic FTS5 projection,
and atomic generation metadata. Do not create or touch any Phase 5/6 table.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_catalog.py tests/test_experience_provenance.py -q`

Expected: PASS, including alias currentness, conflict quarantine, append-only
observed evaluations, and provenance fingerprints.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/experience/catalog.py tradingagents/experience/provenance.py tradingagents/experience/diagnostics.py tests/test_experience_catalog.py tests/test_experience_provenance.py
git commit -m "feat: add phase 8 catalog and alias provenance"
```

## Task 4 — Pre-decision feature extraction and deterministic trust classification

**Files:**

- Create: `tradingagents/experience/features.py`
- Create: `tradingagents/experience/trust.py`
- Test: `tests/test_experience_features.py`
- Test: `tests/test_experience_trust.py`

**Interfaces:**

- Produces `MarketStateVector(values: tuple[float, ...], mask: tuple[bool, ...], feature_names, cohort, fingerprint)` and `extract_market_state(decision_row)`.
- Produces `TrustClassification(tier, reasons, policy_version)` and `classify_trust(record, feature_result)`.
- Consumes only persisted `snapshot_json`, decision metadata, and validated
  analysis timestamp; it must not read evaluation/outcome fields.

- [ ] **Step 1: Write the failing feature/trust tests**

```python
def test_feature_order_and_missing_mask_are_deterministic(decision_row: dict) -> None:
    result = extract_market_state(decision_row)
    assert result.feature_names == FEATURE_NAMES_V1
    assert len(result.values) == len(result.mask)
    assert result.mask[result.feature_names.index("M5.average_true_range")] is False

def test_outcome_fields_cannot_change_market_vector(decision_row: dict) -> None:
    changed = {**decision_row, "buy_net_points": 99999, "selected_action": "SELL"}
    assert extract_market_state(decision_row).fingerprint == extract_market_state(changed).fingerprint

def test_incomplete_normalization_is_tier_c(decision_row: dict) -> None:
    record = make_record(decision_row, normalization_status="FAILED")
    assert classify_trust(record, extract_market_state(decision_row)).tier is TrustTier.TIER_C_DIAGNOSTIC_ONLY
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_features.py tests/test_experience_trust.py -q`

Expected: FAIL because the V1 vector extractor and trust policy are missing.

- [ ] **Step 3: Implement the exact V1 vector and trust policy**

Use the ordered fields from the design: `spread_points`, each M1/M5/M15/H1
return/range/close-position/ATR/direction field, and UTC hour sine/cosine.
Encode UP/FLAT/DOWN as 1/0/-1 and INSUFFICIENT_DATA as a masked value. Preserve
source JSON paths and missing reasons. Reject malformed point/digits,
non-finite values, missing symbol/profile/timestamp, or invalid schema with a
typed extraction diagnostic; never infer a value from future prices.

Classify Tier A only for executed=0, COMPLETE context, exact normalized action,
valid UTC timestamps/quotes/features, and consistent provenance. Classify
limited but usable rows as Tier B. Classify failures, conflicts, invalid
provenance, executed rows, and insufficient features as Tier C. Include exact
symbol/profile/timeframe/schema/extractor in the cohort key.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_features.py tests/test_experience_trust.py -q`

Expected: PASS, including action/outcome independence, masks, exact feature
order, and all Tier A/B/C reason codes.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/experience/features.py tradingagents/experience/trust.py tests/test_experience_features.py tests/test_experience_trust.py
git commit -m "feat: add phase 8 features and trust policy"
```

## Task 5 — Cohort-scoped robust normalization profiles

**Files:**

- Create: `tradingagents/experience/normalization.py`
- Test: `tests/test_experience_normalization.py`

**Interfaces:**

- Produces `NormalizationCohortV1(resolved_symbol, analysis_profile, analysis_timeframe, feature_schema_version, feature_extractor_version)`.
- Produces `SimilarityProfileV1` with median, IQR, MAD/fallback scale,
  epsilon, weights, clipping, ordered feature/mask policy, trust-tier
  population, cohort, version, and population fingerprint.
- Produces `build_profile(rows, cohort, trust_tiers, as_of)` and
  `query_normalization_fingerprint(profile, cohort, trust_tiers, as_of)`.
- Consumes only eligible active/pre-as-of feature rows and never actions,
  outcomes, evaluations, account fields, or unrelated cohorts.

- [ ] **Step 1: Write the failing normalization tests**

```python
def test_zero_iqr_uses_mad_then_versioned_fallback(rows: list[FeatureRow]) -> None:
    profile = build_profile(rows, cohort=EURUSD_M15, trust_tiers=AB, as_of=None)
    assert profile.scales["spread_points"] == profile.mad_scales["spread_points"]
    assert profile.fallback_policy_version == "mad-then-feature-fallback.v1"

def test_unrelated_symbol_profile_and_timeframe_do_not_change_eurusd_profile(rows) -> None:
    base = build_profile(rows_for("EURUSD", "INTRADAY", "M15"), EURUSD_M15, AB, None)
    expanded = build_profile(rows + rows_for("XAUUSD", "INTRADAY", "M15") + rows_for("EURUSD", "SWING", "H1"), EURUSD_M15, AB, None)
    assert base.population_fingerprint == expanded.population_fingerprint
    assert base.medians == expanded.medians

def test_tier_a_only_changes_population_and_fingerprint(rows) -> None:
    ab = build_profile(rows, EURUSD_M15, AB, None)
    a = build_profile(rows, EURUSD_M15, (TrustTier.TIER_A_HIGH_TRUST,), None)
    assert a.population_fingerprint != ab.population_fingerprint
    assert a.trust_tiers == (TrustTier.TIER_A_HIGH_TRUST,)

def test_extreme_future_row_cannot_change_historical_profile(rows) -> None:
    cutoff = utc("2026-01-02T00:00:00Z")
    before = build_profile(rows_before(cutoff), EURUSD_M15, AB, cutoff)
    after = build_profile(rows_before(cutoff) + extreme_rows_after(cutoff), EURUSD_M15, AB, cutoff)
    assert before.to_fingerprint() == after.to_fingerprint()

def test_historical_profile_keeps_retained_tombstone(catalog, service) -> None:
    query = ExperienceQuery(state, as_of=utc("2026-01-03T00:00:00Z"))
    before = service.search(query)
    catalog.mark_last_alias_removed("exp1")
    rebuild_projection(catalog)
    after = service.search(query)
    assert before.query_normalization_fingerprint == after.query_normalization_fingerprint
    assert [(h.experience_id, h.distance) for h in before.hits] == [(h.experience_id, h.distance) for h in after.hits]
    assert after.hits[0].currently_tombstoned is True
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_normalization.py -q`

Expected: FAIL because cohort profiles, fallback scales, and fingerprints are not implemented.

- [ ] **Step 3: Implement profile construction**

Define the exact cohort tuple. For current queries (`as_of is None`), filter
rows by exact cohort, requested Tier A/B trust tiers, and at least one CURRENT
source alias. For historical queries (`as_of` supplied), use retained accepted
experiences with a valid decision fingerprint, no conflict/provenance failure,
the exact cohort and requested Tier A/B trust tiers, and both analysis snapshot
and decision completion strictly before `as_of`; current alias state must not
exclude an experience merely because it was removed later. For current queries
use the persisted exact-cohort A+B profile when present; otherwise compute the
same cohort-local view. For non-default trust tiers compute a query-local view.
Reject Tier C numeric normalization in V1.

Use scale order IQR when greater than configured epsilon, otherwise
MAD-derived scale when greater than epsilon, otherwise a persisted per-feature
fallback scale. Never use `1e-12` as the effective scale. Persist the
configured epsilon, MAD conversion, fallback policy/version, feature order,
mask policy, weights, clipping, cohort, trust tiers, and eligible population
fingerprint. Include all requested cohort/trust/cutoff/policy fields in
`query_normalization_fingerprint`.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_normalization.py -q`

Expected: PASS, including zero-IQR behavior, cohort isolation, trust-tier
population changes, current-alias filtering, retained historical tombstones,
and future-row invariance.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/experience/normalization.py tests/test_experience_normalization.py
git commit -m "feat: add cohort scoped experience normalization"
```

## Task 6 — Exact NumPy similarity index and read-only query service

**Files:**

- Create: `tradingagents/experience/similarity.py`
- Create: `tradingagents/experience/query.py`
- Test: `tests/test_experience_similarity.py`
- Test: `tests/test_experience_query.py`

**Interfaces:**

- Produces `ExactSimilarityIndex.search(query_vector, query_mask, candidate_ids, top_k, profile) -> tuple[SimilarityHit, ...]`.
- Produces `ExperienceQueryService.search(query: ExperienceQuery) -> ExperienceSearchResult`.
- Consumes catalog/projection readers, `SimilarityProfileV1`, and exact feature
  vectors from Tasks 3–5; never opens source Phase 5/6 databases or MT5.

- [ ] **Step 1: Write the failing query tests**

```python
def test_default_similarity_excludes_tier_c_and_tombstones(service) -> None:
    result = service.search(ExperienceQuery(market_state=state))
    assert all(hit.trust_tier in {"TIER_A_HIGH_TRUST", "TIER_B_LIMITED"} for hit in result.hits)
    assert all(not hit.currently_tombstoned for hit in result.hits)

def test_completion_time_and_as_of_are_strict(service) -> None:
    cutoff = utc("2026-01-02T00:00:00Z")
    result = service.search(ExperienceQuery(state, as_of=cutoff))
    assert "analysis-before-completion-after-cutoff" not in {h.experience_id for h in result.hits}

def test_historical_tombstone_is_reproducible(service, catalog) -> None:
    query = ExperienceQuery(state, as_of=utc("2026-01-03T00:00:00Z"))
    before = service.search(query)
    catalog.mark_last_alias_removed("exp1")
    after = service.search(query)
    assert [(h.experience_id, h.distance) for h in before.hits] == [(h.experience_id, h.distance) for h in after.hits]
    assert after.hits[0].currently_tombstoned is True

def test_equal_distance_order_is_stable(service) -> None:
    result = service.search(ExperienceQuery(state, top_k=10))
    assert [h.experience_id for h in result.hits] == sorted_ids_by_time_then_id(result.hits)
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_similarity.py tests/test_experience_query.py -q`

Expected: FAIL because the exact index, compatibility gates, and query service are not present.

- [ ] **Step 3: Implement gates, distance, and result envelope**

Apply exact symbol, profile, timeframe, feature schema, extractor, trust-tier,
current/tombstone, and strict as-of completion gates before distance. For
historical queries allow retained accepted evidence after later alias removal
when not conflicted/provenance-invalid, and set `currently_tombstoned=true`.
Require at least eight comparable dimensions and at least 50% overlap of both
vectors. Use masked float32 NumPy weighted RMS with profile normalization and
score `1/(1+distance)`. Sort by distance, analysis snapshot timestamp, then
experience ID. Do not use action/outcome fields.

Return `ExperienceSearchResult` with the query normalization fingerprint,
active generation, candidate count, and exclusion counts. Reject Tier C from
numeric similarity rather than silently adding it.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_similarity.py tests/test_experience_query.py -q`

Expected: PASS, including strict gates, minimum overlap, deterministic ties,
cohort-safe normalization, tombstones, and fingerprint metadata.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/experience/similarity.py tradingagents/experience/query.py tests/test_experience_similarity.py tests/test_experience_query.py
git commit -m "feat: add exact experience similarity queries"
```

## Task 7 — Incremental importer, alias reconciliation, staging, rebuild, and recovery

**Files:**

- Create: `tradingagents/experience/importer.py`
- Modify: `tradingagents/experience/catalog.py` (only to expose importer transaction seams)
- Test: `tests/test_experience_importer.py`
- Test: `tests/test_experience_recovery.py`

**Interfaces:**

- Produces `ExperienceImporter.import_sources(source_paths) -> ImportReport`.
- Produces `ExperienceImporter.reconcile_removed_aliases(successful_source_scan)`.
- Produces `ExperienceRebuilder.rebuild(fail_after_stage: str | None = None) -> GenerationManifest`.
- Consumes `ReadonlySourceReader`, identity, feature/trust, and catalog APIs;
  never writes to source DBs.

- [ ] **Step 1: Write the failing importer tests**

```python
def test_first_import_is_idempotent(importer, source_db) -> None:
    first = importer.import_sources((source_db,))
    second = importer.import_sources((source_db,))
    assert first.indexed_count > 0
    assert second.unchanged_count == first.indexed_count
    assert second.experience_count == first.experience_count

def test_recovered_complete_without_prior_snapshot_is_not_fabricated(importer, source_db) -> None:
    report = importer.import_sources((source_db_with_only_recovered_complete,))
    assert report.fabricated_evaluation_snapshot_count == 0
    snapshots = importer.catalog.evaluation_snapshots("exp-recovered")
    assert len(snapshots) == 1
    assert snapshots[0].evaluation_status == "COMPLETE"
    assert snapshots[0].recovered_from_unavailable_at == utc("2026-01-02T15:00:00Z")
    assert snapshots[0].previous_unavailable_reason == "NO_QUOTE"
    assert snapshots[0].source_evaluation_fingerprint == "complete-fp"
    assert not any(snapshot.source_evaluation_fingerprint == "fake-prior-fp" for snapshot in snapshots)

def test_failed_scan_does_not_remove_alias(importer, catalog) -> None:
    importer.import_sources((source_db_a,))
    importer.import_sources((missing_or_unavailable_source_db,))
    assert catalog.current_alias_count("d1") == 1

def test_rebuild_failure_preserves_previous_generation(rebuilder) -> None:
    old = rebuilder.rebuild()
    with pytest.raises(RuntimeError):
        rebuilder.rebuild(fail_after_stage="features")
    assert rebuilder.catalog.active_generation_id() == old.generation_id

def test_rebuild_retains_feature_evidence_for_historical_tombstone(rebuilder, catalog, service) -> None:
    cutoff = utc("2026-01-03T00:00:00Z")
    before = service.search(ExperienceQuery(state, as_of=cutoff))
    catalog.mark_last_alias_removed("exp1")
    rebuilder.rebuild()
    after = service.search(ExperienceQuery(state, as_of=cutoff))
    assert before.query_normalization_fingerprint == after.query_normalization_fingerprint
    assert [(h.experience_id, h.distance) for h in before.hits] == [(h.experience_id, h.distance) for h in after.hits]
    assert after.hits[0].currently_tombstoned is True
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_importer.py tests/test_experience_recovery.py -q`

Expected: FAIL because import, alias reconciliation, staging, and generation recovery do not exist.

- [ ] **Step 3: Implement the importer state machine**

For each source, read a complete immutable snapshot, fingerprint rows, join
watcher/evaluation provenance, extract safe decision/market fields, classify
trust, and write one Phase 8 transaction. Unchanged fingerprints emit
`UNCHANGED`; duplicate database copies create aliases; conflicting decision
fingerprints quarantine without replacement. Append evaluation snapshots only
for states actually seen by this importer. If a source row is absent after a
successful scan, mark that alias removed; if the scan fails, preserve it.

Use `locks/import.lock`, `.staging/<run-id>`, RUNNING/INTERRUPTED events, and
atomic generation publication. Rebuild feature matrices, exact-cohort
profiles, diagnostics, and generation metadata from retained catalog evidence;
feature projections remain retained for historical as-of queries after the last
source alias is removed, while current queries exclude the resulting
tombstone. On failure keep the prior generation active. Never
initialize/migrate source stores and never synthesize an old recovered
evaluation.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_importer.py tests/test_experience_recovery.py -q`

Expected: PASS, including incremental skip, duplicate aliases, confirmed
removals, failed-scan preservation, conflict quarantine, recovery semantics,
staging, locking, and last-generation preservation.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/experience/importer.py tradingagents/experience/catalog.py tests/test_experience_importer.py tests/test_experience_recovery.py
git commit -m "feat: add incremental experience importer"
```

## Task 8 — Explicit basis/horizon outcome statistics

**Files:**

- Create: `tradingagents/experience/outcomes.py`
- Test: `tests/test_experience_outcomes.py`

**Interfaces:**

- Produces `OutcomeStatsCalculator.calculate(request: OutcomeStatsRequest) -> OutcomeStatistics`.
- Consumes only retained Phase 8 evaluation snapshots and trust/provenance
  records for the requested experience IDs; does not alter similarity scores.

- [ ] **Step 1: Write the failing statistics tests**

```python
def test_training_null_does_not_exclude_complete_descriptive_row(calculator) -> None:
    result = calculator.calculate(OutcomeStatsRequest(("exp1",), "ANALYSIS_SNAPSHOT", 300))
    assert result.eligible_count == 1

def test_basis_and_horizon_are_exact(calculator) -> None:
    result = calculator.calculate(OutcomeStatsRequest(("exp1",), "DECISION_REFERENCE", 900))
    assert result.requested_basis == "DECISION_REFERENCE"
    assert result.requested_horizon_seconds == 900
    assert result.excluded_counts["BASIS_OR_HORIZON_MISMATCH"] == 1

def test_hold_uses_counterfactual_opportunity_cost(calculator) -> None:
    result = calculator.calculate(OutcomeStatsRequest(("hold-exp",), "ANALYSIS_SNAPSHOT", 300))
    assert result.hold.opportunity_cost_points == (0.0,)
    assert result.hold.normal_win_rate is None

def test_as_of_excludes_late_recovery_without_prior_snapshot(calculator) -> None:
    request = OutcomeStatsRequest(("exp-recovered",), "ANALYSIS_SNAPSHOT", 300, as_of=utc("2026-01-02T14:00:00Z"))
    result = calculator.calculate(request)
    assert result.excluded_counts["EVALUATION_NOT_YET_AVAILABLE"] == 1

def test_recovered_complete_participates_after_recovery(calculator) -> None:
    request = OutcomeStatsRequest(("exp-recovered",), "ANALYSIS_SNAPSHOT", 300, as_of=utc("2026-01-02T16:00:00Z"))
    assert calculator.calculate(request).eligible_count == 1

def test_genuine_prior_unavailable_snapshot_is_used_before_recovery(calculator) -> None:
    request = OutcomeStatsRequest(("exp-with-prior",), "ANALYSIS_SNAPSHOT", 300, as_of=utc("2026-01-02T14:00:00Z"))
    result = calculator.calculate(request)
    assert result.excluded_counts["EVALUATION_NOT_YET_AVAILABLE"] == 1
    assert result.excluded_counts["DATA_UNAVAILABLE"] == 1
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_outcomes.py -q`

Expected: FAIL because the statistics calculator and eligibility rules do not exist.

- [ ] **Step 3: Implement the calculator**

Require allowed trust tier, exact basis, exact horizon, COMPLETE status,
`source_context_eligible == 1`, finite required fields, and historical
evaluation availability when `as_of` is present. Do not inspect
`training_eligible` to include/exclude a descriptive row. Select
`evaluation_available_at` as recovered timestamp, otherwise evaluated timestamp,
otherwise created timestamp; require trustworthy UTC and observation timestamp
no later than `as_of`. If no earlier Phase 8 snapshot was observed, report
`EVALUATION_NOT_YET_AVAILABLE` and never fabricate one.

Report BUY and SELL counterfactuals separately, preserve ANALYSIS_SNAPSHOT vs
DECISION_REFERENCE and each horizon, and keep HOLD separate from normal win
rate. Use the existing `hold_opportunity_cost_points` semantics and expose
eligible denominators and exclusions by reason/status/tier.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_outcomes.py -q`

Expected: PASS, including `training_eligible=NULL`, basis/horizon isolation,
late recovery handling, directional counters, HOLD semantics, and provenance.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/experience/outcomes.py tests/test_experience_outcomes.py
git commit -m "feat: add explicit experience outcome statistics"
```

## Task 9 — Read-only Evidence Orchestrator

**Files:**

- Create: `tradingagents/experience/orchestrator.py`
- Modify: `tradingagents/experience/models.py` only if protocol typing needs a forward declaration
- Test: `tests/test_experience_orchestrator.py`

**Interfaces:**

- Produces `EvidenceOrchestrator(knowledge_service, experience_service, statistics_calculator).query(request: EvidenceRequest) -> EvidenceBundle`.
- Consumes only the Phase 7 `KnowledgeQueryService.search(KnowledgeQuery)`
  contract and the Phase 8 `ExperienceQueryService.search(ExperienceQuery)`
  plus `OutcomeStatsCalculator`; no parser, writer, LLM, MT5, or source reader.

- [ ] **Step 1: Write the failing orchestration tests**

```python
def test_as_of_is_forwarded_unchanged(orchestrator, knowledge, experience, stats) -> None:
    cutoff = utc("2026-01-02T03:04:05.123456Z")
    orchestrator.query(EvidenceRequest(market_state=state, evaluation_basis="ANALYSIS_SNAPSHOT", horizon_seconds=300, as_of=cutoff))
    assert experience.last_query.as_of == cutoff
    assert stats.last_request.as_of == cutoff

def test_both_sources_keep_separate_scores(orchestrator) -> None:
    bundle = orchestrator.query(EvidenceRequest(research_question="order flow", market_state=state))
    assert bundle.knowledge and bundle.experience
    assert not hasattr(bundle, "combined_score")
    assert bundle.status == "COMPLETE"

def test_one_source_failure_is_partial(orchestrator_with_failed_knowledge) -> None:
    bundle = orchestrator_with_failed_knowledge.query(EvidenceRequest(market_state=state, research_question="OFI"))
    assert bundle.status == "PARTIAL"
    assert bundle.errors[0].source == "knowledge"
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_orchestrator.py -q`

Expected: FAIL because the orchestrator and source-status handling are missing.

- [ ] **Step 3: Implement independent composition**

Build a KnowledgeQuery only when `research_question` is present and an
ExperienceQuery only when `market_state` is present. Pass `EvidenceRequest.as_of`
unchanged into both the ExperienceQuery and OutcomeStatsRequest. Calculate
statistics only after retrieval and only for returned experience IDs. Keep
KnowledgeHit scores and ExperienceHit similarity scores in separate fields;
never fuse, rank, summarize, recommend, or generate prompts. Return COMPLETE,
PARTIAL, EMPTY, or FAILED with typed source errors and provenance, including
the query normalization fingerprint.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_orchestrator.py -q`

Expected: PASS, including knowledge-only, experience-only, combined, partial,
failed, empty, strict `as_of`, and no-recommendation behavior.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/experience/orchestrator.py tradingagents/experience/models.py tests/test_experience_orchestrator.py
git commit -m "feat: add read-only evidence orchestrator"
```

## Task 10 — Explicit Experience CLI

**Files:**

- Create: `tradingagents/experience/cli.py`
- Modify: `pyproject.toml` (add `experience = "tradingagents.experience.cli:main"`; leave `tradingagents`, `forex-shadow`, `forex-evaluate`, and `forex-watch` entries unchanged)
- Test: `tests/test_experience_cli.py`

**Interfaces:**

- Produces argparse subcommands `experience import`, `rebuild`, `status`,
  `list`, `show`, `similar`, `stats`, `quarantine`, and `evidence`.
- `import` and `rebuild` are the only commands allowed to write under
  `data_cache/experience`; status/list/show/similar/stats/quarantine/evidence
  use read-only catalog/index access.
- Consumes public Phase 8 services and the Phase 7 KnowledgeQueryService only
  for evidence queries; no stock CLI or forex collector import is changed.

- [ ] **Step 1: Write the failing CLI tests**

```python
def test_status_does_not_construct_parser_embedder_or_writer(monkeypatch, runner) -> None:
    monkeypatch.setattr("tradingagents.knowledge.docling_parser.DoclingDocumentParser", fail_if_constructed)
    monkeypatch.setattr("tradingagents.knowledge.embeddings.FastEmbedProvider", fail_if_constructed)
    assert runner(["status", "--artifact-root", str(tmp_path)]).exit_code == 0

def test_similar_never_constructs_any_embedding_provider(monkeypatch, runner) -> None:
    monkeypatch.setattr("tradingagents.knowledge.embeddings.FastEmbedProvider", fail_if_constructed)
    monkeypatch.setattr("tradingagents.knowledge.docling_parser.DoclingDocumentParser", fail_if_constructed)
    monkeypatch.setattr("tradingagents.knowledge.vector_index.VectorIndexWriter", fail_if_constructed)
    monkeypatch.setattr("tradingagents.knowledge.lexical_index.LexicalIndexWriter", fail_if_constructed)
    assert runner(["similar", "--market-state-json", str(state_file)]).exit_code == 0

def test_evidence_market_state_only_never_constructs_embedding_provider(monkeypatch, runner) -> None:
    monkeypatch.setattr("tradingagents.knowledge.embeddings.FastEmbedProvider", fail_if_constructed)
    assert runner(["evidence", "--market-state-json", str(state_file)]).exit_code == 0

def test_evidence_question_may_use_phase7_embedder_but_no_writers(monkeypatch, runner) -> None:
    monkeypatch.setattr("tradingagents.knowledge.docling_parser.DoclingDocumentParser", fail_if_constructed)
    monkeypatch.setattr("tradingagents.knowledge.vector_index.VectorIndexWriter", fail_if_constructed)
    monkeypatch.setattr("tradingagents.knowledge.lexical_index.LexicalIndexWriter", fail_if_constructed)
    assert runner(["evidence", "--question", "order flow"]).exit_code == 0

def test_cli_has_no_trading_or_model_options(runner) -> None:
    help_text = runner(["--help"]).stdout
    assert "--mt5" not in help_text and "--model" not in help_text
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_cli.py -q`

Expected: FAIL because the `experience` command surface and entry point are missing.

- [ ] **Step 3: Implement the explicit CLI**

Use argparse and stable JSON output. Resolve artifact paths without creating
directories for metadata-only commands. `status`, `list`, `show`, and
`quarantine` must not instantiate parser/embedder/ingestor/writers. `similar`
must construct only the Phase 8 catalog/projection readers, normalization
service, `ExactSimilarityIndex`, and `ExperienceQueryService`; Phase 8 numeric
similarity has no embedding model. `evidence --market-state-json` likewise
constructs no embedder. Only `evidence --question` may indirectly instantiate
the existing Phase 7 local query embedder, and it must still construct no
parser, ingestor, vector writer, or lexical writer. Expose exact filters for
symbol/profile/timeframe/trust/as-of/basis/horizon, but no MT5, order, model,
prompt, training, or execution options. Return typed errors and bounded
diagnostics without source reports or reasoning.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_cli.py -q`

Expected: PASS, including command routing, read-only metadata boundaries,
JSON output, and unchanged stock/forex entry-point behavior.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/experience/cli.py pyproject.toml tests/test_experience_cli.py
git commit -m "feat: add explicit experience cli"
```

## Task 11 — Leakage, isolation, and deterministic retrieval-quality gates

**Files:**

- Create: `tests/test_experience_leakage.py`
- Create: `tests/test_experience_quality.py`
- Create: `tests/test_experience_isolation.py`

**Interfaces:**

- Consumes the public query, normalization, outcomes, catalog, and orchestrator
  contracts from Tasks 1–10.
- Produces deterministic benchmark assertions and machine-readable counts of
  violations, exclusions, denominators, Recall@K/MRR, exact-keyword/provenance
  checks where applicable, stable ordering, latency, and bounded memory.

- [ ] **Step 1: Write the failing leakage tests**

```python
def test_outcome_change_does_not_change_similarity(service, fixture_corpus) -> None:
    before = service.search(fixture_corpus.query)
    fixture_corpus.change_outcome("exp1", net_points=999, mfe=999, mae=-999)
    after = service.search(fixture_corpus.query)
    assert projection(before) == projection(after)

def test_action_change_does_not_change_default_similarity(service, fixture_corpus) -> None:
    before = service.search(fixture_corpus.query)
    fixture_corpus.change_action("exp1", "SELL")
    assert projection(service.search(fixture_corpus.query)) == projection(before)

def test_all_six_future_leakage_counts_are_zero(benchmark) -> None:
    report = benchmark.run()
    assert report.violation_counts == {
        "OUTCOME": 0, "ACTION": 0, "CANDIDATE": 0,
        "COMPLETION": 0, "NORMALIZATION": 0, "EVALUATION": 0,
    }

def test_no_forbidden_phase8_imports() -> None:
    assert scan_imports("tradingagents.experience").forbidden == set()
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_leakage.py tests/test_experience_quality.py tests/test_experience_isolation.py -q`

Expected: FAIL until the benchmark fixtures, import scanner, and all leakage
assertions are connected to the implemented services.

- [ ] **Step 3: Implement the deterministic benchmark tests**

Cover outcome, action, candidate-time, completion-time, normalization-future,
evaluation-future, and cohort leakage exactly as listed in the design. Include
extreme future rows, unrelated XAUUSD/profile/timeframe rows, Tier A-only
normalization, recovered evaluations with and without prior observed
snapshots, and later alias removal. Assert zero violations, complete
provenance, exact compatibility filters, Recall@K/MRR, stable repeated order,
and explicit denominators. Scan package imports to forbid MT5, forex, graph,
agent, Qwen/Ollama, execution, training, and Phase 7 writer modules.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_leakage.py tests/test_experience_quality.py tests/test_experience_isolation.py -q`

Expected: PASS with all six leakage violation counts equal to zero and no
forbidden imports.

- [ ] **Step 5: Commit**

```bash
git add tests/test_experience_leakage.py tests/test_experience_quality.py tests/test_experience_isolation.py
git commit -m "test: add phase 8 leakage and quality gates"
```

## Task 12 — Bounded real local Phase 8 acceptance smoke

**Files:**

- Create: `scripts/experience_phase8_smoke.py`
- Create: `tests/test_experience_task12_scripts.py`

**Interfaces:**

- Produces a bounded JSON/text report from an existing Phase 5/6 validation
  SQLite path, Phase 8 artifact root, existing Phase 7 artifact root, and
  existing local Phase 7 embedding model path. The command requires explicit
  arguments equivalent to `--source-db`, `--experience-artifact-root`,
  `--knowledge-artifact-root`, `--knowledge-embedding-model-path`, and
  `--offline`.
- Consumes `ReadonlySourceReader`, importer, feature/trust/catalog/query,
  outcomes, and Evidence Orchestrator services. It must not construct MT5,
  TradingAgents, Ollama/Qwen, or any source writer. The final smoke must use
  the actual Phase 7 `KnowledgeConfig(artifact_root=..., embedding_model_path=...)`,
  `KnowledgeCatalog`, `VectorIndexReader`, `LexicalIndexReader`,
  `FastEmbedProvider.from_config`, and `KnowledgeQueryService` read contracts;
  it must not instantiate `DoclingDocumentParser`, `SourceScanner`,
  `KnowledgeIngestor`, `VectorIndexWriter`, or `LexicalIndexWriter`.

- [ ] **Step 1: Write the failing harness tests**

```python
def test_smoke_records_source_integrity_and_counts(tmp_path, real_fixture_db, knowledge_root, embedding_model_path) -> None:
    report = run_smoke(real_fixture_db, experience_artifact_root=tmp_path / "experience", knowledge_artifact_root=knowledge_root, knowledge_embedding_model_path=embedding_model_path, offline=True)
    assert report["source_hash_before"] == report["source_hash_after"]
    assert report["source_unchanged"] is True
    assert set(report) >= {
        "decision_count", "evaluation_count", "experience_count",
        "tier_counts", "quarantine_count", "alias_count",
        "feature_population_count", "active_generation_id",
        "normalization_fingerprint", "similarity_examples",
        "statistics", "knowledge_hit_count", "evidence_status",
        "network_attempts",
    }

def test_smoke_does_not_run_new_analysis(monkeypatch, real_fixture_db, tmp_path, knowledge_root, embedding_model_path) -> None:
    monkeypatch.setattr("tradingagents.forex.runner.ForexShadowRunner", fail_if_constructed)
    report = run_smoke(real_fixture_db, experience_artifact_root=tmp_path / "experience", knowledge_artifact_root=knowledge_root, knowledge_embedding_model_path=embedding_model_path, offline=True)
    assert report["analysis_invocations"] == 0

def test_smoke_requires_real_phase7_query_configuration(tmp_path, real_fixture_db) -> None:
    with pytest.raises(SystemExit):
        run_cli(["--source-db", str(real_fixture_db), "--experience-artifact-root", str(tmp_path / "experience"), "--offline"])

def test_network_guard_blocks_unexpected_connect(monkeypatch, real_fixture_db, tmp_path, knowledge_root, embedding_model_path) -> None:
    with OfflineNetworkGuard() as guard:
        with pytest.raises(NetworkAttempt):
            socket.create_connection(("203.0.113.1", 9), timeout=0.01)
    assert guard.attempt_count == 1

def test_smoke_installs_guard_before_query_service_construction(monkeypatch, real_fixture_db, tmp_path, knowledge_root, embedding_model_path) -> None:
    def constructor_that_connects(*args, **kwargs):
        socket.create_connection(("203.0.113.1", 9), timeout=0.01)
    monkeypatch.setattr("tradingagents.knowledge.embeddings.FastEmbedProvider.from_config", constructor_that_connects)
    with pytest.raises(NetworkAttempt):
        run_smoke(real_fixture_db, experience_artifact_root=tmp_path / "experience", knowledge_artifact_root=knowledge_root, knowledge_embedding_model_path=embedding_model_path, offline=True)
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_experience_task12_scripts.py -q`

Expected: FAIL because the acceptance harness does not exist.

- [ ] **Step 3: Implement the bounded real smoke**

Require an existing source DB path, existing Phase 8 artifact root, existing
Phase 7 artifact root, and existing local embedding model path; never launch a
new analysis. Record source SQLite SHA-256, size, mtime, and WAL SHA-256/size
before and after. Set `KNOWLEDGE_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1` (plus the Phase 7 local-model flags in its current
configuration) and install the Phase 7-style `OfflineNetworkGuard` before
constructing either query service. Any unexpected socket/URL connection raises
`NetworkAttempt` and fails the smoke; the report records the observed attempt
count, which must be zero.

Run the read-only import, real feature extraction, trust classification,
exact-cohort normalization, representative exact similarity queries, explicit
basis/horizon statistics where available, and a mandatory combined Phase 7
knowledge + Phase 8 evidence query. Construct Phase 7 only through its actual
read contracts (`KnowledgeConfig`, `KnowledgeCatalog`, `VectorIndexReader`,
`LexicalIndexReader`, `FastEmbedProvider.from_config`, and
`KnowledgeQueryService`); do not construct its parser, source scanner,
ingestor, or writers. Report decision/evaluation/import/experience counts,
Tier A/B/C counts, quarantine and duplicate/alias counts, feature population,
generation IDs, normalization cohort/population/fingerprint, Phase 7
generation and embedding-spec/artifact identity, top IDs and comparable
dimensions, statistic eligibility/exclusions, Knowledge and Experience hit
counts, EvidenceBundle status, bounded latency, and network attempts. If no
eligible COMPLETE outcome exists, report the honest exclusion reason rather
than fabricating eligibility.

The harness must prove source hashes/sizes/WAL metadata are unchanged and
must not write under the source path or `new books`. Keep the smoke bounded to
the existing data and local services; no Qwen, MT5, execution, or unguarded
network call is allowed. Missing or incompatible Phase 7 local query artifacts
returns `PHASE 8 NOT COMPLETE`; the combined EvidenceBundle gate may not be
skipped.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_experience_task12_scripts.py -q`

Expected: PASS for fixture validation. During final acceptance run the script
once against an already-produced real Phase 5/6 validation DB and archive its
report as verification evidence.

- [ ] **Step 5: Commit**

```bash
git add scripts/experience_phase8_smoke.py tests/test_experience_task12_scripts.py
git commit -m "test: add phase 8 real acceptance smoke"
```

## Task 13 — Final verification, source-integrity audit, and Phase 8 handoff

**Files:**

- Create: `docs/phase8/2026-09-11-phase-8-verification.md`
- Test: all Phase 8 tests created above; existing repository suite remains unchanged.

**Interfaces:**

- Consumes all public Phase 8 services and the Task 12 smoke report.
- Produces the final implementation SHA, verification report, generation and
  provenance evidence, and an explicit `PHASE 8 COMPLETE` or `PHASE 8 NOT COMPLETE` decision.

- [ ] **Step 1: Run focused Phase 8 tests**

Run:

```bash
pytest tests/test_experience_*.py -q
```

Expected: PASS, including all model, source, catalog, feature, trust,
normalization, similarity, importer, outcome, orchestrator, CLI, leakage,
isolation, and smoke tests.

- [ ] **Step 2: Run repository-wide verification**

Run:

```bash
pytest -q
ruff check tradingagents/experience tests/test_experience_*.py scripts/experience_phase8_smoke.py
python -m compileall tradingagents/experience scripts/experience_phase8_smoke.py
git diff --check
```

Expected: PASS. Any missing optional dependency must be installed according to
the repository configuration before verification; no network-dependent test,
MT5 test, Ollama test, or CUDA requirement may be introduced.

- [ ] **Step 3: Run the real smoke once and audit source integrity**

Run the supported script against one existing Phase 5/6 validation DB, record
its exact path and report, and verify source SQLite/WAL hashes, sizes, and
timestamps remain unchanged. Confirm the real smoke's network-attempt count is
zero and that the Phase 7 query used only its existing read-only public API.

- [ ] **Step 4: Inspect scope and forbidden boundaries**

Run:

```bash
git diff --name-only a878fb39c8df22e4dfbe8f049a5e52861233bfaa..HEAD
git status --short
```

Expected: only Phase 8 package/tests/smoke/CLI metadata and explicitly scoped
documentation are changed; no `tradingagents/forex`, MT5 provider, execution,
watcher, TradingAgents/LangGraph, Qwen prompt, training, or Phase 7 ingestion
file is modified. The worktree is clean after committing.

- [ ] **Step 5: Commit the final verification report**

```bash
git add docs/phase8/2026-09-11-phase-8-verification.md
git commit -m "docs: record phase 8 verification"
```

Do not claim completion unless the real smoke passes the required source
integrity and read-only checks. If any required gate fails, report
`PHASE 8 NOT COMPLETE` with the exact failing evidence and preserve the last
valid generation.

## Final plan self-review

- Every approved design requirement maps to Tasks 1–13.
- `OutcomeStatsRequest.as_of` is defined in Task 1, consumed in Task 8, and
  propagated unchanged by Task 9.
- Normalization cohorts, trust tiers, fingerprints, and historical cutoffs
  are consistent across Tasks 5, 6, 7, 9, and 11.
- Current normalization requires a CURRENT alias; historical normalization uses
  retained accepted evidence available before `as_of` even after later alias
  removal, and Task 7 rebuild preserves that feature evidence.
- Tier C is diagnostic-only and cannot enter V1 numeric similarity.
- Outcome/action, candidate-time, completion-time, normalization, evaluation,
  and cohort leakage gates are explicit and require zero violations.
- Recovery without a prior observed Phase 8 snapshot never fabricates history.
- Duplicate alias removal and historical tombstone semantics are explicit.
- `training_eligible` remains provenance only; HOLD remains separate.
- Knowledge and Experience stores, scores, and provenance remain separate.
- The bounded real smoke is mandatory, uses existing data only, and requires a
  real Phase 7 read query with explicit artifact/model paths.
- The real smoke installs an active offline network guard before constructing
  query services and fails on any unexpected network attempt.
- Catalog tombstone state is derived via `is_tombstoned`; only query hits carry
  `currently_tombstoned`.
- Task 12 stages only its declared smoke script and test files; Task 13 owns
  the separate verification report.
- No task modifies MT5, TradingAgents/LangGraph, forex-watch, Qwen/Ollama,
  execution, training, Phase 7 ingestion, or Phase 9.
- No unfinished markers or unspecified owner remains in this plan.

## Final verification command list

```bash
pytest tests/test_experience_*.py -q
pytest -q
ruff check tradingagents/experience tests/test_experience_*.py scripts/experience_phase8_smoke.py
python -m compileall tradingagents/experience scripts/experience_phase8_smoke.py
git diff --check
git diff --name-only a878fb39c8df22e4dfbe8f049a5e52861233bfaa..HEAD
git status --short
```

Final handoff must report the corrected design baseline SHA, final
implementation SHA, branch/worktree, commit list, files changed, test totals,
Ruff/compileall/diff results, real source DB and integrity evidence, experience
count, Tier distribution, feature/schema/profile versions, normalization
cohorts, generation ID, all leakage-gate results, benchmark results,
basis/horizon statistics evidence, Knowledge result evidence, EvidenceBundle
result, network attempts, and working-tree status. State exactly
`PHASE 8 COMPLETE` or `PHASE 8 NOT COMPLETE`; do not begin Phase 9.
