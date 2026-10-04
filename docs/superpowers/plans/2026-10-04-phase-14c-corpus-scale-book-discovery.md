# Phase 14C — Corpus-Scale Book Strategy Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover source-grounded, HFT-suitable strategy specifications across the frozen Phase 7 corpus, evaluate only complete candidates through the unchanged causal Phase 14 replay/gate, and publish an auditable Phase 14C report.

**Architecture:** Use a read-only Phase 7 inventory and pinned-generation check, then run a versioned multi-query retrieval bank with deterministic diversity and hard selection caps. Reuse the existing bounded local Ollama presence/concept/atomic extraction and cache contracts, validate every selected claim against exact source spans, assemble specs/conflicts conservatively, and route only complete specs through the finite deterministic registry and existing replay gate. An explicit offline CLI owns fresh artifacts, exact-identity resume, status, and reporting; no runtime trading module calls it.

**Tech Stack:** Python, SQLite read-only connections, existing Phase 7 `KnowledgeQueryService`/FastEmbed/LanceDB/FTS5 stack, existing strict Pydantic schemas and local loopback-only Ollama adapter, existing Phase 14 `StrategySpec`/registry/replay/gate, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-04-phase-14c-corpus-scale-book-discovery-design.md`

## Global Constraints

- Phase 7 is frozen: no parse, source, catalog, embedding, vector, lexical, generation, or manifest writes.
- Only preinstalled local FastEmbed and the explicit loopback-only Qwen 2B adapter are used; no network/cloud provider.
- Phase 11A, Phase 14A, and Phase 12D sources are read-only; published knowledge and trading experience remain separate.
- No model call or retrieval occurs on the live HFT tick path.
- Unsupported, ambiguous, incomplete, conflicting, out-of-domain, or unavailable-data rules fail closed.
- Book-sourced rules and research parameters remain distinct and separately labeled.
- Existing live strategies, risk settings, watcher, broker gateway, and execution code are untouched.
- Candidate outcomes cannot exceed `SHADOW_CHALLENGER`; real-money execution remains impossible.
- The corpus universe is re-read at run time; design-time counts are never treated as current authority.
- Presence selection is capped at 750 groups; hitting a cap produces an incomplete/budget-limited result, never an exhaustive-coverage claim.
- Phase 14B's existing query defaults, schema, prompts, output limits, cache identity, and `extract(hits)` behavior remain backward compatible.

## Review Focus

1. **Stale or mismatched Phase 7 generation/projections:** fail before embedding or model construction; pinned-generation and population-hash mismatch tests belong to Task 1.
2. **Sibling query formulations returning the same chunk repeatedly:** preserve every query-family/formulation reference while counting one unique source chunk; deterministic dedup and provenance-retention tests belong to Tasks 2 and 4.
3. **Selection caps deferring eligible evidence:** report exact deferred counts and an incomplete state rather than implying classifier exhaustion; cap-boundary tests belong to Task 4.
4. **Malformed model result, cache mismatch, or interrupted batch:** fail closed, retain only safe telemetry, and resume only under an identical run identity; tests belong to Tasks 3 and 8.
5. **Duplicate/contradictory source passages or changed replay inputs:** never count copied passages as independent support, merge incompatible stages, or publish replay results after source mutation; tests belong to Tasks 5 and 7.

---

## File and Module Map

- `tradingagents/self_enhancement/phase14c_models.py` — immutable Phase 14C inventory, query, selection, assembly, mapping, run, and report records plus typed failure codes.
- `tradingagents/self_enhancement/phase14c_inventory.py` — read-only SQLite generation/inventory validation and fresh-root boundary checks.
- `tradingagents/self_enhancement/phase14c_queries.py` — closed versioned family/formulation bank, retrieval fan-out, and provenance-bearing retrieval records.
- `tradingagents/self_enhancement/book_atomic_extraction.py` — backward-compatible bounded sentence-selection option and public presence-only/actionable-only seams around the existing stages.
- `tradingagents/self_enhancement/phase14c_selection.py` — deterministic exact/near-duplicate clustering, family/document/section diversity, and bounded evidence-group selection.
- `tradingagents/self_enhancement/phase14c_assembly.py` — semantic anchors, cross-source compatibility/conflicts, research-parameter requests, deterministic draft construction, and assembly records.
- `tradingagents/self_enhancement/book_pipeline.py` — extract/reuse the existing exact source-claim validation as a public, tested boundary without changing Phase 14B defaults.
- `tradingagents/self_enhancement/phase14c_mapping.py` — versioned code-derived contract snapshots and evidence-backed incumbent component comparisons.
- `tradingagents/self_enhancement/phase14c_runner.py` — offline discovery orchestration, source fingerprint guards, candidate filtering, replay orchestration, and safe aggregate report.
- `cli/phase14c.py` — explicit `status`, `plan-only`, `discover`, and `evaluate` commands; local model construction only in explicit `discover` after preflight.
- `docs/phase14c-corpus-discovery.md` — operator usage, artifact/resume rules, statuses, limits, and interpretation of counts.
- `tests/test_phase14c_*.py` — deterministic fixtures/fakes for every new boundary; no Ollama, network, MT5, model download, or source mutation in CI.
- Existing Phase 14B and Phase 7 production modules/artifacts are not rewritten or rebuilt. `watch_dashboard.py` is unrelated and must remain untouched.

## Implementation Tasks

### Task 1: Pinned read-only inventory and run preflight

**Files:**
- Create: `tradingagents/self_enhancement/phase14c_models.py`
- Create: `tradingagents/self_enhancement/phase14c_inventory.py`
- Test: `tests/test_phase14c_inventory.py`

**Interfaces:**
- `read_phase7_inventory(knowledge_root: Path, *, expected_generation_id: str, expected_generation_fingerprint: str, expected_population_hash: str) -> tuple[Phase14CGenerationPin, CorpusInventory]`
- `validate_fresh_artifact_root(artifact_root: Path, *, protected_roots: Sequence[Path]) -> None`
- `Phase14CGenerationPin` records generation ID, full generation fingerprint, population hash, readiness/status, vector/lexical locations, and inventory schema version. `CorpusInventory` records unique sources, indexed/queryable documents, aliases, OCR-only/unavailable resources, active chunks, and deterministic source-manifest fingerprint.
- `phase14c_models.py` also owns the shared immutable `DiscoveryQuery`, `QueryReference`, `DiscoveryHitRecord`, `DiscoveryRetrievalReport`, `DiscoverySelectionLimits`, `DuplicateClusterIndex`, `EvidenceSelectionReport`, `SourceConflict`, `ResearchParameterRequest`, `AssemblyReport`, `Phase14CSourcePaths`, `Phase14CReplayPreflight`, `CandidateEvaluationRecord`, `Phase14CReport`, and `Phase14CEvaluationReport` records. Fields are specified at the consuming task below; enums use closed values. `EvidenceGroup` and its classification records live beside `EvidenceSentence` in `book_atomic_extraction.py` to avoid a module cycle.

- [ ] **Step 1: Write failing read-only inventory tests** for the approved generation, 97 unique-source accounting with aliases and `NEEDS_OCR`, an unknown schema/state, a population/fingerprint mismatch, and a source/catalog byte fingerprint that must remain unchanged.
- [ ] **Step 2: Run the focused tests and verify RED** because the Phase 14C inventory API does not exist; mismatches must fail before any model/embedder or artifact-root creation.
- [ ] **Step 3: Implement the inventory boundary** using SQLite URI `mode=ro`, `PRAGMA query_only=ON`, explicit table/column/state checks, and canonical `IndexGeneration` fingerprinting. Do not call catalog initialization or any write method.
- [ ] **Step 4: Add fresh-root containment tests** proving existing roots and paths overlapping Phase 7, the atomic cache, or source databases are rejected without deleting or modifying anything; run `pytest tests/test_phase14c_inventory.py -q`.
- [ ] **Step 5: Commit** as `feat: add read-only phase 14c inventory preflight`.

### Task 2: Versioned discovery query bank and retrieval fan-out

**Files:**
- Create: `tradingagents/self_enhancement/phase14c_queries.py`
- Modify: `tradingagents/self_enhancement/phase14c_models.py`
- Test: `tests/test_phase14c_queries.py`

**Interfaces:**
- `build_discovery_query_bank() -> tuple[DiscoveryQuery, ...]`
- `query_bank_fingerprint(bank: Sequence[DiscoveryQuery]) -> str`
- `retrieve_discovery_evidence(query_service: KnowledgeQueryService, bank: Sequence[DiscoveryQuery], *, pinned_generation_id: str, pinned_population_hash: str, top_k_per_formulation: int = 10) -> DiscoveryRetrievalReport`
- `DiscoveryQuery` fields are `family_id`, `formulation_id`, `text`, and optional `content_types`; `QueryReference` fields are family/formulation IDs, rank, semantic/lexical/fused/rerank scores; `DiscoveryHitRecord` holds one validated `KnowledgeHit` and all its sorted `QueryReference` values. `DiscoveryRetrievalReport` contains pinned generation/population identity, bank version/fingerprint, ordered records, raw/unique counts, and fixed query-failure records/status.

- [ ] **Step 1: Write failing tests** for deterministic bank order/fingerprint, required family coverage and sibling formulations, version/hash changes, exact source provenance checks, and one `KnowledgeQuery(top_k=10)` per formulation.
- [ ] **Step 2: Verify RED** on the missing bank/retrieval APIs; a hit with mismatched generation, population, source hash, chunk text, or inactive catalog state must be rejected, not repaired.
- [ ] **Step 3: Implement the closed versioned bank and sequential fan-out** through the existing hybrid query service only. Keep retrieval text as a search request, not a support claim; do not make model calls in this task.
- [ ] **Step 4: Verify** sibling formulations retain all query references and query-service failure yields an explicit retrieval failure with no partial success claim; run `pytest tests/test_phase14c_queries.py -q`.
- [ ] **Step 5: Commit** as `feat: add versioned phase 14c discovery queries`.

### Task 3: Reusable presence-only and actionable-only Phase 14B extraction seams

**Files:**
- Modify: `tradingagents/self_enhancement/book_atomic_extraction.py`
- Modify: `tradingagents/self_enhancement/phase14c_models.py`
- Test: `tests/test_phase14b_atomic_extraction.py`
- Test: `tests/test_phase14c_classification.py`

**Interfaces:**
- Extend `prepare_evidence_sentences(hits: Sequence[Any], *, max_selected_sentences: int = 60) -> tuple[EvidenceSentence, ...]`; the default 60 and existing evidence IDs/order remain unchanged.
- Add `EvidenceGroup(family_id: str, group_id: str, sentences: tuple[EvidenceSentence, ...])` and `AtomicStrategyExtractor.classify_presence(groups: Sequence[EvidenceGroup]) -> PresenceClassificationReport`; each supplied group receives `ACTIONABLE`, `NONE`, `AMBIGUOUS`, or a fixed failure code with safe call telemetry.
- Add `AtomicStrategyExtractor.extract_actionable(sentences: Sequence[EvidenceSentence]) -> AtomicExtractionReport`; it runs only concept grouping and atomic extraction and never repeats presence classification.
- Keep `AtomicStrategyExtractor.extract(hits)` behavior, schema, stage prompt, token limits, cache identity for unchanged batches, and bounded split retry backward compatible by delegating to the new seams.
- `PresenceGroupResult` contains the original `EvidenceGroup`, one closed status or failure code, and no generated prose. `PresenceClassificationReport` contains ordered group results and scalar `ModelCallTelemetry` only.

- [ ] **Step 1: Write failing compatibility tests** for the default 60-sentence cap, exact existing Phase 14B classification/output, and the new explicit larger cap.
- [ ] **Step 2: Write failing Phase 14C tests** proving `NONE`, `AMBIGUOUS`, and failed groups never reach downstream stages; `ACTIONABLE` evidence does; `extract_actionable` performs zero presence calls; and the same ordered evidence group reuses a validated Phase 14B presence cache entry because family metadata is not part of the unchanged presence prompt.
- [ ] **Step 3: Verify RED** with fake structured responses; malformed output remains fail-closed and the existing one-level split retry is not increased.
- [ ] **Step 4: Implement the public stage seams** without modifying schemas, prompts, output/context limits, local-only transport, or existing Phase 14B defaults; preserve safe scalar telemetry only.
- [ ] **Step 5: Run** `pytest tests/test_phase14b_atomic_extraction.py tests/test_phase14c_classification.py -q` and commit as `refactor: expose bounded phase 14b extraction stages`.

### Task 4: Deterministic diversity, duplicate handling, and hard selection bounds

**Files:**
- Create: `tradingagents/self_enhancement/phase14c_selection.py`
- Modify: `tradingagents/self_enhancement/phase14c_models.py`
- Test: `tests/test_phase14c_selection.py`

**Interfaces:**
- `select_diverse_evidence(report: DiscoveryRetrievalReport, *, limits: DiscoverySelectionLimits = DEFAULT_DISCOVERY_LIMITS) -> EvidenceSelectionReport`; output contains exact `EvidenceGroup` batches partitioned by family plus all unclassified/deferred counts.
- `DuplicateClusterIndex` maps each exact `(document_id, chunk_id, source_hash)` identity to a stable cluster ID. `EvidenceSelectionReport` contains ordered groups, the duplicate index, unique raw/selected counts, eligible/selected/deferred group counts by family/document, and `coverage_complete`.
- V1 limits: maximum two distinct chunks per document/family; two chunks per section/family; eight three-sentence groups per document; forty groups per family; 750 groups total. Ordering is deterministic round-robin by family → document → section, then retrieval score and stable IDs.
- Duplicate grouping uses NFKC/casefold/whitespace-normalized exact text and deterministic five-token shingle Jaccard `>= 0.90` for near-identical passages; it preserves all provenance and never adds duplicate documents to independent-source counts.

- [ ] **Step 1: Write failing tests** for exact chunk dedup across sibling queries, stable tie ordering, per-document/per-section/family caps, duplicate-cluster provenance retention, and independent-source counts across aliases/copies.
- [ ] **Step 2: Verify RED** and add boundary tests where 750 groups are accepted, group 751 is deferred, and `coverage_complete=False` with the exact residual count; assert each group is family-tagged and contains at most three evidence sentences.
- [ ] **Step 3: Implement deterministic duplicate clustering and round-robin selection** using Task 3's `EvidenceSentence`/`EvidenceGroup` APIs; no passage is erased from its provenance records and no cap is silently raised.
- [ ] **Step 4: Verify** identical input produces byte-stable selection and report; run `pytest tests/test_phase14c_selection.py -q`.
- [ ] **Step 5: Commit** as `feat: bound and diversify phase 14c evidence`.

### Task 5: Exact-source canonical assembly, conflicts, and parameter requests

**Files:**
- Create: `tradingagents/self_enhancement/phase14c_assembly.py`
- Modify: `tradingagents/self_enhancement/book_pipeline.py`
- Modify: `tradingagents/self_enhancement/phase14c_models.py`
- Test: `tests/test_phase14c_assembly.py`
- Test: `tests/test_phase14b_book_strategies.py`

**Interfaces:**
- Extract the existing claim-validation body into `validate_strategy_draft(draft: StrategyDraft, *, hit_by_alias: Mapping[str, KnowledgeHit], generation_id: str, generation_fingerprint: str, model_id: str, available_features: Iterable[str]) -> StrategyValidationResult`; `StrategyValidationResult` has `spec: StrategySpec | None` and `rejected_rules: tuple[RuleValidationResult, ...]`. Keep `BookStrategyPipeline._spec_from_draft` as a compatibility wrapper using the same validator.
- `assemble_atomic_concepts(concepts: Sequence[StrategyConcept], *, generation_id: str, generation_fingerprint: str, model_id: str, available_features: Iterable[str], duplicate_clusters: DuplicateClusterIndex) -> AssemblyReport`
- `SourceConflict` contains the family/anchor/stage and all conflicting evidence identities; `ResearchParameterRequest` contains the incomplete stage, required parameter kind/unit, evidence refs, and a closed reason. Each assembled record contains its semantic anchor, independent document IDs, exact rule provenance, assembly mode, executable eligibility, and optional `StrategySpec`. `AssemblyReport` contains these records plus conflict/parameter rows and complete/partial/rejected counts.

- [ ] **Step 1: Write failing tests** for exact atomic quote/offset-to-claim conversion, full document/hash/chunk/span provenance, deterministic `implementation_confidence=0.0`, and existing Phase 14B draft validation parity.
- [ ] **Step 2: Write failing semantic tests** proving compatible sources combine only with a shared supported entry/confirmation/invalidation anchor; incompatible same-stage rules become variants/conflicts; name similarity alone never merges; aliases and duplicate passages never count as independent sources.
- [ ] **Step 3: Write failing omission tests** proving a missing numeric parameter produces a separate `RESEARCH_PARAMETER_REQUIRED` record and no invented `StrategyRuleClaim`, and composite cross-source hypotheses cannot be executable.
- [ ] **Step 4: Implement assembly** by resolving every atomic evidence ID back to the exact `EvidenceSentence` and Phase 7 `KnowledgeHit`, then use the existing strict rule grammar and `StrategySpec` validation. Do not change `StrategySpec` fields or validators.
- [ ] **Step 5: Verify** specs round-trip with unchanged hashes and invalid/contradictory evidence is rejected; run `pytest tests/test_phase14c_assembly.py tests/test_phase14b_book_strategies.py -q` and commit as `feat: assemble source-grounded phase 14c specs`.

### Task 6: Suitability, finite registry routing, and incumbent component audit

**Files:**
- Create: `tradingagents/self_enhancement/phase14c_mapping.py`
- Modify: `tradingagents/self_enhancement/phase14c_models.py`
- Create: `tests/test_phase14c_mapping.py`
- Test: `tests/test_phase14c_suitability.py`
- Reuse without broadening: `tradingagents/self_enhancement/book_pipeline.py`, `book_strategies.py`, `book_factory.py`

**Interfaces:**
- `capture_current_strategy_contracts() -> CurrentStrategyContractSnapshot` derives the reviewed `range_rejection` and `momentum_continuation` entry/confirmation/expected-move/exit/risk/holding-horizon values from current offline strategy defaults and exit profiles; the snapshot is versioned and fingerprinted.
- `map_current_strategies(specs: Sequence[StrategySpec], *, snapshot: CurrentStrategyContractSnapshot) -> tuple[CurrentStrategyMapping, ...]` returns each required component as `MATCHED`, `PARTIALLY_MATCHED`, or `NO_DIRECT_MATCH` with exact source references only for matched components.
- The snapshot records schema/version, the two strategy IDs, their six named components and canonical code values, and a SHA-256 fingerprint. Each `CurrentStrategyMapping` records strategy ID and exactly six component results; each result contains status, matched source-rule fingerprints, and reason code.

- [ ] **Step 1: Write failing tests** for component-by-component matching and exact evidence binding; a strategy-name mention alone yields no match.
- [ ] **Step 2: Add drift tests** pinning current values (momentum 2 points / 0.6 persistence / 3 ticks; range 0.2 edge / 2 points / 4 ticks) and exit-profile identity; a changed runtime contract must invalidate the snapshot fingerprint before reporting a match.
- [ ] **Step 3: Add suitability/registry tests** proving missing L2/queue-position data, unsupported operators/families, partial stages, conflicts, and horizons over 60 seconds cannot produce an executable candidate; only existing `BookStrategyRegistry` mappings are accepted.
- [ ] **Step 4: Implement the isolated Phase 14C mapping and finite-registry check**; do not edit live forex/HFT strategies, engines, risk, watcher, or execution paths.
- [ ] **Step 5: Verify** `pytest tests/test_phase14c_mapping.py tests/test_phase14c_suitability.py -q` and commit as `feat: audit phase 14c suitability and strategy mappings`.

### Task 7: Read-only replay preflight and bounded candidate evaluation

**Files:**
- Create: `tradingagents/self_enhancement/phase14c_runner.py`
- Modify: `tradingagents/self_enhancement/phase14c_models.py`
- Test: `tests/test_phase14c_replay.py`
- Reuse unchanged: `tradingagents/self_enhancement/orchestrator.py`, `BookCandidateExperiment`, `CandidatePromotionGate`

**Interfaces:**
- `preflight_phase14_sources(phase14a_path: Path, hft_path: Path, demo_path: Path) -> Phase14CReplayPreflight` returns source fingerprints, verified/quarantined counts, causal tick/segment counts, commission status, and a closed readiness/reason code.
- `evaluate_phase14c_candidates(specs: Sequence[StrategySpec], *, artifact_root: Path, source_paths: Phase14CSourcePaths, source_commit: str, max_candidate_runs: int = 5) -> tuple[CandidateEvaluationRecord, ...]` evaluates no more than 10, default 5; incomplete, non-HFT-suitable, unimplemented, or conflicted specs are reported `NOT_RUN` with reason codes.
- `Phase14CSourcePaths` contains explicit Phase 14A/HFT/DEMO `Path` values only. `Phase14CReplayPreflight` records those paths' fingerprints, verified/quarantined experience counts, valid causal tick/segment counts, commission status, readiness, and one fixed reason code. Each `CandidateEvaluationRecord` contains spec/candidate ID, status, reason, replay artifact path, gate decision/reasons, and source fingerprints.

- [ ] **Step 1: Write failing tests** for a ready preflight, no verified Phase 14A experience, demo reconciliation uncertainty, invalid causal HFT data, and explicit path/fingerprint reporting.
- [ ] **Step 2: Write failing safety tests** that mutate a source between preflight and post-run verification and prove the experiment is failed/rejected; assert all three source file fingerprints remain identical on success.
- [ ] **Step 3: Write failing candidate tests** proving no incomplete spec reaches `BookCandidateExperiment`, no more than five run by default/ten by explicit bound, no broker API or MT5 import/call, and highest state is at most `SHADOW_CHALLENGER`.
- [ ] **Step 4: Implement by adapting the existing `BookCandidateExperiment` call shape** and unchanged gate; preserve development/validation/walk-forward/unseen isolation, cost scenarios, sample floors, and `commission=UNKNOWN`.
- [ ] **Step 5: Verify** `pytest tests/test_phase14c_replay.py tests/test_phase14b_book_replay.py -q`; commit as `feat: evaluate bounded phase 14c candidates offline`.

### Task 8: Explicit CLI, artifact lifecycle, and exact-identity resume

**Files:**
- Create: `cli/phase14c.py`
- Modify: `tradingagents/self_enhancement/phase14c_runner.py`
- Modify: `tradingagents/self_enhancement/phase14c_models.py`
- Test: `tests/test_phase14c_cli.py`
- Test: `tests/test_phase14c_resume.py`
- Create: `docs/phase14c-corpus-discovery.md`

**Interfaces:**
- `phase14c status --artifact-root PATH [--json]` reads only the manifest/report and never constructs the embedder, model, replay preflight, or trading DB reader.
- `phase14c plan-only` requires the explicit pinned Phase 7 root/generation/fingerprint and local embedding path; it runs inventory and deterministic retrieval/selection only, performs zero Ollama calls, creates no run artifact root, and reports pool sizes/deferred counts and a conservative classifier runtime estimate.
- `phase14c discover` requires explicit Phase 7 pin, fresh Phase 14C root, explicit `qwen3.5:2b`-compatible loopback model/provider settings, and an external cache path; model construction happens only after preflight and all path/offline checks.
- `phase14c evaluate --artifact-root PATH` accepts only complete, integrity-checked discovery artifacts and explicit Phase 14A/HFT/DEMO paths; it cannot construct Ollama or the Phase 7 embedder.
- `phase14c discover --resume --phase14c-artifact-root PATH` resumes only when generation, query-bank/selection/assembly versions, model/version/settings, cache identity, source fingerprints, and artifact schemas exactly match.
- CLI source options are required, not inferred from newest-file heuristics: `--knowledge-root`, `--expected-generation`, `--expected-fingerprint`, `--expected-population-hash`, `--phase14c-artifact-root`, `--phase14a-db`, `--hft-db`, and `--demo-db`; `discover` additionally requires `--embedding-model-path`, `--atomic-cache-path`, explicit `--model`, loopback `--ollama-endpoint`, and bounded timeout/output/context settings. `Phase14CReport` and `Phase14CEvaluationReport` use the design's full safe metric sets and never store raw prompts/completions/reasoning.

- [ ] **Step 1: Write failing CLI tests** showing `status` imports/constructs no embedder/model and reads no market/trading DB; `plan-only` does no model calls and writes no artifacts; `evaluate` does not construct model/embedder.
- [ ] **Step 2: Write failing safety tests** for non-loopback endpoint rejection, missing required paths, artifact-root collision/overlap, atomic cache under Phase 7/source DB, and existing artifact roots preserved unchanged.
- [ ] **Step 3: Write failing resume tests** for matching identity success, each generation/model/query/config/source mismatch rejection, interrupted last-record handling, and no duplicate processing of completed exact cache keys.
- [ ] **Step 4: Implement explicit commands and atomic safe artifacts** (`run-manifest.json`, inventory, query bank, retrieval, selection, validated classification, concepts/conflicts/specs, mappings, evaluations, report). Never store prompts, completions, reasoning, credentials, or unvalidated prose; never silently overwrite an existing root.
- [ ] **Step 5: Document exact command examples, output statuses, artifact paths, caps, and interpretation**; run `pytest tests/test_phase14c_cli.py tests/test_phase14c_resume.py -q` and commit as `feat: add explicit phase 14c discovery cli`.

### Task 9: Full orchestration/report integration and real offline acceptance

**Files:**
- Modify: `tradingagents/self_enhancement/phase14c_runner.py`
- Test: `tests/test_phase14c_integration.py`
- Modify: `docs/phase14c-corpus-discovery.md`

**Interfaces:**
- `run_discovery(...) -> Phase14CReport` composes Tasks 1–6 in fixed order; `run_evaluation(...) -> Phase14CEvaluationReport` composes Task 7 only after a valid discovery artifact exists.
- `Phase14CReport` persists source-universe/coverage, retrieval/dedup/cap, presence/extraction/cache/telemetry, conflicts/assembly, spec completeness/suitability, mapping, replay/gate, source-fingerprint, and Phase 7 before/after metrics required by the spec.

- [ ] **Step 1: Write a fake-boundary integration test** over a tiny synthetic catalog with aliases, OCR-only source, duplicate passages, actionable/ambiguous groups, one conflict, one partial spec, and one candidate; assert the entire order of gates and exact provenance in the final report.
- [ ] **Step 2: Assert failure-state semantics**: zero complete specs → `DISCOVERY_COMPLETE_NO_COMPLETE_SPEC`; only unsuitable specs → `DISCOVERY_COMPLETE_NO_HFT_SUITABLE_SPEC`; blocked replay → no evaluation claim; no candidate passes → `CANDIDATES_EVALUATED_NONE_PASSED`; no status exceeds `SHADOW_CHALLENGER`.
- [ ] **Step 3: Assert safety end to end** with no MT5/network/model calls in fixture mode and identical Phase 7, Phase 14A, HFT, and DEMO fingerprints before/after.
- [ ] **Step 4: Run targeted and full verification**: `python -m pytest -q tests/test_phase14c_*.py tests/test_phase14b_*.py`; then `python -m pytest -q`, `python -m ruff check tradingagents/self_enhancement cli tests`, `python -m compileall -q tradingagents/self_enhancement cli`, and `git diff --check`.
- [ ] **Step 5: Run real acceptance once** only after all deterministic checks pass: first `plan-only` against `C:\p7fast` and the approved generation/fingerprint, review measured retrieval pool and conservative runtime estimate, then run explicit offline `discover` with the local Qwen 2B and reusable Phase 14B atomic cache under a new Phase 14C root; run `evaluate` only if complete HFT-suitable candidates exist. Verify Phase 7 identity/inventory and all three source DB fingerprints are unchanged; report exact state/counts, never claim exhaustive classification if any groups were deferred.
- [ ] **Step 6: Commit** the integrated implementation only after final verification, using `feat: add corpus-scale phase 14c strategy discovery`.

## Self-Review

- **Spec coverage:** pinned inventory/identity (Task 1); versioned multi-query bank and retrieval (Task 2); local staged extraction/cache (Task 3); source-aware diversity/caps/dedup (Task 4); strict provenance, assembly, conflict and parameter semantics (Task 5); HFT suitability, finite registry, current-strategy mapping/drift (Task 6); causal replay and unchanged promotion gate (Task 7); explicit offline CLI/artifacts/resume/status (Task 8); metrics, real acceptance, immutability, tests and final report (Task 9).
- **Step granularity:** every implementation task has an isolated test boundary, a concrete interface, a RED check, a minimal implementation, a GREEN command, and its own commit; the real-corpus run is last and cannot start before deterministic verification.
- **Type consistency:** retrieval produces `DiscoveryRetrievalReport`; selection consumes it and produces family-tagged `EvidenceGroup` batches; presence classification consumes those groups; actionable-only extraction produces `StrategyConcept` values; assembly consumes concepts and creates existing `StrategySpec` values; mapping/replay consume only validated specs.
- **Review focus:** each of the five high-risk inputs above has an explicit test in its owning task; no untested review-focus item remains.
- **Proportion:** the plan adds only isolated Phase 14C modules and the narrow Phase 14B stage/validator seams required to reuse existing behavior; Phase 7, live Forex/HFT, Phase 11A, schemas, and promotion gates are not redesigned.
