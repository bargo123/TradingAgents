# Phase 11A — HFT/FX Book Knowledge Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (or `superpowers:executing-plans`) to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build an offline-capable, grounded, immutable Phase 7-to-knowledge-lesson factory without changing trading or Phase 10 behavior.

**Architecture:** A read-only Phase 7 catalog adapter feeds deterministic bounded source packets into an injectable structured teacher. Independent validators enforce provenance, grounding, copyright, safety, deduplication, and grouped splits before an immutable generation writer publishes accepted and excluded lessons. A separate curriculum adapter exposes knowledge train/validation rows to Phase 11 without merging them with trading experiences.

**Tech Stack:** Python 3.10 dataclasses, canonical JSON/JSONL, SQLite read-only catalog access, existing Phase 7 models, pytest fixtures; optional teacher adapters remain lazy and offline by default.

**Spec:** `docs/superpowers/specs/2026-09-15-phase-11a-book-knowledge-distillation-design.md`

## Global Constraints

* Phase 7 is the sole knowledge source; its catalog, chunks, and artifacts are read only.
* PDF/EPUB parsing, OCR, embedding, RAG, MT5, TradingAgents decisions, execution, Phase 8 memory, and Phase 10 eligibility are out of scope.
* Lesson IDs, packets, canonical files, splits, hashes, and manifests are deterministic once teacher output is accepted.
* Every factual claim has explicit Phase 7 source references; unsupported claims are excluded.
* No raw-book dumping, excessive verbatim copying, fabricated BUY/SELL/HOLD or future outcomes, secrets, prompts, completions, or hidden reasoning may enter outputs.
* All repository tests are offline and use fake teachers; no test downloads models or requires CUDA.
* Existing `tradingagents`, stock CLI, Phase 5–10 packages, and Phase 11 decision trainer remain behaviorally unchanged.
* Published generations are immutable and create-only; failures leave bounded diagnostics and never overwrite prior data.

---

### Task 1: Versioned contracts and errors

**Files:**
- Create: `tradingagents/distillation/__init__.py`
- Create: `tradingagents/distillation/models.py`
- Create: `tradingagents/distillation/errors.py`
- Test: `tests/test_phase11a_models.py`

**Interfaces:** `SourceRef`, `GroundingClaim`, `SourceBlock`, `SourcePacket`, `SourcePlan`, `TeacherConfig`, `KnowledgeExample`, `DatasetExclusion`, `DistillationManifest`, `ValidationReport`, enums, version constants, `canonical_json`, and `canonical_hash`.

- [ ] **Step 1: Write the failing test** asserting version constants, closed enums, required provenance, deterministic ID inputs, bounded text, JSON-safe serialization, and rejection of sensitive/future/action fields.
- [ ] **Step 2: Run the focused test** with `python -m pytest -q tests/test_phase11a_models.py`; it must fail because the contracts do not exist.
- [ ] **Step 3: Implement the frozen dataclass contracts** with canonical serialization, strict versions, source-ref validation, closed lesson/difficulty/reason/status enums, and deterministic `example_id` derivation from canonical source refs/instructions/policies.
- [ ] **Step 4: Re-run the focused test** and confirm it passes without importing optional ML/provider modules.
- [ ] **Step 5: Commit** with `feat: add phase11a knowledge contracts`.

### Task 2: Read-only Phase 7 adapter

**Files:**
- Create: `tradingagents/distillation/phase7.py`
- Test: `tests/test_phase11a_phase7.py`

**Interfaces:** `Phase7KnowledgeSource.open(root, expected_generation_id=None)`, `.generation_id`, `.source_fingerprints`, `.documents()`, `.blocks()`, and packet-independent `SourceBlock` conversion.

- [ ] **Step 1: Write the failing test** using a temporary SQLite/catalog fixture and spies proving opening/reading current chunks never constructs parsers, embedders, writers, or writes the Phase 7 root; assert complete `SourceRef` values and generation identity.
- [ ] **Step 2: Run the focused test** and verify the missing adapter fails.
- [ ] **Step 3: Implement the read-only adapter** over `KnowledgeCatalog`, active `IndexGeneration`, `current_document_ids`, `get_document`, and `chunks_for_document`; derive stable catalog/projection fingerprints and fail typed on missing/incomplete artifacts.
- [ ] **Step 4: Re-run the focused test** and verify the source tree/artifact bytes are unchanged.
- [ ] **Step 5: Commit** with `feat: add read-only phase7 distillation source`.

### Task 3: Deterministic packet planner

**Files:**
- Create: `tradingagents/distillation/planning.py`
- Test: `tests/test_phase11a_planning.py`

**Interfaces:** `SourcePacketPlanner.plan(source, topics, config) -> SourcePlan` and bounded packet settings.

- [ ] **Step 1: Write the failing test** proving repeated planning yields identical packet IDs/order, groups equation-definition and table-caption blocks, respects character/block/token limits, and does not emit a whole-book packet.
- [ ] **Step 2: Run the focused test** and confirm failure.
- [ ] **Step 3: Implement lexical topic scoring, durable section grouping, structural bundles, stable tie-breaks, packet bounds, and a request fingerprint containing Phase 7/policy/selection settings.
- [ ] **Step 4: Re-run the focused test** and confirm deterministic plans and explicit `SOURCE_PACKET_TOO_LARGE` diagnostics.
- [ ] **Step 5: Commit** with `feat: add deterministic phase7 source planning`.

### Task 4: Teacher protocol and structured fake

**Files:**
- Create: `tradingagents/distillation/teacher.py`
- Test: `tests/test_phase11a_teacher.py`

**Interfaces:** `Teacher.generate(packet, config) -> TeacherResult`, `TeacherResult`, `FakeTeacher`, and `teacher_from_environment()`.

- [ ] **Step 1: Write the failing test** for configurable provider/model/temperature/limit/timeout, deterministic fake responses, no teacher when unconfigured, bounded teacher errors, and zero implicit network calls.
- [ ] **Step 2: Run the focused test** and confirm failure.
- [ ] **Step 3: Implement the protocol and fake; keep any optional provider boundary lazy and return `DISTILLATION_TEACHER_NOT_CONFIGURED` when absent.
- [ ] **Step 4: Re-run the focused test** and confirm provider identity is metadata only until a caller explicitly supplies a teacher.
- [ ] **Step 5: Commit** with `feat: add injectable distillation teacher boundary`.

### Task 5: Grounding and safety validation

**Files:**
- Create: `tradingagents/distillation/grounding.py`
- Create: `tradingagents/distillation/quality.py`
- Test: `tests/test_phase11a_grounding.py`

**Interfaces:** `GroundingValidator.validate(candidate, packet)`, `QualityPolicy.validate(candidate, packet)`, and exclusion reason diagnostics.

- [ ] **Step 1: Write the failing test** covering accepted grounded paraphrase, missing/foreign refs, unsupported claims, contradictions, incomplete targets, action/future fields, hidden reasoning/sensitive fields, excessive verbatim overlap, and lesson-length bounds.
- [ ] **Step 2: Run the focused test** and verify failure.
- [ ] **Step 3: Implement independent claim-to-ref checks, bounded recursive safety checks, action/future rejection, normalized-token overlap guard, and closed reason codes; never repair or infer content.
- [ ] **Step 4: Re-run the focused test** and confirm all bad candidates become explicit exclusions.
- [ ] **Step 5: Commit** with `feat: enforce grounded knowledge lesson quality`.

### Task 6: Deduplication and grouped splits

**Files:**
- Create: `tradingagents/distillation/dedup.py`
- Create: `tradingagents/distillation/splits.py`
- Test: `tests/test_phase11a_dedup_splits.py`

**Interfaces:** `canonical_example_id`, `DedupIndex.check`, `GroupedSplitter.assign`, and `validate_no_group_leakage`.

- [ ] **Step 1: Write the failing test** for exact/equivalent duplicate rejection, deterministic near-duplicate detection, grouped document-section assignments, no leakage, approximately 70/15/15 splits when possible, and explicit `INSUFFICIENT_DATA` when groups are insufficient.
- [ ] **Step 2: Run the focused test** and verify failure.
- [ ] **Step 3: Implement normalized exact keys, deterministic local similarity, stable group hashing/greedy assignment, and split validation; never split a source group.
- [ ] **Step 4: Re-run the focused test** and confirm results are order-independent.
- [ ] **Step 5: Commit** with `feat: add phase11a deduplication and grouped splits`.

### Task 7: Immutable generation writer and validator

**Files:**
- Create: `tradingagents/distillation/writer.py`
- Test: `tests/test_phase11a_writer.py`

**Interfaces:** `write_generation(output_root, examples, exclusions, assignments, metadata) -> Path` and `validate_generation(path) -> ValidationReport`.

- [ ] **Step 1: Write the failing test** for canonical JSONL, manifest distributions/counters, all file hashes, source-index provenance, atomic publication, interrupted staging, and create-only immutability.
- [ ] **Step 2: Run the focused test** and verify failure.
- [ ] **Step 3: Implement staged create-only publication of `manifest.json`, `examples.jsonl`, `excluded.jsonl`, `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `source_index.json`; bind Phase 7 fingerprints, teacher/request identity, policies, safety counters, and generation identity.
- [ ] **Step 4: Re-run the focused test** and confirm tampering, unknown versions, missing refs, and destination collisions fail closed.
- [ ] **Step 5: Commit** with `feat: publish immutable phase11a generations`.

### Task 8: Distillation factory orchestration

**Files:**
- Create: `tradingagents/distillation/factory.py`
- Test: `tests/test_phase11a_factory.py`

**Interfaces:** `plan(source, config)`, `distill(plan, teacher, output_root)`, `validate(path)`, and bounded `BuildReport` telemetry.

- [ ] **Step 1: Write the failing test** for positive fake-teacher generation across all required lesson types, accepted/rejected counts, no-CoT/no-action safety, duplicate handling, retries/failures, zero MT5/network calls, and no teacher configured status.
- [ ] **Step 2: Run the focused test** and verify failure.
- [ ] **Step 3: Implement orchestration in deterministic packet order: teacher result → schema → grounding → quality → dedup → grouped split → immutable writer; record every failed candidate without fabricating lessons.
- [ ] **Step 4: Re-run the focused test** and confirm accepted lessons remain source-grounded and no raw source dump is emitted.
- [ ] **Step 5: Commit** with `feat: add phase11a distillation factory`.

### Task 9: Explicit Phase 11 curriculum adapter

**Files:**
- Create: `tradingagents/distillation/curriculum.py`
- Modify: `tradingagents/finetuning/__init__.py` only if a lazy export is needed
- Test: `tests/test_phase11a_curriculum.py`

**Interfaces:** `KnowledgeDatasetBinding.open(path)`, `KnowledgeTrainingSource`, and `CurriculumPlan` with explicit `KNOWLEDGE` then `EXPERIENCE` stages.

- [ ] **Step 1: Write the failing test** proving only train/validation knowledge rows load, test rows are inaccessible to the trainer, source type is explicit, Phase 10 remains separate, and an empty Phase 10 generation remains empty.
- [ ] **Step 2: Run the focused test** and verify failure.
- [ ] **Step 3: Implement read-only binding and stage metadata without modifying Phase 10 eligibility or the Phase 11 trainer.
- [ ] **Step 4: Re-run the focused test** and confirm source separation and test protection.
- [ ] **Step 5: Commit** with `feat: add explicit knowledge curriculum handoff`.

### Task 10: Standalone CLI

**Files:**
- Create: `tradingagents/distillation/cli.py`
- Modify: `pyproject.toml` to add only `knowledge-distill`
- Test: `tests/test_phase11a_cli.py`

**Interfaces:** `knowledge-distill plan|distill|validate|inspect` with bounded JSON output.

- [ ] **Step 1: Write the failing test** for command routing, read-only plan, unconfigured-teacher failure without fabricated output, validate/inspect without teacher or writer construction, and unchanged stock/knowledge CLIs.
- [ ] **Step 2: Run the focused test** and verify failure.
- [ ] **Step 3: Implement lazy command handlers and machine-readable statuses; never auto-start distillation or training.
- [ ] **Step 4: Re-run the focused test** and confirm optional dependencies/providers are not imported for metadata commands.
- [ ] **Step 5: Commit** with `feat: add standalone phase11a CLI`.

### Task 11: Deterministic HFT/FX fixtures and end-to-end offline acceptance

**Files:**
- Create: `tests/fixtures/phase11a_knowledge.py`
- Create: `tests/test_phase11a_acceptance.py`

- [ ] **Step 1: Write the failing fixture acceptance test** with order-flow imbalance, liquidity, market impact, inventory risk, volatility, equations, tables, and fake teacher lessons across the closed lesson types.
- [ ] **Step 2: Run the focused acceptance test** and verify the implementation is incomplete.
- [ ] **Step 3: Implement only fixture helpers needed to exercise public contracts; do not add production shortcuts.
- [ ] **Step 4: Re-run the focused acceptance test** and confirm train/validation/test counts, provenance, safety counters, split leakage, no action/future/CoT leakage, and hash validation.
- [ ] **Step 5: Commit** with `test: add phase11a offline acceptance fixtures`.

### Task 12: Real Phase 7 read-only planning acceptance

**Files:**
- Create: `scripts/phase11a_real_acceptance.py`
- Create: `tests/test_phase11a_real_acceptance.py`
- Create: `docs/phase11a-book-distillation.md`

- [ ] **Step 1: Write the failing opt-in acceptance test** that opens the configured real Phase 7 root, records source fingerprints before/after, reports documents/chunks/packets, and returns `DISTILLATION_TEACHER_NOT_CONFIGURED` when no teacher is supplied.
- [ ] **Step 2: Run the bounded test**; skip only when the documented real artifact path is unavailable, and never fabricate lessons.
- [ ] **Step 3: Implement the script/operator guide with explicit paths, offline behavior, teacher configuration, inspect/validate commands, recovery, and Phase 12 handoff.
- [ ] **Step 4: Re-run the bounded acceptance and confirm source/artifact fingerprints are unchanged and network/MT5 counters are zero.
- [ ] **Step 5: Commit** with `docs: add phase11a operator workflow`.

### Task 13: Whole-branch verification and review

**Files:**
- Modify: `.superpowers/sdd/2026-09-15-phase-11a-book-knowledge-distillation/progress.md`

- [ ] **Step 1: Run Phase 11A tests**, then affected Phase 10/11 compatibility tests.
- [ ] **Step 2: Run `python -m ruff check tradingagents cli scripts tests` and `python -m compileall -q tradingagents cli scripts`.
- [ ] **Step 3: Run `git diff --check` and inspect the branch diff against `fd80d4cc67a03558d30bc4837253de2932d24476`; verify no Phase 7 source/artifact, Phase 10 eligibility, MT5, stock CLI, or execution changes.
- [ ] **Step 4: Run the final whole-branch code review using the review package and resolve Critical/Important findings before completion.
- [ ] **Step 5: At final closure only, run the full suite and report the six known unchanged Phase 9 Windows spawn-timeout failures exactly if they recur.
- [ ] **Step 6: Commit final documentation/ledger only after all verification is recorded.

