# Phase 14 Self-Enhancement Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an isolated, offline, fail-closed self-enhancement plane that imports verified DEMO experience, evaluates bounded exit-policy candidates causally, and records promotion/challenger/rollback decisions without changing live trading.

**Architecture:** Add `tradingagents.self_enhancement` with immutable contracts, a dedicated SQLite catalog, read-only ledger import, bounded candidates, causal replay/evaluation, promotion gates, and an explicit CLI. Reuse existing HFT features/strategies/risk/fills and Phase 7/8 provenance interfaces; keep all artifacts outside their roots.

**Tech Stack:** Python 3.10+, dataclasses/enums, SQLite, existing `tradingagents.forex.hft` primitives, existing `tradingagents.knowledge` and `tradingagents.experience` read-only seams, pytest/Ruff.

**Spec:** `docs/superpowers/specs/2026-10-02-phase-14-self-enhancement-design.md`

## Global Constraints

- Live HFT tick path is unchanged and has no LLM/RAG calls.
- Phase 7, Phase 8, Phase 10, and Phase 11A artifacts are read-only.
- Every Phase 14 result is `real_money=false`; Phase 14 has no MT5 mutation API.
- Automatic mutation is limited to typed, bounded exit-policy parameters.
- Replay is chronological and causal; future data is evaluator-only.
- Insufficient, stale, conflicted, or uncertain data never promotes.
- Existing stock and forex trading CLIs remain unchanged; Phase 14 has its own CLI.

## Review Focus

1. A tick after the candidate entry must not influence an earlier feature or split — tests in `test_phase14_replay.py`.
2. A test-only or reconciliation-uncertain DEMO row must never enter verified experience — tests in `test_phase14_importer.py`.
3. A candidate with equal dimensions but incompatible provenance/config must not promote — tests in `test_phase14_promotion.py`.
4. A crashed promotion must not leave an active candidate without a rollback package — tests in `test_phase14_store.py`.
5. No Phase 14 code path may expose REAL execution or an MT5 mutation callable — tests in `test_phase14_safety.py`.

---

### Task 1: Typed contracts and bounded mutation surface

**Files:**
- Create: `tradingagents/self_enhancement/__init__.py`
- Create: `tradingagents/self_enhancement/models.py`
- Test: `tests/test_phase14_models.py`

**Interfaces:**
- Produces `ParameterSpec`, `ExitPolicyConfig`, `StrategyVersion`, `ExperienceTrade`, `DataQualityReport`, `CandidateSpec`, `SplitFingerprint`, `MetricReport`, and enums for experiment/candidate/promotion states.

- [ ] Write failing tests for bounded parameter validation, immutable strategy/config hashes, safe execution metadata, and trade provenance requirements.
- [ ] Run `python -m pytest -q tests/test_phase14_models.py`; expect failures because the package/contracts do not exist.
- [ ] Implement frozen dataclasses and deterministic canonical hashing; reject unknown parameters, out-of-range values, non-UTC timestamps, REAL/CONTEST/UNKNOWN execution, and test-only experience.
- [ ] Re-run the focused tests; expect all passing.

### Task 2: Dedicated transactional catalog

**Files:**
- Create: `tradingagents/self_enhancement/store.py`
- Test: `tests/test_phase14_store.py`

**Interfaces:**
- `SelfEnhancementStore(path).initialize()`
- `record_experience(trade: ExperienceTrade) -> None`
- `create_experiment(...) -> str`
- `record_evaluation(...) -> None`
- `record_promotion(...) -> None`
- `recover_incomplete_experiments() -> tuple[str, ...]`
- `snapshot() -> dict[str, int]`

- [ ] Test schema creation, idempotent experience writes, immutable experiment history, transactional promotion+rollback package, and crash recovery.
- [ ] Run focused tests to verify RED.
- [ ] Implement a separate SQLite schema with experience, quality, candidates, experiments, evaluations, deployments, rollbacks, and trigger tables; use one transaction for each lifecycle transition.
- [ ] Re-run focused tests to verify GREEN.

### Task 3: Read-only experience importer and quality audit

**Files:**
- Create: `tradingagents/self_enhancement/importer.py`
- Create: `tradingagents/self_enhancement/quality.py`
- Test: `tests/test_phase14_importer.py`

**Interfaces:**
- `audit_hft_ticks(path, symbol) -> DataQualityReport`
- `import_verified_demo(hft_path, demo_path, store, *, source_run_ids=None) -> ImportReport`

- [ ] Test query-only source access, source fingerprints, closed DEMO matching, exclusion of canaries/synthetic/real-money/reconciliation uncertainty, duplicate handling, and data-quality flags.
- [ ] Run focused tests to verify RED.
- [ ] Implement read-only SQLite connections and deterministic import mapping; never update source DBs and quarantine rejected rows with reason codes.
- [ ] Re-run focused tests to verify GREEN.

### Task 4: Bounded candidates and frozen Phase 7 book bridge

**Files:**
- Create: `tradingagents/self_enhancement/candidates.py`
- Create: `tradingagents/self_enhancement/book_factory.py`
- Test: `tests/test_phase14_candidates.py`

**Interfaces:**
- `CandidateGenerator.generate(incumbent: StrategyVersion) -> tuple[CandidateSpec, ...]`
- `BookStrategyFactory.from_hits(hits) -> tuple[CandidateSpec, ...]`

- [ ] Test deterministic candidate IDs, bounded exit variants, no code/source mutation, and exact Phase 7 `KnowledgeHit` provenance.
- [ ] Run focused tests to verify RED.
- [ ] Implement baseline/expected-target/MFE/micro-reversal/trailing/no-progress variants and an adapter that accepts read-only knowledge hits without importing the query service into tick evaluation.
- [ ] Re-run focused tests to verify GREEN.

### Task 5: Causal replay and chronological splits

**Files:**
- Create: `tradingagents/self_enhancement/replay.py`
- Test: `tests/test_phase14_replay.py`

**Interfaces:**
- `chronological_splits(ticks, guard_gap_seconds) -> ChronologicalSplits`
- `ReplayEvaluator.evaluate(ticks, candidate, *, split) -> ReplayMetrics`

- [ ] Test strict monotonicity, future-tick rejection, split immutability, guard gaps, and candidate isolation.
- [ ] Run focused tests to verify RED.
- [ ] Implement replay using existing `TickFeatureEngine`, strategies, `SignalArbiter`, `HftExecutionEngine`, `RiskEngine`, `ShadowFillEngine`, and `AccountSimulator`; candidate parameters affect only the supplied exit policy.
- [ ] Re-run focused tests to verify GREEN.

### Task 6: Metrics, walk-forward, and cost sensitivity

**Files:**
- Create: `tradingagents/self_enhancement/evaluation.py`
- Test: `tests/test_phase14_evaluation.py`

**Interfaces:**
- `summarize_trades(...) -> MetricReport`
- `walk_forward(evaluator, ticks, candidate, windows) -> WalkForwardReport`
- `cost_sensitivity(evaluator, ticks, candidate, scenarios) -> CostSensitivityReport`

- [ ] Test all metric edge cases, missing commission markers, long/short/session/regime breakdowns, and no false statistical significance for tiny samples.
- [ ] Run focused tests to verify RED.
- [ ] Implement descriptive metrics and deterministic aggregation with persisted dataset/split fingerprints; no metric alone can approve.
- [ ] Re-run focused tests to verify GREEN.

### Task 7: Promotion gate, challenger isolation, rollback

**Files:**
- Create: `tradingagents/self_enhancement/promotion.py`
- Test: `tests/test_phase14_promotion.py`
- Test: `tests/test_phase14_safety.py`

**Interfaces:**
- `CandidatePromotionGate.evaluate(incumbent, candidate, reports) -> PromotionDecision`
- `RollbackController.rollback(store, deployment_id, reason) -> None`
- `ShadowChallenger` remains a pure offline/same-tick recorder and has no broker API.

- [ ] Test minimum-sample rejection, Pareto/risk gates, unseen/cost failures, schema/runtime regressions, challenger isolation, automatic rollback, and real-money impossibility.
- [ ] Run focused tests to verify RED.
- [ ] Implement deterministic multi-gate decisions, immutable deployment packages, and transactional rollback to the previous approved version.
- [ ] Re-run focused tests to verify GREEN.

### Task 8: Orchestrator and explicit CLI

**Files:**
- Create: `tradingagents/self_enhancement/orchestrator.py`
- Create: `tradingagents/self_enhancement/cli.py`
- Modify: `pyproject.toml` (add `self-enhancement` entry point only)
- Test: `tests/test_phase14_orchestrator.py`
- Test: `tests/test_phase14_cli.py`

**Interfaces:**
- `SelfEnhancementOrchestrator.run_once(...) -> ExperimentReport`
- CLI commands: `self-enhancement status`, `self-enhancement import`, `self-enhancement experiment`, `self-enhancement rollback`.

- [ ] Test trigger/sample insufficiency produces `NO_EXPERIMENT`, explicit paths are required, no MT5/LLM is constructed, and status is scalar-only.
- [ ] Run focused tests to verify RED.
- [ ] Implement a one-shot asynchronous-plane workflow with persisted lifecycle states and no busy loop; keep live CLIs untouched.
- [ ] Re-run focused tests to verify GREEN.

### Task 9: First offline autonomous experiment and closure verification

**Files:**
- Create: `scripts/phase14_first_experiment.py`
- Test: `tests/test_phase14_first_experiment.py`

- [ ] Add a deterministic fixture experiment proving development, validation, unseen, walk-forward, cost sensitivity, and a promotion or fail-closed rejection.
- [ ] Run the fixture first and verify no live artifacts changed.
- [ ] Run one bounded real offline experiment against a read-only snapshot if sufficient verified ticks/trades exist; otherwise persist `INSUFFICIENT_EVIDENCE`.
- [ ] Run focused Phase 14 tests, the full suite, Ruff, compileall, and `git diff --check`; confirm the current supervisor was not restarted and Phase 7/8/11A fingerprints are unchanged.
