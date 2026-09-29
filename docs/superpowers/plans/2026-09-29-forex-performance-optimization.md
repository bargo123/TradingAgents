# Forex Shadow Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a causally safe replay benchmark and use measured, test-backed optimizations to bring representative Forex shadow analysis under the existing 900-second freshness budget without weakening any safety or reasoning contract.

**Architecture:** A read-only replay harness will invoke the existing `ForexShadowRunner.analyze(..., persist=False)` and graph factory against immutable saved snapshots, emitting versioned scalar benchmark artifacts outside the live database. Each candidate optimization is isolated behind RED/GREEN tests and compared on identical replay cases before any supervisor restart or live canary.

**Tech Stack:** Python, pytest, LangGraph/TradingAgentsGraph, existing Forex telemetry/callbacks, local Ollama, SQLite read-only snapshot access, JSON benchmark artifacts.

**Spec:** `docs/superpowers/specs/2026-09-29-forex-performance-optimization-design.md`

## Global Constraints

- Do not modify Phase 7 or Phase 11A artifacts, code, or data.
- Do not call MT5 mutation APIs or add an order path; all runs remain read-only and `executed=False`.
- Do not weaken schemas, grounding, risk controls, freshness, temporal validity, normalization, or fail-closed behavior.
- Replay/benchmark output is explicitly `REPLAY`/`BENCHMARK` and never writes live watcher, decision, or evaluation tables.
- Do not persist prompts, completions, reasoning, credentials, or hidden chain-of-thought.
- Keep the existing supervisor lease, one loaded Ollama model, and one request at a time unless a test and resource measurement prove a safer alternative.
- Never delete or rewrite existing shadow database rows.
- Do not commit `watch_dashboard.py`.

## Review Focus

- Replay must not write live SQLite tables or mutate the source snapshot; test this in Task 1.
- Critical-path duration must account for overlap and dependencies rather than summing node times; test this in Task 1.
- Parallel risk execution must preserve reducer state and speaker order when dependencies exist; test this in Task 3.
- PM/RM compaction must retain every context-integrity artifact and strict structured field; test this in Task 4.
- Temporal polling must reject old ticks and never substitute application time; test this in Task 7.

### Task 1: Replay case and scalar benchmark contracts

**Files:**
- Create: `tradingagents/forex/performance.py`
- Create: `cli/forex_benchmark.py`
- Create: `tests/test_forex_performance.py`
- Modify: `tradingagents/forex/runner.py` only where a metadata-only replay seam is required

**Interfaces:**
- `ReplayCase` identifies a read-only source run, snapshot fingerprint, symbol, timeframe, and causal timestamp.
- `BenchmarkConfig` selects replay cases, provider/models, evidence mode, and output path.
- `BenchmarkReport` serializes versioned scalar totals, node/call timings, dependency edges, critical path, token counts, retries, tool time, model metadata, validation status, and replay label.
- `run_benchmark(config: BenchmarkConfig) -> BenchmarkReport` invokes the existing non-persisting runner path and never opens a write connection to the source DB.

- [ ] **Step 1: Write failing isolation tests**

  Add tests proving a benchmark case loads from a saved `forex_watch_runs.snapshot_json`, invokes `ForexShadowRunner.analyze(..., persist=False)`, leaves source bytes and row counts unchanged, marks output `run_kind="REPLAY"`, and rejects missing/future/non-causal case data.

- [ ] **Step 2: Run the tests to verify RED**

  Run: `python -m pytest -q tests/test_forex_performance.py`

  Expected: failure because the replay contracts and CLI do not yet exist.

- [ ] **Step 3: Implement the minimal replay contracts**

  Deserialize only the stored causal snapshot fields needed by `ForexMarketSnapshot`; use the existing saved-snapshot provider seam and callback/state telemetry. Emit scalar JSON under a dedicated benchmark artifact path. Keep provider calls explicit and opt-in.

- [ ] **Step 4: Run the tests to verify GREEN**

  Run: `python -m pytest -q tests/test_forex_performance.py`

  Expected: all replay isolation and serialization tests pass.

- [ ] **Step 5: Commit**

  `git add tradingagents/forex/performance.py cli/forex_benchmark.py tradingagents/forex/runner.py tests/test_forex_performance.py && git commit -m "feat: add forex replay performance harness"`

### Task 2: Critical-path and provider telemetry

**Files:**
- Modify: `tradingagents/forex/telemetry.py`
- Modify: `tradingagents/forex/runner.py`
- Modify: `tradingagents/forex/performance.py`
- Test: `tests/test_forex_performance.py`

**Interfaces:**
- `critical_path_from_intervals(nodes, edges) -> CriticalPathReport` returns longest dependency-respecting duration, overlap, idle/wait, and unknown timing components.
- `scalar_call_metrics(...)` records model, prompt/completion token counts, finish status, retry count, tool time, and load/swap time without content.

- [ ] **Step 1: Write failing interval/overlap tests**

  Cover overlapping Market/News, sequential Research Manager → Trader, risk dependency edges, missing timing metadata, and non-negative idle/wait calculations.

- [ ] **Step 2: Run RED**

  Run: `python -m pytest -q tests/test_forex_performance.py -k critical_path`

- [ ] **Step 3: Implement pure critical-path accounting and metadata-only provider fields**

  Do not infer missing provider timings as zero. Preserve existing live telemetry keys for compatibility and add only versioned benchmark fields.

- [ ] **Step 4: Run GREEN and lint**

  Run: `python -m pytest -q tests/test_forex_performance.py -k critical_path; python -m ruff check tradingagents/forex/telemetry.py tradingagents/forex/runner.py tradingagents/forex/performance.py tests/test_forex_performance.py`

- [ ] **Step 5: Commit**

  `git add tradingagents/forex/telemetry.py tradingagents/forex/runner.py tradingagents/forex/performance.py tests/test_forex_performance.py && git commit -m "feat: report forex replay critical paths"`

### Task 3: Risk dependency audit and safe scheduling decision

**Files:**
- Inspect/modify only if proven safe: `tradingagents/graph/setup.py`, `tradingagents/graph/parallel_analysts.py`, `tradingagents/graph/conditional_logic.py`
- Inspect: risk analyst factories under `tradingagents/agents`
- Test: `tests/test_forex_latency.py`, new focused tests in `tests/test_forex_performance.py`

**Interfaces:**
- `risk_dependency_report() -> Mapping[str, Any]` documents whether Aggressive, Conservative, and Neutral consume prior speaker output or only the common Trader state.
- If and only if independent, a bounded `run_parallel_risk_analysts(...)` preserves deterministic reducer merge order and branch failure semantics.

- [ ] **Step 1: Add a RED test for the proven dependency contract**

  Use real prompt/state construction to assert whether each risk node reads prior risk history. If any node depends on the previous speaker, assert that parallel execution is rejected and the existing sequential graph remains unchanged. If all are independent, assert concurrent completion with deterministic merge and complete histories.

- [ ] **Step 2: Run RED**

  Run: `python -m pytest -q tests/test_forex_latency.py tests/test_forex_performance.py -k risk`

- [ ] **Step 3: Implement only the proven branch**

  Preserve state reducers, risk rounds, speaker routing, and stock mode. A sequential result is an acceptable outcome when semantic dependencies are real; record that finding in benchmark metadata rather than forcing parallelism.

- [ ] **Step 4: Run focused GREEN tests**

  Run: `python -m pytest -q tests/test_forex_latency.py tests/test_forex_graph_mode.py tests/test_forex_performance.py -k risk`

- [ ] **Step 5: Commit only if production scheduling changes**

  Use `perf: parallelize independent forex risk analysis` or commit the dependency audit test alone if no scheduling change is safe.

### Task 4: Lossless PM/RM context compaction

**Files:**
- Inspect/modify: `tradingagents/agents/managers/portfolio_manager.py`
- Inspect/modify: `tradingagents/agents/managers/research_manager.py`
- Inspect/modify only where handoff is defined: `tradingagents/graph/propagation.py`, `tradingagents/graph/setup.py`
- Test: `tests/test_forex_phase42_prompts.py`, `tests/test_forex_phase43_context_integrity.py`, `tests/test_ollama_forex_structured_nodes.py`

**Interfaces:**
- A forex-only compact handoff must retain Market, News, Bull, Bear, Research Manager, Trader, all risk debate artifacts, evidence context, and runtime profile metadata.
- `ForexPortfolioDecision` remains unchanged and is still validated strictly after mapping.

- [ ] **Step 1: Add RED prompt/context tests**

  Capture in-memory prompt metadata only and assert unique required artifacts appear once, duplicated debate/risk copies are absent, and context-integrity traces remain COMPLETE. Assert stock and non-Ollama prompts are byte-for-byte behavior-compatible where applicable.

- [ ] **Step 2: Run RED**

  Run: `python -m pytest -q tests/test_forex_phase42_prompts.py tests/test_forex_phase43_context_integrity.py tests/test_ollama_forex_structured_nodes.py -k compact`

- [ ] **Step 3: Implement minimal compact handoffs**

  Remove only content proven redundant by the state graph. Use labeled structured summaries/references rather than truncation. Do not remove unique evidence, risk reports, or required runtime context.

- [ ] **Step 4: Run GREEN and strict-output tests**

  Run: `python -m pytest -q tests/test_forex_phase42_prompts.py tests/test_forex_phase43_context_integrity.py tests/test_ollama_forex_structured_nodes.py`

- [ ] **Step 5: Commit**

  `git add tradingagents/agents tradingagents/graph tests/test_forex_phase42_prompts.py tests/test_forex_phase43_context_integrity.py tests/test_ollama_forex_structured_nodes.py && git commit -m "perf: compact forex decision context without loss"`

### Task 5: News, Market, and Research Manager contract audit

**Files:**
- Inspect/modify only when the replay gate proves a safe change: `tradingagents/agents/analysts/news_analyst.py`, `tradingagents/agents/analysts/market_analyst.py`, `tradingagents/agents/managers/research_manager.py`, `tradingagents/forex/context.py`, and their prompt tests
- Test: `tests/test_news_analyst_prompt.py`, `tests/test_forex_prompts.py`, `tests/test_forex_phase42_prompts.py`, `tests/test_forex_performance.py`

**Interfaces:**
- Deterministic market features are causal, snapshot-keyed, typed, and reproducible; the LLM interprets them instead of recomputing them.
- News and Research Manager outputs retain downstream-required evidence and do not persist reasoning.

- [ ] **Step 1: Add RED tests for causal feature ownership and required evidence**

  Prove every proposed deterministic feature uses only snapshot-time bars/ticks, and prove concise News/RM outputs still satisfy downstream context-integrity and grounding requirements.

- [ ] **Step 2: Run RED**

  Run: `python -m pytest -q tests/test_news_analyst_prompt.py tests/test_forex_prompts.py tests/test_forex_phase42_prompts.py tests/test_forex_performance.py -k causal`

- [ ] **Step 3: Implement the smallest safe contract change**

  Prefer typed feature blocks and concise, explicit evidence fields. If no safe reduction is proven, keep production behavior unchanged and record the negative finding in the benchmark report.

- [ ] **Step 4: Run GREEN and affected tests**

  Run: `python -m pytest -q tests/test_news_analyst_prompt.py tests/test_forex_prompts.py tests/test_forex_phase42_prompts.py tests/test_forex_performance.py`

- [ ] **Step 5: Commit only accepted changes**

  Use a focused `perf:` commit naming the actual contract changed; do not commit speculative prompt edits.

### Task 6: Same-case model-routing and output-budget benchmark

**Files:**
- Modify: `tradingagents/forex/performance.py`, `cli/forex_benchmark.py`
- Test: `tests/test_forex_performance.py`

**Interfaces:**
- `benchmark_models(case_ids, candidates) -> tuple[ModelBenchmarkResult, ...]` compares qwen3.5:2b and qwen3.5:4b on identical saved contexts, with bounded output and one request at a time.
- Results include runtime, input/output tokens, schema/context/evidence/risk validity, action, retry/failure classification, and configuration fingerprint.

- [ ] **Step 1: Add RED result-schema and no-auto-selection tests**

  Assert both candidates are reported factually, a faster model is not selected automatically, malformed output fails closed, and no live DB rows are written.

- [ ] **Step 2: Run RED**

  Run: `python -m pytest -q tests/test_forex_performance.py -k model`

- [ ] **Step 3: Implement bounded candidate execution**

  Support explicit local Ollama opt-in and bounded timeout/retry configuration. Preserve deep qwen3.5:4b for Research Manager, Trader, and PM unless measured equivalence is demonstrated.

- [ ] **Step 4: Run GREEN, then one bounded real benchmark**

  Run focused tests first. Then run the CLI against the fixed representative case set, sequentially, with artifacts outside `data_cache` and no MT5 access.

- [ ] **Step 5: Commit benchmark support**

  `git add tradingagents/forex/performance.py cli/forex_benchmark.py tests/test_forex_performance.py && git commit -m "feat: benchmark forex model routing safely"`

### Task 7: Temporal polling audit and regression coverage

**Files:**
- Inspect/modify only if a configuration/implementation defect is proven: `tradingagents/forex/runner.py`, `tradingagents/forex/runtime_config.py`
- Test: existing runner/reference tests plus `tests/test_forex_performance.py`

**Interfaces:**
- Existing `_fresh_reference_quote_until_post_completion` remains bounded and broker-time based.
- Telemetry reports attempts, wait, final broker delay, and status.

- [ ] **Step 1: Add RED tests for configured polling and freshness cap**

  Cover an old first tick followed by a qualifying tick, exact completion timestamp, all-old timeout, provider failure, zero remaining freshness budget, and no application-time substitution.

- [ ] **Step 2: Run RED**

  Run: `python -m pytest -q tests/test_forex_read_only_timeout_validation.py tests/test_forex_shadow_runner.py tests/test_forex_performance.py -k reference`

- [ ] **Step 3: Fix only a proven acquisition defect**

  Preserve strict `INVALID_TEMPORAL` behavior and bounded polling. If current one-attempt/zero-wait behavior is correctly caused by exhausted freshness, make no production change.

- [ ] **Step 4: Run GREEN**

  Run the same focused command and verify all temporal tests pass.

- [ ] **Step 5: Commit only if required**

  Use `fix: restore bounded decision-reference polling` only for an actual acquisition/configuration regression.

### Task 8: Replay baseline, candidate comparison, and directional audit

**Files:**
- Modify: `tradingagents/forex/performance.py`, `cli/forex_benchmark.py`
- Test: `tests/test_forex_performance.py`, `tests/test_forex_revision_validation.py`

**Interfaces:**
- `compare_benchmarks(baseline, candidate) -> ComparisonReport` rejects mismatched case/config fingerprints and reports runtime, critical path, token, validity, and safety deltas.
- `directional_distribution(reports) -> Mapping[str, Any]` reports Research/Trader/PM distributions and transitions without labels or trading outcomes.

- [ ] **Step 1: Add RED comparison and directional tests**

  Assert mismatched causal cases fail closed, HOLD is reported rather than penalized, and only scalar metadata is retained.

- [ ] **Step 2: Run RED**

  Run: `python -m pytest -q tests/test_forex_performance.py tests/test_forex_revision_validation.py -k comparison`

- [ ] **Step 3: Implement comparison and audit reports**

  Run the current-head replay sample, then each accepted candidate on the identical sample. Include a negative finding when the hardware/model lower bound remains above 900 seconds.

- [ ] **Step 4: Run GREEN and major focused suite**

  Run: `python -m pytest -q tests/test_forex_performance.py tests/test_forex_latency.py tests/test_forex_graph_mode.py tests/test_forex_revision_validation.py`

- [ ] **Step 5: Commit**

  `git add tradingagents/forex/performance.py cli/forex_benchmark.py tests/test_forex_performance.py tests/test_forex_revision_validation.py && git commit -m "feat: compare forex replay candidates"`

### Task 9: Full verification and controlled live canary

**Files:**
- Modify only if verification exposes a defect; otherwise no production files
- Test: full affected Forex suite and repository suite

- [ ] **Step 1: Run focused and affected verification**

  Run focused tests for every changed module, Ruff on changed files, compileall on changed modules, and `git diff --check`.

- [ ] **Step 2: Run full pytest at the major checkpoint**

  Run: `python -m pytest -q`

  Record every pass, skip, or failure exactly; do not reinterpret environment-gated skips.

- [ ] **Step 3: Verify replay acceptance**

  Require a materially safe sub-900-second representative replay, or record the measured lower-bound proof and required architecture/model change.

- [ ] **Step 4: Safely restart the single supervisor if needed**

  Wait for idle, stop only the existing supervisor, preserve its lease/database, start one process from the final commit, and verify persisted health is `HEALTHY` with no active run.

- [ ] **Step 5: Run exactly one live canary only after replay acceptance**

  Record commit provenance, run/decision IDs, runtime, freshness, context, normalization, Research/Trader/PM actions, reference status, `executed=False`, and database integrity. Do not retry a graph failure.

- [ ] **Step 6: Commit final accepted implementation**

  Use focused commits already created by the tasks; do not squash unrelated history or add `watch_dashboard.py`.

## Final handoff

Report starting/final HEAD, commits, harness path/command, baseline and candidate total/critical-path timings, per-node and token tables, model routing, parallelism, deterministic features, caching, PM/Risk/News/Market/RM findings, replay sample/distributions/transitions, temporal polling evidence, live canary, database integrity, supervisor PID/health, remaining bottleneck/risks, sustainable sub-900 assessment, realistic sub-750 assessment, and `NO LIVE ORDER WAS SENT.`
