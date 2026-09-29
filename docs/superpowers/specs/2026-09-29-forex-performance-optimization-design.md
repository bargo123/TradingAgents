# Forex Shadow Performance Optimization Design

## Status

Design approved in conversation on 2026-09-29. Implementation is not covered by this document; it begins only after the implementation plan is reviewed.

## Goals

- Reduce representative EURUSD M15 Forex shadow analysis below the existing 900-second freshness budget, with a preferred p95 target below 750 seconds.
- Measure the real critical path rather than optimizing isolated node timings.
- Preserve reasoning quality, strict schemas, evidence grounding, risk controls, temporal validity, normalization, and fail-closed behavior.
- Make before/after comparisons repeatable on stored causal inputs without waiting for live opportunities.
- Keep one MT5-using operation at a time, one supervisor, one loaded Ollama model, and no order path.

## Non-goals and hard boundaries

- No Phase 7 or Phase 11A changes, rebuilds, or data mixing.
- No MT5 mutation, order execution, leverage/risk changes, martingale, grid, or recovery logic.
- No freshness or temporal-gate weakening; an old broker tick remains invalid.
- No schema weakening, prose parsing, heuristic recovery, or hidden reasoning persistence.
- No strategy changes motivated only by a small HOLD sample.
- No cloud fine-tuning, model training, or automatic model downgrade.
- No deletion, rewriting, or migration of existing shadow decisions.
- Replay results are labeled `REPLAY`/`BENCHMARK` and never enter live watcher metrics or evaluation tables.

## Current baseline

The baseline commit is `27610cae509c2195741eaa4a2104fed55b43a6e5`. The live configuration uses a dedicated Ollama endpoint, 16K context, qwen3.5:2b for quick prose nodes, qwen3.5:4b for deep/structured nodes, one loaded model, and one request at a time. The baseline canary is approximately 1179 seconds with 12 calls; Market and News overlap, while debate and risk stages retain state dependencies.

## Architecture

### Replay benchmark boundary

Add a read-only benchmark/replay entry point that consumes a stored, causally valid snapshot and deterministic upstream fixture state. It invokes the existing Forex graph/factory where practical, but runs under an explicit `run_kind=REPLAY`/`BENCHMARK` context. It must not write `forex_watch_runs`, `shadow_decisions`, evaluation rows, or live telemetry, and must not call order APIs or expose future prices.

The harness accepts a stable replay-case identifier, model/provider configuration, and an optional candidate configuration. It emits a scalar JSON report containing case fingerprint, commit, graph mode, total wall time, per-node intervals, call count, prompt/completion tokens, model, finish status, retries, tool time, model load/swap observations, overlap/dependency edges, and validation outcomes. Prompts, completions, chain-of-thought, credentials, and raw market data are not persisted in benchmark output.

### Critical-path accounting

Telemetry records monotonic start/end times for every node and call. The report derives:

- sequential dependency edges from graph transitions;
- overlap windows for concurrent branches;
- critical-path duration as the longest dependency-respecting path;
- idle/wait time between dependency completion and consumer start;
- model load/swap, prompt ingestion, generation, and tool/network components when provider metadata exposes them.

Missing provider timings remain explicitly unknown; they are never inferred as zero.

### Optimization sequence

Each candidate is isolated behind a focused test and benchmarked against the same replay cases:

1. Prove the three risk analysts' state dependencies from code and prompts. Parallelize only independent opinions; preserve any genuine debate ordering.
2. Remove only demonstrably duplicated PM/RM/risk context. Use compact structured handoffs or references while retaining every artifact required by context-integrity checks and the strict `ForexPortfolioDecision` schema.
3. Audit News and Market outputs/tools. Concise output or deterministic features are allowed only when downstream-required evidence and provenance remain available.
4. Benchmark qwen3.5:2b versus qwen3.5:4b on the same stored contexts for candidate mechanical nodes. Research Manager, Trader, and Portfolio Manager remain deep unless equivalence is demonstrated by validity, evidence, risk, and directional checks.
5. Investigate repeated immutable context and safe per-cycle caching with explicit keys. No cache may cross snapshots or symbols.
6. Diagnose reference polling independently. Acquisition may be corrected only if implementation/configuration is wrong; the temporal gate itself is unchanged.

No optimization is retained merely because one node is faster. A candidate must preserve structured validity, context completeness, evidence/provenance, risk fields, causal inputs, and fail-closed behavior on the same replay cases.

## Data and provenance

Replay cases are immutable copies or read-only views of existing approved snapshots. Each case has a deterministic fingerprint, source timestamp, symbol, timeframe, and source-run identifier. Benchmark artifacts live outside the live SQLite database. They include tool/provider/model versions and configuration fingerprints, but no prompts or completions.

## Directional audit

The harness may summarize Research Manager, Trader, and Portfolio Manager actions and transitions for a larger current-head replay sample. It must not add quotas, HOLD penalties, random directions, future labels, or hindsight features. If HOLD remains dominant, investigation follows the first proven signal-loss boundary; no prompt or strategy change is made without a semantic defect.

## Temporal reference behavior

The system first reads a broker tick, then performs only a small bounded poll when its broker timestamp precedes decision completion. It accepts only a broker timestamp at or after completion. If no qualifying tick arrives within the configured bound or freshness would be violated, the result remains `INVALID_TEMPORAL`. Application time is never substituted for market time. Replay reports polling attempts, wait, final broker delay, and status as scalar metadata.

## Testing strategy

- Unit tests for replay isolation, deterministic case fingerprints, scalar telemetry, critical-path calculation, and no live-table writes.
- RED/GREEN tests for each scheduling or prompt-contract change, including reducer/state preservation and fail-closed malformed outputs.
- Regression tests proving stock/non-Ollama paths and supervisor health verification remain unchanged.
- Same-case baseline/candidate comparisons for structured validity, required context, evidence, risk, temporal causality, and action transitions.
- Focused Forex tests, affected-suite tests, Ruff, compileall, and `git diff --check` after each change; full pytest at a major checkpoint.
- A live canary is permitted only after replay demonstrates a material safe improvement and the active supervisor is idle. It runs once, with provenance and `executed=False` verification.

## Acceptance criteria

The pass succeeds only if one of the following is proven:

1. A representative replay and one controlled live canary are comfortably below 900 seconds without any safety or quality regression; or
2. The replay lab proves a hardware/model lower bound above 900 seconds, identifies the measured bottleneck, and documents the concrete architectural/model change required to go lower.

The final report includes baseline/final commits, harness design, critical-path and node/token tables, model routing, parallelism, caching, PM/Risk/News/Market/RM findings, replay distributions/transitions, temporal polling evidence, live canary status, database integrity, supervisor PID/health, remaining risks, and the explicit statement `NO LIVE ORDER WAS SENT.`

## Future seams

The benchmark report schema is versioned and provider-neutral so later model or hardware tests can reuse it. Replay remains separate from live collection and from Phase 7 knowledge and Phase 11A distillation. Any future strategy or training work requires a separate approved phase.
