# Phase 4.3 Forex Context Integrity Implementation Plan

**Implementation status:** Completed and validated 2026-09-08. The RED/GREEN
evidence and final verification are recorded in the Phase 4.3 report.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove and repair propagation of normal TradingAgents research/risk artifacts through the standalone forex shadow graph, then persist an explicit context-integrity result without changing stock semantics.

**Architecture:** Trace the existing shared state schema, reducers, conditional graph edges, and forex prompt wrappers before changing behavior. Reproduce the missing-context symptom with deterministic structured stubs, fix the source field/reducer/edge mismatch, and validate integrity from the resulting state rather than copying prose into prompts. Persist `COMPLETE` or `INCOMPLETE` alongside the existing strict normalized shadow decision; no execution or Phase 5 behavior is added.

**Tech Stack:** Python 3.10+, LangGraph/LangChain state annotations and reducers, Pydantic, SQLite, pytest, Ruff.

**Baseline:** Phase 4.2 final commit `1d55066b7bdb7e3e275730ec26543de1fe5f78d1`.

## Global Constraints

- Keep `forex-shadow` standalone, read-only, and `executed=False`; do not add order APIs, execution, RiskGovernor, RAG, training, ONNX, or Phase 5 evaluation.
- Do not alter MT5 provider/models, stock CLI behavior, stock prompts, or stock graph/state semantics.
- Preserve the existing required flow: Market + News → Bull + Bear → Research Manager → Trader → three Risk Analysts → Portfolio Manager.
- Use only the existing normal reports/structured state fields; never persist or log private chain-of-thought.
- Instrument boundaries with metadata only: presence booleans, character/count sizes, node names, and field names.
- The forex default remains `market,news`, one cached MT5 snapshot, and the `INTRADAY` profile.
- Context-integrity failure must remain explicit and exclude the row from future training/evaluation by status; it must not be relabeled as complete.

## Task 1: Root-cause state and reducer trace

**Files to inspect:**

- `tradingagents/agents/utils/agent_states.py`
- `tradingagents/agents/utils/agent_utils.py`
- `tradingagents/agents/researchers/bull_researcher.py`
- `tradingagents/agents/researchers/bear_researcher.py`
- `tradingagents/agents/managers/research_manager.py`
- `tradingagents/agents/trader/trader.py`
- `tradingagents/agents/risk_mgmt/aggressive_debator.py`
- `tradingagents/agents/risk_mgmt/conservative_debator.py`
- `tradingagents/agents/risk_mgmt/neutral_debator.py`
- `tradingagents/agents/managers/portfolio_manager.py`
- `tradingagents/graph/setup.py`
- `tradingagents/graph/propagation.py`
- `tradingagents/graph/conditional_logic.py`
- `tradingagents/forex/runner.py`

- [x] Record the exact state fields each node reads and writes, including reducer/append annotations and conditional edge targets.
- [x] Compare the same boundaries in stock mode and identify every forex-only wrapper or state initialization transformation.
- [x] Add metadata-only boundary instrumentation to the deterministic trace seam (presence and size/count only; no report content).
- [x] Run the deterministic trace once and document the first boundary where a non-empty artifact becomes absent or renamed.

## Task 2: Deterministic failing propagation test (RED)

**Files:**

- Create or modify: `tests/test_forex_phase43_context_integrity.py`
- Reuse: existing fake provider/graph/state helpers in `tests/test_forex_shadow_runner.py` and `tests/test_forex_graph_mode.py`

- [x] Build structured stub outputs for Bull, Bear, Research Manager, Trader, all three Risk Analysts, and Portfolio Manager.
- [x] Execute the real node functions or graph state transitions in deterministic order, not a prose-copying test double.
- [x] Assert Bull/Bear history fields, Research Manager input, Trader input, risk debate history, and Portfolio Manager input are populated at every boundary.
- [x] Assert the test fails against the current implementation at the first missing field and record the failure as the root-cause evidence.

## Task 3: Minimal root-cause propagation fix (GREEN)

**Files:** Only the production files identified by Task 1.

- [x] Correct the source field name, reducer behavior, edge/state merge, or forex wrapper that Task 1 proves responsible; do not inject fabricated summaries.
- [x] Preserve existing stock state keys, reducers, prompts, and graph routing exactly.
- [x] Re-run the deterministic test and a stock regression test; both must pass.
- [x] Add assertions that required normal reports are passed unchanged (or through the existing renderer) to the downstream node that consumes them.

## Task 4: Persist explicit context-integrity status

**Files:**

- `tradingagents/forex/shadow.py`
- `tradingagents/forex/runner.py`
- `cli/forex_shadow.py`
- `tests/test_forex_shadow_contract.py`
- `tests/test_forex_shadow_runner.py`

- [x] Define a backward-compatible `decision_context_status` with `COMPLETE` and `INCOMPLETE` values (and an optional diagnostic reason/metadata field if needed).
- [x] Require non-empty expected artifacts from Market, News, Bull, Bear, Research Manager, Trader, risk debate, and Portfolio Manager for `COMPLETE`.
- [x] Persist incomplete evidence with the genuine raw PM result and strict normalization status, without relabeling an incomplete run as training-quality.
- [x] Add SQLite migration, round-trip, runner, and CLI evidence tests; retain `executed=False` invariant.

## Task 5: Validation and report

- [x] Run the deterministic trace and focused propagation/integrity tests.
- [x] Run the full suite, Ruff, compileall, and `git diff --check`.
- [x] Verify stock-mode regression tests and the MT5 mutation-surface guard.
- [x] Evaluate the existing real EURUSD v3 evidence; no second ~40-minute local run was practical, so the deterministic trace is the bounded correctness proof.
- [x] Update `docs/forex-shadow.md` and create a Phase 4.3 validation report with the actual root cause and evidence.
- [x] Commit the completed Phase 4.3 work and stop; do not begin Phase 5.
