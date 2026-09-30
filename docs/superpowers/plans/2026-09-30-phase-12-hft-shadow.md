# Phase 12 HFT-Style Forex Shadow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, replayable, read-only HFT-style Forex shadow engine and validate it offline before an explicitly bounded MT5 shadow run.

**Architecture:** New `tradingagents.forex.hft` modules own typed plans, causal features, deterministic engines, realistic shadow fills, persistence, replay, and MT5 shadow runtime. Existing stock CLI, Phase 5/6 tables, Phase 7 artifacts, and Phase 11A remain unchanged.

**Tech Stack:** Python dataclasses, SQLite, existing MT5 provider, argparse CLIs, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-30-phase-12-hft-shadow-design.md`

## Global Constraints

- Every action/fill is shadow-only and persists `executed=False`.
- No MT5 mutation API or order path may be imported or called.
- The fast path contains no LLM calls; strategic plans are validated and atomically swapped.
- Replay and walk-forward are causal and use the same engines as shadow runtime.
- Phase 7/11A and existing Phase 5/6 rows are read-only and isolated.
- Phase 12D is not implemented or exposed.
- Do not commit `watch_dashboard.py`.

## Review Focus

- A malformed/expired plan must never produce an entry; tests live in `test_phase12_models.py` and `test_phase12_plan_store.py`.
- A BUY/SELL fill must use the correct side of the quote with slippage; tests live in `test_phase12_execution.py`.
- A replay tick must never read future history; tests live in `test_phase12_replay.py`.
- A risk reject must be persisted without creating a position; tests live in `test_phase12_risk.py`.
- MT5 shadow CLI must refuse active watcher leases before provider construction and expose no mutation API; tests live in `test_phase12_runtime.py`.

### Task 1: Typed plans, ticks, and atomic plan store

**Files:**
- Create: `tradingagents/forex/hft/__init__.py`, `models.py`, `plan_store.py`
- Test: `tests/test_phase12_models.py`, `tests/test_phase12_plan_store.py`

**Interfaces:** `StrategicExecutionPlan`, `Tick`, `FastAction`, `RiskDecision`, `ShadowPosition`, `utc`, `AtomicPlanStore.replace/current`.

- [x] Write RED tests for bounds, UTC, expiry, symbol mismatch, and atomic replacement.
- [x] Run `python -m pytest -q tests/test_phase12_models.py tests/test_phase12_plan_store.py` and observe the missing-contract failure.
- [x] Implement bounded frozen dataclasses and copy-on-write plan replacement.
- [x] Re-run focused tests to GREEN and commit `feat: add phase12 execution contracts`.

### Task 2: Causal features and fast entry/exit engines

**Files:**
- Create: `features.py`, `engines.py`
- Test: `tests/test_phase12_features.py`, `tests/test_phase12_engines.py`

**Interfaces:** `TickFeatureEngine.update`, `FastExecutionEngine.on_tick`, `ExitPolicy`.

- [x] Add RED tests for causal rolling values, monotonic timestamps, spread/momentum gates, exits, and millisecond telemetry.
- [x] Run focused tests and confirm RED.
- [x] Implement bounded deque features and deterministic entry/exit rules with no LLM calls.
- [x] Run tests GREEN and commit `feat: add deterministic phase12 fast engine`.

### Task 3: Risk, fills, positions, and account simulation

**Files:**
- Create: `risk.py`, `execution.py`, `account.py`
- Test: `tests/test_phase12_risk.py`, `tests/test_phase12_execution.py`, `tests/test_phase12_account.py`

**Interfaces:** `RiskEngine.evaluate`, `ShadowFillEngine.fill`, `ShadowPositionLedger`, `AccountSimulator`, `CompoundingMode`.

- [x] Add RED tests for every risk limit, bid/ask fill side, slippage/latency, lifecycle, MAE/MFE, sizing, drawdown, and 10%-benchmark reporting.
- [x] Verify RED.
- [x] Implement deterministic bounded checks and transactional in-memory state.
- [x] Run focused tests GREEN and commit `feat: add phase12 shadow risk and fills`.

### Task 4: SQLite ledger and crash recovery

**Files:**
- Create: `store.py`
- Test: `tests/test_phase12_store.py`

**Interfaces:** `HftShadowStore.initialize`, `record_tick`, `record_action`, `record_risk`, `record_fill`, `snapshot`, `recover_open_positions`.

- [x] Add RED tests for schema version, idempotent writes, executed=false, row isolation, and crash recovery.
- [x] Verify RED.
- [x] Implement explicit schema/version migrations and transaction boundaries.
- [x] Run GREEN and commit `feat: persist phase12 shadow ledger`.

### Task 5: Causal replay, walk-forward, and strategy lab

**Files:**
- Create: `replay.py`
- Create: `cli/forex_hft_replay.py`
- Test: `tests/test_phase12_replay.py`, `tests/test_phase12_walk_forward.py`

**Interfaces:** `TickReplay`, `ReplayReport`, `walk_forward_splits`, `StrategyFamily`.

- [x] Add RED tests for source fingerprints, future/lookahead rejection, chronological splits, identical runtime engines, and scalar metrics.
- [x] Verify RED.
- [x] Implement CSV/JSONL tick loading, two bounded strategy families, replay reports, and explicit train/dev/validation/test partitions.
- [x] Run GREEN and commit `feat: add phase12 tick replay lab`.

### Task 6: Read-only MT5 shadow runtime and CLI

**Files:**
- Create: `runtime.py`
- Create: `cli/forex_hft_shadow.py`
- Test: `tests/test_phase12_runtime.py`, `tests/test_phase12_cli.py`

**Interfaces:** `HftShadowRuntime.run_once/run`, `ReadOnlyTickSource`, `PlanProvider`.

- [x] Add RED tests proving active lease refusal before provider construction, no mutation methods, one MT5 operation at a time, plan expiry fail-closed, and executed=false.
- [x] Verify RED.
- [x] Implement provider injection, existing serialized gate reuse, bounded loop, and prominent shadow banner.
- [x] Run GREEN and commit `feat: add phase12 read-only shadow runtime`.

### Task 7: Supervisor/dashboard read-only metrics seam

**Files:**
- Modify: `tradingagents/forex/supervisor.py`, `tradingagents/forex/dashboard.py`, `cli/forex_dashboard.py` only where tests prove a compatible read-only seam.
- Test: `tests/test_phase12_supervisor.py`, `tests/test_phase12_dashboard.py`

- [x] Add RED tests for health/status, HFT SHADOW metrics, and duplicate-process refusal.
- [x] Implement minimal scalar read-only health/dashboard integration without changing stock CLI or watcher semantics.
- [x] Run focused tests GREEN and commit `feat: expose phase12 shadow health`.

### Task 8: Acceptance verification and bounded 12C smoke

**Files:** tests and reports only unless a defect is proven.

- [x] Run all Phase 12 focused tests, Ruff, compileall, and diff check.
- [x] Run the full suite at the major checkpoint and record all skips/failures exactly.
- [x] Run a deterministic replay acceptance on a fixture or existing read-only tick sample; require no lookahead, realistic fills, risk safety, and sub-millisecond/millisecond-class fast-path telemetry.
- [x] Defer the bounded MT5 shadow smoke because the existing supervisor lease is active; the active-lease refusal and no-mutation tests pass, so no second MT5 owner was started.
- [x] Final review explicitly confirms no Phase 7/11A changes, no order API, no 12D path, and clean worktree except pre-existing `watch_dashboard.py`.
