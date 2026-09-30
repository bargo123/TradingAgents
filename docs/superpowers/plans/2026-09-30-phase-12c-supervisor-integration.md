# Phase 12C Supervisor-Owned Shadow HFT Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing Forex supervisor the single owner of the strategic watcher and the read-only Phase 12C HFT shadow engine, with validated strategic-plan handoff and no execution path.

**Architecture:** Keep the existing strategic watcher lease authoritative for the supervisor process and add a separate HFT lease in the HFT ledger so two HFT engines cannot run concurrently. The supervisor starts one HFT worker in the same process as the watcher; all MT5 operations use a shared serialized gate at individual provider-operation boundaries so a slow LLM analysis does not pause the fast engine. Completed normalized shadow decisions feed only typed, provenance-bound, expiring plans into an atomic plan store.

**Tech Stack:** Existing Python watcher/supervisor, dataclasses, SQLite, `SerializedMt5OperationGate`, read-only `MT5Provider`, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-30-phase-12-hft-shadow-design.md`

## Global Constraints

- No MT5 mutation API, order path, `order_send`, position close/modify, or live execution.
- Every HFT action, fill, position, plan, and account row has `executed=False`.
- The strategic watcher lease remains authoritative; a valid lease is never bypassed.
- A strategic HOLD/NONE decision produces a valid no-direction plan and therefore no HFT entry.
- Invalid, incomplete, stale, or unproven-provenance plans fail closed to `NO_ACTION`.
- Phase 7, Phase 8/9/10, and Phase 11A artifacts remain read-only and isolated.
- Phase 12D is not implemented or exposed.

## Review Focus

- A second supervisor/HFT owner must be rejected before MT5/provider construction.
- A HOLD/NONE decision must never become a directional plan.
- A failed or incomplete strategic decision must not replace the last valid plan.
- A plan must expire into `NO_ACTION` rather than inventing direction.
- A slow strategic analysis must not block tick processing between read-only MT5 operations.

### Task 1: Lease domains and shared MT5 operation boundary

**Files:**
- Modify: `tradingagents/forex/hft/store.py`, `tradingagents/forex/hft/runtime.py`, `cli/forex_hft_shadow.py`
- Modify: `tradingagents/forex/watcher.py`, `cli/forex_watch.py`
- Test: `tests/test_phase12_runtime.py`, `tests/test_phase12_cli_runtime.py`, new lease tests

- [ ] Write RED tests for HFT lease acquisition, active-owner refusal, expiry recovery, and the standalone CLI refusing before provider construction.
- [ ] Verify RED with the focused Phase 12 lease tests.
- [ ] Add a versioned HFT lease table and bounded acquire/heartbeat/release methods; retain the strategic watcher lease unchanged.
- [ ] Add a read-only provider proxy that gates each MT5 operation individually and exposes only read-only methods.
- [ ] Verify GREEN and commit the lease/resource-boundary change.

### Task 2: Provenance-bound strategic plan contract and feed

**Files:**
- Modify: `tradingagents/forex/hft/models.py`, `tradingagents/forex/hft/plan_store.py`, `tradingagents/forex/hft/store.py`
- Create: `tradingagents/forex/hft/plan_feed.py`
- Modify: `tradingagents/forex/watcher.py`
- Test: new `tests/test_phase12_plan_feed.py`, `tests/test_phase12_models.py`, `tests/test_phase12_plan_store.py`

- [ ] Write RED tests for BUY/SELL/HOLD mapping, runtime metadata, source decision/run provenance, schema version, expiry, and invalid-decision rejection.
- [ ] Verify RED.
- [ ] Extend `StrategicExecutionPlan` with `valid_from`, `source_decision_id`, `source_run_id`, `git_commit`, `analysis_profile`, `plan_schema_version`, and a provenance validity check without changing existing replay constructors.
- [ ] Implement `build_plan_from_shadow_decision()` and a watcher completion callback; only normalized complete decisions may replace the current plan.
- [ ] Ensure HOLD maps to `Direction.NONE`, and failed/incomplete decisions leave the previous plan unchanged.
- [ ] Verify GREEN and commit the plan-feed change.

### Task 3: Continuous HFT worker and account/persistence integration

**Files:**
- Create: `tradingagents/forex/hft/supervisor.py`
- Modify: `tradingagents/forex/hft/runtime.py`, `tradingagents/forex/hft/store.py`, `tradingagents/forex/hft/account.py`, `tradingagents/forex/hft/dashboard.py`
- Test: `tests/test_phase12_runtime.py`, new `tests/test_phase12_hft_supervisor.py`, `tests/test_phase12_store.py`, `tests/test_phase12_dashboard.py`

- [ ] Write RED tests for worker start/stop, plan swaps while ticks continue, account snapshots, daily 10% benchmark fields, and no-action with HOLD/expired plans.
- [ ] Verify RED.
- [ ] Add bounded `stop_event` support, HFT lease heartbeat, continuous tick metrics, account updates, and plan provenance persistence to `HftShadowRuntime`.
- [ ] Implement a supervisor-owned worker that initializes only the read-only provider, runs the same deterministic engine, and shuts down without mutating MT5.
- [ ] Verify GREEN and commit the worker/persistence change.

### Task 4: One-command supervisor integration

**Files:**
- Modify: `tradingagents/forex/supervisor.py`, `cli/forex_supervisor.py`, `cli/forex_watch.py`
- Test: `tests/test_forex_supervisor.py`, `tests/test_forex_watch_cli.py`, new integration tests

- [ ] Write RED tests proving `forex-supervisor run --hft-shadow` owns watcher plus HFT, refuses duplicate HFT ownership, forwards plan completions, and keeps legacy supervisor mode unchanged.
- [ ] Verify RED.
- [ ] Add explicit safe HFT options (`--hft-shadow`, `--hft-db-path`, bounded poll/max-ticks) without adding any execution flag.
- [ ] Start/stop the HFT worker within the existing supervisor lifecycle; do not create a second strategic watcher or bypass the watcher lease.
- [ ] Verify GREEN and commit the supervisor integration.

### Task 5: Health/dashboard and crash recovery

**Files:**
- Modify: `tradingagents/forex/supervisor.py`, `cli/forex_supervisor.py`, `cli/forex_dashboard.py`, `tradingagents/forex/hft/dashboard.py`
- Test: `tests/test_phase12_supervisor.py`, `tests/test_phase12_dashboard.py`, new crash-recovery tests

- [ ] Write RED tests for separate strategic/HFT/Ollama/MT5-read-only health, crash recovery in FLAT/LONG/SHORT states, and persisted scalar-only dashboard output.
- [ ] Verify RED.
- [ ] Implement additive health fields and dashboard section labelled `PHASE 12C — SHADOW HFT`, including `EXECUTED = FALSE` and the 10% benchmark comparison.
- [ ] Verify GREEN and commit the health/recovery change.

### Task 6: Scale replay and sensitivity research

**Files:**
- Modify: `tradingagents/forex/hft/replay.py`, `tradingagents/forex/hft/account.py`
- Test: `tests/test_phase12_replay.py`, new cost/risk sensitivity tests

- [ ] Write RED tests for independent momentum/range/combined reports, cost and risk grids, walk-forward train/development/validation/unseen partitions, and descriptive daily compounding fields.
- [ ] Verify RED.
- [ ] Add bounded research helpers without optimizing on the unseen partition and without fabricating tick data.
- [ ] Run the largest valid repository tick dataset; if none exists, record the unavailable-data result and retain the deterministic fixture smoke only.
- [ ] Verify GREEN and commit the research helpers if code changed.

### Task 7: Controlled transition and final verification

**Files:** reports/tests only unless a defect is proven.

- [ ] Run focused Phase 12C tests, affected Forex tests, full pytest, Ruff, compileall, and diff check.
- [ ] Inspect the active PID/lease; wait for a safe boundary before transitioning the old supervisor.
- [ ] Stop the old supervisor gracefully, start exactly one supervisor-owned HFT runtime, and verify one strategic lease plus one HFT lease.
- [ ] Run one bounded real MT5 read-only canary only after all gates pass; capture positions/orders before and after without calling mutation APIs.
- [ ] Report replay scale, walk-forward, cost/risk sensitivity, runtime throughput/latency, dashboard/health, database integrity, and explicit Phase 12D exclusion.

