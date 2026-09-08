# Phase 5 Shadow Outcome Evaluator — Implementation Report

**Date:** 2026-09-09
**Repository:** `C:\\AITrading\\TradingAgents`
**Accepted Phase 4.3 baseline:** `3d9262c639ca1ede3ac830626361500b760d3a29`
**Implementation tip before this report:** `417a27e20b6992f62b6006feccb7df53c3f6675b`

## Delivered scope

Phase 5 is implemented as a standalone, deterministic, read-only outcome
evaluator. It does not rerun TradingAgents, make LLM calls, alter the stock
CLI, or expose an execution path.

- Added `tradingagents/forex/evaluation.py` with pure outcome math, per-basis
  records, SQLite persistence, maturity handling, recovery transitions, and
  batch/single-decision evaluation services.
- Added read-only `MT5Provider.get_ticks_range()` with UTC normalization,
  `time_msc` preference, bounded reads, and typed data errors.
- Added `forex-evaluate` as a separate CLI; the existing `tradingagents` stock
  entry point was not modified.
- Extended persisted shadow decisions with explicit analysis, completion, and
  fresh broker-reference timestamps/quotes plus delay and status evidence.
- Added unit, migration, CLI, safety, deterministic integration, and guarded
  real-terminal tests.

Implementation commits after the accepted baseline:

```text
cf3453f docs: specify phase 5 shadow outcome evaluation
6e817a1 docs: plan phase 5 shadow outcome evaluation
42f9b5e docs: refine phase 5 reference recovery semantics
e56f6a6 feat: persist shadow decision temporal evidence
0dd9e6c feat: add read-only MT5 historical ticks
9a23129 feat: add deterministic forex outcome math
53abaac feat: persist basis-aware shadow evaluations
dcc18d7 feat: add deterministic shadow outcome evaluator
cc0ff1b feat: add standalone forex outcome CLI
128e93a test: guard read-only shadow evaluation
417a27e fix: harden shadow evaluator batch lifecycle
```

## Temporal and evaluation semantics

Every new decision keeps two distinct quote sets:

- `analysis_snapshot_*` is the quote captured before the graph runs.
- `decision_completed_timestamp` is the application UTC completion time.
- `decision_reference_*` is one fresh read-only quote after the structured
  Portfolio Manager result. Its timestamp comes from the broker tick
  (`Mt5Spread.timestamp`, with a read-only tick fallback), never from the
  application clock. `decision_reference_delay_seconds` is calculated against
  completion, and older broker data is retained as `INVALID_TEMPORAL`.

The evaluator persists one row per
`(decision_id, evaluation_basis, horizon_seconds)` for both
`ANALYSIS_SNAPSHOT` and `DECISION_REFERENCE`. It leaves rows `PENDING` until
`target + observation_tolerance`; after that deadline it accepts only the
first valid quote in the closed tolerance interval. MFE/MAE uses the exact
anchor-to-target interval and never the tolerance tail.

For each completed horizon both directional counterfactuals are retained:

```text
buy_net  = future_bid - entry_ask
sell_net = entry_bid - future_ask
```

HOLD selected trading net is zero, while
`hold_opportunity_cost_points = max(0, buy_net_points, sell_net_points)`.
Cost-aware MFE is signed and may remain negative when price never overcomes
the entry spread; it is not floored.

`source_context_eligible` is separate from nullable `training_eligible`.
Phase 5 never assigns a training label. `COMPLETE` and `INELIGIBLE` rows are
immutable. `PENDING` may become `COMPLETE` or `DATA_UNAVAILABLE`, and a later
successful bounded historical read may recover `DATA_UNAVAILABLE` to
`COMPLETE`, preserving the original creation time, recovery/evaluation
timestamps, data-source/version provenance, and prior unavailable reason.

## Verification evidence

Commands were run with the repository `.venv` Python runtime.

| Check | Result |
|---|---|
| Focused Phase 5/MT5/CLI suite | `99 passed, 2 skipped` |
| Full repository suite | `812 passed, 5 skipped, 18 warnings, 71 subtests passed in 115.60s` |
| Ruff | `All checks passed!` |
| Python compile check | `compileall` passed |
| Read-only AST scan | Passed; no direct `MetaTrader5` import or mutation method definitions in evaluator/CLI |
| Guarded local MT5 integration | `2 passed, 2 deselected in 4.00s` with `RUN_MT5_INTEGRATION=1` |
| LLM calls in evaluator | `0` by contract and in deterministic/CLI metrics |

The guarded terminal tests read a bounded historical range and a market
snapshot, resolve the broker symbol, and assert positions and orders are
identical before and after. No order, position, SL/TP, or other mutation API is
defined or called. Test databases use temporary paths; the verification did
not mutate a production shadow database.

## Closure

Phase 5 verification is complete. Phase 6 (corpus construction, training,
evaluation promotion, or any execution path) was not started.
