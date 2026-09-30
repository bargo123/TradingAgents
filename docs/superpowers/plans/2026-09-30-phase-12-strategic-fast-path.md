# Phase 12C strategic freshness fast path

## Scope

Add an explicit, opt-in Phase 12 strategic graph mode while leaving legacy
INTRADAY/stock graph behavior unchanged.  The mode reuses the existing agent
factories and strict schemas, runs only causally independent Bull/Bear and risk
assessments in parallel, records canonical structured handoffs, and preserves
all freshness, context, evidence, and read-only gates.

## Tasks

1. Add RED tests for independent branch merging, explicit mode selection,
   canonical Trader action/PM rejection metadata, and nullable additive DB
   columns.
2. Add a bounded parallel-node helper and a Phase 12 graph construction path
   using existing instrumented nodes.
3. Thread an explicit `phase12_strategic` runtime/config flag through the
   supervisor, watcher, runner, and graph signature; keep the default direct
   graph legacy-compatible.
4. Persist only validated canonical Trader action and a deterministic PM
   rejection classification; never parse prose or alter schemas.
5. Select Phase 12 dependency telemetry and run offline replay benchmarks
   against the current frozen causal snapshot.
6. Run focused/affected/full verification and only then perform a lease-safe
   production restart and one fresh validation; leave the existing HFT path
   read-only and untouched.

## Safety gates

- No MT5 mutation, order API, Phase 7/11A changes, future inputs, or DB
  rewrites/backfills.
- Legacy INTRADAY and stock behavior must remain unchanged.
- Direction remains model-owned; HOLD and failed/incomplete states remain
  fail-closed.
- The 900-second freshness and broker-timestamp validity gates are unchanged.
