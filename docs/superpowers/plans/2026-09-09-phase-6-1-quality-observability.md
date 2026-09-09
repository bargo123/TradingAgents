# Phase 6.1 Quality and Observability Hardening Plan

**Status:** Completed and verified 2026-09-09; implementation is limited to the
accepted Phase 6 collector and stops before Phase 7.

**Baseline:** `cdbedf7911a7f08d599d69146e50a8c83b5ac8b8`

## Goal

Make the accepted Phase 6 collector’s evidence truthful and observable without
changing its architecture, prompts, model routing, MT5 provider, Phase 5
semantics, or stock CLI. Preserve the existing read-only, one-slot
`forex-watch` flow and `executed=False` invariant.

## Investigation findings (before implementation)

The persisted run `3b4443ee-8fa3-4947-b71c-0a5445194d0d` / source run
`623ec407-20d2-4d72-88b5-98272c400b0e` contains metadata-only boundaries. The
first substantive loss is the Bull Researcher **after** boundary: its field is
15 characters (`"\\nBull Analyst: "`) with zero visible content. Bear then has
the same label-only result. Aggressive and Neutral Risk behave similarly;
Conservative has substantive visible content. The Research Manager, Trader, and
Portfolio Manager artifacts are present. Because the after-boundary is captured
inside the instrument wrapper before LangGraph applies any later reducer, this
is not downstream state loss. Bull/Bear/Risk use direct `response.content`
extraction, not a schema parser, and the exact label-only lengths prove the
visible provider responses were empty. Hidden reasoning is neither assumed nor
persisted. The accepted Phase 4.3 deterministic graph test with non-empty stub
responses passes through the same `GraphSetup` and state fields, proving the
collector path is not a different graph. No prompt/model change is authorized
in this pass; the real row remains `INCOMPLETE`.

The watcher row reports `llm_calls=0`, `tool_calls=0`, and zero token counts
because `cli.forex_watch._make_coordinator` did not construct/pass the existing
`StatsCallbackHandler`. The runner already accepts callbacks and reports its
numeric, per-agent-safe stats. The evaluator’s explicit `llm_calls=0` remains
correct because it is a zero-LLM Phase 5 operation.

A short live diagnostic initialized the local MetaQuotes-Demo terminal, read
EURUSD five times, and observed normalized `time_msc` values advancing. Quote
age was approximately 8–11 seconds; positions and orders were equal before and
after. The old reference quote being 35.158 seconds before application
completion is therefore genuine transient/stale-feed evidence. Keep
`INVALID_TEMPORAL`; never substitute an application timestamp or loosen the
Phase 5 check.

## Scope constraints

- Keep MT5 read-only; do not add or call execution/mutation APIs.
- Do not alter prompts, models, graph routing, reducers, context validator, or
  Phase 5 evaluator semantics.
- Do not fabricate reports, promote incomplete context, or relabel the real
  decision.
- Do not persist prompts, completions, hidden reasoning, credentials, or raw
  callback payloads.
- Missing analysis telemetry must be `NULL`/unavailable, not numeric zero.
  Numeric zero remains valid for the existing zero-LLM evaluator.
- Keep `forex-shadow` and the stock `tradingagents` CLI behavior unchanged.

## Task 1 — deterministic RED tests

1. Add a runner test showing that a run with no callback reports telemetry
   fields as unavailable (`None`) rather than fabricated zero values. Keep the
   existing supplied-callback test asserting exact numeric propagation.
2. Add a watcher CLI/coordinator test proving production coordinator creation
   constructs one `StatsCallbackHandler` and passes it to
   `WatcherCoordinator`; use monkeypatched factories so no MT5, graph, or LLM
   resource is initialized.
3. Add/retain a deterministic context test that records only artifact
   presence/character counts and confirms non-empty stub Bull/Bear/Risk values
   reach Research Manager, Trader, Risk, and Portfolio Manager. Keep the
   label-only case explicitly `INCOMPLETE`.
4. Run the focused tests and record the expected RED failures before editing
   production code.

## Task 2 — minimal GREEN implementation

1. Change `tradingagents.forex.runner._callback_metrics` defaults for analysis
   telemetry to `None` and include an explicit availability marker (for
   example, `telemetry_status=UNAVAILABLE`). When a valid callback exposes
   `get_stats()`, mark it available and copy only numeric aggregate fields and
   safe per-agent numeric/model identifiers. Ignore malformed callbacks without
   inventing values. Ensure evaluator metrics continue to expose numeric
   `llm_calls=0` through their existing path.
2. In `cli.forex_watch._make_coordinator`, construct the existing
   `StatsCallbackHandler` and pass it as the coordinator’s callback sequence.
   `WatcherCoordinator._run_claimed` will then reuse the already-supported
   runner callback path; no second telemetry implementation is introduced.
3. Keep watcher SQLite nullable metric columns and evidence mappings as-is so
   unavailable analysis telemetry persists as SQL `NULL` and the safe metrics
   JSON contains no private text. Update only user-facing status formatting if
   needed to display unavailable rather than zero.

## Task 3 — verification and report

Run, in order:

- focused context, runner, watcher, CLI, and telemetry tests;
- Phase 4 forex tests, Phase 5 evaluator tests, and Phase 6 watcher tests;
- the full pytest suite;
- Ruff, `compileall`, and `git diff --check`;
- the opt-in guarded MT5 read-only integration, including before/after
  positions/orders and mutation-surface scans.

Run no second 40-minute Qwen graph automatically. Since the context finding is
provider/model output inadequacy rather than a remaining propagation defect,
stop after deterministic proof and report that no final real `forex-watch once`
was repeated. Include the prior real run’s node/artifact metadata, telemetry
availability correction, broker-clock diagnostic, test/static results, baseline
and final SHAs, and explicit `executed=False`/unchanged positions-orders
evidence. Do not begin Phase 7.
