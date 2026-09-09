# Phase 6.1 Quality and Observability Report

Date: 2026-09-09  
Scope baseline: `cdbedf7911a7f08d599d69146e50a8c83b5ac8b8`  
Implementation commit: `7e6dbdb650c1405291f7f5109a01631b8f2c98bf`  
Mode: standalone `forex-watch` / `forex-shadow`, read-only MT5  
Phase 7: not started

## Root-cause investigation

The reviewed real row is decision
`3b4443ee-8fa3-4947-b71c-0a5445194d0d`, source run
`623ec407-20d2-4d72-88b5-98272c400b0e`, from
`data_cache/phase6-broker-clock-qwen-once-20260909.db`. The row remains a
genuine structured Portfolio Manager `Hold`, strict normalization
`NORMALIZED`, `executed=False`, and `decision_context_status=INCOMPLETE`.

The persisted boundary trace contains 20 metadata-only entries (before/after
for each expected LLM-bearing node). It reports:

| Stage | Present | Raw chars / fields | Visible content chars |
| --- | ---: | ---: | ---: |
| Market Analyst | yes | 800 | 800 |
| News Analyst | yes | 1,476 | 1,476 |
| Bull Researcher | no | 15 | 0 |
| Bear Researcher | no | 15 | 0 |
| Research Manager | yes | 1,280 | 1,280 |
| Trader | yes | 531 | 531 |
| Aggressive Risk | no | 21 | 0 |
| Conservative Risk | yes | 1,144 | 1,121 |
| Neutral Risk | no | 18 | 0 |
| Portfolio Manager structured result | yes | 7 fields | — |

The first substantive loss is the Bull Researcher **after** boundary. The
Bull node builds `"Bull Analyst: {response.content}"` directly; a blank
visible response therefore produces the observed 15-character wrapper. Bear
and the two empty risk responses show the same pattern. This establishes:

- **A — yes:** the actual Qwen prose nodes returned empty visible
  `response.content`, leaving label-only state values.
- **B — no:** the first failing after-boundary is already empty, before any
  downstream reducer or prompt hand-off can remove content.
- **C — no:** these affected prose nodes do not pass through schema
  normalization; they use direct response-content extraction. The structured
  Research Manager, Trader, and Portfolio Manager outputs are present.
- **D — no truncation evidence:** the exact label-only lengths prove zero
  visible content, but the old run has no callback token/finish metadata from
  which to claim a token-limit cause. Hidden reasoning is not promoted or
  persisted.
- **E — no path divergence:** the watcher calls the existing
  `ForexShadowRunner`, which constructs the same forex `TradingAgentsGraph` /
  `GraphSetup` path used by the deterministic Phase 4.3 proof. That proof
  reaches all fields with non-empty stub reports; the real row reflects model
  output quality, not a separate watcher graph.

The Phase 4.3 reducer and Portfolio Manager prompt fix is present and was not
changed in Phase 6.1. The validator continues to reject label-only context as
`INCOMPLETE`; no fake summaries, hidden reasoning, or artificial action were
introduced.

## Telemetry fix

The old watcher row recorded zero analysis calls because
`cli.forex_watch._make_coordinator` did not instantiate the existing
`StatsCallbackHandler`. Phase 6.1 now creates one safe numeric callback per
coordinator and passes it through `WatcherCoordinator` to the existing runner
callback path. The callback records aggregate/per-agent call, token, timing,
and model identifiers only.

`ForexShadowRunner` now reports `telemetry_status=AVAILABLE` with the supplied
numeric values when a usable callback returns stats. Without a usable callback,
analysis fields are `None` (`NULL` in watcher SQLite) and status is
`UNAVAILABLE`; the status CLI prints `unknown` rather than a fabricated zero.
The existing Phase 5 evaluator remains explicitly zero-LLM and continues to
report numeric `llm_calls=0`. Prompts, completions, private reasoning, and
credentials are not stored.

## Broker reference diagnostic

The original decision completed at
`2026-09-09T10:14:15.734231Z`; its broker reference tick was
`2026-09-09T10:13:40.576Z`, 35.158 seconds earlier, so its persisted status is
`INVALID_TEMPORAL`. A bounded zero-LLM MetaQuotes-Demo diagnostic initialized
EURUSD and read five ticks:

```text
normalized tick ages (seconds): 8.308, 8.397, 10.555, 8.564, 10.719
time_msc: 1788961441388, 1788961443462, 1788961443462,
          1788961447635, 1788961447635
time_msc advances: true
broker clock: CALIBRATED, offset +10800s, server MetaQuotes-Demo
positions before/after: 0 / 0, unchanged=True
orders before/after: 0 / 0, unchanged=True
```

This classifies the old reference as transient stale-feed evidence. Phase 5
did not receive an application-timestamp substitution or a weakened temporal
rule.

## TDD and verification

The new RED tests failed before production changes for (1) missing callback
telemetry defaulting to zero/no status and (2) watcher coordinator construction
omitting callbacks. After the minimal fix, the focused suite passed, including
runner numeric propagation, unavailable-to-NULL persistence, callback wiring,
metadata-only context traces, and no-private-text assertions.

```text
focused Phase 4/5/6/provider suite: 173 passed, 1 skipped
full pytest suite: 898 passed, 6 skipped, 18 warnings, 71 subtests
Ruff: passed
compileall: passed
git diff --check: passed
guarded real MT5 integration (RUN_MT5_INTEGRATION=1): 9 passed
```

The guarded integration covered normalized snapshot/history reads,
before/after positions and orders, the one-snapshot adapter, and static
mutation-surface checks. No order or execution API was added or called.

No second 40-minute Qwen graph was run. The real decision and its eight
`INELIGIBLE` Phase 5 rows were not modified or promoted. Phase 6.1 ends here;
Phase 7, training, execution, and model/prompt changes remain out of scope.
