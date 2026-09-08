# Phase 5 Shadow Outcome Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, read-only `forex-evaluate` workflow that evaluates persisted forex shadow decisions at separate analysis-snapshot and decision-reference bases without LLM calls or execution.

**Architecture:** Extend `ShadowTradeDecision` and `ForexShadowRunner` with immutable analysis/completion/reference timestamps and quote sets. Add a read-only `MT5Provider.get_ticks_range()` seam, pure bid/ask evaluation helpers, and a SQLite `ShadowEvaluationStore` keyed by decision, basis, and horizon. `ShadowOutcomeEvaluator` will evaluate only mature horizons, use one bounded historical read per decision where possible, and expose separate status/metrics for `ANALYSIS_SNAPSHOT` and `DECISION_REFERENCE`; a standalone argparse `forex-evaluate` CLI will call that service.

**Tech Stack:** Python 3.10 dataclasses and typing, SQLite via `sqlite3`, timezone-aware `datetime`, existing `MT5Provider`/`Mt5Tick`, pytest fixtures and mocks, argparse, Ruff, and `compileall`.

**Spec:** `docs/superpowers/specs/2026-09-09-phase-5-shadow-outcome-evaluator-design.md`

## Global Constraints

- Keep the existing stock `tradingagents` CLI unchanged.
- Keep `forex-shadow` read-only and preserve `executed=False`.
- Do not add `order_send`, order lifecycle, position mutation, SL/TP, execution modes, RiskGovernor, RAG, training, ONNX, or outcome-label promotion.
- The evaluator must not import `MetaTrader5`; all terminal access goes through `MT5Provider`.
- Evaluation performs zero LLM calls and never re-runs the TradingAgents graph.
- Preserve both `ANALYSIS_SNAPSHOT` and `DECISION_REFERENCE` quote/timestamp sets; never substitute one basis for the other.
- Default horizons are `(300, 900, 1800, 3600)` seconds and default observation tolerance is `30` seconds.
- A horizon remains `PENDING` until `target_timestamp + observation_tolerance_seconds`; after that deadline the first valid tick in `[target, deadline]` is selected, otherwise `DATA_UNAVAILABLE`.
- MFE/MAE use ticks from the basis anchor through the exact target, never tolerance-window ticks.
- Both BUY and SELL counterfactuals are always stored. HOLD selected PnL is zero and its opportunity cost is `max(0, buy_net_points, sell_net_points)`.
- `source_context_eligible`, evaluation/data quality, and future `training_eligible` are independent; Phase 5 leaves `training_eligible` null and defers labeling to a corpus builder.
- Existing legacy `future_evaluation_status` and `outcome_*` fields are not changed by the evaluator.
- Provider failures leave terminal evidence untouched and leave unresolved horizons pending. `COMPLETE` and `INELIGIBLE` rows are immutable; `PENDING` may become `COMPLETE`/`DATA_UNAVAILABLE`, and a later successful MT5 read may recover `DATA_UNAVAILABLE` to `COMPLETE` with provenance preserved.
- MT5 integration tests are opt-in and must compare positions/orders before and after.

## File Map

- Modify `tradingagents/forex/shadow.py`: temporal decision dataclass fields, validation, SQLite migration, record serialization, and backward-compatible row conversion.
- Modify `tradingagents/forex/runner.py`: completion timestamp and exactly one fresh read-only quote after structured Portfolio Manager output.
- Modify `tradingagents/dataflows/mt5/provider.py`: UTC-normalized, read-only historical tick-range method.
- Create `tradingagents/forex/evaluation.py`: public basis/config/evaluation records, pure quote math, SQLite evaluation store, evaluator service, and result/metrics types.
- Modify `tradingagents/forex/__init__.py`: re-export public Phase 5 types.
- Create `cli/forex_evaluate.py`: separate read-only outcome-evaluation command.
- Modify `pyproject.toml`: register only the `forex-evaluate` entry point.
- Modify `docs/forex-shadow.md`: document temporal evidence, bases, maturity, formulas, HOLD semantics, safety, and CLI usage.
- Modify `tests/test_forex_shadow_contract.py`: temporal fields and migration/round-trip tests.
- Create `tests/test_forex_shadow_runner.py`: fresh-reference call ordering, failure, and temporal-invalid tests.
- Modify `tests/test_mt5_provider.py`: historical tick-range mapping and read-only safety tests.
- Create `tests/test_forex_shadow_evaluation.py`: deterministic evaluator math, status, maturity, and idempotency tests.
- Create `tests/test_forex_evaluate_cli.py`: CLI parsing/output and stock-CLI isolation tests.
- Create `tests/test_forex_shadow_evaluation_integration.py`: guarded real-terminal historical read and unchanged positions/orders.

---

### Task 1: Extend the decision record with temporal evidence

**Files:**
- Modify: `tests/test_forex_shadow_contract.py`
- Modify: `tests/test_forex_shadow_runner.py`
- Modify: `tradingagents/forex/shadow.py:272-368`
- Modify: `tradingagents/forex/runner.py:205-390`

**Interfaces:**
- Consumes: existing `ShadowTradeDecision`, `ShadowDecisionStore.record/get`, `ForexMarketSnapshot`, `MT5Provider.get_spread`, and structured Portfolio Manager output.
- Produces: optional-at-end `ShadowTradeDecision` fields `analysis_snapshot_timestamp`, `analysis_snapshot_bid`, `analysis_snapshot_ask`, `analysis_snapshot_spread`, `analysis_snapshot_spread_points`, `decision_completed_timestamp`, `analysis_latency_seconds`, `decision_reference_timestamp`, `decision_reference_bid`, `decision_reference_ask`, `decision_reference_spread`, `decision_reference_spread_points`, `decision_reference_status: Literal["AVAILABLE", "UNAVAILABLE", "INVALID_TEMPORAL"]`, `decision_reference_delay_seconds`, and `decision_reference_error`; `ShadowDecisionStore` persists and reconstructs them while treating legacy `snapshot_timestamp`/`reference_*` as analysis aliases. The reference timestamp must be the broker tick timestamp returned by `Mt5Spread.timestamp` or the smallest equivalent read-only tick seam, never the local completion time.

- [ ] **Step 1: Write the failing contract tests.** Add a round-trip test that constructs a decision with distinct analysis, completion, and broker-reference timestamps/quotes and asserts every field survives SQLite serialization. Add a legacy-schema test that initializes a database without the new columns and asserts the read object exposes analysis aliases while reference status is unavailable. Add runner tests with a fake provider/graph that record calls and assert `get_spread(resolved_symbol)` is called once after graph invocation, never before, that its `Mt5Spread.timestamp` is persisted as the reference timestamp (not the local completion timestamp), and that no second graph/LLM call occurs. Add tests for quote failure (`UNAVAILABLE` with null reference quote and a safe error) and an older broker reference timestamp (`INVALID_TEMPORAL` with retained raw quote).

- [ ] **Step 2: Run the focused tests to verify RED.**

Run: `pytest tests/test_forex_shadow_contract.py tests/test_forex_shadow_runner.py -q`

Expected: FAIL because the temporal dataclass fields, database columns, and post-graph reference capture do not yet exist.

- [ ] **Step 3: Implement the smallest temporal contract.** Append defaulted dataclass fields so existing positional construction remains valid. Validate every provided timestamp as timezone-aware UTC, finite quote values, status vocabulary, non-negative analysis latency, and `executed is False`; use the legacy fields as read-time fallbacks without rewriting old values. Extend `initialize()` with idempotent nullable columns and extend `record()`/`_row_to_decision()` with explicit serialization. In `ForexShadowRunner.run()`, retain the snapshot fields as the analysis set, capture `_utc_now()` immediately after structured PM extraction/normalization, compute latency, call `provider.get_spread()` exactly once, copy the provider's broker tick timestamp from `Mt5Spread.timestamp`, mark failure or temporal-invalid status explicitly, and persist the decision even when the fresh read fails. Do not feed the fresh quote back into graph state or substitute the local completion timestamp for a market timestamp.

- [ ] **Step 4: Run the focused tests to verify GREEN.**

Run: `pytest tests/test_forex_shadow_contract.py tests/test_forex_shadow_runner.py -q`

Expected: PASS, including old Phase 4 contract tests and all new temporal/reference assertions.

- [ ] **Step 5: Commit the temporal contract.**

```bash
git add tradingagents/forex/shadow.py tradingagents/forex/runner.py tests/test_forex_shadow_contract.py tests/test_forex_shadow_runner.py
git commit -m "feat: persist shadow decision temporal evidence"
```

### Task 2: Add the read-only MT5 historical tick seam

**Files:**
- Modify: `tests/test_mt5_provider.py`
- Modify: `tradingagents/dataflows/mt5/provider.py:63-330`

**Interfaces:**
- Consumes: existing `MT5Provider`, `_utc_timestamp`, symbol resolution, and injected MT5 module fakes.
- Produces: `MT5Provider.get_ticks_range(symbol: str, start: datetime, end: datetime, *, flags: int | None = None) -> tuple[Mt5Tick, ...]`, using `copy_ticks_range` only.

- [ ] **Step 1: Write the failing provider tests.** Add fake `copy_ticks_range` data containing `time_msc` and `time` rows, assert exact-first/prefix/suffix symbol resolution, UTC timestamps, chronological preservation, and default `COPY_TICKS_ALL` (or `-1` only when the fake has no constant). Add tests for aware non-UTC inputs, reversed ranges, `None` response, malformed rows, and empty valid windows. Extend the mutation-surface scan to include the new method and assert no `order_send`, buy/sell, close, modify, or other mutation API is exposed.

- [ ] **Step 2: Run the focused tests to verify RED.**

Run: `pytest tests/test_mt5_provider.py -q`

Expected: FAIL because `get_ticks_range` is missing and the fake cannot observe the required call.

- [ ] **Step 3: Implement the read-only method.** Validate initialized/connected state, resolve the symbol through `ensure_symbol`, require aware datetimes, normalize both to UTC, reject `end < start`, select the module `COPY_TICKS_ALL` constant when present, call `copy_ticks_range(resolved, start_utc, end_utc, flags)`, map rows to `Mt5Tick` with `time_msc` preferred and `time` fallback, and raise typed `Mt5DataError` for `None` or malformed rows while returning `()` for a valid empty response. Do not add any mutation wrapper or direct evaluator dependency.

- [ ] **Step 4: Run the focused tests to verify GREEN.**

Run: `pytest tests/test_mt5_provider.py -q`

Expected: PASS, including all existing symbol/timeframe/safety tests.

- [ ] **Step 5: Commit the provider seam.**

```bash
git add tradingagents/dataflows/mt5/provider.py tests/test_mt5_provider.py
git commit -m "feat: add read-only MT5 historical ticks"
```

### Task 3: Define pure evaluation records, eligibility, and quote mathematics

**Files:**
- Create: `tests/test_forex_shadow_evaluation.py`
- Create: `tradingagents/forex/evaluation.py`

**Interfaces:**
- Consumes: `ShadowTradeDecision`, `Mt5Tick`, `Mt5SymbolInfo` metadata from `snapshot_json["symbol_metadata"]`, and injected UTC `now`.
- Produces: `EvaluationBasis = Literal["ANALYSIS_SNAPSHOT", "DECISION_REFERENCE"]`; `EvaluationStatus = Literal["PENDING", "COMPLETE", "DATA_UNAVAILABLE", "INELIGIBLE"]`; `EvaluationConfig`; immutable `ShadowOutcomeEvaluation`; `EligibilityResult`; `evaluate_directional_outcomes(...)`; and MFE/MAE helpers with no provider or LLM imports.

- [ ] **Step 1: Write deterministic failing tests for the pure layer.** Use fixed UTC anchors and synthetic ticks to assert: exact `(300, 900, 1800, 3600)` validation; target/deadline calculation per basis; pending before `target+tolerance`; first valid tick at/after target within tolerance; no interpolation/session jump; BUY `future_bid-entry_ask`; SELL `entry_bid-future_ask`; zero-spread preservation; point/digits are read from metadata; both directions are calculated for every terminal row; BUY/SELL selected PnL; HOLD selected PnL zero and `max(0,buy,sell)` opportunity cost when both directions lose; strict best-action/tie behavior; cost-aware MFE remains negative when spread is not overcome (no zero floor); and MFE/MAE excludes ticks after the exact target even when such a tick is the selected terminal observation. Assert source eligibility and nullable/deferred training fields are independent.

- [ ] **Step 2: Run the pure tests to verify RED.**

Run: `pytest tests/test_forex_shadow_evaluation.py -q`

Expected: FAIL because `tradingagents.forex.evaluation` and its public records/helpers do not yet exist.

- [ ] **Step 3: Implement the pure layer.** Define frozen/slotted records and validation for basis/status/config. Implement `decision_source_eligibility()` with the exact COMPLETE/NORMALIZED/BUY|SELL|HOLD/UTC contract. Implement quote validation (`finite`, `ask >= bid`, positive finite point, valid digits), target/deadline maturity, first-tick selection, bid/ask directional formulas, HOLD semantics, and signed MFE/MAE over `[anchor,target]` only. Keep training eligibility `None` and set only the reason `DEFERRED_TO_CORPUS_BUILDER` or `SOURCE_CONTEXT_INELIGIBLE`; do not infer missing metadata or normalize ambiguous actions.

- [ ] **Step 4: Run the pure tests to verify GREEN.**

Run: `pytest tests/test_forex_shadow_evaluation.py -q`

Expected: PASS with zero MT5 imports and zero LLM/provider calls.

- [ ] **Step 5: Commit the pure evaluation layer.**

```bash
git add tradingagents/forex/evaluation.py tests/test_forex_shadow_evaluation.py
git commit -m "feat: add deterministic forex outcome math"
```

### Task 4: Implement the idempotent SQLite evaluation store

**Files:**
- Modify: `tests/test_forex_shadow_evaluation.py`
- Modify: `tradingagents/forex/evaluation.py`

**Interfaces:**
- Consumes: `ShadowOutcomeEvaluation` records and the existing SQLite path used by `ShadowDecisionStore`.
- Produces: `ShadowEvaluationStore(path)`, `initialize()`, `get(decision_id, evaluation_basis, horizon_seconds)`, `list_for_decision(decision_id)`, `upsert_pending(records)`, and transactional status-aware persistence keyed by `(decision_id, evaluation_basis, horizon_seconds)`.

- [ ] **Step 1: Write failing store tests.** Assert table creation on a Phase 4 database, nullable `training_eligible`, the basis check constraint, triple-key uniqueness (both bases at the same horizon coexist; duplicate reruns do not duplicate), allowed transitions `PENDING -> COMPLETE`, `PENDING -> DATA_UNAVAILABLE`, and `DATA_UNAVAILABLE -> COMPLETE`, immutable COMPLETE/INELIGIBLE rows, recovery timestamps/provenance/previous-unavailable-reason retention, and migration of an older evaluation table missing optional columns. Assert that unavailable/ineligible rows may have null target/entry/future fields while complete rows retain all quote/calculation fields.

- [ ] **Step 2: Run the store tests to verify RED.**

Run: `pytest tests/test_forex_shadow_evaluation.py -q -k "store or migration or idempot"`

Expected: FAIL because `ShadowEvaluationStore` and the evaluation table do not exist.

- [ ] **Step 3: Implement the SQLite store.** Create `shadow_decision_evaluations` with the spec's checks, nullable fields, `created_at`, `evaluated_at`, `recovered_from_unavailable_at`, and `previous_unavailable_reason`, plus the triple primary key. Add idempotent column migration for an existing evaluation table; when the old table lacks `evaluation_basis`/the triple key, rebuild it transactionally and map legacy rows to `ANALYSIS_SNAPSHOT`, leaving new audit columns nullable for those rows. Serialize UTC timestamps with the repository's `Z` convention and nullable numeric/text values. Use one transaction for batches and a status-aware upsert that permits `PENDING -> COMPLETE`, `PENDING -> DATA_UNAVAILABLE`, and `DATA_UNAVAILABLE -> COMPLETE`, preserves the original `created_at`, records the recovery/evaluation timestamp, copies the old unavailable reason into audit metadata, and never overwrites COMPLETE or INELIGIBLE evidence. Keep legacy `shadow_decisions.future_evaluation_status` untouched.

- [ ] **Step 4: Run the store tests to verify GREEN.**

Run: `pytest tests/test_forex_shadow_evaluation.py -q -k "store or migration or idempot"`

Expected: PASS with both evaluation bases independently queryable and reruns idempotent.

- [ ] **Step 5: Commit the evaluation store.**

```bash
git add tradingagents/forex/evaluation.py tests/test_forex_shadow_evaluation.py
git commit -m "feat: persist basis-aware shadow evaluations"
```

### Task 5: Build the zero-LLM evaluator service and batch behavior

**Files:**
- Modify: `tests/test_forex_shadow_evaluation.py`
- Modify: `tradingagents/forex/evaluation.py`
- Modify: `tradingagents/forex/__init__.py`

**Interfaces:**
- Consumes: `ShadowDecisionStore.get/list_pending`, `ShadowEvaluationStore`, `MT5Provider.get_ticks_range`, `EvaluationConfig`, and pure helpers from Task 3.
- Produces: `ShadowOutcomeEvaluator.evaluate_decision(decision_id, *, now=None, terminal_path=None) -> ShadowDecisionEvaluationResult` and `.evaluate_pending(*, now=None, terminal_path=None) -> ShadowEvaluationBatchResult`; result fields include per-basis records/status, diagnostics, `decisions_scanned`, `horizons_evaluated`, `historical_ticks_processed`, `database_seconds`, `mt5_read_seconds`, `total_runtime_seconds`, and `llm_calls == 0`.

- [ ] **Step 1: Write failing service tests.** Inject fake stores/providers and fixed `now`. Assert no MT5 initialization when no horizon is mature; one provider lifecycle and at most one bounded tick-range call per mature decision when both bases are ready; no LLM calls; separate analysis/reference anchors; missing/invalid reference produces DATA_UNAVAILABLE only for the reference basis; provider failure leaves pending rows and existing terminal rows unchanged; ineligible decisions create all configured basis/horizon rows without provider access; partial maturity leaves future rows PENDING; empty/malformed historical data produces DATA_UNAVAILABLE after deadline; a later successful read recovers a prior DATA_UNAVAILABLE row to COMPLETE without replacing COMPLETE evidence; batches reuse one provider lifecycle while preserving per-decision bounds; metrics are populated.

- [ ] **Step 2: Run the service tests to verify RED.**

Run: `pytest tests/test_forex_shadow_evaluation.py -q -k "evaluator or batch or maturity or provider"`

Expected: FAIL because the evaluator service and result types are not implemented.

- [ ] **Step 3: Implement the service.** Load the decision, derive both basis anchors without fallback, create missing PENDING/INELIGIBLE rows for every configured combination, and skip provider initialization when no mature eligible horizon exists. For mature bases, compute the smallest shared UTC range (`min(anchor)` to `max(target+tolerance)`), call `get_ticks_range` once per decision where possible, sort/filter/validate ticks, select terminal observations independently, and use only anchor-to-exact-target ticks for excursions. Persist rows through the status-aware store, preserve COMPLETE/INELIGIBLE rows, leave unresolved horizons PENDING on retryable failures, and allow a later successful read to replace only a prior DATA_UNAVAILABLE row with COMPLETE while storing recovery timestamps/provenance and its prior reason. Reuse one provider lifecycle in `evaluate_pending`, collect timing/row/tick/error metrics, and hard-code `llm_calls = 0`.

- [ ] **Step 4: Run the service tests to verify GREEN.**

Run: `pytest tests/test_forex_shadow_evaluation.py -q`

Expected: PASS for single-decision, partial-maturity, unavailable-data, ineligible, retry, idempotency, basis-separation, and batch cases.

- [ ] **Step 5: Commit the service.**

```bash
git add tradingagents/forex/evaluation.py tradingagents/forex/__init__.py tests/test_forex_shadow_evaluation.py
git commit -m "feat: add deterministic shadow outcome evaluator"
```

### Task 6: Add the standalone `forex-evaluate` CLI and documentation

**Files:**
- Create: `cli/forex_evaluate.py`
- Create: `tests/test_forex_evaluate_cli.py`
- Modify: `pyproject.toml`
- Modify: `docs/forex-shadow.md`

**Interfaces:**
- Consumes: `ShadowOutcomeEvaluator`, `ShadowDecisionStore`, argparse, `--db-path`, `--terminal-path`, `--observation-tolerance-seconds`, and exactly one of `--decision-id`/`--pending`.
- Produces: installed `forex-evaluate` command with banner `MT5 FOREX — OUTCOME EVALUATION (READ ONLY)` and `NO ORDER WILL BE SENT`, separate basis statuses/rows, zero-LLM metrics, and non-zero exit for configuration/provider failure.

- [ ] **Step 1: Write failing CLI tests.** Invoke `main(argv)` with a temporary database and injected evaluator seam. Assert mutually exclusive decision/pending parsing, banner text, separate `ANALYSIS_SNAPSHOT`/`DECISION_REFERENCE` labels, maturity/status/metrics output, no credentials or execution options, and non-zero return on evaluator/provider failure. Assert `cli/main.py` remains byte-for-byte unchanged by the Phase 5 change.

- [ ] **Step 2: Run the CLI tests to verify RED.**

Run: `pytest tests/test_forex_evaluate_cli.py -q`

Expected: FAIL because the new module and entry point do not exist.

- [ ] **Step 3: Implement the CLI and entry point.** Parse the required mutually exclusive selector and optional database/terminal/tolerance flags, construct `EvaluationConfig` and `ShadowOutcomeEvaluator`, invoke the selected service method, print both safety lines before results, print each basis separately with per-horizon status and metrics including `llm_calls=0`, and return non-zero with a concise error for invalid configuration/provider failure. Add only `forex-evaluate = "cli.forex_evaluate:main"` to `[project.scripts]`; do not edit `cli/main.py` or add MT5 options to the stock CLI.

- [ ] **Step 4: Update documentation and run CLI tests.** Document the two temporal quote sets, analysis-vs-decision-reference semantics, staleness evidence, maturity/tolerance, exact formulas, HOLD opportunity cost, MFE/MAE window, training deferral, idempotency, read-only boundary, and example commands. Run `pytest tests/test_forex_evaluate_cli.py -q` and expect PASS.

- [ ] **Step 5: Commit the CLI/docs.**

```bash
git add cli/forex_evaluate.py tests/test_forex_evaluate_cli.py pyproject.toml docs/forex-shadow.md
git commit -m "feat: add standalone forex outcome CLI"
```

### Task 7: Add guarded real-MT5 integration and safety regression coverage

**Files:**
- Create: `tests/test_forex_shadow_evaluation_integration.py`
- Modify: `tests/test_forex_shadow_contract.py`
- Modify: `tests/test_forex_shadow_integration.py`
- Modify: `tests/test_mt5_provider.py`

**Interfaces:**
- Consumes: `RUN_MT5_INTEGRATION=1`, optional `MT5_TERMINAL_PATH`, `MT5_SHADOW_DB`, `MT5_SYMBOL`, `ShadowOutcomeEvaluator`, and the read-only provider methods.
- Produces: opt-in bounded historical-range proof with UTC ticks, exact symbol resolution, metrics, and unchanged positions/orders; safety scans covering all new Phase 5 modules.

- [ ] **Step 1: Write the guarded tests.** Add a test that skips unless `RUN_MT5_INTEGRATION=1`, initializes `MT5Provider`, captures positions/orders before, reads a bounded recent UTC historical range for resolved EURUSD, evaluates only a test decision when one is available, captures positions/orders after, and asserts equality plus `llm_calls == 0`. Add static scans asserting no `MetaTrader5` import in `evaluation.py`/CLI, no mutation method names, no `order_send`, and `executed` remains false in persisted decisions.

- [ ] **Step 2: Run the guarded tests without external access.**

Run: `pytest tests/test_forex_shadow_evaluation_integration.py -q`

Expected: SKIPPED unless `RUN_MT5_INTEGRATION=1`; no test may fabricate a provider result when the terminal is unavailable.

- [ ] **Step 3: Run the guarded test with the local demo terminal when configured.**

Run: `$env:RUN_MT5_INTEGRATION='1'; pytest tests/test_forex_shadow_evaluation_integration.py -q -m integration`

Expected: PASS with resolved symbol, UTC historical ticks, zero LLM calls, and identical positions/orders; otherwise a clear provider/configuration failure is reported and no decision is mislabeled.

- [ ] **Step 4: Commit integration/safety coverage.**

```bash
git add tests/test_forex_shadow_evaluation_integration.py tests/test_forex_shadow_contract.py tests/test_forex_shadow_integration.py tests/test_mt5_provider.py
git commit -m "test: guard read-only shadow evaluation"
```

### Task 8: Full verification and handoff

**Files:**
- Modify: any Phase 5 files only when a verification failure identifies a concrete defect.
- Create: `docs/superpowers/reports/2026-09-09-phase-5-shadow-outcome-evaluator-report.md`

**Interfaces:**
- Consumes: all Phase 5 tests, the accepted baseline SHA `3d9262c639ca1ede3ac830626361500b760d3a29`, and the final implementation tree.
- Produces: evidence-backed implementation report with final SHA, scope diff, test/lint/compile results, guarded MT5 result, safety checks, and explicit Phase 5 stop boundary.

- [ ] **Step 1: Run the complete automated verification.**

Run:

```bash
pytest -q
ruff check tradingagents/forex/evaluation.py tradingagents/forex/shadow.py tradingagents/forex/runner.py tradingagents/dataflows/mt5/provider.py cli/forex_evaluate.py tests/test_forex_shadow_evaluation.py tests/test_forex_shadow_contract.py tests/test_forex_shadow_runner.py tests/test_mt5_provider.py tests/test_forex_evaluate_cli.py
python -m compileall -q tradingagents cli
git diff --check 3d9262c639ca1ede3ac830626361500b760d3a29...HEAD
```

Expected: full suite and Ruff pass, compileall completes, and the scoped diff contains no stock CLI/agent-registration changes or execution APIs.

- [ ] **Step 2: Inspect the scoped diff and safety surface.** Verify only the file map changed, `forex-shadow` still persists `executed=False`, no direct `MetaTrader5` import appears in evaluator/CLI, no order/mutation method was added, legacy outcome columns remain untouched, and evaluation rows keep both bases plus nullable/deferred training fields.

- [ ] **Step 3: Write the implementation report.** Record exact commands/results, baseline/final SHAs, database path used by tests, per-basis statuses, horizon/tolerance behavior, quote/math evidence, zero LLM calls, optional MT5 positions/orders before/after, and limitations including stale analyses not being executable-quality evidence. State explicitly that Phase 5 evaluation is complete and Phase 6+ work was not started.

- [ ] **Step 4: Commit the report and final changes.**

```bash
git add docs/superpowers/reports/2026-09-09-phase-5-shadow-outcome-evaluator-report.md
git commit -m "docs: report phase 5 shadow evaluation"
```

## Plan Self-Review

- **Spec coverage:** Tasks 1–2 cover temporal decision/reference capture from the broker tick timestamp and the read-only provider seam; Tasks 3–5 cover bases, eligibility, maturity, tolerance, formulas, HOLD, negative cost-aware MFE, recoverable status transitions, storage, idempotency, batch reuse, metrics, and zero LLM calls; Task 6 covers the separate CLI and docs; Task 7 covers guarded real-terminal/safety proof; Task 8 covers the required final evidence and baseline-scoped diff.
- **Placeholder scan:** No step relies on “TBD”, “TODO”, “fill in”, “implement later”, or an unspecified “handle edge cases” instruction; each implementation step names the files, interfaces, validation, and command that proves it.
- **Type consistency:** The evaluator service consumes the `ShadowOutcomeEvaluation` and `EvaluationConfig` types created in Task 3, persists them through the `ShadowEvaluationStore` created in Task 4, and exposes the exact result methods consumed by the Task 6 CLI. Both public basis/status literals match the spec and the SQLite checks.
