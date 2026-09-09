# MT5 Broker Clock Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a calibrated, fail-closed broker-clock boundary so read-only MT5 ticks, bars, snapshots, spreads, and historical ranges exposed to TradingAgents are true UTC instants without using the computer's timezone or a broker-specific hard-coded offset.

**Architecture:** Keep literal Unix-epoch decoding in one pure helper, then apply an immutable `Mt5BrokerClock` calibration (derived from fresh live tick samples) at the provider boundary. The provider will calibrate lazily per session before exposing timestamped data, translate true-UTC historical requests into the terminal's calibrated clock, and persist safe calibration provenance inside the existing snapshot JSON. Phase 5 and Phase 6 continue to consume true UTC and retain their strict temporal checks.

**Tech Stack:** Python 3.10+, dataclasses, `datetime`/`timezone`, MetaTrader5 Python API, pytest, SQLite-backed existing shadow stores.

**Spec:** User-approved broker-clock normalization requirements in the Phase 6 blocker request (2026-09-09), including the documented historical-data limitation.

## Global Constraints

- MT5 remains standalone, read-only, and mutation APIs remain absent.
- Do not hard-code Jordan, UTC+3, MetaQuotes-Demo, `-3 hours`, or any deployment-local timezone.
- Calibration uses fresh live broker ticks, a bounded 15-minute candidate grid over -14:00 through +14:00, multiple samples, freshness/residual validation, and fail-closed statuses.
- `decision_completed_timestamp >= analysis_snapshot_timestamp` remains strict; Phase 5 maturity and excursion math are unchanged.
- Historical requests translate true UTC to broker-clock time and normalize returned rows back to true UTC.
- No credentials are persisted; clock provenance is safe metadata only.
- Do not start Phase 7 or run a new Qwen analysis until zero-LLM calibration verification proves true-UTC behavior.

---

### Task 1: Add immutable broker-clock contracts and pure calibration

**Files:**
- Create: `tradingagents/dataflows/mt5/clock.py`
- Modify: `tradingagents/dataflows/mt5/errors.py`
- Modify: `tradingagents/dataflows/mt5/__init__.py`
- Test: `tests/test_mt5_broker_clock.py`

**Interfaces:**
- `Mt5BrokerClock` is an immutable dataclass with `offset_seconds`, `status`, `calibrated_at_utc`, `server`, `symbol`, `sample_count`, `max_residual_seconds`, and `source`.
- `BrokerClockConfig` owns the explicit candidate grid, sample count, freshness, future-skew, and calibration-age limits.
- `BrokerClockSample` carries aware UTC observation-before/after times and a literal raw-epoch UTC decode.
- `decode_mt5_epoch(value, milliseconds=False)` performs only literal `datetime.fromtimestamp(..., timezone.utc)` conversion.
- `calibrate_broker_clock(samples, metadata, config)` returns `CALIBRATED`, `UNAVAILABLE`, `AMBIGUOUS`, or `STALE_MARKET` without consulting local timezone state.
- `Mt5BrokerClock.normalize_epoch()` subtracts the calibrated offset; `to_broker_datetime()` adds it for outgoing historical requests; `ensure_fresh()` rejects stale/invalid calibration.
- Export `Mt5BrokerClockError`, the model, config, sample, decoder, and calibrator from the MT5 package.

- [ ] **Step 1: Write deterministic red tests**

  Add tests proving exact second and millisecond decoding, UTC+3/UTC+2/UTC+5:45 calibration, local-timezone independence by monkeypatching the module clock rather than relying on the host timezone, stale/ambiguous/unavailable fail-closed statuses, zero-offset behavior, and inverse request translation. Assert production source contains no Jordan/MetaQuotes/`-3` offset constant.

- [ ] **Step 2: Run the broker-clock tests and verify the expected failures**

  Run `pytest tests/test_mt5_broker_clock.py -q`. The new imports/methods must fail for missing production contracts, not because of a test typo.

- [ ] **Step 3: Implement the pure decoder, model, candidate search, and errors**

  Use a 900-second grid from -50,400 through +50,400 seconds. Accept a candidate only when every sample's normalized timestamp is no more than the configured future skew ahead of the observation window and no older than the configured market-age limit. Reject no-candidate samples as `STALE_MARKET` when all candidates are old, as `AMBIGUOUS` when samples cannot agree, and as `UNAVAILABLE` when no usable sample exists. Never call `round(delta / 3600)` or read the local timezone.

- [ ] **Step 4: Run the broker-clock tests green**

  Run `pytest tests/test_mt5_broker_clock.py -q` and confirm all new cases pass.

- [ ] **Step 5: Commit the standalone clock contracts**

  Commit with `git add tradingagents/dataflows/mt5/clock.py tradingagents/dataflows/mt5/errors.py tradingagents/dataflows/mt5/__init__.py tests/test_mt5_broker_clock.py && git commit -m "feat: add calibrated MT5 broker clock"`.

### Task 2: Integrate calibration and normalization into MT5Provider

**Files:**
- Modify: `tradingagents/dataflows/mt5/provider.py`
- Modify: `tradingagents/dataflows/mt5/models.py`
- Modify: `tests/test_mt5_provider.py`
- Modify: `tests/test_mt5_models.py`

**Interfaces:**
- Extend `MT5Provider.__init__` with optional `broker_clock`, `clock_config`, and injectable UTC clock arguments while preserving existing call sites.
- Add `MT5Provider.broker_clock` and `MT5Provider.calibrate_broker_clock(symbol)` read-only accessors.
- Keep `ensure_symbol()` as the public symbol resolver, but make it select the symbol and ensure a fresh calibrated clock before timestamped data is returned.
- Every timestamped provider method (`get_tick`, `get_spread`, bars, positions, orders, snapshots, and historical ticks) uses the same calibrated helper exactly once.

- [ ] **Step 1: Write failing provider integration tests**

  Add fake-API tests for lazy multi-sample calibration, exact-once tick/bar/spread/snapshot normalization, stale calibration rejection, no-calibration rejection before timestamp exposure, and snapshot provenance serialization. Update fake-provider fixtures to inject an explicit zero-offset test clock where old synthetic epochs are intentionally used.

- [ ] **Step 2: Run the provider tests red**

  Run `pytest tests/test_mt5_provider.py tests/test_mt5_models.py -q` and verify the new calibration/provenance assertions fail before provider integration exists.

- [ ] **Step 3: Implement provider calibration and centralized normalization**

  Capture application UTC immediately before and after each raw `symbol_info_tick()` sample, decode `time_msc` first and `time` second, call the pure calibrator, and raise `Mt5BrokerClockError` for non-calibrated status. Preserve the calibrated model on the provider. Keep symbol resolution and selection separate from calibration to avoid recursion.

- [ ] **Step 4: Normalize all incoming timestamped data**

  Route tick, spread, bar, position, order, snapshot, and `copy_ticks_range` row timestamps through `Mt5BrokerClock.normalize_epoch()`. Leave account/terminal metadata unchanged. Add the optional clock to `ForexMarketSnapshot` after existing fields so positional constructors remain compatible.

- [ ] **Step 5: Run provider/model tests green**

  Run `pytest tests/test_mt5_provider.py tests/test_mt5_models.py -q` and confirm existing and new cases pass.

- [ ] **Step 6: Commit provider integration**

  Commit with `git add tradingagents/dataflows/mt5/provider.py tradingagents/dataflows/mt5/models.py tests/test_mt5_provider.py tests/test_mt5_models.py && git commit -m "feat: normalize MT5 data with broker clock"`.

### Task 3: Translate historical requests and persist safe clock provenance

**Files:**
- Modify: `tradingagents/dataflows/mt5/provider.py`
- Modify: `tradingagents/forex/context.py`
- Modify: `tests/test_forex_shadow_context_tools.py`
- Modify: `tests/test_forex_shadow_evaluation.py`
- Modify: `tests/test_forex_shadow_runner.py`

**Interfaces:**
- `get_ticks_range(symbol, start, end)` accepts true-UTC aware datetimes, adds the calibrated broker offset before calling `copy_ticks_range`, and subtracts it from returned rows.
- `snapshot_to_dict()` includes a JSON-safe `broker_clock` object containing offset, status, calibration timestamp, server, symbol, sample count, residual, and source whenever a snapshot has one.
- Existing `ShadowTradeDecision.snapshot_json` persists this metadata without schema-breaking columns or credentials.

- [ ] **Step 1: Write failing historical/provenance tests**

  Assert a +3-hour fake clock sends `start + 3h`/`end + 3h` to the API while returned ticks normalize back to true UTC, zero-offset requests remain unchanged, and persisted snapshot JSON contains all safe clock fields. Assert evaluator requests remain true-UTC at its public boundary.

- [ ] **Step 2: Run the tests red**

  Run `pytest tests/test_forex_shadow_context_tools.py tests/test_forex_shadow_evaluation.py tests/test_forex_shadow_runner.py -q` and verify the new range/provenance assertions fail before the integration changes.

- [ ] **Step 3: Implement inverse range translation and JSON provenance**

  Translate only at the MT5 provider boundary; do not modify Phase 5 evaluator timestamps or maturity logic. Serialize the immutable clock with UTC ISO formatting and finite numeric values.

- [ ] **Step 4: Run the tests green**

  Re-run the same command and confirm all tests pass.

- [ ] **Step 5: Commit historical/provenance integration**

  Commit with `git add tradingagents/dataflows/mt5/provider.py tradingagents/forex/context.py tests/test_forex_shadow_context_tools.py tests/test_forex_shadow_evaluation.py tests/test_forex_shadow_runner.py && git commit -m "feat: translate MT5 history and persist clock provenance"`.

### Task 4: Document calibration and historical-data boundaries

**Files:**
- Modify: `docs/forex-shadow.md`
- Test: `tests/test_forex_watch_cli.py` or a new documentation assertion in `tests/test_mt5_broker_clock.py`

- [ ] **Step 1: Add documentation assertions**

  Require documentation to state that broker-clock calibration is live, fail-closed, independent of the computer timezone, and that current short horizons use the calibrated session offset while months/years of historical HFT data require separate DST/timezone-regime handling.

- [ ] **Step 2: Update the documentation**

  Include the safe provenance fields and the fact that no execution path is added.

- [ ] **Step 3: Run documentation tests**

  Run the targeted documentation test and confirm it passes.

- [ ] **Step 4: Commit documentation**

  Commit with `git add docs/forex-shadow.md tests/test_forex_watch_cli.py tests/test_mt5_broker_clock.py && git commit -m "docs: describe MT5 broker clock limits"`.

### Task 5: Run zero-LLM real calibration and bounded history verification

**Files:**
- No production files; use an inline read-only Python diagnostic.
- Evidence: terminal output captured in the final report.

- [ ] **Step 1: Run calibration with no TradingAgents/LLM construction**

  Capture application UTC before/after raw samples, raw `time`/`time_msc`, inferred offset/status/residual/age, normalized tick, raw and normalized M1/M5/M15 bars, and positions/orders.

- [ ] **Step 2: Verify outgoing range translation experimentally**

  Request a bounded true-UTC range through `MT5Provider.get_ticks_range()`, report the actual broker-clock range observed by an injected or instrumented API seam, and show normalized returned timestamps.

- [ ] **Step 3: Require the fail-closed acceptance gate**

  Proceed only if the normalized live tick is close to application UTC, bars are plausible, residuals are within policy, positions/orders are unchanged, and LLM calls are zero. If calibration remains unavailable/ambiguous/stale, stop without launching Qwen.

### Task 6: Regression verification and exactly one collector attempt

**Files:**
- No further production changes unless a deterministic regression exposes a defect in Tasks 1–5.

- [ ] **Step 1: Run focused and full verification**

  Run MT5 provider/model tests, Phase 4/5 temporal tests, Phase 6 watcher tests, full pytest, Ruff, compileall, `git diff --check`, and the guarded real MT5 read-only integration. Confirm stock CLI files and agent registration remain unchanged.

- [ ] **Step 2: Run exactly one `forex-watch once` only after the acceptance gate**

  Use EURUSD, INTRADAY, `market,news`, and configured Qwen 2B/4B. Capture decision ID, source run ID, true-UTC snapshot/completion/reference timestamps, latency/delay, clock provenance, context/normalization/action, `executed=False`, Phase 5 statuses/zero evaluator LLM calls, and positions/orders before/after.

- [ ] **Step 3: Stop on any new defect**

  Add a deterministic red regression and smallest fix before any retry; never repeat a long Qwen run solely to obtain a complete example. Do not start Phase 7.

