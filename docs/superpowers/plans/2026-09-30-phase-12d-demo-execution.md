# Phase 12D-Demo Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strictly DEMO-account-only MT5 execution boundary for Phase 12 while preserving SHADOW/TEST_ONLY behavior and all existing phase boundaries.

**Architecture:** Extend the normalized MT5 account seam with authoritative trade-mode metadata. Add typed DEMO order intents/results, a separate transactional DEMO ledger, and a `VerifiedDemoExecutionGateway` containing the only production `order_send` call. Integrate it into the deterministic HFT worker only behind an explicit `--demo-execute` opt-in and preserve the existing serialized MT5 gate, plan validation, risk gates, lease ownership, and recovery state.

**Tech Stack:** Python dataclasses/enums, existing MT5 provider, SQLite, argparse, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-30-phase-12d-demo-execution-design.md`

## Global Constraints

- Account mode must equal the authoritative MetaTrader DEMO trade-mode constant before request construction and immediately before `order_send`.
- No REAL/LIVE mode or generic `--execute`, `--live`, or `--real` flag exists.
- `order_send` may appear only inside `VerifiedDemoExecutionGateway`.
- SHADOW and TEST_ONLY remain broker-mutation-free and preserve `executed=False`.
- Initial DEMO scope is EURUSD with at most one owned open position, default risk 0.25%, hard risk ceiling 0.5%.
- Existing freshness, provenance, plan, deterministic HFT, risk, lease, and MT5 serialization gates remain mandatory.
- Phase 5/6, Phase 7, Phase 11A, and the stock CLI are not modified or mixed with DEMO records.
- No credentials, prompts, completions, or private model reasoning are persisted.

## Review Focus

- A REAL/CONTEST/unknown account or a changed account mode immediately before send must produce `DEMO_EXECUTION_FORBIDDEN` with zero broker calls; covered in `tests/test_phase12d_gateway.py`.
- A stale, invalid, NONE/HOLD, mismatched, or expired plan must not create an order intent; covered in `tests/test_phase12d_runtime.py`.
- Broker rejection, partial fill, invalid stops, and connection failure must be persisted without fabricating a fill or creating an unowned position; covered in `tests/test_phase12d_gateway.py` and `tests/test_phase12d_ledger.py`.
- Restart/disconnect reconciliation must block new entries when broker and ledger ownership disagree; covered in `tests/test_phase12d_reconciliation.py`.
- Existing SHADOW/TEST_ONLY runtime and stock/non-Ollama behavior must remain unchanged; covered by existing Phase 12 tests plus `tests/test_phase12d_isolation.py`.

---

### Task 1: Authoritative DEMO account metadata

**Files:**
- Modify: `tradingagents/dataflows/mt5/models.py`
- Modify: `tradingagents/dataflows/mt5/provider.py`
- Modify: `tradingagents/forex/context.py` only to preserve serialization compatibility
- Test: `tests/test_mt5_models.py`, `tests/test_mt5_provider.py`, `tests/test_phase12d_gateway.py`

**Interfaces:**
- `Mt5AccountInfo.trade_mode: int | None`
- `MT5Provider.get_account_info()` maps the raw `trade_mode` field without assuming a server-name convention.
- `DemoAccountSnapshot`/account verification consumes the normalized field.

- [ ] Write RED tests proving the normalized account retains `trade_mode`, missing mode is distinguishable from DEMO, and existing account fixtures remain valid.
- [ ] Run the focused tests and confirm the missing field/metadata failure.
- [ ] Add the optional validated field and provider mapping; keep old callers compatible when fixtures omit it.
- [ ] Run the focused MT5 tests to GREEN.
- [ ] Commit `feat: expose authoritative mt5 account trade mode`.

### Task 2: Typed DEMO execution contracts and mode isolation

**Files:**
- Create: `tradingagents/forex/hft/demo_models.py`
- Modify: `tradingagents/forex/hft/models.py` only if a shared `ExecutionMode` enum is required
- Test: `tests/test_phase12d_models.py`, `tests/test_phase12d_isolation.py`

**Interfaces:**
- `ExecutionMode` values exactly `SHADOW`, `TEST_ONLY`, `DEMO`.
- `DemoAccountSnapshot`, `DemoOrderIntent`, `DemoOrderResult`, `DemoPositionRecord`, and `DemoHealth` with bounded UTC, numeric, symbol, direction, ticket, and `real_money=False` validation.
- `DemoOrderResult.classification` values include `FILLED`, `PARTIAL`, `REJECTED`, `REQUOTE`, `PRICE_CHANGED`, `MARKET_CLOSED`, `NO_MONEY`, `INVALID_STOPS`, `CONNECTION_ERROR`, `UNKNOWN`.

- [ ] Write RED contract tests for valid LONG/SHORT records, invalid directions/volumes/timestamps, forbidden REAL mode, and explicit shadow/test semantics.
- [ ] Verify RED.
- [ ] Implement frozen dataclasses/enums with no broker calls.
- [ ] Verify GREEN and assert `real_money` is always false.
- [ ] Commit `feat: add phase12 demo execution contracts`.

### Task 3: Separate DEMO ledger and circuit state

**Files:**
- Create: `tradingagents/forex/hft/demo_store.py`
- Test: `tests/test_phase12d_ledger.py`

**Interfaces:**
- `DemoExecutionStore(path).initialize()` and `integrity_check()`.
- `record_run`, `record_account_snapshot`, `record_order_intent`, `record_order_result`, `record_position`, `record_exit`, `record_reconciliation`, `read_owned_positions`, `read_open_positions`, `read_circuit_state`, and `set_circuit_state`.
- Default path is `data_cache/live-market-clean-20260923.demo.sqlite3`; all writes carry `execution_mode=DEMO` and `real_money=0`.

- [ ] Write RED tests for schema creation, transaction/idempotency keys, mode separation, account snapshot fields, result persistence, and DB integrity.
- [ ] Verify RED.
- [ ] Implement versioned SQLite tables with transactional writes and no writes to the strategic/HFT databases.
- [ ] Verify GREEN.
- [ ] Commit `feat: add phase12 demo execution ledger`.

### Task 4: Account hard-lock and broker request boundary

**Files:**
- Create: `tradingagents/forex/hft/demo_gateway.py`
- Modify: `tradingagents/forex/hft/__init__.py`
- Test: `tests/test_phase12d_gateway.py`

**Interfaces:**
- `DemoExecutionForbidden` typed exception with code `DEMO_EXECUTION_FORBIDDEN`.
- `VerifiedDemoExecutionGateway(provider, store, gate, *, demo_trade_mode, magic, comment, volume_cap, clock)`.
- `verify_demo_account() -> DemoAccountSnapshot`.
- `submit(intent: DemoOrderIntent) -> DemoOrderResult`.
- `close(position, intent metadata) -> DemoOrderResult`.

- [ ] Write RED tests for DEMO success, REAL/CONTEST/unknown/missing account refusal, server-name spoofing refusal, and account-mode change between preflight and send.
- [ ] Add a source-level RED test showing the gateway is the only module allowed to reference `order_send`.
- [ ] Verify RED.
- [ ] Implement the final hard lock, request construction, one `order_send` call, result-code classification, and safe scalar diagnostics.
- [ ] Verify GREEN; malformed/missing broker results fail closed without fabricated fills.
- [ ] Commit `feat: add verified demo execution gateway`.

### Task 5: Volume, stops, ownership, and entry/exit validation

**Files:**
- Modify: `tradingagents/forex/hft/demo_gateway.py`
- Modify: `tradingagents/forex/hft/risk.py` only for explicit bounded DEMO risk inputs
- Test: `tests/test_phase12d_gateway.py`, `tests/test_phase12d_risk.py`

**Interfaces:**
- `normalize_volume(raw, minimum, maximum, step, cap) -> float`.
- `validate_demo_entry(plan, tick, symbol_info, account, risk, owned_positions) -> DemoOrderIntent | None`.
- `validate_demo_exit(position, tick, reason) -> DemoOrderIntent`.

- [ ] Write RED tests for min/max/step/cap normalization, invalid stop distances, fresh tick, one-position limit, duplicate/conflicting request, risk rejection, daily -2% circuit, and three-loss cooldown.
- [ ] Verify RED.
- [ ] Implement conservative bounded validation and deterministic circuit state; never fall back to an unprotected order.
- [ ] Verify GREEN.
- [ ] Commit `feat: gate phase12 demo entries and exits`.

### Task 6: DEMO-aware deterministic runtime

**Files:**
- Modify: `tradingagents/forex/hft/runtime.py`
- Modify: `tradingagents/forex/hft/supervisor.py`
- Test: `tests/test_phase12d_runtime.py`, `tests/test_phase12d_reconciliation.py`

**Interfaces:**
- `HftShadowConfig.execution_mode` remains default `SHADOW`; DEMO requires an explicit gateway and `execution_mode=DEMO`.
- `HftShadowRuntime` continues using shadow fills for SHADOW/TEST_ONLY and delegates only validated entry/exit actions to the DEMO gateway.
- `DemoExecutionWorker`/worker hooks recover owned broker positions before accepting entries.

- [ ] Write RED tests for NONE/HOLD no-action, natural LONG/SHORT entry, stale/expired plan refusal, valid exit, one-position limit, no LLM calls, and mode-specific `executed`/`broker_order_sent` semantics.
- [ ] Write RED restart tests for matching owned position, unknown broker position, ledger mismatch, and disconnect with an open position.
- [ ] Verify RED.
- [ ] Implement the smallest runtime seam; preserve existing SHADOW/TEST_ONLY code paths byte-for-byte where practical.
- [ ] Verify GREEN and confirm no duplicate order after restart.
- [ ] Commit `feat: integrate phase12 demo runtime safely`.

### Task 7: Supervisor/CLI opt-in and truthful DEMO health

**Files:**
- Modify: `tradingagents/forex/supervisor.py`
- Modify: `tradingagents/forex/hft/dashboard.py`
- Modify: `cli/forex_supervisor.py` or add `cli/forex_hft_demo.py` without changing the stock CLI
- Test: `tests/test_phase12d_cli.py`, `tests/test_phase12d_health.py`

**Interfaces:**
- Explicit `--demo-execute` opt-in only; no generic live/real/execute flag.
- `DEMO_EXECUTION_HEALTH`: `DISABLED`, `READY`, `POSITION_OPEN`, `DEGRADED`, `RECONCILIATION_REQUIRED`, `OPERATOR_REVIEW_REQUIRED`.
- Startup must verify the current account before constructing the gateway; failed verification never initializes an order path.

- [ ] Write RED tests for active-lease refusal, absent opt-in, account verification before gateway construction, DEMO banner, health states, and stock CLI unchanged.
- [ ] Verify RED.
- [ ] Implement the opt-in handoff and scalar health persistence without interrupting active strategic analysis.
- [ ] Verify GREEN.
- [ ] Commit `feat: add phase12 demo supervisor opt-in`.

### Task 8: Static order-boundary audit and acceptance fixtures

**Files:**
- Test: `tests/test_phase12d_boundary.py`, `tests/test_phase12d_acceptance.py`
- Modify: documentation only if required

- [ ] Add a static audit proving the only production `order_send` reference is the gateway and no REAL mode exists.
- [ ] Add a deterministic full lifecycle fixture: natural validated LONG/SHORT -> DEMO fill -> owned exit -> realized P&L, plus rejection/partial paths.
- [ ] Verify all Phase 12 tests and the new Phase 12D tests.
- [ ] Run Ruff, compileall, and `git diff --check`.
- [ ] Run the full suite; record skips exactly.
- [ ] Verify Phase 7/11A fingerprints and existing strategic/HFT DB integrity.
- [ ] Commit `test: verify phase12 demo safety boundaries`.

### Task 9: Current-account gate and safe runtime handoff

**Files:** tests/reports only unless a proven defect requires code.

- [ ] Stop no process while strategic analysis is active; wait for a safe lifecycle point.
- [ ] Verify current account metadata through the provider: login, server, company, currency, authoritative trade mode.
- [ ] If and only if trade mode is DEMO, start exactly one supervisor with explicit `--demo-execute` and a dedicated DEMO DB.
- [ ] If account verification is not DEMO, leave execution disabled and report `DEMO_EXECUTION_FORBIDDEN`; do not weaken the gate.
- [ ] Monitor only natural plans; never manufacture direction or send a test order against the live terminal.
- [ ] Record process/lease/health/account/position/order/DB evidence and stop engineering unless a real defect appears.
