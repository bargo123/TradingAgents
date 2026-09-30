# Phase 12D-Demo Execution Design

## Status

Design for review. This document does not authorize a broker order and does
not add an execution path by itself.

## Goal

Add a separately gated, DEMO-account-only execution boundary for the existing
Phase 12 strategic-plan and deterministic HFT pipeline. A legitimate,
freshly validated LONG or SHORT plan may eventually produce a real MT5 demo
order, while a real, contest, unknown, unavailable, or unverifiable account
can never reach an order API.

## Non-goals and permanent boundaries

- There is no REAL/LIVE execution mode and no generic `--execute`, `--live`, or
  `--real` switch.
- SHADOW and TEST_ONLY behavior remains unchanged and continues to persist
  `executed=False`.
- Phase 7 knowledge, Phase 11A distillation, Phase 5/6 evaluation, and the
  stock TradingAgents CLI are not modified or mixed with DEMO execution.
- No direction is manufactured. `NONE`/HOLD remains no-action.
- No risk increase, forced frequency, martingale, grid, averaging down,
  hedging, or 10% benchmark optimization is allowed.
- DEMO execution does not make a trade real-money execution; every execution
  record carries `execution_mode=DEMO` and `real_money=False`.

## Architectural options

### A. Add order calls to the existing HFT runtime

This is the smallest diff, but it couples shadow fills, broker requests,
reconciliation, and demo safety in one module. It makes it too easy for a
future caller to bypass the account gate and it would blur the existing
SHADOW/TEST_ONLY contracts.

### B. Add a dedicated `VerifiedDemoExecutionGateway` and DEMO ledger

The deterministic HFT runtime remains the decision/risk owner. A typed
`DemoOrderIntent` crosses one explicit gateway only after account, plan,
tick, position, risk, volume, and stop checks. The gateway owns the final
account re-check and the sole `order_send` call. A separate SQLite ledger
records requests, broker results, positions, account snapshots, and
reconciliation state. This adds a small seam but gives the required hard
lock, audit trail, and crash/disconnect semantics.

### C. Run a separate DEMO worker process

This isolates broker execution and would simplify process-level ownership,
but it introduces a second MT5 lifecycle, cross-process plan transport, and
more failure modes before the first controlled demo order.

### Recommendation

Use **B**. Reuse the existing supervisor, serialized MT5 gate, validated
`StrategicExecutionPlan`, deterministic `FastExecutionEngine`, and
`RiskEngine`, but keep broker mutation behind one gateway and one new DEMO
ledger. This preserves the proven shadow path and makes the safety boundary
auditable.

## Runtime data flow

```text
fresh strategic decision
        -> validated StrategicExecutionPlan (LONG/SHORT only)
        -> fresh MT5 tick + deterministic HFT engine
        -> deterministic risk decision and bounded sizing
        -> VerifiedDemoExecutionGateway
        -> DEMO MT5 order_send
        -> broker result validation
        -> owned position reconciliation/management
        -> DEMO ledger and shadow-vs-demo evidence
```

The gateway is constructed only after the CLI has explicitly selected
`--demo-execute`, the current provider has been initialized, and authoritative
account metadata has been verified. The lowest gateway boundary repeats the
account check immediately before `order_send`.

## Account hard lock

The provider's authoritative `account_info()` result is required. The
gateway accepts only the MetaTrader trade-mode value corresponding to
`ACCOUNT_TRADE_MODE_DEMO`; server/company strings are secondary evidence and
never sufficient. It records login, server, company, trade mode, and currency
without credentials.

Missing account info, unknown/unavailable metadata, REAL, CONTEST, or any
trade-mode mismatch raises a typed `DEMO_EXECUTION_FORBIDDEN` failure before
request construction and before `order_send`. A flag or environment variable
cannot override this decision.

## Execution modes and typed contracts

Introduce an explicit execution-mode enum with `SHADOW`, `TEST_ONLY`, and
`DEMO`; do not add `REAL`. The existing shadow/test contracts remain
unchanged. New typed records include:

- `DemoOrderIntent`: intent ID, plan/source IDs, strategy, symbol, LONG/SHORT,
  normalized volume, requested quote, SL/TP, deviation, timestamps, commit,
  execution mode, account trade mode, and `real_money=False`.
- `DemoOrderResult`: request/retcode, order/deal ticket, fill price/volume,
  broker comment/timestamp, and a closed result classification.
- owned-position and exit-intent records with ticket, magic, comment, and
  ledger ownership.

Every record includes execution mode. A DEMO broker request sets
`broker_order_sent=True` only after a request was actually submitted;
`executed` is not overloaded to mean both shadow and broker execution.

## Entry gates

Only a fresh, provenance-valid, unexpired plan with `primary_direction` LONG
or SHORT may enter. The existing 900-second strategic freshness contract,
valid reference state, symbol check, current tick, spread/micro confirmation,
risk approval, one-open-position limit, and no-conflicting-order checks remain
mandatory. The initial EURUSD position limit is one; no pyramiding, hedging,
or averaging down is allowed.

Initial risk is bounded to a configurable default of 0.25% simulated/account
equity, with a hard ceiling of 0.5%. Volume must obey broker min/max/step and
the configured demo cap. SL and TP distances must validate against symbol
metadata. Invalid stops fail the entry; there is no unprotected fallback.

## Broker boundary and result handling

`order_send` may occur only inside `VerifiedDemoExecutionGateway`. A source
audit must prove that no other production path calls it. The request uses a
dedicated Phase 12D magic number and `TradingAgents-P12D-DEMO` comment.

Broker results are classified from actual retcodes as FILLED, PARTIAL,
REJECTED, REQUOTE, PRICE_CHANGED, MARKET_CLOSED, NO_MONEY, INVALID_STOPS,
CONNECTION_ERROR, or UNKNOWN. Missing tickets/fills are never fabricated.

Exit requests use the same gateway, verify the broker-owned position first,
and persist exit intent, request/result, broker fill, realized demo P&L, and
reason. Manual or other-EA positions are never modified or closed.

## DEMO ledger

Use a separate database, by default
`data_cache/live-market-clean-20260923.demo.sqlite3`. It is not mixed with
SHADOW/TEST_ONLY rows. Versioned tables record:

- execution runs and mode;
- account snapshots (balance, equity, margin, free margin, margin level);
- order intents and broker requests/results;
- deals/fills and owned positions;
- exit intents/results;
- risk/circuit state and reconciliation events;
- shadow-vs-demo comparison references.

Writes are transactional and idempotent by intent/ticket identity. Existing
strategic and HFT databases are read-only inputs for this path and are never
rewritten.

## Recovery and health

On startup and after reconnect, query broker positions and match only the
dedicated magic/comment and ledger-owned tickets. A broker/local mismatch
sets `RECONCILIATION_REQUIRED` and blocks new entries. A disconnect with an
owned position sets `BROKER_STATE_UNKNOWN`, suspends new entries, reconnects
under the serialized MT5 gate, and reconciles before resuming. It never
assumes a position disappeared.

Expose `DEMO_EXECUTION_HEALTH` states: `DISABLED`, `READY`, `POSITION_OPEN`,
`DEGRADED`, `RECONCILIATION_REQUIRED`, and `OPERATOR_REVIEW_REQUIRED`.
Unexpected positions, repeated rejects, database failures, risk failures,
provenance mismatches, and unresolved disconnects stop new entries. Existing
owned positions remain subject to safe reconciliation/management.

Add a daily-loss circuit at -2% of demo account equity and a three-consecutive-
losing-trade cooldown. These circuits pause new entries; they do not
liquidate blindly or alter strategy parameters. The 10% daily-compounding
number remains a descriptive benchmark only.

## CLI and supervisor handoff

Add an explicit DEMO opt-in command/flag such as `--demo-execute` to the
Phase 12 supervisor/HFT path. Do not alter the stock CLI or expose a generic
execution switch. Startup prints a prominent DEMO banner and the account
verification result. If verification fails, the process remains non-executing
and reports `DEMO_EXECUTION_FORBIDDEN`.

The handoff must stop the old runtime only at a safe lifecycle point, reconcile
strategic and HFT leases, start exactly one supervisor/watcher/HFT worker and
one execution gateway, and preserve active strategic analysis. A direction is
not requested or fabricated to make the first DEMO order happen.

## Tests and acceptance

Tests must cover account-mode hard locks, final revalidation, valid LONG/SHORT
requests, stale/invalid plans, risk and stop rejection, volume normalization,
broker rejection/partial/fill persistence, ownership filtering, crash
recovery, disconnect-with-position, duplicate prevention, daily/consecutive
loss circuits, and explicit execution-mode semantics. Static source tests
must prove there is no REAL mode and that `order_send` exists only in the
gateway.

Before any live DEMO handoff: focused tests, all Phase 12 tests, the full
suite, Ruff, compileall, diff check, DB integrity, and a current authoritative
account verification must pass. A first DEMO order is allowed only when a
natural fresh LONG/SHORT plan and all deterministic gates pass. If the current
account is not authoritatively DEMO, no order is sent and the phase remains
blocked without weakening the gate.

Phase 7 and Phase 11A fingerprints must be unchanged before and after the
implementation.
