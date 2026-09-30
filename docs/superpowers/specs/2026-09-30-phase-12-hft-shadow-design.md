# Phase 12 HFT-Style Forex Shadow Design

## Status

Approved implementation scope supplied on 2026-09-30. Phase 12D is explicitly out of scope.

## Goals

- Separate slow strategic reasoning from a deterministic, tick-driven execution path.
- Provide one implementation of features, entry/exit, risk, fills, positions, account metrics, and replay for offline research and read-only shadow mode.
- Make every result causal, replayable, provenance-bearing, and fail closed.
- Produce measurable evidence for the 10%-daily research benchmark without treating it as a target or guarantee.

## Non-goals and hard safety boundaries

- No `order_send`, order modification, close-position, or other MT5 mutation API.
- No live positions, real-money execution, leverage changes, martingale, grid, revenge, or forced trade quota.
- Phase 7 Knowledge and Phase 11A distillation remain untouched and are never mixed into the Phase 12 database.
- No LLM call on a tick. The strategic plan is supplied by an explicit plan source and may be refreshed only at a bounded schedule boundary.
- No future prices or future-derived features in replay, optimization, or walk-forward reports.
- Phase 12D remains unimplemented and unavailable from every CLI.

## Architecture

```text
Strategic plan source (validated, slow)
              |
        atomic PlanStore
              |
  causal TickFeatureEngine (no LLM)
              |
  FastExecutionEngine (entry/exit/invalidation)
              |
  Deterministic RiskEngine
              |
  ShadowFillEngine -> ShadowPositionLedger -> AccountSimulator
              |
  SQLite HFT ledger + scalar telemetry
```

The same `FastExecutionEngine`, `RiskEngine`, `ShadowFillEngine`, and
`AccountSimulator` are used by `forex_hft_replay` and `forex_hft_shadow`.
Only the tick source differs: replay reads an immutable tick file/database;
shadow reads `MT5Provider.get_tick()` through the existing serialized read-only
MT5 gate. The strategic plan is immutable after validation and replaced atomically.

## Module boundaries

- `tradingagents/forex/hft/models.py`: enums, bounded typed contracts, plan and tick/decision records.
- `features.py`: causal rolling tick features with bounded history.
- `plan_store.py`: validated atomic plan replacement and expiry.
- `engines.py`: entry/exit decisioning and deterministic risk checks.
- `execution.py`: realistic bid/ask fills, slippage/latency, position lifecycle, MAE/MFE.
- `account.py`: balance/equity/drawdown and fixed/compounding/volatility-adjusted sizing.
- `store.py`: versioned SQLite shadow ledger and crash-safe transactions.
- `replay.py`: causal tick loading, walk-forward splits, strategy lab and scalar reports.
- `runtime.py`: read-only MT5 shadow loop, plan refresh seam, and one-operation gate.
- `cli/forex_hft_replay.py` and `cli/forex_hft_shadow.py`: explicit commands; no stock CLI changes.

## StrategicExecutionPlan

The plan contains symbol, creation/expiry/allowed-until timestamps, timeframe,
regime, `primary_direction` (`LONG`, `SHORT`, `BOTH`, `NONE`), confidence,
strategy family, entry constraints, invalidation, spread/momentum/volatility
bounds, risk posture, stop/take-profit/time-stop policy, and session constraints.
All timestamps are timezone-aware UTC. Confidence, prices, distances, and
durations are finite and bounded. Expired or mismatched-symbol plans yield
`NO_ACTION` and cannot be swapped into the active store.

## Deterministic features and engines

Features include bid/ask/mid/spread, one-step and short-window returns,
velocity/acceleration, rolling volatility/range, high/low distance, spread
expansion, tick frequency/burst, direction persistence, M1/M5 context when
provided, session label, and plan age. A feature only uses ticks at or before
the current tick.

The entry scorer requires both strategic direction and confirmation: spread
ceiling, momentum threshold, volatility band, and risk approval. Exit logic
checks stop, target, time stop, trailing/breakeven policy, thesis invalidation,
momentum/volatility/spread shock, and session end. All outputs are typed
`NO_ACTION`, `ENTER_LONG`, `ENTER_SHORT`, `EXIT`, `REDUCE`, or `INVALIDATE`.

## Risk and sizing

Risk checks are independent of direction and enforce bounded risk-per-trade,
open exposure, daily loss, drawdown, consecutive-loss, spread/slippage, stale
tick, plan expiry, and session limits. Every rejection has a stable code.
Sizing supports `FIXED_RISK`, `COMPOUNDING_RISK`, and
`VOLATILITY_ADJUSTED` with explicit caps. The 10%-daily report is a mathematical
benchmark (achieved/not achieved, drawdown, trade count, and risk), never a
control input.

## Shadow execution semantics

BUY entry uses ask and BUY exit uses bid; SELL entry uses bid and SELL exit uses
ask. Slippage and latency are explicit assumptions. Missing ticks, gaps, and
spread shocks are recorded rather than filled at midpoint. Position states are
`FLAT`, `PENDING_ENTRY`, `LONG`, `SHORT`, `PENDING_EXIT`, and `CLOSED`.

## Persistence

Each HFT artifact root has a SQLite database with schema version, run metadata,
plans, ticks/features, actions, risk decisions, shadow positions/fills, and
account snapshots. Writes are transactional and idempotent by run/tick/action
identity. Source replay files are read-only. No existing Phase 5/6 tables are
rewritten.

## Replay and walk-forward

`python -m cli.forex_hft_replay` feeds immutable normalized ticks through the
same runtime engines. It rejects non-monotonic/future input and records a case
fingerprint. Splits are chronological TRAIN/DEV/VALIDATION/UNSEEN_TEST with
guard gaps; parameters are selected only on TRAIN/DEV. The initial strategy lab
ships two bounded families (momentum continuation and range rejection) behind a
common interface; additional families are future extensions, not automatic
optimization.

Replay reports p50/p95/p99/max tick latency, trade count, win rate, profit
factor, expectancy, drawdown, daily returns, compounding result, 10%-plus day
count, and risk-of-ruin only when statistically meaningful. Metrics are
descriptive and do not imply profitability.

## Read-only shadow runtime

`python -m cli.forex_hft_shadow run` is opt-in, requires a validated plan and
explicit artifact root, and prints `MT5 FOREX — HFT SHADOW MODE` and
`NO ORDER WILL BE SENT`. It acquires the existing watcher lease before creating
an MT5 provider, uses the existing serialized MT5 operation gate, and calls
only `get_tick`/read-only data methods. It persists `executed=False` for every
action/fill and stops on lease loss, stale data, expired plan, or storage error.

## Supervisor/dashboard seams

The core runtime is independent and testable. Supervisor integration is an
explicit subcommand/health seam that refuses duplicate MT5 ownership; it does
not alter the stock CLI. Dashboard additions read the HFT ledger read-only and
label every metric `SHADOW`.

## Acceptance gates

Before 12C validation: replay integrity, no lookahead, realistic bid/ask fills,
risk limits, deterministic millisecond-class processing, SQLite crash recovery,
and CLI safety tests must pass. 12C requires one bounded read-only shadow smoke
with unchanged MT5 positions/orders. 12D has no implementation and requires a
separate future approval plus unseen-data evidence.

