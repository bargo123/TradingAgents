# Phase 5 Shadow Outcome Evaluator Design

**Status:** Approved and corrected design; implementation follows in the Phase 5 plan.

**Date:** 2026-09-09

**Repository:** `C:\\AITrading\\TradingAgents`

**Accepted baseline:** `3d9262c639ca1ede3ac830626361500b760d3a29` (Phase 4.3)

## Goal

Build a deterministic, reusable evaluator that measures what happened after a
persisted forex shadow decision. The evaluator reads the original
`ShadowTradeDecision`, waits for each configured horizon to mature, obtains
historical quotes through the read-only `MT5Provider`, calculates cost-aware
BUY/SELL counterfactuals plus MFE/MAE, and persists auditable evidence. It
makes zero LLM calls and never changes how a decision is produced.

## Non-goals and safety boundary

- The stock `tradingagents` CLI is unchanged.
- `forex-shadow` remains a decision-producing, read-only command. This phase
  adds a separate `forex-evaluate` command and does not route evaluation into
  the graph.
- No `order_send`, order lifecycle, position mutation, SL/TP, live/demo
  execution, RiskGovernor, RAG, training, ONNX, or outcome-label promotion is
  added.
- The evaluator never imports the `MetaTrader5` package. All terminal access
  goes through `MT5Provider`.
- Existing incomplete decisions remain incomplete forever for training
  eligibility, even if later market data is available. Outcome evidence and
  context quality are separate dimensions.

## Current repository contracts

Phase 4.3 provides:

- `ShadowTradeDecision` with UTC `snapshot_timestamp`, requested/resolved
  symbols, the analysis-snapshot bid/ask/spread/spread-points, normalized
  action, raw Portfolio Manager result, `normalization_status`,
  `decision_context_status`, and immutable `executed=False`. In Phase 5 this
  record is extended with explicit analysis-completion and decision-reference
  fields; the legacy names remain readable aliases for old rows.
- `ShadowDecisionStore` backed by SQLite. Its legacy
  `future_evaluation_status`/`outcome_*` fields remain untouched for
  compatibility; Phase 5 evidence lives in a separate table.
- `ForexMarketSnapshot`/`Mt5Tick` and normalized symbol metadata containing
  `digits` and `point`.
- `MT5Provider` with symbol resolution, UTC tick normalization, and read-only
  tick/bar/account/position/order access, but no historical tick-range method.

## Architecture

The evaluator is a standalone service layer:

```text
forex-evaluate CLI or Python caller
        |
ShadowOutcomeEvaluator
        |-- ShadowDecisionStore (read the decision)
        |-- ShadowEvaluationStore (persist per-basis/per-horizon evidence)
        `-- MT5Provider.get_ticks_range (one bounded read per decision when possible)
```

The primary new module is `tradingagents/forex/evaluation.py`. The provider
gets the smallest necessary read-only extension in
`tradingagents/dataflows/mt5/provider.py`; no raw terminal import appears in
the evaluator. `tradingagents/forex/__init__.py` re-exports the public service,
configuration, result, and record types. `cli/forex_evaluate.py` is a separate
entry point registered as `forex-evaluate` in `pyproject.toml`.

The Phase 5 compatibility portion of `ShadowTradeDecision` and
`ForexShadowRunner` records temporal evidence at decision creation. The runner
captures the analysis snapshot first, records `decision_completed_timestamp`
immediately after the graph returns its structured Portfolio Manager result,
then performs exactly one fresh read-only `get_spread(resolved_symbol)` call.
The quote's broker-provided tick timestamp is stored as the decision
reference; the application completion time is never copied into the market
timestamp. The fresh quote is not fed back into the graph and it never
triggers another LLM call.

## Temporal evidence, eligibility, and training safety

Phase 4.3's `snapshot_timestamp` and `reference_*` values describe the market
state analyzed by the LLM. They must not be treated as an executable entry if
the graph took a long time to finish. Phase 5 therefore extends the decision
record with these separate fields:

```text
analysis_snapshot_timestamp
analysis_snapshot_bid
analysis_snapshot_ask
analysis_snapshot_spread
analysis_snapshot_spread_points
decision_completed_timestamp
analysis_latency_seconds
decision_reference_timestamp
decision_reference_bid
decision_reference_ask
decision_reference_spread
decision_reference_spread_points
decision_reference_status       # AVAILABLE, UNAVAILABLE, INVALID_TEMPORAL
decision_reference_delay_seconds
decision_reference_error
```

The existing `snapshot_timestamp`/`reference_*` columns remain populated as
backward-compatible aliases of the analysis snapshot. New rows populate both
the explicit fields and the legacy aliases. A legacy row without explicit
decision-reference values can still be evaluated on the `ANALYSIS_SNAPSHOT`
basis, but its `DECISION_REFERENCE` basis is explicitly unavailable; the
analysis quote is never silently substituted.

`decision_completed_timestamp` is captured immediately after the graph returns
the structured Portfolio Manager result using the application clock.
`analysis_latency_seconds` is the UTC difference between that timestamp and
`analysis_snapshot_timestamp`.
`decision_reference_delay_seconds` is the difference between the fresh quote
timestamp and `decision_completed_timestamp` (negative when the terminal
returns an older quote, which makes the reference `INVALID_TEMPORAL`). The
source timestamps also make the full analysis-to-reference interval
reconstructible. These raw values make staleness auditable without inventing a
universal stale threshold. A decision may be structurally valid for offline
signal-quality research while remaining unsuitable as execution-quality
evidence; the evaluator never labels anything as live or executable
performance.

The evaluator computes a separate result for each `EvaluationBasis`:

- `ANALYSIS_SNAPSHOT`: signal/reasoning evidence anchored at the analyzed
  snapshot. It is never described as an executable shadow entry.
- `DECISION_REFERENCE`: actionable counterfactual evidence anchored at the
  fresh quote taken after completion. It is available only when the reference
  quote is present, finite, UTC-aware, and not earlier than
  `decision_completed_timestamp`; otherwise its rows are
  `DATA_UNAVAILABLE` with an explicit reason.

`EvaluationBasis` is the closed vocabulary
`Literal["ANALYSIS_SNAPSHOT", "DECISION_REFERENCE"]`. It is required on every
evaluation result and database row.

Source/decision context eligibility is true only when all of these hold:

1. `decision_context_status == "COMPLETE"`;
2. `normalization_status == "NORMALIZED"`;
3. `action` is exactly `BUY`, `SELL`, or `HOLD`;
4. `analysis_snapshot_timestamp` (or the validated legacy alias) is
   timezone-aware UTC.

This is stored as `source_context_eligible`, independently from evaluation
status and data quality. An ineligible decision is not sent to MT5; it gets
one `INELIGIBLE` row for every configured basis/horizon combination. If an
invalid legacy timestamp prevents deriving a target, that row keeps
`target_timestamp` null and records the eligibility reason rather than being
silently omitted.

The evaluator does **not** set `training_eligible=1`. The evaluation table
stores nullable `training_eligible` plus
`training_eligibility_reason`; Phase 5 leaves the value `NULL` and records
`DEFERRED_TO_CORPUS_BUILDER` for source-eligible rows or
`SOURCE_CONTEXT_INELIGIBLE` otherwise. A future corpus builder must combine
source eligibility, evaluation completeness/data quality, temporal staleness
policy, and strategy rules before assigning a training label. Pending,
data-unavailable, temporally invalid, or incomplete outcomes are never
pre-labeled as training examples.

## Configurable horizons and maturity

`EvaluationConfig` defaults to the current `INTRADAY` profile:

```python
horizons_seconds = (300, 900, 1800, 3600)
observation_tolerance_seconds = 30
```

The horizon tuple is validated as positive, strictly increasing integers and
is injectable for future profiles. No swing horizons are included in Phase 5.

For each basis and horizon:

```text
target_time = basis_anchor_timestamp + horizon_seconds
observation_deadline = target_time + tolerance
```

For `ANALYSIS_SNAPSHOT`, `basis_anchor_timestamp` is
`analysis_snapshot_timestamp`; for `DECISION_REFERENCE`, it is
`decision_reference_timestamp`.

Before the deadline, the horizon is `PENDING`, including the interval between
`target_time` and `observation_deadline`. Once the deadline has passed, the
evaluator chooses the first valid tick whose UTC timestamp satisfies:

```text
target_time <= tick.timestamp <= observation_deadline
```

`observation_lag_ms` is the selected timestamp minus `target_time`. A missing
valid tick after the deadline is terminal `DATA_UNAVAILABLE`; the evaluator
never jumps to a later session or interpolates a price.

The evaluator's clock is injectable for deterministic tests and defaults to a
timezone-aware UTC `now`. It requests no data beyond the latest mature
observation deadline for the basis being evaluated and filters returned ticks
to that bound, preventing lookahead even if a broker returns extra rows. A
decision-reference horizon whose fresh quote is missing or temporally invalid
has no safe anchor and is `DATA_UNAVAILABLE`, rather than falling back to the
analysis snapshot.

## Historical MT5 provider extension

Add this public, read-only method to `MT5Provider`:

```python
def get_ticks_range(
    self,
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    flags: int | None = None,
) -> tuple[Mt5Tick, ...]:
    ...
```

The method:

- requires an initialized, connected provider;
- resolves the requested symbol using existing exact-first,
  ambiguity-safe rules;
- requires timezone-aware `start` and `end`, normalizes them to UTC, and
  rejects `end < start`;
- calls only `copy_ticks_range(resolved, start_utc, end_utc, flags)`;
- defaults `flags` to the API's `COPY_TICKS_ALL` constant (or `-1` only when
  that constant is unavailable in an injected fake);
- maps every row to `Mt5Tick`, preferring `time_msc` and falling back to
  `time`, with UTC-aware timestamps;
- raises the existing typed `Mt5DataError` for a `None` API response or
  malformed rows, while allowing an empty tuple for a valid empty market
  window.

No mutation method is added. The provider's existing `digits`/`point`
normalization is reused; the evaluator never infers one from the other.

## Quote normalization, decision completion, and one-read strategy

The runner preserves two immutable quote sets on every new decision. The
analysis set is captured before any LLM work:

```text
analysis_snapshot_timestamp = ForexMarketSnapshot.timestamp
analysis_snapshot_bid       = ForexMarketSnapshot.bid
analysis_snapshot_ask       = ForexMarketSnapshot.ask
analysis_snapshot_spread    = ForexMarketSnapshot.spread
analysis_snapshot_spread_points = ForexMarketSnapshot.spread_points
```

Immediately after the graph returns its structured Portfolio Manager result,
the runner captures `decision_completed_timestamp` and derives
`analysis_latency_seconds`. It then calls the existing read-only
`provider.get_spread(resolved_symbol)` exactly once. On success, that result
must expose the actual broker tick timestamp through `Mt5Spread.timestamp`
(or the smallest equivalent read-only tick seam if a provider implementation
does not expose it); that timestamp populates
`decision_reference_timestamp`. The local completion timestamp is never
assigned to the market quote. The same fresh result supplies bid, ask, spread,
and spread-points.
On failure, the decision is still persisted with the analysis result,
`decision_reference_status=UNAVAILABLE`, null reference quote fields, and a
non-secret error string. No second LLM call and no mutation is allowed. If the
fresh quote's timestamp precedes completion, its raw values are retained but
the status is `INVALID_TEMPORAL`; it cannot qualify as actionable evidence.

The legacy `snapshot_timestamp` and `reference_*` fields are populated with
the analysis set for old readers. Phase 5 always selects an explicit
`evaluation_basis` and never treats those aliases as a decision reference.

`point` and `digits` come from `snapshot_json["symbol_metadata"]`. Missing,
non-finite, non-positive `point`, or invalid `digits` produces explicit
`DATA_UNAVAILABLE` evidence; it is never repaired or inferred. A broker
reported zero spread is stored as zero in whichever quote set reported it.

For each decision, the evaluator fetches at most one historical range whenever
both bases have mature horizons:

```text
start = min(mature basis anchor timestamps)
end = max(target_time + tolerance for mature horizons across both bases)
```

Each basis filters that shared range to its own target/deadline and exact MFE/
MAE window, so the extra rows in the superset cannot leak across bases or past
an individual target. Already terminal horizon rows are not recomputed. If no
horizon is ready, no provider is initialized and all configured rows remain
`PENDING`. For a batch, one provider lifecycle is reused, while each decision
gets its own bounded range because anchor times differ.

Returned ticks are sorted by UTC timestamp and validated for finite bid/ask
with `ask >= bid`. Invalid rows are excluded from calculations and counted in
diagnostics; no valid terminal observation means `DATA_UNAVAILABLE`.

`ShadowDecisionStore` adds the temporal/decision-reference columns listed
above through an idempotent SQLite migration. Existing rows receive
logical `analysis_snapshot_*` values from their legacy
`snapshot_timestamp`/`reference_*` fields on read, while
`decision_completed_timestamp`, latency, and decision-reference fields remain
unknown (`NULL`) unless independently recorded. The migration never rewrites
the old quote values or marks an old row as actionable.

## Counterfactual calculations

For every terminal `COMPLETE` horizon, both hypothetical directions are
calculated regardless of the AI's selected action:

```text
BUY:
    buy_net_price  = future_bid - entry_ask
    buy_net_points = buy_net_price / point

SELL:
    sell_net_price  = entry_bid - future_ask
    sell_net_points = sell_net_price / point
```

These formulas use executable sides and therefore include the entry/exit
spread cost. Mid-to-mid movement is not used for profitability.

`selected_action` is copied from the eligible decision. For `BUY` and `SELL`,
`selected_action_net_price/points` equals that direction's net result. For
`HOLD`, no position is opened: selected trading price/points are explicitly
zero, and

```text
hold_opportunity_cost_points = max(0, buy_net_points, sell_net_points)
```

If both directional counterfactuals lose, the opportunity cost is zero. The
best directional counterfactual remains stored separately, and HOLD is not
labeled correct by the zero trading PnL.

`best_counterfactual_action` is `BUY` or `SELL` for a strict maximum and
`TIE` when the two net-point values are equal. Its value is evidence, not a
quality label. `best_counterfactual_net_points` always stores the maximum.

## MFE and MAE

The excursion window is `[basis_anchor_timestamp, target_time]`, never the
post-target observation tolerance. The basis entry quote is included as an
anchor; the terminal observation is included only when its timestamp is at or
before the exact target. A tick from `target_time+1s` through
`target_time+30s` may select the terminal return quote but must not affect MFE
or MAE. Signed price and point values are both persisted:

```text
BUY (bid is the executable close side):
    buy_mfe_price   = max(bid_in_window) - entry_ask
    buy_mae_price   = min(bid_in_window) - entry_ask
    buy_mfe_points  = buy_mfe_price / point
    buy_mae_points  = buy_mae_price / point

SELL (ask is the executable close side):
    sell_mfe_price  = entry_bid - min(ask_in_window)
    sell_mae_price  = entry_bid - max(ask_in_window)
    sell_mfe_points = sell_mfe_price / point
    sell_mae_points = sell_mae_price / point
```

Under these conventions adverse excursion is normally negative; the initial
spread may make the anchor itself adverse. Cost-aware MFE can remain negative
when price never moves far enough to overcome the entry spread. It is not
floored to zero; a conventional spread-free excursion metric would be a
separate future field. No tick after the target may affect MFE or MAE.

## Evaluation records and statuses

The new immutable Python record is `ShadowOutcomeEvaluation`. There is one row
per `(decision_id, evaluation_basis, horizon_seconds)`, so signal evidence and
actionable counterfactual evidence can never be mistaken for one another. Its
fields are:

```text
decision_id, resolved_symbol, evaluation_basis, horizon_seconds,
evaluation_version, market_data_source,
source_context_eligible, training_eligible, training_eligibility_reason,
target_timestamp, observation_timestamp, observation_lag_ms,
entry_timestamp, entry_bid, entry_ask, entry_spread, entry_spread_points,
future_bid, future_ask, future_spread, future_spread_points,
point, digits,
buy_net_price, buy_net_points, sell_net_price, sell_net_points,
selected_action, selected_action_net_price, selected_action_net_points,
best_counterfactual_action, best_counterfactual_net_points,
hold_opportunity_cost_points,
buy_mfe_price, buy_mfe_points, buy_mae_price, buy_mae_points,
sell_mfe_price, sell_mfe_points, sell_mae_price, sell_mae_points,
evaluation_status, unavailable_reason, created_at, evaluated_at,
recovered_from_unavailable_at, previous_unavailable_reason
```

`entry_*` is the quote set for the row's basis: analysis snapshot for
`ANALYSIS_SNAPSHOT`, decision reference for `DECISION_REFERENCE`. The source
decision itself retains both complete quote sets and both timestamps.

Per-horizon `evaluation_status` values:

- `PENDING`: target/deadline has not matured, or a retryable provider failure
  left the row pending;
- `COMPLETE`: a valid future observation and all calculations were persisted;
- `DATA_UNAVAILABLE`: the bounded window closed without a valid observation or
  required market metadata was malformed;
- `INELIGIBLE`: the source decision failed the context/action/timestamp
  eligibility contract.

The evaluator derives an aggregate status independently for each basis, without
collapsing signal and actionable evidence:

- `INELIGIBLE` if eligibility failed;
- `PENDING` if no horizon is terminal and at least one is pending;
- `PARTIAL` if terminal and pending horizons coexist;
- `COMPLETE` if every horizon is terminal and at least one is complete
  (unavailable horizons remain visible);
- `DATA_UNAVAILABLE` if every horizon is terminal unavailable.

A `COMPLETE` or `INELIGIBLE` evaluation row is never overwritten. A retryable
MT5 initialization or read failure leaves existing valid rows untouched and
leaves unresolved horizons pending, with an operation error in the result
metrics. A `PENDING` row may transition to `COMPLETE` or
`DATA_UNAVAILABLE`. After a temporary data gap has produced
`DATA_UNAVAILABLE`, a later successful bounded MT5 read may transition that
same row to `COMPLETE` when a valid observation and calculations are
available. Recovery preserves evaluation timestamps, provider/data-source
provenance, and the prior unavailable reason in audit metadata; valid
`COMPLETE` evidence cannot be replaced.

## SQLite schema and idempotency

`ShadowEvaluationStore` uses the same SQLite path as `ShadowDecisionStore` and
creates `shadow_decision_evaluations`:

```sql
CREATE TABLE IF NOT EXISTS shadow_decision_evaluations (
    decision_id TEXT NOT NULL,
    evaluation_basis TEXT NOT NULL CHECK (evaluation_basis IN ('ANALYSIS_SNAPSHOT','DECISION_REFERENCE')),
    horizon_seconds INTEGER NOT NULL CHECK (horizon_seconds > 0),
    resolved_symbol TEXT NOT NULL,
    evaluation_version TEXT NOT NULL,
    market_data_source TEXT NOT NULL CHECK (market_data_source = 'MT5'),
    source_context_eligible INTEGER NOT NULL CHECK (source_context_eligible IN (0, 1)),
    training_eligible INTEGER CHECK (training_eligible IS NULL OR training_eligible IN (0, 1)),
    training_eligibility_reason TEXT NOT NULL,
    target_timestamp TEXT,
    observation_timestamp TEXT,
    observation_lag_ms INTEGER,
    entry_timestamp TEXT,
    entry_bid REAL,
    entry_ask REAL,
    entry_spread REAL,
    entry_spread_points REAL,
    future_bid REAL,
    future_ask REAL,
    future_spread REAL,
    future_spread_points REAL,
    point REAL,
    digits INTEGER,
    buy_net_price REAL,
    buy_net_points REAL,
    sell_net_price REAL,
    sell_net_points REAL,
    selected_action TEXT CHECK (selected_action IS NULL OR selected_action IN ('BUY','SELL','HOLD')),
    selected_action_net_price REAL,
    selected_action_net_points REAL,
    best_counterfactual_action TEXT CHECK (best_counterfactual_action IS NULL OR best_counterfactual_action IN ('BUY','SELL','TIE')),
    best_counterfactual_net_points REAL,
    hold_opportunity_cost_points REAL,
    buy_mfe_price REAL,
    buy_mfe_points REAL,
    buy_mae_price REAL,
    buy_mae_points REAL,
    sell_mfe_price REAL,
    sell_mfe_points REAL,
    sell_mae_price REAL,
    sell_mae_points REAL,
    evaluation_status TEXT NOT NULL CHECK (evaluation_status IN ('PENDING','COMPLETE','DATA_UNAVAILABLE','INELIGIBLE')),
    unavailable_reason TEXT,
    created_at TEXT NOT NULL,
    evaluated_at TEXT,
    recovered_from_unavailable_at TEXT,
    previous_unavailable_reason TEXT,
    PRIMARY KEY (decision_id, evaluation_basis, horizon_seconds)
);
```

`target_timestamp` is nullable only for an invalid legacy timestamp that
cannot safely be expanded; valid decisions always persist it. Quote,
calculation, and observation fields are nullable for pending, unavailable, or
ineligible rows. A status-aware upsert may update an existing `PENDING` row to
`COMPLETE` or `DATA_UNAVAILABLE`, or recover an existing
`DATA_UNAVAILABLE` row to `COMPLETE`; it must never replace `COMPLETE` or
`INELIGIBLE` evidence. `created_at` is the first row timestamp and
`evaluated_at` records the current evaluation attempt for new/retried rows;
legacy rows created before this column may keep it null. When a
`DATA_UNAVAILABLE` row recovers to `COMPLETE`, `recovered_from_unavailable_at`
records the recovery attempt and `previous_unavailable_reason` preserves the
prior terminal reason; the new observation timestamp, evaluation version, and
market-data provenance are stored alongside it. Batch writes use one SQLite
transaction. Table creation is the
backward-compatible migration for existing Phase 4 databases, and tests also
cover an older evaluation table missing newly added optional columns. If an
older Phase 5 table lacks `evaluation_basis` and the triple key, the migration
rebuilds it transactionally, maps its rows to `ANALYSIS_SNAPSHOT`, and leaves
decision-reference rows to be created on the next evaluation; nullable audit
columns remain null for legacy rows. The
triple primary key makes reruns idempotent while allowing both bases for every
horizon.

The legacy `shadow_decisions.future_evaluation_status` is not changed by this
phase. Overall evaluation status is returned by the service and can be
re-derived from the per-horizon rows, avoiding a second mutable status source.

## Service and CLI contracts

The reusable service exposes:

```python
class ShadowOutcomeEvaluator:
    def evaluate_decision(
        self,
        decision_id: str,
        *,
        now: datetime | None = None,
        terminal_path: str | None = None,
    ) -> ShadowDecisionEvaluationResult: ...

    def evaluate_pending(
        self,
        *,
        now: datetime | None = None,
        terminal_path: str | None = None,
    ) -> ShadowEvaluationBatchResult: ...
```

Both result types include the decision/horizon records, a
`status_by_basis` mapping, error diagnostics, and numeric metrics:
`decisions_scanned`, `horizons_evaluated`, `historical_ticks_processed`,
`database_seconds`, `mt5_read_seconds`, `total_runtime_seconds`, and
`llm_calls` (always zero). A batch may also expose an operational aggregate
status, but callers must inspect each basis separately; no aggregate may merge
an analysis quote with a decision-reference quote.

`forex-evaluate` accepts exactly one of `--decision-id ID` or `--pending`, plus
`--db-path`, `--terminal-path`, and `--observation-tolerance-seconds`. It
prints:

```text
MT5 FOREX — OUTCOME EVALUATION (READ ONLY)
NO ORDER WILL BE SENT
```

It prints each basis's per-horizon/aggregate status and runtime metrics,
returns non-zero on a configuration or provider failure, and never accepts
credentials or execution options. The output labels analysis-snapshot and
decision-reference results separately and reports zero LLM calls.

## Error and closure behavior

The evaluator explicitly handles a missing decision, incomplete/unnormalized
decision, future target, terminal initialization failure, unresolved symbol,
empty historical range, market closure/weekend, quote gaps beyond tolerance,
malformed point/digits, malformed ticks, invalid UTC input, and a missing or
temporally invalid decision-reference quote. It never fabricates observations,
repairs zero spread, or mutates existing terminal or database evidence after a
read failure. A slow analysis is reported through its raw latency and
analysis-to-reference delay; it is not silently converted into an executable
entry or a profitability claim.

## Test and validation contract

All normal tests use synthetic UTC ticks and injected provider/store seams; no
LLM or live terminal is required. Tests cover the separate analysis/completion/
reference timestamps and latency, eligibility/training isolation, exact
horizons, UTC conversion, maturity/deadline/tolerance, no-lookahead,
BUY/SELL bid-ask arithmetic, zero/non-zero spread, point/digits, HOLD
opportunity cost (including both-negative counterfactuals), both
counterfactuals, signed MFE/MAE excluding tolerance ticks, gaps and missing
data, partial/complete aggregate status, idempotent retries, migration and
triple-key uniqueness, zero LLM calls, no direct MetaTrader5 import, no
mutation APIs, stock CLI regression, and `forex-shadow` read-only regression. A
guarded real MT5 integration test reads a bounded historical range and checks
UTC quotes, symbol resolution, calculations, and unchanged positions/orders; it
never sends an order.

The final validation runs the full suite, Ruff, `compileall`,
`git diff --check`, the opt-in MT5 integration guard, and an execution-safety
scan. Outcome numbers are objective evidence, not a correctness label or a
profitability guarantee.
