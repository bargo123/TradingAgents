# Phase 6 Automatic Forex Shadow Collector — Design Specification

**Status:** Design only; implementation is not approved or included
**Date:** 2026-09-09
**Repository:** `C:\\AITrading\\TradingAgents`
**Phase 5 SHA stated in request:** `5093c754bed8ad862b6006feccb7df53c3f6675b` (not present in this checkout)
**Accepted Phase 5 baseline in this checkout:** `5093c754bed8ad86210702700974d0c9f4a5c50c`

## 1. Purpose and scope

Phase 6 adds an autonomous, shadow-only collector around the completed Phase 4
and Phase 5 contracts. Its purpose is to collect real, timestamped forex
analysis evidence and later attach deterministic MT5 outcomes to that
evidence. It is an observation/data-collection service, not a trading engine.

The intended data path is:

```text
read-only MT5 availability probe
        |
        v
one bounded opportunity claim
        |
        v
existing ForexShadowRunner (one cached market snapshot)
        |
        v
existing TradingAgents forex-safe graph
        |
        v
ShadowDecisionStore (executed=False)
        |
        v
existing zero-LLM ShadowOutcomeEvaluator
        |
        v
per-basis/per-horizon outcome evidence
```

The operational default is deliberately small:

| Setting | Default |
|---|---|
| Requested symbol set | `EURUSD` |
| Analysis profile | `INTRADAY` |
| Analyst allow-list | `market,news` |
| Schedule anchor | completed `M15` UTC bar |
| Maximum active analyses | `1` |
| Execution | none; every decision is `executed=False` |

The design keeps symbols, cadence, analyst set, and provider/model settings
configurable for later experiments, but the first deployment must not fan out
to a large symbol basket.

## 2. Non-goals and hard boundaries

Phase 6 does not add or enable any of the following:

- `order_send`, opening or closing positions, pending orders, SL/TP changes,
  position modification, demo execution, or live execution;
- a second analysis implementation or a second graph;
- a second outcome-calculation implementation;
- RAG, fine-tuning, training-label promotion, model-weight changes, ONNX,
  RiskGovernor, or strategy selection;
- changes to the existing stock `tradingagents` CLI;
- a GUI, distributed queue, Redis/Kafka/Postgres, or a market-event bus;
- an assumption that a completed shadow decision is executable-quality
  evidence;
- a claim that a decision is profitable or training eligible merely because
  an outcome row exists.

The watcher may persist operational metadata and skipped opportunities, but it
must not persist prompts, private chain-of-thought, passwords, API keys, or
unredacted provider configuration containing credentials. Normal structured
Portfolio Manager output remains governed by the existing Phase 4 decision
contract; watcher logs contain only IDs, statuses, sizes/counts, timings, and
safe identifiers.

## 3. Existing contracts the design must preserve

The current repository already provides the boundaries Phase 6 should compose:

1. `MT5Provider` is read-only, resolves broker symbols fail-closed, maps
   timestamps to UTC, and exposes normalized `ForexMarketSnapshot`, ticks,
   account/position/order reads, and historical tick ranges. It has no
   mutation/execution method.
2. `ForexShadowRunner` resolves one symbol, captures one cached snapshot, uses
   the existing forex-safe TradingAgents graph, performs strict structured
   Portfolio Manager normalization, records context-integrity status, obtains
   one fresh broker-timestamped reference quote, persists a
   `ShadowTradeDecision`, and enforces `executed=False`.
3. `ShadowDecisionStore` persists the decision and its separate analysis and
   decision-reference temporal evidence. Its legacy outcome columns remain
   untouched by Phase 5/6.
4. `ShadowOutcomeEvaluator` reads decisions, makes zero LLM calls, obtains
   bounded historical ticks through `MT5Provider.get_ticks_range`, and writes
   one row per `(decision_id, evaluation_basis, horizon_seconds)`.
5. Phase 5 distinguishes `ANALYSIS_SNAPSHOT` from `DECISION_REFERENCE`, keeps
   `PENDING` until `target + tolerance`, retains both BUY/SELL
   counterfactuals, excludes tolerance-tail ticks from MFE/MAE, and allows
   `DATA_UNAVAILABLE -> COMPLETE` recovery. It leaves `training_eligible`
   deferred/null.
6. Existing `StatsCallbackHandler` can report numeric LLM/tool/token metrics
   without retaining prompt or completion content.
7. `DEFAULT_CONFIG` and its `TRADINGAGENTS_*` environment override system are
   the existing source for provider/model configuration. Phase 6 must consume
   that system and never require a credential command-line argument.

The only runner seam proposed below is an optional caller-supplied
`source_run_id`. It preserves all existing callers while allowing a watcher
run to reconcile a decision persisted immediately before a process crash.

## 4. Recommended architecture

### 4.1 Components

The watcher is a small coordinator with explicit injected boundaries:

- `WatcherCoordinator` owns the state machine, polling loop, circuit breakers,
  one-slot analysis executor, and evaluation cadence.
- `WatcherStore` owns the Phase 6 SQLite tables, atomic lease acquisition,
  opportunity claims, run status, restart reconciliation, and summary queries.
  It uses a separate SQLite connection per operation, a bounded busy timeout,
  and WAL mode where the existing database permits it.
- `SchedulePolicy` converts an injected UTC clock into completed-bar
  opportunities. It has no MT5 or LLM dependency.
- `ReadOnlyMarketProbe` performs only `initialize`, `ensure_symbol`, and a
  finite bid/ask tick read. It prevents a graph launch when the broker is
  unavailable or cannot provide a usable quote. It is a preflight quote, not
  the analysis snapshot.
- `SingleSlotAnalysisExecutor` is an explicit one-worker executor with a
  zero-capacity queue. It calls the existing `ForexShadowRunner` exactly once
  for a claimed opportunity. A busy executor rejects submission; the
  coordinator records the opportunity as skipped instead of queueing it.
- `LeaseHeartbeat` renews the watcher lease and records active-run heartbeat
  and soft-timeout metadata while the worker is running. It is a named,
  observable component, not an unbounded background queue.
- `Mt5OperationGate` is a non-reentrant injected guard around the read-only
  probe, the runner call, and the Phase 5 evaluator. The coordinator never
  waits on it while `ANALYZING`; the gate exists to make the one-operation
  invariant testable and to fail closed if a future code path attempts overlap.
- `OutcomeCoordinator` calls the existing `ShadowOutcomeEvaluator`; it never
  imports `MetaTrader5`, invokes an LLM, or reimplements outcome formulas.
- `StatusReporter` runs the same read-only SQL summary for the CLI and for
  optional structured status logs.

The foreground watcher loop remains the owner of scheduling decisions. The
single worker exists only so the foreground loop can observe bar closes and
record skips during a 40-minute analysis. There is never more than one active
runner task in a watcher instance, and the SQLite lease prevents a second
watcher instance from becoming active for the same database.

The Phase 6 v1 MT5 serialization invariant is **at most one MT5-using
operation at a time**. While the lifecycle is `ANALYZING`, the foreground loop
may renew the lease, observe schedule buckets, and record skipped opportunities,
but it must not run `ReadOnlyMarketProbe`, create another MT5 provider/session,
or invoke `ShadowOutcomeEvaluator`. A due evaluation is retained as a durable
pending fact and is run immediately after the analysis worker has returned and
its runner-owned MT5 session has shut down, before any next analysis preflight
or submission. A small injected operation gate is used as a defensive assertion
around probe, runner, and evaluator calls; it is not a queue and never permits
overlap.

### 4.2 Why one bounded worker instead of a queue

The current Qwen workflow can take about 40 minutes. A normal thread or timer
queue would accumulate stale jobs while the model is busy. The executor has
exactly one active slot and no pending queue. A schedule opportunity is either
claimed immediately or recorded as `SKIPPED`; it is never placed in a future
work list for late processing. A faster future model uses the same executor:
completion releases the slot, the cooldown/bar policy selects the next current
opportunity, and the old safety invariant remains unchanged.

The analysis worker is not a second graph. Its callable is the existing
`ForexShadowRunner.run` with the configured symbol, profile, analyst set,
callbacks, terminal path, database path, and watcher run ID.

### 4.3 Data flow for one opportunity

1. The scheduler observes a newly completed UTC bar bucket and derives a
   deterministic opportunity key.
2. `WatcherStore` inserts the opportunity with a unique key. If it already
   exists, no second decision is permitted.
3. If another analysis is active, the new opportunity is terminally marked
   `SKIPPED` with `ANALYSIS_ALREADY_RUNNING`; it is not queued.
4. If the watcher is cooling down, degraded, unable to resolve/probe MT5, or
   the bucket has expired, the opportunity is marked with the corresponding
   explicit skip reason.
5. Otherwise the store atomically claims the opportunity and creates a
   `RUNNING` run row. The claim contains the owner token and attempt number.
6. The worker calls `ForexShadowRunner.run`. The runner captures exactly one
   `ForexMarketSnapshot`, runs the existing forex graph, persists the genuine
   structured decision, and returns its metrics. No watcher code parses prose
   or copies reasoning into prompts.
7. The coordinator verifies the returned decision ID, `executed is False`,
   the source run ID, and the persisted row. It records safe provenance and
   marks the opportunity `DECISION_SAVED` (or the run `SUCCEEDED_SLOW` when the
   soft runtime threshold was exceeded).
8. If the runner returned, finalize its run first. If evaluation was due while
   the worker was active, clear the retained pending flag only after the
   existing `ShadowOutcomeEvaluator` completes (or records an evaluation
   error). The evaluator runs before any next analysis preflight/submission and
   makes zero LLM calls.
9. The next loop reports current state and considers only a current, unseen
   bar opportunity after cooldown. It never replays old timer events.

## 5. Architecture diagram

```text
                         +-----------------------------+
                         | forex-watch run / once      |
                         | foreground WatcherLoop       |
                         +-------------+---------------+
                                       |
                 +---------------------+----------------------+
                 |                                            |
                 v                                            v
       SchedulePolicy (UTC M15)                    WatcherStore / SQLite
                 |                             lease + opportunities + runs
                 v                                            ^
       current opportunity key <--------- atomic claim --------+
                 |
       if busy/unavailable/cooldown
                 +-------------> SKIPPED(reason), no queue
                 |
                 v
       ReadOnlyMarketProbe
                 |
                 v
       SingleSlotAnalysisExecutor (capacity=1, queue=0)
                 |
                 v
       ForexShadowRunner.run(source_run_id=watch_run_id)
                 |
                 +--> one ForexMarketSnapshot
                 +--> existing forex-safe TradingAgents graph
                 +--> ShadowDecisionStore / executed=False
                 +--> numeric StatsCallbackHandler metrics
                 |
                 v
       DECISION_SAVED / SUCCEEDED_SLOW
                 |
                 v
       OutcomeCoordinator (when due)
                 |
                 v
       existing ShadowOutcomeEvaluator (zero LLM, read-only MT5 history)
                 |
                 v
       shadow_decision_evaluations

       LeaseHeartbeat -- renews owner/run lease and soft-timeout facts
       StatusReporter -- SQL summary; no MT5/LLM unless explicitly probed
```

`forex-watch status` reads the store only. `forex-watch evaluate` delegates to
the existing evaluator only. The existing `forex-shadow` and stock
`tradingagents` commands remain separate entry points.

## 6. Scheduling recommendation

### 6.1 Comparison

| Option | Advantages | Problems at ~40-minute runtime | Decision |
|---|---|---|---|
| A. Fixed hourly timer | Easy to understand; predictable wall-clock cadence | Timer events occur while analysis is active; arbitrary seconds can capture mid-bar context; naïve implementations queue stale work or silently lose evidence | Rejected as the primary policy |
| B. Completion-driven | Never needs overlap; naturally adapts to model speed | Decisions drift relative to meaningful bar closes; a long run can complete just after a poor market window; no explicit record of bar opportunities missed during the run unless another observer exists | Useful as a gating signal, not sufficient alone |
| C. Bar-close/event-driven | Anchors context to M5/M15/H1 closes; meaningful for offline comparisons | A slow model misses many bars; a full event engine is unnecessary; without a single-slot gate it creates a backlog | Useful schedule input, not sufficient alone |
| D. Hybrid/completion-aware | Uses bar closes, completion/cooldown, a single active slot, and explicit skip rows; adapts when models become faster without changing safety | Requires a small deterministic scheduler and one explicit observer loop | **Recommended** |

### 6.2 Chosen policy

The default is a completion-aware hybrid:

- observe completed `M15` UTC buckets every `15` seconds;
- wait `30` seconds after the bucket close for a settled quote;
- allow at most one active analysis;
- after a decision finishes, wait a `60`-second cooldown and select only the
  newest eligible unseen bucket;
- while analysis is active, each newly observed bucket is recorded once as
  `SKIPPED / ANALYSIS_ALREADY_RUNNING`; no bucket is queued;
- if multiple buckets accumulated during downtime, materialize at most
  `8` recovery buckets and record one bounded gap summary for anything older;
- if a model becomes faster, a later current bar can run after cooldown; an
  already skipped bucket is never resurrected.

This is not a market-event engine. It is a UTC bucket calculation with a
polling loop. MT5 quote availability remains authoritative; the scheduler
never infers “market open” from a weekday or fabricates a quote.

### 6.3 Bar-close calculation

For `M5`, `M15`, or `H1`, the policy maps a UTC instant to the most recent
completed bucket, never the currently forming bucket. `eligible_after` equals
`bucket_close + bar_close_settle_seconds`. The key uses the bucket's UTC start
timestamp, not the time at which the timer noticed it. MT5's normalized UTC
timestamps are used for the eventual snapshot; no broker-local timezone is
guessed.

## 7. Exact state machines

### 7.1 Watcher lifecycle state

Persisted lifecycle states are:

```text
STOPPED
  -> STARTING             process launched, schema/config checks
  -> IDLE                 lease acquired and no active analysis

IDLE
  -> CHECK_MARKET         each scheduler poll
  -> START_ANALYSIS       one current opportunity passed all gates
  -> DEGRADED             a circuit breaker is open
  -> STOPPING             shutdown requested

CHECK_MARKET
  -> IDLE                 no new opportunity or explicit skip recorded
  -> START_ANALYSIS       claim and submit exactly one opportunity
  -> DEGRADED             breaker opens during checks

START_ANALYSIS
  -> ANALYZING            atomic claim succeeded and worker started
  -> IDLE                 claim/preflight/submission failed safely

ANALYZING
  -> ANALYZING            heartbeat/soft-timeout observation; still one slot
  -> DECISION_SAVED        worker returned and persisted a decision
  -> IDLE                 decision finalized and optional evaluation attempted
  -> STOPPING              shutdown requested; finish current run before exit

DECISION_SAVED
  -> IDLE                 watcher metadata committed

DEGRADED
  -> CHECK_MARKET          cooldown elapsed and a read-only health check passes
  -> STOPPING              shutdown requested

STOPPING
  -> STOPPED               active worker finished or process is deliberately
                           terminated by the operator
```

`ANALYZING` is never entered if `max_concurrent_analyses != 1` validation
fails; Phase 6 v1 rejects any value other than `1`. A soft timeout adds a
`runtime_alert` fact and may lead to `SUCCEEDED_SLOW` after completion; it does
not create a second worker or kill the existing runner.

### 7.2 Opportunity state

Each unique schedule opportunity has one of:

```text
ELIGIBLE
  -> RUNNING              atomic claim by the current lease owner
  -> SKIPPED              busy/cooldown/market/breaker/expiry reason

RUNNING
  -> DECISION_SAVED       decision row confirmed by source_run_id
  -> FAILED               runner/provider/DB exception, no valid decision row
  -> ABANDONED             stale owner recovered after crash

DECISION_SAVED, SKIPPED, FAILED, ABANDONED
  -> terminal for this key in v1
```

An `INCOMPLETE` context or `FAILED` normalization is not silently converted
to a successful-quality decision. If the runner persisted it, the opportunity
is still `DECISION_SAVED`, while the linked decision retains its independent
`decision_context_status` and `normalization_status`. Dashboards count those
quality states separately.

Automatic retries of the same opportunity are disabled by default
(`max_attempts_per_opportunity=1`). A future explicit operator retry would
need a new versioned opportunity/config key and would be visible as a distinct
attempt; Phase 6 does not use retries to create a backlog.

### 7.3 Evaluation state

Evaluation is independent of the analysis lifecycle:

```text
NOT_DUE / NO_MATURE_HORIZONS
  -> DUE
  -> EVALUATING
  -> EVALUATED             per-basis/per-horizon rows persisted
  -> EVALUATION_ERROR      read/provider/DB error; retry on next interval
```

The durable states remain the Phase 5 row values `PENDING`, `COMPLETE`,
`DATA_UNAVAILABLE`, and `INELIGIBLE`. `COMPLETE` and `INELIGIBLE` rows remain
immutable. `DATA_UNAVAILABLE` may recover to `COMPLETE` through the existing
Phase 5 store transition. The watcher never changes those meanings or writes
training labels.

## 8. Normal iteration and failure behavior

### 8.1 Startup

1. Parse and validate watcher configuration. Reject credentials or unknown
   analyst names before opening a run.
2. Initialize the shared SQLite schema with a short busy timeout. Do not
   delete, rewrite, or migrate valid `shadow_decisions` evidence except for
   the existing idempotent stores' normal schema setup.
3. Generate a random owner token and record process ID, host, and UTC start
   time. Acquire the singleton lease in `BEGIN IMMEDIATE` transaction.
4. If a non-expired owner exists, exit with `WATCHER_ALREADY_RUNNING`; do not
   probe MT5 or start an LLM.
5. If the lease is expired but the exact old same-host process is provably
   alive, exit with `WATCHER_OPERATOR_REVIEW_REQUIRED`; do not start a second
   watcher. Otherwise reconcile expired/stale runs before setting lifecycle
   state to `IDLE`.
6. Start the explicit heartbeat and scheduler components; emit a safe
   `WATCHER_STARTED` event.

### 8.2 Normal poll

On each injected-clock poll:

1. Renew the lease and update `last_loop_at`.
2. Materialize newly observed completed buckets, bounded by the recovery cap.
3. Mark opportunities that occurred while the slot was occupied as
   `SKIPPED / ANALYSIS_ALREADY_RUNNING`.
4. If an analysis future is active, do not run any probe or evaluator and do
   not create another MT5 provider/session. If an evaluation interval elapsed,
   set `evaluation_due_pending=1`; the loop does only heartbeat, schedule
   observation, and skip recording until the worker returns.
5. If an analysis future completed, finalize its run and linked opportunity,
   then run the retained/due zero-LLM evaluator immediately, before considering
   another analysis.
6. If no analysis is active and evaluation is due, run the evaluator before any
   new probe or analysis submission; clear `evaluation_due_pending` only after
   the evaluator returns or records a bounded error.
7. If the circuit is open, do not submit analysis; retain status and continue
   only the serialized zero-LLM evaluation path when no runner is active.
8. Otherwise select only the newest eligible current opportunity, run the
   read-only market probe, claim it atomically, and submit it to the empty
   analysis slot. The probe must finish and release its provider/session before
   the runner starts.

### 8.3 Analysis running and another schedule tick

The foreground loop continues while the single analysis worker runs. When a
new M15 bucket closes, it creates the deterministic opportunity row and marks
it `SKIPPED / ANALYSIS_ALREADY_RUNNING`. The row is not retained as a job and
is never submitted after the 40-minute analysis finishes. This gives an
auditable reason for the missing example without using a stale market quote.
During this interval, an evaluation deadline is recorded as
`evaluation_due_pending=1`, but no historical MT5 read is started. Once the
runner returns (and its provider has shut down), the coordinator evaluates
pending decisions before starting another probe or runner.

### 8.4 Runner/LLM/MT5 failures

- A runner exception is captured as `FAILED` with a closed error code and a
  short sanitized message. The worker slot is released only after the future
  is finalized; no overlapping retry starts.
- MT5 initialization, symbol resolution, snapshot, or quote failure increments
  the MT5 failure counter and may open the MT5 circuit. The next schedule
  opportunity gets a fresh preflight; no market-open state is fabricated.
- Ollama/hosted LLM or graph failure increments the analysis failure counter.
  The existing runner's `finally` shutdown behavior remains authoritative.
- A persisted incomplete context or failed normalization remains a real
  decision record with its raw structured result/status. It increments the
  corresponding quality counter and is not rerun automatically to “repair”
  the label.
- A database lock uses bounded retries. If the claim or final metadata write
  cannot commit, the run is marked `FAILED / DB_LOCKED` when possible and the
  next loop retries only the operational state, never an old market bucket.

### 8.5 Graceful shutdown

`SIGINT`/`SIGTERM` sets a stop request in the foreground loop. The watcher
stops claiming new opportunities, writes `STOPPING`, and by default waits for
the active runner future to return so the genuine decision can be persisted.
It then runs one best-effort zero-LLM evaluation pass, releases the lease, and
writes `STOPPED`. If an operator forcibly terminates the process, startup
reconciliation handles the stale run; safety does not depend on orderly
shutdown.

## 9. Lease, heartbeat, crash, and restart recovery

### 9.1 Lease rules

The database has one collector lease per database. Deployments that need
independent collector configurations use independent database paths. A lease
contains an owner token, PID, host, process start time, acquisition time,
heartbeat time, and expiry time. Suggested defaults are a `15`-second
heartbeat and a `9000`-second lease TTL; the TTL must be greater than the
heartbeat interval and greater than the configured soft analysis timeout plus
grace when a deployment chooses a longer timeout.

Lease acquisition is an atomic SQLite transaction:

1. lock the database with `BEGIN IMMEDIATE`;
2. read the singleton lease;
3. if an owner exists and `lease_expires_at > now`, return
   `WATCHER_ALREADY_RUNNING` regardless of PID/process-liveness observations;
   do not probe MT5, reconcile runs, or start an LLM;
4. only when there is no owner or the lease has expired may takeover/recovery
   begin. For an expired same-host lease, compare the recorded PID and exact
   process-start identity when the platform can prove them. If that exact old
   process is still provably alive, return
   `WATCHER_OPERATOR_REVIEW_REQUIRED` and fail closed rather than starting a
   second watcher. If the process is proven dead or cannot be proven alive,
   reconcile stale runs and write a new random owner token and expiry;
5. commit before any MT5 or LLM operation.

Every watcher status/run update includes the owner token. A stale owner cannot
overwrite a newly acquired owner's state. If a heartbeat write cannot commit,
the owner does not start any new analysis; it continues only the already
active runner and exits/degrades when the lease is definitively lost.

Lease expiry is the gate for every takeover. Process-liveness is supplemental
evidence used only after expiry: it can block a takeover when the exact
same-host PID plus process-start identity is provably still alive, but it can
never bypass a valid lease or justify stealing one. A missing, unverifiable, or
reused PID after expiry is not proof of a live owner; the expired lease may be
reconciled with the stored owner token and run IDs. PID reuse and host clock
anomalies must not permit a second active owner while the lease remains valid.

### 9.2 Heartbeat behavior

`LeaseHeartbeat` is an explicit component with a bounded loop:

- renew owner lease and active run heartbeat every `15` seconds;
- record `runtime_alert_at` once when monotonic elapsed time exceeds
  `analysis_timeout_seconds`;
- never enqueue work, call an LLM, or call an order API;
- stop and join on normal shutdown.

The heartbeat stores only operational metadata. It does not log prompts,
reports, or private reasoning. The main loop remains the only component that
selects or claims new opportunities.

### 9.3 Crash reconciliation

After the lease has expired and takeover safety checks permit recovery, any
`RUNNING` run whose owner lease is stale is reconciled before new scheduling.
The old process-start identity check is part of lease acquisition, not a reason
to bypass a non-expired lease:

1. Query `ShadowDecisionStore` by the run's unique `source_run_id`.
2. If exactly one decision exists, verify `executed=False`, link its decision
   ID, retain its stored context/normalization/temporal evidence, and mark the
   opportunity `DECISION_SAVED`. Missing watcher metrics are recorded as
   `unknown`, never fabricated.
3. If no decision exists, mark the run and opportunity `ABANDONED` with
   `STALE_RUN_RECOVERED`; do not resubmit the old bucket.
4. If more than one decision exists for the source ID, mark
   `RECONCILIATION_AMBIGUOUS`, open the circuit, and require operator review;
   no new analysis is launched for that key.
5. Clear the stale current-run pointer only under the new owner token.

This handles Python crashes, Windows restarts, MT5 restarts, Ollama exits, and
forced process termination without allowing a stale `RUNNING` row to block
collection forever or cause a duplicate historical analysis.

## 10. Decision deduplication

### 10.1 Canonical opportunity identity

The key is the SHA-256 digest of canonical UTF-8 JSON with sorted keys:

```json
{
  "requested_symbol": "EURUSD",
  "analysis_profile": "INTRADAY",
  "schedule_timeframe": "M15",
  "anchor_timestamp": "2026-09-09T12:00:00Z",
  "analysis_config_version": "forex-shadow.v1",
  "config_fingerprint": "<sha256 of safe effective config>"
}
```

The requested symbol is normalized to uppercase and trimmed. The resolved
broker suffix/prefix is stored after resolution but is not used to create a
second key; a broker symbol variant cannot create a duplicate for the same
requested opportunity. The key intentionally includes profile, timeframe,
anchor, and model/config version. A deliberate model/config version change is
an explicit new experiment, not an accidental restart duplicate.

`git_commit` is stored as provenance but is not silently folded into the key;
the workflow/prompt contract version must be bumped when behavior changes. A
restart with the same effective contract therefore cannot create a duplicate.

### 10.2 Claim and restart rules

`forex_watch_opportunities.opportunity_key` is the primary key. Insertion and
claim happen in one transaction with the owner token. A row in any terminal
state (`DECISION_SAVED`, `SKIPPED`, `FAILED`, `ABANDONED`) is never claimed
again in v1. A live `RUNNING` row may be finalized only by its owner or by
stale-run reconciliation. The runner receives the unique `watch_run_id` as
`source_run_id`, which makes a decision written just before a crash
reconcilable. The existing `ShadowDecisionStore` gets a small read-only
`find_by_source_run_id()` query seam for this reconciliation; it does not
change decision mutability or add any execution API.

The watcher never deduplicates by quote time, action text, or raw prose. Those
fields can differ while the schedule opportunity is the same.

## 11. Proposed SQLite schema

The existing `shadow_decisions` and Phase 5
`shadow_decision_evaluations` tables remain authoritative for decision and
outcome evidence. Phase 6 adds separate tables in the same SQLite file. The
new store must use idempotent `CREATE TABLE IF NOT EXISTS`/column migrations,
preserve unknown existing tables, and use one transaction for each claim or
state transition.

### 11.1 `forex_watcher_state` singleton

```sql
CREATE TABLE forex_watcher_state (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    lifecycle_status TEXT NOT NULL CHECK (
        lifecycle_status IN ('STOPPED','STARTING','IDLE','CHECK_MARKET',
                             'START_ANALYSIS','ANALYZING','DECISION_SAVED',
                             'DEGRADED','STOPPING')
    ),
    owner_token TEXT,
    owner_pid INTEGER,
    owner_host TEXT,
    process_started_at TEXT,
    lease_acquired_at TEXT,
    heartbeat_at TEXT,
    lease_expires_at TEXT,
    current_run_id TEXT,
    current_opportunity_key TEXT,
    last_loop_at TEXT,
    last_analysis_completed_at TEXT,
    next_eligible_at TEXT,
    last_evaluation_at TEXT,
    last_evaluation_status TEXT,
    evaluation_due_pending INTEGER NOT NULL DEFAULT 0 CHECK (evaluation_due_pending IN (0,1)),
    last_error_code TEXT,
    last_error TEXT,
    circuit_reason TEXT,
    circuit_opened_at TEXT,
    consecutive_mt5_failures INTEGER NOT NULL DEFAULT 0,
    consecutive_analysis_failures INTEGER NOT NULL DEFAULT 0,
    consecutive_incomplete_decisions INTEGER NOT NULL DEFAULT 0,
    consecutive_normalization_failures INTEGER NOT NULL DEFAULT 0,
    consecutive_runtime_exceeded INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
```

Only the current owner may update lease/current-run fields. Error text is
sanitized and bounded; full exception tracebacks belong in local operator logs,
not in the evidence database.

### 11.2 `forex_watch_opportunities`

```sql
CREATE TABLE forex_watch_opportunities (
    opportunity_key TEXT PRIMARY KEY,
    requested_symbol TEXT NOT NULL,
    analysis_profile TEXT NOT NULL,
    analyst_set_json TEXT NOT NULL,
    schedule_timeframe TEXT NOT NULL CHECK (schedule_timeframe IN ('M5','M15','H1')),
    anchor_timestamp TEXT NOT NULL,
    bar_close_timestamp TEXT NOT NULL,
    eligible_after TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('ELIGIBLE','RUNNING','DECISION_SAVED','SKIPPED',
                   'FAILED','ABANDONED')
    ),
    skip_reason TEXT,
    skip_detail TEXT,
    run_id TEXT,
    decision_id TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    first_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (requested_symbol, analysis_profile, schedule_timeframe,
            anchor_timestamp, config_fingerprint)
);
```

The unique composite constraint mirrors the canonical key and makes accidental
key-construction regressions visible. `skip_reason` is a controlled value
such as `ANALYSIS_ALREADY_RUNNING`, `COOLDOWN`, `CIRCUIT_BREAKER`,
`MT5_UNAVAILABLE`, `SYMBOL_NOT_FOUND`, `STALE_BUCKET`, `WATCHER_NOT_RUNNING`,
or `DB_LOCKED`; `skip_detail` is short sanitized context.

### 11.3 `forex_watch_runs`

```sql
CREATE TABLE forex_watch_runs (
    run_id TEXT PRIMARY KEY,
    opportunity_key TEXT NOT NULL REFERENCES forex_watch_opportunities(opportunity_key),
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    owner_token TEXT NOT NULL,
    run_status TEXT NOT NULL CHECK (
        run_status IN ('RUNNING','SUCCEEDED','SUCCEEDED_SLOW','FAILED','ABANDONED')
    ),
    requested_symbol TEXT NOT NULL,
    resolved_symbol TEXT,
    started_at TEXT NOT NULL,
    heartbeat_at TEXT,
    runtime_alert_at TEXT,
    completed_at TEXT,
    decision_id TEXT,
    source_run_id TEXT NOT NULL UNIQUE,
    failure_code TEXT,
    failure_detail TEXT,
    decision_context_status TEXT CHECK (
        decision_context_status IS NULL OR decision_context_status IN ('COMPLETE','INCOMPLETE')
    ),
    normalization_status TEXT CHECK (
        normalization_status IS NULL OR normalization_status IN ('NORMALIZED','FAILED')
    ),
    normalized_action TEXT CHECK (
        normalized_action IS NULL OR normalized_action IN ('BUY','SELL','HOLD')
    ),
    analysis_snapshot_timestamp TEXT,
    decision_completed_timestamp TEXT,
    analysis_latency_seconds REAL,
    decision_reference_timestamp TEXT,
    decision_reference_status TEXT,
    decision_reference_delay_seconds REAL,
    stale_by_completion INTEGER CHECK (stale_by_completion IS NULL OR stale_by_completion IN (0,1)),
    freshness_budget_seconds INTEGER,
    llm_provider TEXT,
    quick_model TEXT,
    deep_model TEXT,
    analyst_set_json TEXT NOT NULL,
    analysis_profile TEXT NOT NULL,
    prompt_config_version TEXT NOT NULL,
    application_version TEXT NOT NULL,
    git_commit TEXT,
    collector_contract_version TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    safe_config_json TEXT NOT NULL,
    runtime_seconds REAL,
    llm_calls INTEGER,
    tool_calls INTEGER,
    tokens_in INTEGER,
    tokens_out INTEGER,
    reasoning_tokens INTEGER,
    metrics_json TEXT,
    UNIQUE (opportunity_key, attempt_number)
);
```

The full quote sets, raw Portfolio Manager result, snapshots, context
artifacts, and execution invariant remain in `shadow_decisions`; the run table
stores only joinable temporal/status summaries and safe numeric provenance.
The `source_run_id` unique constraint is the crash-reconciliation anchor.

`safe_config_json` is built from an allow-list of non-secret effective values:
provider/model names, analyst set, profile, debate/risk round counts,
temperature/max-token settings, prompt/collector versions, and relevant
forex-safe flags. API keys, authorization headers, and credential-bearing URL
query values are never included. A missing `git_commit` is stored as
`unknown`, not guessed.

### 11.4 Retention and indexes

Phase 6 does not delete decision, outcome, opportunity, or run evidence. Add
indexes for `(status, eligible_after)`, `(requested_symbol, anchor_timestamp)`,
`(run_status, heartbeat_at)`, and `(decision_id)` to keep polling and status
queries bounded. Operational event logs may be rotated outside the evidence
database; the skip/opportunity row itself is retained as the canonical audit
fact.

## 12. Phase 5 integration

### 12.1 Recommended placement

The watcher owns an `OutcomeCoordinator` in the same process because it can
reuse the same database path, clock, terminal path, and lifecycle policy. It
calls `ShadowOutcomeEvaluator.evaluate_pending()` only when:

- the evaluation interval has elapsed;
- no analysis worker is active (if one is active, it records
  `evaluation_due_pending=1` instead); and
- the collector lease is still owned.

The coordinator never calls the graph or an LLM. If a long analysis is active,
the evaluator waits without opening a historical MT5 read; immediately after
the runner returns and releases its MT5 session, the coordinator runs the
evaluator before any next probe or runner. A shared injected MT5 operation gate
asserts that probe, runner, and evaluator calls cannot overlap. There is no
second MT5 reader competing with the runner in the default single-slot process.

The existing `forex-evaluate` command remains the independent recovery/manual
path. It can evaluate a database after the watcher is stopped or after a crash;
it must refuse with `WATCHER_ALREADY_RUNNING` when a non-expired watcher lease
owns that database, so an external evaluator cannot race the active runner.
Both paths use the same Phase 5 store/upsert rules;
duplicate evaluation reads are idempotent, and terminal rows cannot be
overwritten.

### 12.2 Maturity and evidence semantics

The coordinator passes the injected UTC clock to the evaluator. The evaluator
continues to create rows for both bases and all configured horizons, leaves
immature rows `PENDING`, accepts the first valid quote in the closed tolerance
window only after that window matures, and preserves
`DATA_UNAVAILABLE -> COMPLETE` recovery. It does not change
`future_evaluation_status` in `shadow_decisions`.

The watcher summary must show the bases separately. For example:

```text
decision A: ANALYSIS_SNAPSHOT 5m=COMPLETE, 15m=COMPLETE,
            30m=PENDING, 60m=PENDING
later:      ANALYSIS_SNAPSHOT 30m=COMPLETE, 60m=COMPLETE
```

`DECISION_REFERENCE` can be independently `DATA_UNAVAILABLE` when the fresh
post-completion quote was missing or temporally invalid. It is never silently
replaced with the analysis snapshot.

### 12.3 Quality and training separation

The watcher reports, but does not assign:

- source decision context (`COMPLETE`/`INCOMPLETE`);
- strict normalization (`NORMALIZED`/`FAILED`);
- evaluation status/data availability;
- temporal staleness indicators;
- training eligibility.

`training_eligible` remains `NULL`/deferred under the Phase 5 contract.
`COMPLETE` outcomes are evidence, not labels that a future corpus builder must
accept. Slow decisions can be useful for offline signal-quality research and
simultaneously be ineligible as execution-quality evidence.

## 13. Model, configuration, and provenance

The watcher reads provider/model settings through `DEFAULT_CONFIG` and the
existing `TRADINGAGENTS_*` environment override mechanism. Non-secret CLI
schedule overrides are merged after environment configuration; credentials
remain runtime-only in the provider/SDK environment.

Every `forex_watch_runs` row records:

- `llm_provider`, `quick_model`, `deep_model` from the persisted decision or
  effective runner config;
- ordered analyst set and `INTRADAY`/other validated forex profile;
- `prompt_config_version` and `collector_contract_version`;
- safe config fingerprint and redacted safe config JSON;
- application package version and optional source commit;
- runner start/completion/runtime and analysis/reference timestamps/delay;
- callback-derived LLM/tool/token/reasoning counts when available;
- decision context, normalization, action, resolved symbol, and decision ID.

The config fingerprint is stable for the same safe effective configuration and
is used in the opportunity key. A model or analyst-set change is therefore a
new intentional opportunity version. A code revision change requires a
prompt/workflow contract version bump when it changes behavior; the commit is
still retained for audit.

No provenance field includes API keys, account passwords, authorization
headers, full credential-bearing URLs, prompts, completions, or hidden
reasoning. The existing persisted raw Portfolio Manager result remains the
normal structured result required by Phase 4; the watcher does not add a
second copy to logs.

## 14. Staleness and temporal evidence

Phase 6 preserves the Phase 5 distinction without claiming live performance:

| Evidence | Meaning |
|---|---|
| `analysis_snapshot_timestamp` + analysis quote | Market state actually supplied to the graph |
| `decision_completed_timestamp` | Application UTC time at which structured PM output became available |
| `analysis_latency_seconds` | Completion minus analysis snapshot; raw reasoning latency |
| `decision_reference_timestamp` + reference quote | Actual broker tick time from the post-completion read |
| `decision_reference_delay_seconds` | Broker reference timestamp minus application completion |
| `stale_by_completion` | Optional configured freshness indicator, not a training/live label |

The default `freshness_budget_seconds` is `900` for an operational warning;
the value is configurable and versioned. `stale_by_completion=1` means only
that analysis latency exceeded this declared budget. It does not invalidate the
analysis-snapshot signal-quality outcome, and it does not make a stale quote an
executable entry. The raw timestamps and quote sets remain authoritative for
future research.

If the broker reference timestamp is older than completion, Phase 5 retains it
as `INVALID_TEMPORAL`; the watcher reports that status and never marks it as a
fresh/actionable entry. If the reference quote is unavailable, the watcher
does not manufacture a local completion quote.

## 15. Market availability and symbol handling

`ReadOnlyMarketProbe` is intentionally small:

1. construct the configured `MT5Provider`;
2. initialize and verify connected account/terminal;
3. resolve the requested symbol with exact-first, ambiguity-safe rules;
4. call the read-only tick API and require finite `bid`, `ask`, and a UTC
   timestamp with `ask >= bid`;
5. shut down the probe provider in `finally`.

The probe does not call `get_market_snapshot`, so the runner still owns the
single cached analysis snapshot. It does not infer market-open status from a
calendar. A failed probe produces an explicit skip/failure reason and no LLM
construction. A successful probe is not persisted as the analysis quote; the
runner's snapshot and post-completion broker reference remain the evidence.

The scheduler accepts multiple symbols in the schema, but the default config
contains only `EURUSD`. The single current-candidate selection plus
`max_concurrent_analyses=1` enforces one analysis slot; if an operator opts into
multiple symbols, the policy selects one current candidate per slot and marks
other simultaneous candidates explicitly skipped rather than creating a
queue. There is no separate per-interval symbol-count setting in Phase 6 v1.
Multi-symbol fairness is therefore observable but not a Phase 6 throughput
guarantee.

## 16. Configuration schema and defaults

The future implementation should expose a typed `WatcherConfig` validated at
startup. Recommended defaults are:

| Field | Default | Rules/meaning |
|---|---:|---|
| `symbols` | `("EURUSD",)` | non-empty, normalized forex requests |
| `analysis_profile` | `INTRADAY` | existing forex profile resolver |
| `analysts` | `("market", "news")` | exact forex-safe allow-list; no fundamentals/social |
| `schedule_timeframe` | `M15` | one of `M5`, `M15`, `H1` |
| `poll_interval_seconds` | `15` | positive; injected scheduler in tests |
| `bar_close_settle_seconds` | `30` | non-negative |
| `cooldown_seconds` | `60` | starts after runner completion |
| `max_concurrent_analyses` | `1` | Phase 6 v1 rejects values other than one; the only analysis slot |
| `analysis_timeout_seconds` | `7200` | soft alert/classification threshold; no thread kill |
| `lease_heartbeat_seconds` | `15` | positive and less than lease TTL |
| `lease_ttl_seconds` | `9000` | must exceed heartbeat and deployment timeout policy |
| `evaluation_interval_seconds` | `60` | zero-LLM evaluator cadence |
| `evaluation_enabled` | `True` | false only to run a collector without in-process evaluation |
| `max_attempts_per_opportunity` | `1` | prevents duplicate decisions/backlog |
| `max_recovery_buckets` | `8` | bounded restart catch-up |
| `freshness_budget_seconds` | `900` | staleness warning only; not a label |
| `max_consecutive_mt5_failures` | `3` | opens MT5 circuit |
| `max_consecutive_analysis_failures` | `3` | opens analysis circuit |
| `max_consecutive_incomplete` | `5` | opens quality circuit |
| `max_consecutive_normalization_failures` | `3` | opens normalization circuit |
| `max_consecutive_runtime_exceeded` | `2` | opens slow-analysis circuit |
| `circuit_breaker_cooldown_seconds` | `1800` | degraded state duration before probe |
| `db_busy_timeout_seconds` | `5` | bounded SQLite lock wait |
| `db_path` | configured `data_cache_dir/shadow_decisions.db` | same path as shadow/evaluator stores |
| `terminal_path` | `None` | optional local MT5 executable path |

`WatcherConfig` must reject booleans where integers are required, negative
intervals, unsupported analysts/timeframes, zero/negative timeouts, and
credential-bearing CLI fields. It should expose a safe canonical mapping for
the config fingerprint. Existing provider/model environment variables remain
the source of truth for LLM selection.

The `lease_ttl_seconds` default is intentionally longer than the normal poll
interval and the 7,200-second soft timeout (with deployment grace), and can be
raised when a deployment chooses a larger soft timeout. A heartbeat thread
keeps a live process from being mistaken for a crashed one; the lease still
expires after a true process failure.

## 17. CLI design

Add a separate `forex-watch` console script. Do not add subcommands or options
to `forex-shadow` or the stock `tradingagents` CLI.

### 17.1 `forex-watch run`

Long-running foreground collector. It prints prominently before any external
connection:

```text
MT5 FOREX — SHADOW COLLECTOR
NO ORDER WILL BE SENT
```

Recommended options:

```text
forex-watch run
    [--db-path PATH]
    [--terminal-path PATH]
    [--symbols EURUSD[,USDJPY...]]
    [--analysis-profile INTRADAY]
    [--analysts market,news]
    [--schedule-timeframe M15]
    [--poll-interval-seconds N]
    [--cooldown-seconds N]
    [--analysis-timeout-seconds N]
    [--evaluation-interval-seconds N]
    [--no-evaluate]
    [--json-logs]
```

Schedule/runtime options are non-secret. Provider/model values continue to be
read from the existing environment/config system; the watcher does not accept
API keys, secrets, order flags, or stock-specific options.

### 17.2 `forex-watch once`

Perform one scheduling decision at the current injected/real UTC time:

- observe the latest eligible completed bar;
- run at most one current analysis if all gates pass;
- record an explicit skip otherwise;
- perform one optional due evaluation pass;
- exit after the active worker completes.

It does not wait for a future bar and never replays old opportunities. This is
the deterministic smoke/deployment-canary entry point.

### 17.3 `forex-watch status`

Read-only SQLite summary; no MT5 or LLM connection by default. It reports:

- lifecycle/lease state, owner activity, current run/opportunity, last loop,
  next eligible time, circuit reason;
- total decisions and context-complete/incomplete counts;
- normalized BUY/SELL/HOLD counts and normalization failures;
- average/min/max analysis latency and count over the freshness budget;
- skipped opportunities by reason and failed/abandoned runs;
- per-basis/per-horizon evaluation counts for `PENDING`, `COMPLETE`,
  `DATA_UNAVAILABLE`, and `INELIGIBLE`;
- whether a due evaluation is pending behind an active analysis;
- fully terminal outcome rows/decision IDs, without calling them training
  examples;
- last evaluation status/error and zero LLM calls for evaluation.

`--json` returns stable field names suitable for a monitor. `--probe` is an
explicit optional read-only MT5 availability check and is not part of the
default status command.

### 17.4 `forex-watch evaluate`

Invoke the existing `ShadowOutcomeEvaluator` for pending decisions. It prints
the Phase 5 read-only banner, reports zero LLM calls, preserves separate
analysis/reference bases, and returns nonzero on provider/configuration
failure. It first verifies that no non-expired watcher lease owns the database;
otherwise it returns `WATCHER_ALREADY_RUNNING` without opening MT5. It does not
start `ForexShadowRunner` or create a new decision.

## 18. Public interfaces

The following are design-level contracts for the future implementation; they
are not code in this phase.

```python
class WatcherConfig: ...

class Clock(Protocol):
    def now(self) -> datetime: ...  # timezone-aware UTC
    def monotonic(self) -> float: ...

class SchedulePolicy(Protocol):
    def opportunities_between(
        self, previous: datetime | None, now: datetime
    ) -> tuple[ScheduledOpportunity, ...]: ...

class ReadOnlyMarketProbe(Protocol):
    def probe(self, requested_symbol: str) -> MarketProbeResult: ...

class Mt5OperationGate(Protocol):
    def acquire(self, operation: str) -> ContextManager[None]: ...

class ShadowRunner(Protocol):
    def run(
        self, *, symbol: str, source_run_id: str, ...
    ) -> ForexShadowRunResult: ...

class OutcomeEvaluator(Protocol):
    def evaluate_pending(self, *, now: datetime, terminal_path: str | None): ...

class WatcherStore:
    def acquire_lease(self, owner: LeaseOwner, now: datetime) -> LeaseResult: ...
    def heartbeat(self, owner_token: str, now: datetime) -> None: ...
    def mark_evaluation_due(self, owner_token: str, now: datetime) -> None: ...
    def clear_evaluation_due(
        self, owner_token: str, now: datetime, status: str
    ) -> None: ...
    def observe_opportunity(self, opportunity: ScheduledOpportunity, now: datetime): ...
    def claim_opportunity(self, key: str, owner_token: str, now: datetime) -> WatchRun: ...
    def record_skip(self, key: str, reason: SkipReason, now: datetime, detail: str | None): ...
    def finalize_run(self, run_id: str, owner_token: str, result: RunEvidence): ...
    def reconcile_stale_runs(self, owner_token: str, now: datetime): ...
    def summary(self, now: datetime) -> Mapping[str, Any]: ...

class WatcherCoordinator:
    def run_forever(self) -> None: ...
    def run_once(self, *, now: datetime | None = None) -> WatchCycleResult: ...
    def request_shutdown(self) -> None: ...
```

The actual implementation may use dataclasses and concrete classes, but these
boundaries must remain injectable. `Clock`, `SchedulePolicy`, runner,
probe, evaluator, executor, and store fakes are required for deterministic
tests. No public interface includes an order or mutation method.

The runner extension is:

```python
ForexShadowRunner.run(..., source_run_id: str | None = None)
```

`None` preserves the current random UUID behavior. A watcher-provided value is
validated as a non-empty opaque ID and persisted unchanged in
`ShadowTradeDecision.source_run_id` for crash reconciliation.

## 19. Observability and metrics

Each lifecycle event is a structured record with an allow-listed shape:

```json
{
  "event": "analysis_finished",
  "event_timestamp": "2026-09-09T12:41:02Z",
  "opportunity_key": "...",
  "run_id": "...",
  "decision_id": "...",
  "requested_symbol": "EURUSD",
  "resolved_symbol": "EURUSDm",
  "status": "DECISION_SAVED",
  "decision_context_status": "COMPLETE",
  "normalization_status": "NORMALIZED",
  "action": "HOLD",
  "analysis_latency_seconds": 2462.1,
  "decision_reference_delay_seconds": 0.8,
  "stale_by_completion": true,
  "llm_calls": 18,
  "tokens_in": 12345,
  "tokens_out": 2345,
  "reasoning_tokens": 0
}
```

Allowed events include `WATCHER_STARTED`, `LEASE_RECOVERED`,
`OPPORTUNITY_OBSERVED`, `OPPORTUNITY_SKIPPED`, `ANALYSIS_STARTED`,
`ANALYSIS_TIMEOUT_OBSERVED`, `ANALYSIS_FINISHED`, `ANALYSIS_FAILED`,
`EVALUATION_FINISHED`, `EVALUATION_FAILED`, `CIRCUIT_OPENED`,
`SHUTDOWN_REQUESTED`, and `WATCHER_STOPPED`.

The watcher logs IDs, statuses, counts, sizes, timings, and sanitized error
codes only. It does not print the raw Portfolio Manager result, debate text,
prompts, completions, credentials, or private reasoning. The persisted
decision continues to contain the genuine structured PM result required by
Phase 4, but operational logs do not duplicate it.

## 20. Circuit breakers and degradation

Counters are maintained in the singleton state with a rolling/reset policy:

- `MT5_UNAVAILABLE`/symbol/quote failures increment the MT5 counter;
- runner/graph/LLM exceptions increment the analysis counter;
- persisted `INCOMPLETE` context increments the context counter;
- `normalization_status=FAILED` increments the normalization counter;
- a run finishing beyond the soft timeout increments the runtime counter.

When a configured threshold is reached, lifecycle state becomes `DEGRADED`,
the reason is persisted, and no new analysis is submitted until
`circuit_breaker_cooldown_seconds` elapses and the relevant read-only health
probe succeeds. When no runner is active, due Phase 5 evaluation may continue
because it makes zero LLM calls and does not create decisions. It is still
serialized behind the runner and never runs during `ANALYZING`. A healthy
successful run resets only the corresponding counter; unrelated failures
remain visible.

The soft analysis timeout is deliberately non-destructive. The heartbeat
records the alert while the existing runner continues, then the coordinator
classifies a persisted result as `SUCCEEDED_SLOW` or a failed/hung process as
`ABANDONED` after restart. A future hard timeout would require an isolated
worker/process contract and is outside Phase 6; it must not be smuggled into
the watcher as an unsafe thread kill.

## 21. Dataset-quality summary

The status query must keep these concepts separate:

```text
decisions_total
decision_context_complete
decision_context_incomplete
normalized_buy / normalized_sell / normalized_hold
normalization_failed
average_analysis_latency_seconds
stale_by_completion_count
evaluations_by_basis_and_horizon_and_status
fully_terminal_outcome_decision_count
skipped_opportunities_by_reason
failed_runs
abandoned_runs
watcher_lifecycle_status
current_run_id / current_opportunity_key
evaluation_due_pending
last_evaluation_status
```

“Fully terminal outcome” means the configured rows are no longer pending; it
does not mean all are complete, executable, profitable, or training eligible.
An `INELIGIBLE` or `DATA_UNAVAILABLE` row remains visible in the summary.

## 22. Testability and required test matrix

All normal tests use fakes and an injected clock. They never wait real hours,
call `sleep()` to advance time, invoke Qwen/Ollama, or require MT5. The
foreground loop advances by calling `tick()`/`run_once(now=...)`; the analysis
executor uses a deterministic fake future in unit tests.

Required deterministic coverage:

| # | Test | Expected proof |
|---:|---|---|
| 1 | Normal scheduled decision | One current EURUSD opportunity is claimed, runner called once, decision linked and saved |
| 2 | Overlapping opportunity | A new bar while the slot is active becomes `SKIPPED / ANALYSIS_ALREADY_RUNNING` |
| 3 | No unlimited backlog | Hundreds of clock ticks create bounded unique skip rows, no queued runner tasks, and capped restart catch-up |
| 4 | Duplicate opportunity | Same canonical key is ignored across polls/restarts/config object recreation |
| 5 | Runner exception | Run becomes `FAILED`, slot/lease recover, no duplicate retry of the old key |
| 6 | MT5 unavailable | Probe fails, opportunity is explicitly skipped, no graph/LLM call occurs |
| 7 | Ollama/LLM failure | Runner error is recorded with sanitized code; circuit threshold/degraded state works |
| 8 | Incomplete result | Persisted `decision_context_status=INCOMPLETE` remains visible and is not relabeled complete |
| 9 | Complete result | Genuine decision ID, action, provenance, and `executed=False` are linked |
| 10 | Evaluation maturity | Fake clock advances 5m/15m/30m/60m rows from pending to separate basis statuses |
| 11 | Data-unavailable recovery | Existing Phase 5 evaluator performs `DATA_UNAVAILABLE -> COMPLETE`; watcher does not overwrite it |
| 12 | Clean restart | STOPPED state/lease can be reacquired without duplicate opportunity |
| 13 | Crash/stale RUNNING and lease takeover | A non-expired lease always returns `WATCHER_ALREADY_RUNNING`; after expiry, a provably live exact old process returns `WATCHER_OPERATOR_REVIEW_REQUIRED`, while a dead/unverifiable owner is reconciled and persisted decision is linked or run becomes `ABANDONED` |
| 14 | Status summary | Counts actions, context, latency, skips, evaluation statuses, and current state accurately |
| 15 | EURUSD default | Absent symbol config yields exactly `('EURUSD',)` and `INTRADAY`, `market/news` |
| 16 | Provenance | Provider/models/config fingerprint/analysts/profile/versions/tokens are safe and joinable |
| 17 | Zero execution | Public API/source AST scan finds no order/mutation surface; every decision has `executed=False` |
| 18 | Stock CLI unchanged | Existing stock entry point/help/regression tests remain unchanged; only new forex script is registered |
| 19 | Phase 5 reuse and MT5 serialization | Injected evaluator is called only when no runner is active, a due evaluation is retained during `ANALYZING`, it runs after runner completion before the next probe, and a gate/fake proves no MT5 overlap; watcher source contains no copied bid/ask/MFE/MAE formulas |
| 20 | No real-time sleep | Fake clock/scheduler completes all lifecycle tests without wall-clock waiting |

Additional tests should cover lease acquisition races, database busy retry,
owner-token fencing, the strict non-expired-lease rule, expired exact-process
operator review, process-liveness/expiry paths, multiple symbols with one
slot, soft-timeout classification, graceful shutdown while active, evaluation
due during `ANALYZING`, no probe/evaluator/provider creation during an active
runner, immediate post-run evaluation ordering, malformed configuration,
unavailable reference quotes, and redaction of credential-like configuration
values.

An opt-in integration test may run one bounded local MT5 probe and/or one
historical Phase 5 evaluation, asserting UTC timestamps, symbol resolution,
zero mutation methods, and unchanged positions/orders. It must not run a
40-minute LLM workflow as a required CI test.

## 23. Safety analysis

The safety argument is structural and testable:

1. The watcher calls only `ReadOnlyMarketProbe`, `ForexShadowRunner`, and
   `ShadowOutcomeEvaluator`; it has no order client or execution callback.
2. MT5 access remains behind `MT5Provider`/`MT5ToolAdapter`, whose existing
   read-only validator rejects forbidden tool names. The new evaluator/probe
   code does not import `MetaTrader5` directly.
3. `ShadowTradeDecision.__post_init__` and the SQLite `executed=0` check remain
   authoritative. The watcher verifies the invariant before marking a run
   successful.
4. A schedule skip, lease transition, or circuit breaker cannot create a
   position; it only writes SQLite metadata.
5. Provider/LLM failures fail closed: no quote is fabricated, no action is
   guessed, no stale opportunity is replayed, and no order is attempted.
6. The CLI has no order flags, credentials, or stock-specific registration;
   the stock CLI file and behavior remain outside the Phase 6 file map.
7. Static tests scan the proposed production files for forbidden mutation
   definitions/imports, and runtime tests assert forbidden methods are absent
   from provider/runner/evaluator components.
8. Evaluation is zero-LLM and historical-read-only; it cannot trigger a graph
   run or execution path.

The watcher can be stopped, restarted, or degraded without changing the
meaning of already persisted COMPLETE/INELIGIBLE evidence.

## 24. File map for a future implementation

This Phase 6 design commit creates only this specification. The following is
the proposed implementation file map for a later, separately approved plan.

### New files

- `tradingagents/forex/watch_store.py` — SQLite schema, lease, opportunity/run
  transitions, reconciliation, summary queries.
- `tradingagents/forex/watcher.py` — `WatcherConfig`, schedule policy,
  coordinator, single-slot executor, heartbeat, circuit breakers, and
  outcome coordination.
- `cli/forex_watch.py` — separate `forex-watch` subcommand CLI.
- `tests/test_forex_watch_store.py` — schema, dedup, lease, and recovery tests.
- `tests/test_forex_watcher.py` — deterministic scheduler/coordinator tests.
- `tests/test_forex_watch_cli.py` — CLI/options/banner/status tests.
- `tests/test_forex_watch_integration.py` — mocked bounded integration and
  opt-in local MT5 read-only checks.

### Existing files proposed for minimal modification

- `tradingagents/forex/runner.py` — optional `source_run_id` keyword seam only;
  no graph/prompt/execution behavior change.
- `tradingagents/forex/shadow.py` — read-only
  `find_by_source_run_id()` query seam for crash reconciliation only; no
  mutation behavior change.
- `tradingagents/forex/__init__.py` — re-export watcher contracts if useful;
  no stock imports or registration.
- `pyproject.toml` — add only
  `forex-watch = "cli.forex_watch:main"` under project scripts.
- `docs/forex-shadow.md` — document the separate watcher, state/skip semantics,
  status command, and safety banner.

No implementation plan, migration script, execution adapter, or Phase 6 code
is part of this design commit.

## 25. Known limitations

- A local Qwen run taking about 40 minutes means a one-slot collector may
  produce only one or two decisions per hour and will intentionally skip many
  M15 bars. This is expected evidence, not a scheduling bug.
- The default soft timeout observes and degrades after slow completion; it does
  not forcibly kill a stuck runner. An operator may need to terminate a hung
  process, after which stale-run recovery applies.
- SQLite lease/fencing is designed for one host/database. It is not a
  distributed HA coordinator and does not promise multi-machine failover.
- MT5 availability, broker sessions, quote gaps, and historical retention can
  leave outcomes pending or data unavailable. Phase 5 recovery remains
  explicit and auditable.
- UTC bucket alignment is intentionally simple; it is not a broker-calendar or
  economic-event engine.
- Multi-symbol configuration is schema-ready but throughput/fairness under one
  slot is deliberately limited.
- The database grows with decisions, outcomes, skips, and run metadata. Phase 6
  does not silently delete evidence; retention/archival needs a later policy.
- The collector does not strengthen the forex macro/news layer; it uses the
  existing forex-safe Phase 4 analyst configuration.
- No outcome row is a profitability guarantee, executable trade record, or
  training label.

## 26. Alternatives rejected

### Fixed hourly cron as the whole design

Rejected because it cannot express bar-close context, produces ambiguous
behavior when a 40-minute run overlaps the next timer, and encourages stale
backlog replay. The hybrid policy can still be externally woken hourly, but
the persisted opportunity remains a completed-bar key with a single active
slot.

### Completion-driven scheduling without bar identity

Rejected as the sole policy because every decision would drift with model
latency and comparisons would not have stable market anchors. Completion and
cooldown remain gates inside the recommended hybrid.

### A full market-event engine

Rejected for Phase 6 because it adds broker/session/calendar complexity without
solving the single slow model bottleneck. Deterministic UTC bucket polling is
enough to record meaningful M15 opportunities and skips.

### Unlimited in-process queue or thread pool

Rejected because it turns slow inference into stale historical analyses,
consumes memory, and makes duplicate/restart behavior difficult to prove.

### A separate worker process with hard termination

Deferred. Process isolation could enforce a hard timeout, but it introduces
orphan-worker handling, MT5 lifecycle cleanup, Windows process-group behavior,
and cross-process result reconciliation beyond the current need. Phase 6 uses a
single explicit worker slot and a non-destructive soft timeout; a future hard
timeout must be a separately designed safety boundary.

### Redis/Kafka/Postgres

Rejected because SQLite already stores decisions/evaluations, the first
deployment is one host/terminal, and the required lease/dedup workload is
small. A database migration would add operational failure modes without
improving the shadow-only proof.

### A second forex graph or custom evaluator

Rejected because it would split provenance and risk semantic divergence from
the accepted `ForexShadowRunner` and Phase 5 formulas.

### Modifying the stock CLI or overloading `forex-shadow`

Rejected to preserve upstream TradingAgents compatibility, keep the shadow
boundary obvious, and allow future `forex-demo`/`forex-live` discussions to
remain separate (with execution still unavailable here).

## 27. Open questions for review

These are the remaining policy choices worth confirming before an
implementation plan; the recommended defaults above are complete and
internally consistent without them.

1. **Schedule anchor:** approve `M15` as the default, or choose `M5`/`H1` for
   the first operational collector. `M15` is recommended because it balances
   meaningful context with the current 40-minute runtime.
2. **Timeout policy:** approve a non-destructive soft timeout as Phase 6 v1,
   or require a separately supervised worker-process design before collection.
   Soft timeout is recommended to avoid unsafe thread/process termination.
3. **Deployment host:** run the foreground CLI under Windows Task Scheduler,
   a service wrapper, or an operator terminal. The design is host-neutral but
   must retain one database lease owner.
4. **Evidence retention:** choose a future archival/backup policy for long-run
   skip/run metadata. Phase 6 intentionally keeps evidence and does not
   silently prune it.
5. **Version bump policy:** confirm who increments `prompt_config_version` and
   `collector_contract_version` when prompts, graph shape, or safe config
   changes. The design requires an explicit bump rather than silently changing
   the dedup identity.
6. **Evaluation cadence:** the recommended in-process interval is 60 seconds;
   operators may prefer to disable it and run `forex-evaluate` separately when
   MT5 historical reads should be isolated from collection.

## 28. Acceptance criteria for the future implementation

Phase 6 implementation should not begin until this design is approved and a
separate implementation plan exists. That later implementation is acceptable
only if it demonstrates:

- one active runner slot and no pending analysis queue;
- explicit skipped rows for opportunities observed during an active run;
- deterministic key/claim deduplication across polls and restarts;
- stale lease/run reconciliation without duplicate decisions;
- current EURUSD/INTRADAY/market-news defaults;
- real runner/evaluator reuse with one cached snapshot and zero evaluator LLM
  calls;
- safe provider/model/config provenance without secrets;
- separate analysis/reference temporal evidence and staleness metrics;
- visible incomplete/normalization/evaluation statuses with no training labels;
- all Phase 5 status recovery and math delegated to the existing evaluator;
- deterministic tests for all twenty required behaviors;
- full suite, Ruff, compile, diff, and mutation-safety verification;
- no modification to the stock CLI and no execution surface.

This document is the complete Phase 6 design. It deliberately stops before an
implementation plan and before any Phase 6 code.
