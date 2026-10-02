# HFT-Grade DEMO/Research Engine Design

## Status

Proposed design for review. This document authorizes neither implementation nor
broker activity. The current repository remains unchanged except for this
design document.

## 1. Intent and reality boundary

The objective is to make the deterministic Phase 12 tick path materially faster,
more measurable, and more faithful between replay and DEMO operation. The
result will be an HFT-grade *DEMO/research engine* relative to this project;
it must not be described as institutional or exchange-colocated HFT.

The MT5 terminal, retail broker feed, Windows host, Python adapter, and network
round trips impose a latency floor that this repository cannot remove. The
engine therefore reports measured ingress and broker latency separately from
the local deterministic-core latency and never claims a latency guarantee it
has not measured.

## 2. Goals

- Build one deterministic hot-path core for replay, research, shadow, and
  explicitly enabled DEMO operation.
- Remove SQLite writes, LLM calls, slow account reads, and avoidable allocation
  from the per-tick decision path.
- Preserve causal ordering, broker timestamps, monotonic latency measurements,
  and no-lookahead replay parity.
- Provide bounded ingestion, explicit backpressure/drop accounting, and
  fail-closed behavior when the engine cannot keep up safely.
- Preserve the existing verified DEMO gateway as the only broker mutation seam.
- Keep the strategic LLM supervisor off the tick path; it may publish only an
  expiring, validated regime/plan snapshot.
- Make performance, fill, rejection, queue, and recovery behavior auditable
  from scalar telemetry without storing prompts or hidden reasoning.

## 3. Non-goals and hard boundaries

- No real-money or live-account execution. `LIVE`, `REAL`, and `CONTEST`
  execution modes remain unavailable.
- No automatic strategy generation, parameter optimization, training,
  fine-tuning, RAG, Phase 7 work, Phase 11A work, or Phase 8/9/10 data mixing.
- No change to the stock `tradingagents` CLI or the strategic agent graph.
- No synthetic trades, canaries, forced entries, threshold weakening,
  martingale, grid, averaging down, or trade quotas.
- No LLM call, HTTP call, database write, account refresh, or broker history
  query on the hot tick path.
- No assumption that MT5 polling is an exchange-native event feed or that a
  50 ms poll interval makes the system institutional HFT.
- No direct `order_send` call outside
  `VerifiedDemoExecutionGateway`.
- No destructive rewrite of existing Phase 5/6/7/11A artifacts or shadow
  databases.

## 4. Existing baseline

The current Phase 12D-oriented implementation already contains deterministic
features, momentum/range strategies, arbitration, risk, DEMO reconciliation,
leases, and a verified gateway. The current DEMO runtime still performs
provider reads and persistence around the tick loop and can be configured with
an approximately 50 ms poll interval. The new design treats those components
as compatibility seams and moves the deterministic computation behind a
bounded event pipeline rather than duplicating strategy or risk semantics.

## 5. Approaches considered

### Approach A — optimize the existing Python loop

Keep one Python worker and reduce allocations, cache account data, batch
SQLite writes, and tighten the polling loop. This is the lowest-risk change and
is useful as a reference implementation, but the MT5/Python boundary and
database coupling continue to limit throughput and jitter.

### Approach B — hybrid native hot core (selected)

Keep MT5 ingress, leases, orchestration, and DEMO gateway in Python, while
placing the causal feature/strategy/arbitration/risk core behind a stable
native-accelerator interface. A pure-Python reference implementation remains
mandatory for deterministic tests and replay. The DEMO/research runtime uses
the native implementation only when its build and contract fingerprint are
verified; it never silently falls back while claiming HFT-grade operation.

This gives a safe migration path, preserves current behavior for replay, and
removes high-frequency computation and persistence from the slow control plane.

### Approach C — venue-native institutional rewrite

Replace MT5 and the Python broker boundary with a direct venue/ECN feed,
colocated Linux service, native order gateway, and exchange-specific risk
controls. This is the route to institutional HFT but is outside this
repository and cannot be honestly delivered through the current MT5 terminal.

## 6. Selected architecture

```text
MT5 adapter / replay reader
        |
        v
NormalizedTickIngress -- bounded ring --> HotPathCore
        |                                  |
        |                                  +--> Decision/Intent events
        |                                  |        |
        |                                  |        +--> async telemetry WAL
        |                                  |        +--> DEMO gateway (explicit)
        |                                  |
        +--> ingress/drop/clock telemetry  +--> position/risk state

Strategic watcher/LLM -- validated expiring regime --> atomic regime store
```

The ingress adapter is the only component allowed to interact with MT5 for
market data. The hot core consumes normalized ticks and an immutable snapshot
of the latest validated strategic regime. It runs features, both current
strategy families, arbitration, spread/cost gates, deterministic risk, and
position transitions in memory. It emits typed events; it does not write to
SQLite or call an LLM.

The asynchronous telemetry writer owns a separate SQLite connection and WAL
queue. Critical DEMO intents are durably recorded before a broker mutation is
allowed, and the gateway result is durably acknowledged before the intent is
considered complete. If that durability barrier cannot be satisfied, the
gateway call is blocked.

## 7. Runtime modes

The engine has an explicit mode enum:

- `RESEARCH`: immutable replay or synthetic fixture data, no MT5 and no
  mutation.
- `SHADOW`: live read-only MT5 ticks, simulated fills, `executed=False`.
- `DEMO`: live MT5 ticks plus the existing verified DEMO gateway, with
  authoritative DEMO-account checks immediately before every mutation.

There is no `LIVE` mode. `RESEARCH` is the default for benchmarks and tests.
`DEMO` requires the existing watcher/HFT ownership, reconciliation-clean state,
account trade mode equal to the authoritative DEMO constant, fresh ticks,
valid stops/volume/fill mode, and a healthy risk circuit.

## 8. Package and interface boundaries

The implementation will preserve the current modules and add narrow seams:

- `tradingagents/forex/hft/ingress.py`: normalized tick acquisition,
  sequencing, broker-clock validation, and bounded ring-buffer admission.
- `tradingagents/forex/hft/core.py`: reference hot-path contract that composes
  the existing feature engine, strategies, arbiter, regime checks, and risk
  decisions without I/O.
- `tradingagents/forex/hft/native/`: optional native accelerator implementing
  the same versioned core contract and fingerprinting its build.
- `tradingagents/forex/hft/events.py`: scalar-only ingress, decision, risk,
  intent, fill, drop, and latency event records.
- `tradingagents/forex/hft/persistence.py`: bounded asynchronous WAL writer,
  flush barriers, queue metrics, and crash recovery.
- `tradingagents/forex/hft/runtime.py` and `demo_runtime.py`: orchestration,
  mode selection, leases, provider calls, and gateway handoff only.
- `tradingagents/forex/hft/replay.py`: immutable replay reader using the same
  core contract and event semantics.
- `cli/forex_hft.py`: an additive engine CLI for `benchmark`, `replay`,
  `shadow`, and explicitly gated `demo`; the existing stock CLI is untouched.

The existing `demo_gateway.py`, `demo_store.py`, `regime_store.py`, and lease
APIs remain authoritative. No second execution gateway or lease implementation
is introduced.

## 9. Hot-path contracts

Every input tick contains symbol, broker timestamp, bid, ask, point, and a
monotonic ingress sequence. Broker timestamps determine causal ordering;
`time.perf_counter_ns()` values measure local latency and are never used as
market timestamps.

The core consumes:

- the current tick;
- causal rolling feature state;
- one immutable validated regime/plan snapshot;
- current position state and risk state.

The core returns a typed decision with action, strategy family, reason code,
signal strength, position transition, risk decision, and event sequence. It
cannot create an order request directly. An execution intent is created only
after all deterministic gates pass and is then checked again by the DEMO
gateway.

The contract is versioned and includes the strategy/config fingerprint,
feature-window configuration, risk configuration, and native/reference
implementation identifier. A mismatch invalidates the run rather than mixing
semantics.

## 10. Ingress and backpressure

MT5 ingress uses the existing serialized operation gate and the smallest
available normalized tick API. Replay may read immutable batches, but it feeds
the same one-tick core contract. Every tick is classified as accepted,
duplicate, out-of-order, stale, provider-busy, invalid, or queue-dropped.

The queue is bounded. In `RESEARCH`, an explicit drop policy may skip ticks
while recording sequence gaps and preserving deterministic report semantics. In
`SHADOW` and `DEMO`, a queue overflow, clock anomaly, or missing critical tick
causes a fail-closed pause rather than silently trading on an unknown state.
Position-management ticks have priority over new-entry evaluation.

## 11. Strategic regime boundary

The strategic watcher continues to produce validated, provenance-bearing,
time-bounded regime/plan snapshots. The hot core reads an atomic immutable
snapshot and never waits for an LLM. Missing, expired, mismatched, or
untrusted regime state produces `NO_VALID_STRATEGIC_STATE` and blocks new
entries. Existing open-position exit and reconciliation safety remains
independent and fail closed.

The strategic refresh cadence, model routing, and prompts are outside this
design. The only new contract is that a regime snapshot is an input artifact,
not a per-tick call dependency.

## 12. Execution and DEMO safety

The hot core emits `ExecutionIntent` only. In `RESEARCH` and `SHADOW`, the
intent is recorded as non-executed. In `DEMO`, the existing
`VerifiedDemoExecutionGateway` performs the final checks inside the serialized
MT5 operation:

- authoritative account trade mode is DEMO;
- provider and broker clock are healthy;
- symbol, tick freshness, volume, stop/freeze levels, and fill mode are valid;
- no conflicting owned position/order or reconciliation uncertainty exists;
- watcher and HFT leases still belong to this owner;
- risk circuit is READY;
- durable intent persistence succeeded.

Any check failure blocks the mutation and records a bounded error category.
No trade is manufactured to demonstrate activity.

## 13. Persistence and recovery

Hot-path state is held in memory and periodically checkpointed. The async
writer batches non-critical tick/features telemetry and writes critical intent,
broker result, position, and lease events with ordering keys. SQLite uses WAL
mode and one writer connection; readers use separate read-only connections.

Each event has run ID, sequence, broker timestamp, local monotonic timing,
implementation/config fingerprint, and scalar reason/status fields. Prompts,
completions, hidden reasoning, credentials, and raw broker payloads are not
persisted by this engine.

On restart, the owner first acquires the existing lease, verifies account and
reconciliation state, restores only an unambiguous owned position, and resumes
from a new run ID. An ambiguous broker/local state blocks DEMO operation and
requires the existing ownership-aware reconciliation path.

## 14. Replay parity and research artifacts

Replay uses the same normalized tick, core, risk, and event contracts as
SHADOW/DEMO. It rejects non-monotonic timestamps, future-derived features,
missing point information, and incompatible fingerprints. Reports include
sequence gaps, queue/drop behavior, p50/p95/p99/max core latency, end-to-end
latency, fills, costs, spread, slippage assumptions, drawdown, and risk
rejections.

Replay artifacts are separate from Phase 5/6 shadow decisions, Phase 7
Knowledge artifacts, and Phase 11A datasets. No corpus, experience, or model
training consumer is added by this design.

## 15. Measurement and performance targets

Latency is measured with monotonic nanoseconds at ingress, after feature
calculation, after strategy/arbitration, after risk, before gateway submission,
and after the broker response. Broker/MT5 wait time is reported separately from
core compute time.

Initial acceptance targets for the selected machine are measured targets, not
claims about live-market latency:

- native deterministic core p99 at or below 1 ms on the fixed replay fixture;
- reference Python core p99 at or below 5 ms on the same fixture;
- zero unclassified drops and zero causal-order violations;
- bounded queue depth with every intentional drop/rejection classified;
- end-to-end MT5 latency reported separately and never conflated with core
  latency.

If the native target is not met, the run is reported as a research engine and
does not receive an HFT-grade benchmark label. No threshold is weakened to
make the metric pass.

## 16. Configuration

All performance and safety settings are explicit and persisted in the run
fingerprint:

- mode and symbol;
- ingress/queue capacity and overflow policy;
- feature windows and point/digit policy;
- core implementation (`reference` or verified native build fingerprint);
- persistence batch size and flush barrier timeout;
- strategy, cost, stop, risk, and account limits;
- broker poll/read parameters;
- lease and reconnect settings.

The current `poll_interval_seconds`, risk limits, account checks, and gateway
controls remain compatible. A native core is never enabled implicitly by an
unverified build.

## 17. CLI and operational boundary

Additive commands are isolated under `forex_hft`; the stock `tradingagents`
CLI remains unchanged. The CLI prints the selected mode, implementation
fingerprint, queue policy, and execution safety status. `benchmark` and
`replay` never construct MT5 or an execution gateway. `shadow` refuses an
active watcher lease before provider construction. `demo` requires explicit
flags and the existing owner/lease/reconciliation checks.

No command starts a long indexing, training, RAG, or evaluation workflow.

## 18. Failure behavior

Failures are typed and visible:

- invalid or stale tick: reject and record;
- provider busy/disconnected: bounded recovery, then fail closed;
- queue overflow: pause DEMO/SHADOW and record;
- expired/missing regime: block entries;
- persistence barrier failure: block broker mutation;
- native contract/fingerprint mismatch: refuse native mode;
- account mode, lease, or reconciliation uncertainty: `OPERATOR_REVIEW_REQUIRED`;
- broker rejection: record exact retcode/category and leave position state
  reconciled before resuming.

No exception path converts a failure into BUY/SELL/HOLD or into a successful
fill.

## 19. Testing strategy

Deterministic unit and integration tests will cover:

- causal features and no-lookahead parity;
- duplicate/out-of-order/stale/clock-invalid ticks;
- ring-buffer capacity and drop policy;
- reference/native contract equivalence and fingerprint mismatch;
- strategy arbitration, regime expiry, cost gates, and risk limits;
- async persistence ordering, flush barriers, crash recovery, and bounded
  queue telemetry;
- DEMO gateway account/fill/stop/lease/reconciliation safety;
- replay/benchmark determinism and latency metadata;
- CLI mode isolation and no-MT5/no-order guarantees for benchmark/replay;
- stock CLI, Phase 7, Phase 11A, and existing shadow regression tests.

CI remains offline: no Ollama, MT5, network, cloud GPU, or real broker calls.
Real DEMO validation, if separately authorized later, is bounded and must show
positions/orders before and after plus `real_money=false`.

## 20. Rollout and compatibility

The first implementation keeps the current reference Python path available and
adds the native path behind an explicit mode. Existing databases are read
without destructive migration; new HFT-grade runs use a versioned artifact
root. Existing valid decisions and reconciliation records are preserved.

The current supervisor remains the owner of strategic and HFT leases. A second
supervisor, worker, or provider is refused before MT5 construction. Clean
shutdown releases only the exact owned leases; crash recovery remains TTL and
identity based.

## 21. Future extension seams

The interfaces leave room for broker market-depth adapters, additional
strategies, a stronger native implementation, alternative durable stores, and
venue-specific execution gateways. Each extension must preserve the typed core
contract, provenance fingerprint, causality, and DEMO/live safety boundary.

## 22. Explicit Phase 7/11A and later-phase handoff

This design does not read, rewrite, embed, retrieve, distill, or train on books,
Phase 7 artifacts, Phase 10 datasets, or Phase 11A lessons. It does not alter
the current Phase 7 frozen generation or the empty/fail-closed Phase 10
eligibility result. Any future experience/training integration must be a
separate approved design with an explicit provenance boundary.

## 23. Acceptance definition

The HFT-grade DEMO/research engine is accepted only when all of the following
are demonstrated in tests and bounded validation:

1. Reference and native cores produce equivalent typed decisions for the same
   deterministic fixture.
2. Replay and live-adapter event ordering is causal and fingerprinted.
3. No LLM, SQLite write, account refresh, or slow MT5 operation occurs in the
   core tick function.
4. Queue saturation, drops, pauses, and recovery are visible and bounded.
5. Native/reference latency targets are measured and reported separately from
   MT5/broker latency.
6. DEMO mutations, when explicitly enabled, pass through exactly one verified
   gateway and remain impossible for non-DEMO accounts.
7. Existing Phase 5/6/7/11A artifacts, decisions, and stock CLI behavior are
   unchanged.

Until these gates pass, the system must be described as a deterministic MT5
DEMO/research prototype rather than true HFT.
