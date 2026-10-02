# Phase 14 — Autonomous Self-Enhancement Engine Design

## Status and scope

Phase 14 adds an offline, fail-closed learning plane around the existing
deterministic HFT engine. The live DEMO trading plane remains versioned,
deterministic, low-latency, and free of LLM/RAG calls. Phase 7 knowledge,
Phase 8 experience, Phase 10 datasets, and Phase 11A artifacts are read-only
inputs; no existing artifact is rewritten or rebuilt.

The first implementation surface is exit-policy research for
`range_rejection` and `momentum_continuation`. Candidate generation is bounded
to typed configuration and cannot mutate source code or live configuration.

## Architecture

```text
read-only HFT/DEMO ledger + Phase 8 provenance + Phase 7 query
                         |
                 Phase 14 ExperienceImporter
                         |
                 immutable Phase 14 catalog
                         |
            bounded CandidateGenerator / BookStrategyFactory
                         |
                 causal ReplayEvaluator
          development -> validation -> unseen holdout
                         |
          walk-forward + cost sensitivity + quality gates
                         |
          CandidatePromotionGate -> registry decision
                         |
       SHADOW_CHALLENGER -> DEMO_CHALLENGER -> promotion
                         |
                    rollback package
```

All artifacts live under a dedicated Phase 14 root. Experiment writes are
transactional and idempotent. A candidate is only a proposal until every
deterministic gate passes; `APPROVED`/`INCUMBENT` state is never written by an
LLM.

## Module boundaries

- `tradingagents/self_enhancement/models.py`: immutable typed contracts,
  bounded mutation specifications, versions, metrics, and state enums.
- `store.py`: versioned SQLite catalog for experiences, experiments,
  candidates, evaluations, deployments, rollbacks, and triggers.
- `quality.py`: read-only market-data quality audit and minimum-evidence checks.
- `importer.py`: query-only import from validated DEMO/HFT ledgers; excludes
  test-only, synthetic, real-money, uncertain, and incomplete observations.
- `candidates.py`: deterministic bounded candidate generation and optional
  book-evidence adapter; no source-code generation.
- `replay.py`: causal chronological replay using the existing HFT feature,
  strategy, risk, fill, and account primitives.
- `evaluation.py`: metrics, development/validation/unseen evaluation,
  walk-forward aggregation, and cost sensitivity.
- `promotion.py`: multi-metric sample, safety, robustness, and provenance
  gates plus immutable promotion/rollback records.
- `orchestrator.py`: explicit trigger/schedule state machine and crash-safe
  experiment lifecycle; it never runs on the tick path.
- `cli.py`: explicit `self-enhancement status`, `import`, `experiment`, and
  `rollback` commands. Existing stock/forex trading CLIs are unchanged.

## Data and provenance

The Phase 14 catalog is separate from Phase 8 and Phase 10. Each imported
trade stores strategy/config/source commit, timestamps, bid/ask/fill,
spread/slippage, feature snapshot, regime, expected move, risk, MFE/MAE,
exit reason, holding time, gross/net-known result, cost-unknown markers,
session/volatility, execution mode, and source fingerprints. Test-only
canaries and records with reconciliation uncertainty are quarantined, never
counted as verified experience.

Each experiment stores immutable parent/candidate versions, hypothesis,
parameters, source/book evidence, chronological split fingerprints, data
quality, code commit, metrics, decision, rejection reason, deployment and
rollback history. Book claims carry Phase 7 `KnowledgeHit` provenance and are
never copied into the live tick path.

## Bounded mutation surface

Automatic mutation is restricted to explicitly declared parameters with type,
minimum, maximum, default, and safe-range metadata. V1 exposes exit-policy
parameters only: profit-target fraction, protection-arm fraction, giveback
fraction, micro-reversal fraction, no-progress cap, and maximum duration.
Unknown keys, out-of-range values, schema changes, and source-code edits fail
closed.

## Replay and evaluation

Replay consumes monotonic, immutable ticks and only features available at or
before each tick. Splits are chronological DEVELOPMENT, VALIDATION, and
UNSEEN_HOLDOUT with persisted fingerprints and a guard gap. Candidate
selection uses development/validation only; the unseen set is read once per
experiment and is never tuned. Walk-forward windows and cost sensitivities
use real bid/ask, measured slippage, latency, and explicit unknown-commission
scenarios. No zero-cost result is eligible for promotion.

Metrics include trade count, trades/hour, win rate, average win/loss,
expectancy, profit factor, drawdown, return, duration, MFE/MAE, capture ratio,
profit-to-loss flips, spread/slippage, long/short, session/regime, exit
reasons, and broker rejection rate. Metrics remain descriptive; no isolated
win or loss changes configuration.

## Promotion, challengers, and rollback

Promotion requires minimum trades, independent periods, validation and unseen
acceptance, cost sensitivity, no safety/runtime regression, and Pareto-safe
multi-metric comparison to the incumbent. A candidate progresses through
`EXPERIMENTAL`, `REPLAYED`, `VALIDATED`, `UNSEEN_PASSED`,
`SHADOW_CHALLENGER`, `DEMO_CHALLENGER`, and only then `APPROVED`/`INCUMBENT`.
Shadow challengers never call MT5 mutation APIs. DEMO challengers are
explicitly gated and isolated from incumbent positions. Every deployment has
a rollback package; deterministic hard triggers disable the challenger and
restore the previous approved configuration transactionally.

## Failure and crash behavior

Insufficient evidence, missing costs, data gaps, stale ticks, duplicate or
out-of-order data, reconciliation uncertainty, schema/runtime errors, and
conflicting evidence produce quarantine or `NO_EXPERIMENT`, never promotion.
Restart recovery reconciles `RUNNING` experiments into a deterministic
`ABORTED_RECOVERABLE` state. A half-written promotion is impossible because
catalog state and deployment metadata commit in one transaction.

## Safety boundaries

Phase 14 never imports MT5 provider objects into replay, never calls
`order_send`, never enables REAL/CONTEST/UNKNOWN execution, never runs an LLM
or RAG query on a tick, and never rewrites Phase 7/8/10/11A artifacts. The
live HFT worker does not wait for or depend on Phase 14.

## First experiment

The first autonomous experiment compares the incumbent exit policy with a
small deterministic set: baseline, expected-move take-profit, MFE giveback,
micro-reversal, trailing, and no-progress variants. It runs once against the
available verified snapshot, reports all split/cost metrics, and promotes
nothing unless every gate passes. An empty or insufficient source is a valid
fail-closed result.

## Verification

Tests cover chronology and no-lookahead, bounded mutation, source isolation,
provenance, sample gates, metrics, promotion/rejection, challenger/DEMO
separation, rollback, crash recovery, LLM non-authority, and real-money
impossibility. A focused offline experiment is allowed; no live supervisor
restart, MT5 call, Phase 7 rebuild, Phase 11A regeneration, or training run is
part of Phase 14.
