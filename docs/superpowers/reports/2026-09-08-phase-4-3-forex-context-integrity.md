# Phase 4.3 Forex Context-Integrity Validation Report

Date: 2026-09-08
Scope baseline: `1d55066b7bdb7e3e275730ec26543de1fe5f78d1` (accepted Phase 4.2)
Mode: standalone `forex-shadow`, read-only MT5
Phase 5: not started

## Root cause

The state trace established the intended hand-offs:

```text
Market + News -> investment reports
Bull + Bear -> investment_debate_state
Research Manager -> investment_plan
Trader -> trader_investment_plan
Risk Analysts -> risk_debate_state
Portfolio Manager -> final structured result
```

The first RED test failed at the final forex prompt. The Portfolio Manager was
reading the Research Manager plan, Trader proposal, and risk-debate history,
but it was not reading the already-populated Bull/Bear research-debate fields.
That omission made a valid upstream debate invisible at the final boundary.

The nested `investment_debate_state` and `risk_debate_state` channels are plain
mapping channels, so LangGraph replaces a nested mapping on each write. The
research and risk speakers previously rebuilt their mappings without retaining
fields owned by another speaker (notably prior judge metadata). Each update now
starts from the existing mapping and overwrites only the fields owned by that
node. No report text is fabricated or copied from an unrelated source.

The accepted Phase 4.2 v3 evidence also contains a separate provider-output
fact: Bull/Bear state values contain only their speaker labels, with no visible
report body, and the persisted PM result says the risk debate is absent. Phase
4.3 does not promote hidden model reasoning into a report. The new integrity
check therefore labels that evidence `INCOMPLETE` even though the PM rating is
syntactically normal.

## Implementation

- Forex Portfolio Manager now receives `investment_debate_state.history`,
  `bull_history`, and `bear_history` in addition to the existing Research
  Manager, Trader, and risk context.
- All research/risk nested-state updates preserve the prior mapping before
  applying their normal field updates; stock prompts and graph routing are
  unchanged.
- Metadata-only state boundaries are captured around every forex LLM-bearing
  node. Metrics contain node/phase names and presence/character/count values
  only; no prompts, completions, or private chain-of-thought are retained.
- `ShadowTradeDecision.decision_context_status` accepts `COMPLETE` or
  `INCOMPLETE`, defaults old rows to `INCOMPLETE`, and is persisted in SQLite
  with a migration. It is independent of strict PM normalization, so a
  normalized action cannot hide incomplete context.

## Deterministic compiled-graph trace

The test uses the real forex graph/node factories with deterministic visible
reports and structured stubs. It completed all required nodes with 20
before/after boundaries:

| Node | Artifact evidence |
| --- | --- |
| Market Analyst | market report present, 13 chars |
| News Analyst | news report present, 11 chars |
| Bull Researcher | bull history present, 11 body chars |
| Bear Researcher | bear history present, 11 body chars |
| Research Manager | investment plan present, 84 chars |
| Trader | trader plan present, 84 chars |
| Aggressive Analyst | risk history present, 17 body chars |
| Conservative Analyst | risk history present, 19 body chars |
| Neutral Analyst | risk history present, 14 body chars |
| Portfolio Manager | structured result present, 7 raw fields; 145 rendered chars |

Trace result:

```text
decision_context_status=COMPLETE
normalized action=HOLD (covered by runner/persistence tests)
executed=False
report_text_retained=False
```

The label-only regression test returns `INCOMPLETE` and identifies Bull, Bear,
and the risk debate as missing while retaining their non-zero wrapper sizes.

## Accepted real-run evidence reviewed

The Phase 4.2 v3 EURUSD run executed these nodes:

```text
Market Analyst
News Analyst
Bull Researcher
Bear Researcher
Research Manager
Trader
Aggressive Analyst
Conservative Analyst
Neutral Analyst
Portfolio Manager
```

The existing row contains a genuine structured Portfolio Manager result with
`rating=Hold`, `analysis_profile=INTRADAY`, and `valid_for_seconds=3600`; strict
normalization is `NORMALIZED`, action `HOLD`, and `executed=False`. Under the
Phase 4.3 validator its context status is `INCOMPLETE`: Bull/Bear have no
visible report body and the risk-debate body is absent. Market/News and the
manager/trader node calls are evidenced by the run telemetry, but the old row
does not persist their report bodies; they are not treated as proof of a
complete training-quality decision.

A second local Qwen run was not practical within this bounded task because the
prior full run took approximately 41 minutes. The deterministic compiled
trace is the correctness proof; no MT5 mutation or execution call was made.

## Verification

```text
focused Phase 4.3 + forex regressions: 67 passed
full suite: 780 passed, 4 skipped, 71 subtests passed
Ruff: passed
compileall: passed
git diff --check: passed
MT5 integration guard: passed when run with the existing opt-in guard
```

Phase 5 evaluation/outcome labeling, training, RAG, ONNX, RiskGovernor, and
all order-execution paths remain out of scope.
