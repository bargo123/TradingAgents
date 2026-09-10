# Phase 6.3 Qwen3.5 Prose Reliability Benchmark

Status: **complete**. This report covers the Phase 6.3 reliability investigation only; Phase 7 was not started.

Scope: standalone prose-node calls only; no MT5, graph execution, order path, Phase 5 evaluation, or production routing changes.

## Baseline and safety boundary

- Accepted Phase 6.2 baseline SHA: `e4d6c545c323d79e9b26ec2aab7bb21084667b35`.
- Checkout before Phase 6.3 work: `13ea69709eef1b2b6ff638394569b893023a9cd7`.
- The final completion SHA is recorded by the closing `git rev-parse HEAD` check and handoff.
- Changed implementation files are limited to the standalone benchmark utility, its deterministic tests, and Phase 6.3 documentation.
- The existing stock `tradingagents` CLI, MT5 provider, Phase 6 database, prompts, production model routing, context-integrity rules, and execution APIs were not changed.

## Production-equivalent configuration

- Provider: `ollama`
- Quick model: `qwen3.5:2b`
- Backend URL: `http://localhost:11434/v1`
- Quick thinking control: `False`
- Temperature: `0.1`
- Max tokens: `1024`
- Command: `scripts\benchmark_qwen_prose.py --calls-per-agent 20 --model qwen3.5:2b --output data_cache\phase6-3-qwen3.5-2b.jsonl --summary docs\superpowers\reports\2026-09-09-phase-6-3-qwen-reliability.md`

Ollama availability was checked before the run: version `0.33.3`; `qwen3.5:2b` and `qwen3.5:4b` were present in `/api/tags`; `/v1/models` returned HTTP 200. The existing LangChain/OpenAI-compatible client path was used. Each call used a fresh deterministic forex state, an existing full-shape prose-agent factory, and one shared client; calls were strictly sequential. Only scalar metadata was retained (lengths, booleans, finish/usage fields, elapsed time, exception type, and transport status). No prompt, completion, or reasoning text was printed or persisted.

## Per-agent results

| Agent | Calls | GOOD | EMPTY | LABEL_ONLY | ERROR | TRUNCATED | Reliability | Median chars | Min/Max chars | Median sec | P95 sec | Mean input tokens | Mean output tokens | Reasoning present |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Bull Researcher | 20 | 15 | 0 | 1 | 0 | 4 | 75.0% | 4467.5 | 0/5978 | 383.353 | 438.382 | 344.0 | 3368.65 | 0 |
| Bear Researcher | 20 | 10 | 0 | 6 | 0 | 4 | 50.0% | 4150.0 | 0/4949 | 418.542 | 529.091 | 345.0 | 3432.4 | 0 |
| Aggressive Risk Analyst | 20 | 4 | 0 | 10 | 0 | 6 | 20.0% | 448.0 | 0/4768 | 407.912 | 526.266 | 376.0 | 3628.4 | 0 |
| Neutral Risk Analyst | 20 | 16 | 0 | 2 | 0 | 2 | 80.0% | 4784.0 | 0/5511 | 316.516 | 403.611 | 373.0 | 3053.45 | 0 |

## Overall

- Calls: `80`
- GOOD: `45`
- EMPTY: `0`
- LABEL_ONLY: `19`
- ERROR: `0`
- TRUNCATED (evidence-backed only): `16`
- Reliability: `56.25%`
- Median content chars: `4229.5`
- Content min/max chars: `0/5978`
- Median/P95 runtime seconds: `401.847/521.383`
- Mean input/output tokens: `359.5/3370.725`
- Reasoning present count / total chars: `0/0`
- Total measured runtime: `31,260.330` seconds (`521.01` minutes; `8.683` hours).

Only scalar metadata is retained. Prompts, completions, and private reasoning text are not stored.

## Failure evidence (metadata only)

`LABEL_ONLY` means the node wrapper contained only its normal analyst label while the raw `response.content` was empty. `TRUNCATED` is reported only where an explicit length finish reason and matching output-token evidence were present. Iteration numbers are listed without response text:

| Agent | LABEL_ONLY iterations | TRUNCATED iterations |
| --- | --- | --- |
| Bull Researcher | 9 | 3, 5, 8, 13 |
| Bear Researcher | 2, 5, 7, 14, 16, 17 | 4, 8, 13, 15 |
| Aggressive Risk Analyst | 1, 2, 3, 5, 9, 11, 12, 13, 15, 20 | 4, 6, 8, 10, 17, 18 |
| Neutral Risk Analyst | 11, 18 | 9, 14 |

No `EMPTY` or `ERROR` records occurred. All 80 calls reported successful transport.

## Decision-rule result

The Phase 6.2 transport investigation already showed that correctly disabled-thinking OpenAI-compatible/native calls can return visible content, so no transport fix was justified here. The 2B production-equivalent run nevertheless reproduced the Phase 6 symptom as a node/output reliability problem: 19 wrapper-only results and 16 evidence-backed truncations across 80 calls, with only 45 substantive visible reports. This is not sufficient to certify the 2B quick model for a complete forex workflow.

The prescribed material-failure comparison was run only for Bear and Aggressive Risk (the agents with repeated `LABEL_ONLY` outcomes), using `qwen3.5:4b`, 10 sequential calls each, the same fixture and client settings. The separate scalar report is `2026-09-09-phase-6-3-qwen-4b-comparison.md`.

| Agent | 2B GOOD / 20 | 4B GOOD / 10 | 4B result |
| --- | ---: | ---: | --- |
| Bear Researcher | 10 / 20 | 9 / 10 | 90.0% GOOD; 1 LABEL_ONLY |
| Aggressive Risk Analyst | 4 / 20 | 1 / 10 | 10.0% GOOD; 7 LABEL_ONLY; 2 TRUNCATED |

The comparison therefore improves Bear substantially but does not resolve Aggressive Risk. No model switch, retry, prompt edit, or routing change was made. The existing `INCOMPLETE` quarantine/context-integrity boundary remains the safe disposition for affected real runs. Any future retry/format/model experiment is outside Phase 6.3.

## Verification

- Focused Phase 6.3/model-routing/Phase 4–6 tests: **170 passed**, 3 warnings.
- Full suite: **907 passed, 6 skipped**, 18 warnings, 71 subtests passed.
- Ruff: **all checks passed**.
- `compileall`: exit 0.
- `git diff --check`: clean.
- No MT5 calls or MT5 provider sessions, graph constructor, LLM graph run, database writes, or mutation/execution APIs were used by the benchmark; the only external service contacted was the configured local Ollama endpoint.

Stop condition satisfied: Phase 6.3 is complete; do not start Phase 7.
