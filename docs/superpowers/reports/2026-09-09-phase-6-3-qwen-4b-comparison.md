# Phase 6.3 Qwen3.5 Prose Reliability Benchmark

Status: optional material-failure comparison completed; no production model or routing change was made.

Scope: standalone prose-node calls only; no MT5, graph execution, order path, Phase 5 evaluation, or Phase 7.

Accepted Phase 6.2 baseline SHA: `e4d6c545c323d79e9b26ec2aab7bb21084667b35`.

## Production-equivalent configuration

- Provider: `ollama`
- Quick model: `qwen3.5:4b`
- Backend URL: `http://localhost:11434/v1`
- Quick thinking control: `False`
- Temperature: `0.1`
- Max tokens: `1024`
- Command: `scripts/benchmark_qwen_prose.py --calls-per-agent 10 --agents bear,aggressive --model qwen3.5:4b --output data_cache\phase6-3-qwen3.5-4b-failing.jsonl --summary docs\superpowers\reports\2026-09-09-phase-6-3-qwen-4b-comparison.md`

The OpenAI-compatible responses exposed input/output token usage; no Ollama `eval_count` field was available through this `/v1` path, so no eval count is claimed.

## Per-agent results

| Agent | Calls | GOOD | EMPTY | LABEL_ONLY | ERROR | TRUNCATED | Reliability | Median chars | Min/Max chars | Median sec | P95 sec | Mean input tokens | Mean output tokens | Reasoning present |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Bear Researcher | 10 | 9 | 0 | 1 | 0 | 0 | 90.0% | 3945.5 | 0/4499 | 452.731 | 586.122 | 345.0 | 2904.2 | 0 |
| Aggressive Risk Analyst | 10 | 1 | 0 | 7 | 0 | 2 | 10.0% | 0.0 | 0/3042 | 598.194 | 776.315 | 376.0 | 3710.6 | 0 |

## Overall

- Calls: `20`
- GOOD: `10`
- EMPTY: `0`
- LABEL_ONLY: `8`
- ERROR: `0`
- TRUNCATED (evidence-backed only): `2`
- Reliability: `50.0%`
- Median content chars: `2946.5`
- Content min/max chars: `0/4499`
- Median/P95 runtime seconds: `586.809/770.222`
- Mean input/output tokens: `360.5/3307.4`
- Reasoning present count / total chars: `0/0`
- Total measured runtime: `11,176.711` seconds (`186.28` minutes; `3.105` hours).

Only scalar metadata is retained. Prompts, completions, and private reasoning text are not stored.

This comparison was limited to Bear Researcher and Aggressive Risk because those agents had material repeated `LABEL_ONLY` results in the 2B sample. It is diagnostic evidence only; it does not authorize a model switch or alter production routing.
