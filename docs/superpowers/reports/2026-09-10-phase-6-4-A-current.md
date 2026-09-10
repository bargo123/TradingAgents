# Phase 6.3 Qwen3.5 Prose Reliability Benchmark

Scope: standalone prose-node calls only; no MT5, graph, execution, or Phase 7.

## Production-equivalent configuration

- Provider: `ollama`
- Quick model: `qwen3.5:2b`
- Backend URL: `http://localhost:11434/v1`
- Quick thinking control: `False`
- Temperature: `0.1`
- Max tokens: `1024`
- Command: `scripts/benchmark_qwen_prose.py --calls-per-agent 3 --agents bear,aggressive --model qwen3.5:2b --output data_cache\phase6-4-A-current.jsonl --summary docs\superpowers\reports\2026-09-10-phase-6-4-A-current.md`

Wire controls captured with MockTransport: `think=false`; `reasoning_effort` absent; `max_completion_tokens=1024`; `max_tokens` absent; `temperature=0.1`; `model=qwen3.5:2b`.

## Per-agent results

| Agent | Calls | GOOD | EMPTY | LABEL_ONLY | ERROR | TRUNCATED | Reliability | Median chars | Min/Max chars | Median sec | P95 sec | Mean input tokens | Mean output tokens | Reasoning present |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Bear Researcher | 3 | 2 | 0 | 1 | 0 | 0 | 66.67% | 4693 | 0/6083 | 475.922 | 512.075 | 345.0 | 3329.667 | 0 |
| Aggressive Risk Analyst | 3 | 1 | 0 | 1 | 0 | 1 | 33.33% | 4124 | 0/4515 | 438.807 | 467.399 | 376.0 | 3415.333 | 0 |

## Overall

- Calls: `6`
- GOOD: `3`
- EMPTY: `0`
- LABEL_ONLY: `2`
- ERROR: `0`
- TRUNCATED (evidence-backed only): `1`
- Reliability: `50.0%`
- Median content chars: `4319.5`
- Content min/max chars: `0/6083`
- Median/P95 runtime seconds: `454.692/506.05`
- Mean input/output tokens: `360.5/3372.5`
- Reasoning present count / total chars: `0/0`

Only scalar metadata is retained. Prompts, completions, and private reasoning text are not stored.
