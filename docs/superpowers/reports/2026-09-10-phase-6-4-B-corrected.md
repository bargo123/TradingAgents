# Phase 6.3 Qwen3.5 Prose Reliability Benchmark

Scope: standalone prose-node calls only; no MT5, graph, execution, or Phase 7.

## Production-equivalent configuration

- Provider: `ollama`
- Quick model: `qwen3.5:2b`
- Backend URL: `http://localhost:11434/v1`
- Quick thinking control: `None`
- Temperature: `0.1`
- Max tokens: `1024`
- Command: `scripts/benchmark_qwen_prose.py --calls-per-agent 3 --agents bear,aggressive --model qwen3.5:2b --output data_cache\phase6-4-B-corrected.jsonl --summary docs\superpowers\reports\2026-09-10-phase-6-4-B-corrected.md`

Wire controls captured with MockTransport: `think` absent; `reasoning_effort=none`; `max_tokens=1024`; `max_completion_tokens` absent; `temperature=0.1`; `model=qwen3.5:2b`.

## Per-agent results

| Agent | Calls | GOOD | EMPTY | LABEL_ONLY | ERROR | TRUNCATED | Reliability | Median chars | Min/Max chars | Median sec | P95 sec | Mean input tokens | Mean output tokens | Reasoning present |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Bear Researcher | 3 | 0 | 0 | 0 | 0 | 3 | 0.0% | 4417 | 4326/4594 | 137.278 | 143.76 | 347.0 | 1024.0 | 0 |
| Aggressive Risk Analyst | 3 | 2 | 0 | 0 | 0 | 1 | 66.67% | 4226 | 4210/4379 | 136.238 | 140.913 | 378.0 | 1003.333 | 0 |

## Overall

- Calls: `6`
- GOOD: `2`
- EMPTY: `0`
- LABEL_ONLY: `0`
- ERROR: `0`
- TRUNCATED (evidence-backed only): `4`
- Reliability: `33.33%`
- Median content chars: `4352.5`
- Content min/max chars: `4210/4594`
- Median/P95 runtime seconds: `136.758/143.718`
- Mean input/output tokens: `362.5/1013.667`
- Reasoning present count / total chars: `0/0`

Only scalar metadata is retained. Prompts, completions, and private reasoning text are not stored.
