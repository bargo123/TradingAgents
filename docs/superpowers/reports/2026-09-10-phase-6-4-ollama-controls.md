# Phase 6.4 Ollama OpenAI-Compatible Control Investigation

Status: **complete**. Phase 7 was not started and no additional 80-call benchmark was run.

## Scope and baseline

- Phase 6.4 baseline SHA: `01d55304028875d60c5c0eea192af8a31a1e5730`.
- Accepted Phase 6.3 baseline: `e4d6c545c323d79e9b26ec2aab7bb21084667b35`.
- Final completion SHA is recorded in the closing handoff.
- No MT5, graph, execution, database, prompt, context-validation, stock-CLI, or model-routing work outside the Ollama-specific provider seam was performed.
- No API key was added, printed, or persisted.

## Environment and method

- Ollama: `0.33.3`.
- Provider: `ollama`.
- Endpoint: `http://localhost:11434/v1/chat/completions`.
- Model: `qwen3.5:2b` only.
- Temperature: `0.1`.
- Configured output limit: `1024`.
- Existing Bear Researcher and Aggressive Risk Analyst factories were invoked with the full-shape deterministic forex fixture used in Phase 6.3.
- A: 3 calls per agent with the pre-fix production request.
- B: 3 calls per agent after the minimal provider/configuration fix.
- Calls were sequential. Only scalar metadata was retained; prompts, completions, and reasoning text were never printed or persisted.

Official Ollama OpenAI-compatibility documentation lists `max_tokens` and `reasoning_effort` (including `"none"`) as supported `/v1/chat/completions` fields: <https://docs.ollama.com/api/openai-compatibility>.

## Exact wire payload (MockTransport)

The mocked request captured only field names and scalar values; `messages` was not printed.

| Field | A: current production request | B: corrected request |
| --- | --- | --- |
| URL | `/v1/chat/completions` | `/v1/chat/completions` |
| `model` | `qwen3.5:2b` | `qwen3.5:2b` |
| `think` | `False` | absent |
| `reasoning_effort` | absent | `"none"` |
| `reasoning` | absent | absent |
| `max_tokens` | absent | `1024` |
| `max_completion_tokens` | `1024` | absent |
| `temperature` | `0.1` | `0.1` |

The exact translation defect was in the installed `langchain-openai 1.6.0` `ChatOpenAI._get_request_payload` implementation (`.venv/Lib/site-packages/langchain_openai/chat_models/base.py`, source line 3682): it rewrites `max_tokens` to `max_completion_tokens`. The existing `OpenAIClient` also gated `reasoning_effort` to native OpenAI model names, so qwen3.5 never received that field.

## Real A/B results

### A — current request before the fix

| Agent | Calls | GOOD | LABEL_ONLY | TRUNCATED | EMPTY | ERROR | Median sec | Mean completion tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bear Researcher | 3 | 2 | 1 | 0 | 0 | 0 | 475.922 | 3329.667 |
| Aggressive Risk Analyst | 3 | 1 | 1 | 1 | 0 | 0 | 438.807 | 3415.333 |
| Total | 6 | 3 | 2 | 1 | 0 | 0 | 454.692 | 3372.5 |

Total measured runtime: `2553.310` seconds (`42.555` minutes). All six transports succeeded.

### B — `reasoning_effort="none"` plus verified `max_tokens=1024`

| Agent | Calls | GOOD | LABEL_ONLY | TRUNCATED | EMPTY | ERROR | Median sec | Mean completion tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bear Researcher | 3 | 0 | 0 | 3 | 0 | 0 | 137.278 | 1024.0 |
| Aggressive Risk Analyst | 3 | 2 | 0 | 1 | 0 | 0 | 136.238 | 1003.333 |
| Total | 6 | 2 | 0 | 4 | 0 | 0 | 136.758 | 1013.667 |

Total measured runtime: `825.189` seconds (`13.753` minutes). All six transports succeeded. Every call returned visible content; no `LABEL_ONLY` result occurred. `TRUNCATED` is evidence-backed by an explicit length finish reason and output-token count at the configured cap.

Relative to A, B reduced total runtime by approximately `67.7%` and mean completion-token usage by approximately `69.9%`, while reducing wrapper-only results from `2/6` to `0/6`.

## Root-cause decision and fix

The expected decision is confirmed: the current Ollama `/v1` thinking/output-control configuration was materially wrong. This is a transport/configuration defect, not evidence that qwen3.5:2b inherently cannot produce visible prose.

Minimal changes:

1. Added `OllamaChatOpenAI`, which restores documented `max_tokens` after the generic LangChain rewrite, for Ollama only.
2. Allowed `reasoning_effort` for the Ollama provider only; native OpenAI model gating remains unchanged.
3. Forex quick no-thinking requests now send `reasoning_effort="none"` instead of `extra_body={"think": false}`.
4. Forex deep thinking remains on the separate `extra_body={"think": true}` path.
5. The standalone benchmark records exact thinking-control and output-limit field/value scalars in future JSONL rows.

No reasoning text was copied into visible content. No prompts were changed. Hosted providers and the stock TradingAgents path remain unchanged by design and regression coverage.

## Verification

- RED tests reproduced the missing `reasoning_effort`, wrong output-limit field, and outdated quick-control expectation.
- Corrected provider/telemetry/wire tests: **22 passed**, 1 existing warning.
- Benchmark contract tests after scalar-record extension: **10 passed**.
- Full repository suite: **912 passed, 6 skipped**, 18 warnings, 71 subtests passed.
- Ruff: **all checks passed**.
- `compileall`: exit 0.
- `git diff --check`: exit 0.
- Final worktree is clean after commit.
- No MT5 access, graph execution, order path, Phase 5 evaluation, or database mutation occurred.

Stop condition satisfied: Phase 6.4 is complete; do not start Phase 7.
