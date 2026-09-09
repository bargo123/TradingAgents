# Phase 6.2 Ollama/Qwen3.5 Transport Investigation Report

Date: 2026-09-09
Baseline SHA: `e4d6c545c323d79e9b26ec2aab7bb21084667b35`
Final SHA: documentation-only handoff commit reported with this artifact
Scope: isolated qwen3.5:2b calls only; no complete TradingAgents graph
Phase 7: not started

## Result

The requested transport defect was **not reproduced**, so no production
provider/configuration fix was made and no model or prompt was changed.

The current forex quick path is:

```text
TradingAgentsGraph._get_provider_kwargs(role="quick")
  -> {"extra_body": {"think": false}, ...}
  -> create_llm_client("ollama", ...)
  -> OpenAIClient
  -> NormalizedChatOpenAI (langchain_openai.ChatOpenAI)
  -> http://localhost:11434/v1/chat/completions
```

The installed versions are:

| Component | Version / value |
| --- | --- |
| Ollama executable and `/api/version` | 0.33.3 |
| `langchain-openai` | 1.6.0 |
| `openai` | 3.8.0 |
| `langchain-core` | 1.6.2 |
| `httpx` | 0.28.1 |
| Ollama models available | `qwen3.5:2b`, `qwen3.5:4b` |
| Current `.env` provider | `ollama` |
| Current quick model | `qwen3.5:2b` |
| Current deep model | `qwen3.5:4b` |
| Current backend URL | `http://localhost:11434/v1` |
| Current quick/deep thinking config | `false` / `true` |
| Current temperature / max tokens | `0.1` / `1024` |

## Wire-shape evidence

An `httpx.MockTransport` captured the actual request produced by the installed
OpenAI client. Although LangChain’s intermediate payload contains
`extra_body={"think": false}`, the OpenAI SDK flattens that field into the
top-level request body sent to Ollama:

```text
wire URL: http://localhost:11434/v1/chat/completions
wire keys: messages, model, stream, think
wire think: false
wire extra_body: absent
```

The same capture with the deep setting produced top-level `think: true`. A
direct `NormalizedChatOpenAI` capture also showed that
`reasoning_effort="none"` is a top-level request field when supplied directly.
The repository’s `OpenAIClient` intentionally filters `reasoning_effort` for
non-OpenAI model IDs; the B probe below therefore used the proper raw
OpenAI-compatible request field rather than changing that provider policy.

## Isolated probes

Each probe used qwen3.5:2b and the same short Bull-researcher-like synthetic
prompt. Only scalar metadata and lengths were emitted; no response prose or
reasoning text was printed or persisted.

| Probe | HTTP/client path and control | HTTP status | visible content chars | reasoning/thinking chars exposed | finish | token/eval counts | elapsed |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| A current production | `ChatOpenAI` → `/v1/chat/completions`; current `extra_body.think=false`, temperature 0.1, max tokens 1024 | 200 | 1,272 | 0 | stop | prompt 58; completion 1,606; total 1,664 | 176.353 s |
| B explicit disable | `/v1/chat/completions`; top-level `reasoning_effort="none"`, temperature 0.1, max tokens 1024 | 200 | 1,534 | 0 | stop | prompt 60; completion 277; total 337 | 31.575 s |
| C native disable | `/api/chat`; top-level `think=false`, temperature 0.1, `num_predict=1024` | 200 | 1,689 | 0 | stop | prompt eval 60; eval 319 | 36.052 s |

As a separate control demonstrating Qwen’s thinking-field behavior, native
`/api/chat` with `think=true` returned `content_len=0`,
`thinking_len=379`, `done_reason=length`, and `eval_count=128` in 16.256 s.
This confirms that Ollama/Qwen can expose thinking separately when thinking is
enabled; it does not demonstrate that the current production-shaped request
enables thinking.

## Strict classification

The observations do not satisfy Case 1: A did not have zero visible content
with positive exposed reasoning, and both properly disabled controls B and C
returned visible content. Case 2 also does not apply because the disabled
variants succeeded. No Case 3 endpoint/version incompatibility was observed:
Ollama 0.33.3 accepted both `/v1/chat/completions` and native `/api/chat`
controls.

Therefore this bounded test provides no evidence for a transport/thinking
control defect. The Phase 6 empty-node symptom remains unreproduced under the
isolated prompt and should not be “fixed” by copying hidden reasoning into
`response.content` or by switching models. The real Phase 6 row remains
subject to the existing context-integrity safeguards.

## Verification

No MT5 provider was initialized and no complete graph or collector was run.
No orders or positions were read or mutated by this investigation.

```text
focused transport/model-routing/Phase 4-6 suite: 285 passed, 3 skipped
full pytest suite: 898 passed, 6 skipped, 18 warnings, 71 subtests
Ruff: passed
compileall: passed
git diff --check: passed
```

The three skipped focused integrations require `RUN_MT5_INTEGRATION=1`; the
six full-suite skips are the repository’s existing optional-provider/MT5
guards. Since no transport defect was proven, the conditional RED regression,
production fix, and four-call post-fix smoke were correctly skipped.

Phase 6.2 stops here. Phase 7, model changes, prompt changes, full-graph
reruns, execution, and MT5 mutation remain out of scope.
