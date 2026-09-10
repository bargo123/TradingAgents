# Phase 6.3 Qwen3.5 Reliability Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure 20 production-equivalent qwen3.5:2b prose invocations for each affected forex agent type and report visible-content reliability without changing runtime behavior.

**Architecture:** A standalone diagnostic module invokes the existing `OpenAIClient`/`NormalizedChatOpenAI` quick client and the existing Bull, Bear, Aggressive Risk, and Neutral Risk node factories against deterministic synthetic forex state. A recording wrapper captures only response metadata and scalar lengths while the real node factories build their normal prompts and extract `response.content`; no `TradingAgentsGraph` constructor, graph compilation, MT5 provider, database, or execution path is used. The CLI writes metadata-only JSONL and a Markdown summary, with an optional 4B comparison only for materially failing 2B agent types.

**Tech Stack:** Python 3, LangChain/OpenAI-compatible Ollama client, pytest, statistics, argparse, JSONL, Ruff, `compileall`, Git.

**Spec:** Phase 6.3 user request (focused qwen3.5:2b reliability benchmark; stop before Phase 7).

**Baseline:** Requested accepted baseline `e4d6c545c323d79e9b26ec2aab7bb21084667b35`; current checkout is `13ea69709eef1b2b6ff638394569b893023a9cd7` and contains only the Phase 6.2 documentation commit beyond that baseline.

## Global Constraints

- Do not run MT5, the complete TradingAgents graph, Phase 5 evaluation, or any execution path.
- Use the existing production quick path and current configuration: Ollama, `qwen3.5:2b`, `http://localhost:11434/v1`, top-level wire `think=false`, temperature `0.1`, max tokens `1024`.
- Use the existing four prose-agent factories and their full forex prompt shapes; do not simplify prompts to one sentence.
- Run 20 sequential calls per agent, 80 total; do not parallelize.
- Do not change prompts, models, routing, provider configuration, response normalization, or context-integrity rules.
- Do not inspect, print, persist, or transmit prompts, completions, or private reasoning text.
- Persist only scalar metadata: agent, iteration, model/config identifiers, lengths/booleans, finish/usage scalars, elapsed time, exception type, and transport status.
- `GOOD` requires substantive `response.content.strip()`; preserve separate `EMPTY`/`LABEL_ONLY`/`ERROR`/evidence-backed `TRUNCATED` classifications.
- Do not implement retries in the benchmark and do not modify the real Phase 6 database.
- Only run a 4B comparison after material 2B failures, using 10 sequential calls per failing agent and otherwise identical inputs/configuration.
- Keep the stock TradingAgents CLI and all production behavior unchanged.

---

### Task 1: Map the existing invocation seams and benchmark inputs

**Files:**
- Inspect: `tradingagents/graph/trading_graph.py:221-273`
- Inspect: `tradingagents/agents/researchers/bull_researcher.py`
- Inspect: `tradingagents/agents/researchers/bear_researcher.py`
- Inspect: `tradingagents/agents/risk_mgmt/aggressive_debator.py`
- Inspect: `tradingagents/agents/risk_mgmt/conservative_debator.py`
- Inspect: `tradingagents/agents/risk_mgmt/neutral_debator.py`
- Inspect: `tradingagents/default_config.py`

**Interfaces:**
- Consumes: existing factory callables and `_get_provider_kwargs(role="quick")`.
- Produces: a deterministic state fixture containing the quote, M1/M5/M15/H1 summaries, market/news reports, trader plan, and prior debate/risk text required by each factory.

- [ ] **Step 1: Confirm the checkout and configuration without secrets**

Run:

```powershell
git rev-parse HEAD
git status --short
```

Record the actual HEAD separately from the user-provided baseline. Load `.env` through the existing package import and expose only provider/model/backend/thinking/temperature/max-token values.

- [ ] **Step 2: Define the fixed forex state fixture**

Use a fresh mapping for each call with stable values such as:

```python
{
    "company_of_interest": "EURUSD",
    "asset_type": "forex",
    "market_data_mode": "forex_mt5",
    "instrument_context": "EURUSD synthetic MT5 context; bid/ask/spread and M1/M5/M15/H1 features",
    "market_context": "EURUSD bid 1.10000 ask 1.10002 spread 0.00002; M1 trend positive; M5 range-bound; M15 support 1.0990; H1 resistance 1.1050",
    "market_report": "Synthetic market report: deterministic EURUSD quote, momentum, volatility, and timeframe features.",
    "sentiment_report": "Unavailable for forex; do not infer social sentiment.",
    "news_report": "Synthetic broad macro report: central-bank policy, inflation, employment, and geopolitical risk context only.",
    "fundamentals_report": "Unavailable for forex; do not infer company fundamentals.",
    "trader_investment_plan": "Synthetic trader plan: observe only; hypothetical directional bias with explicit spread and invalidation controls.",
    "investment_debate_state": {"history": "Bull and bear debate context.", "bull_history": "Bull context.", "bear_history": "Bear context.", "current_response": "Last debate response.", "judge_decision": "", "count": 2},
    "risk_debate_state": {"history": "Aggressive, conservative, and neutral risk debate context.", "aggressive_history": "Aggressive context.", "conservative_history": "Conservative context.", "neutral_history": "Neutral context.", "latest_speaker": "Neutral", "current_aggressive_response": "Last aggressive response.", "current_conservative_response": "Last conservative response.", "current_neutral_response": "Last neutral response.", "judge_decision": "", "count": 3},
}
```

The fixture is prompt-shape context only; it contains no private reasoning or previous real completion.

### Task 2: Build deterministic benchmark contracts and tests

**Files:**
- Create: `scripts/benchmark_qwen_prose.py`
- Test: `tests/test_qwen_prose_benchmark.py`

**Interfaces:**
- `BenchmarkRecord`: immutable metadata record with `agent`, `iteration`, `model`, `thinking`, `content_len`, `content_empty`, `classification`, `finish_reason`, `input_tokens`, `output_tokens`, `elapsed_seconds`, `exception_type`, `transport_ok`, `reasoning_present`, and `reasoning_len`.
- `classify_visible_output(raw_content: object, wrapped_output: object, agent_label: str) -> str` returns exactly `GOOD`, `EMPTY`, `LABEL_ONLY`, `ERROR`, or `TRUNCATED` only when explicit finish evidence proves truncation.
- `summarize_records(records: Sequence[BenchmarkRecord]) -> dict[str, object]` returns counts, reliability percentage, median/min/max content lengths, median/p95 runtime, and mean available input/output tokens without retaining text.
- `build_synthetic_state(agent: str, iteration: int) -> dict[str, object]` returns a fresh deterministic state mapping for one node call.
- `run_agent_call(llm: object, agent: str, iteration: int) -> BenchmarkRecord` invokes the existing factory closure and records only metadata.

- [ ] **Step 1: Write the failing unit tests**

Cover classification/statistics and safety contracts with deterministic fake responses:

```python
def test_empty_response_is_label_only_after_existing_agent_wrapper():
    assert classify_visible_output("", "Bull Analyst: ", "Bull Analyst") == "LABEL_ONLY"

def test_substantive_content_is_good():
    assert classify_visible_output("  visible report  ", "Bull Analyst: visible report", "Bull Analyst") == "GOOD"

def test_truncation_requires_finish_evidence():
    # The classifier must not call ordinary short output truncated.
    assert classify_visible_output("short", "Bull Analyst: short", "Bull Analyst") == "GOOD"

def test_summary_contains_counts_and_percentiles_without_text():
    records = [
        BenchmarkRecord(agent="bull", iteration=1, model="qwen3.5:2b", thinking=False, content_len=24, content_empty=False, classification="GOOD", finish_reason="stop", input_tokens=10, output_tokens=8, elapsed_seconds=1.0, exception_type=None, transport_ok=True, reasoning_present=False, reasoning_len=0),
        BenchmarkRecord(agent="bull", iteration=2, model="qwen3.5:2b", thinking=False, content_len=0, content_empty=True, classification="LABEL_ONLY", finish_reason="length", input_tokens=10, output_tokens=8, elapsed_seconds=2.0, exception_type=None, transport_ok=True, reasoning_present=True, reasoning_len=12),
    ]
    summary = summarize_records(records)
    assert summary["total_calls"] == 2
    assert summary["good_count"] == 1
    assert "prompt" not in summary
    assert "completion" not in summary

def test_fixture_is_fresh_and_forex_safe():
    first = build_synthetic_state("bull", 1)
    second = build_synthetic_state("bull", 1)
    assert first == second and first is not second
    assert first["asset_type"] == "forex"

def test_recording_wrapper_never_persists_response_text():
    record = run_agent_call(FakeLLM("visible"), "bull", 1)
    assert record.content_len == len("visible")
    assert not hasattr(record, "content")
```

- [ ] **Step 2: Run the focused tests and verify the expected RED failure**

Run:

```powershell
.venv\\Scripts\\python.exe -m pytest tests/test_qwen_prose_benchmark.py -q
```

Expected: collection/import or missing-interface failures because the diagnostic module does not yet exist.

- [ ] **Step 3: Implement only the metadata contracts**

Use `time.perf_counter()` around each node invocation. Extract `response.content` only to compute length/emptiness; read finish/usage/reasoning fields only as scalar metadata. Never include response objects, prompts, or text in `BenchmarkRecord` or serialized output. Mark transport success only after a response is returned; record exception class and no exception message body.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same pytest command and require all tests to pass.

- [ ] **Step 5: Commit the utility and tests**

```powershell
git add scripts/benchmark_qwen_prose.py tests/test_qwen_prose_benchmark.py
git commit -m "test: add qwen forex prose reliability benchmark"
```

### Task 3: Add the standalone sequential benchmark CLI

**Files:**
- Modify: `scripts/benchmark_qwen_prose.py`
- Test: `tests/test_qwen_prose_benchmark.py`

**Interfaces:**
- CLI: `python scripts/benchmark_qwen_prose.py --calls-per-agent 20 --model qwen3.5:2b --output data_cache/phase6-3-qwen3.5-2b.jsonl --summary docs/superpowers/reports/2026-09-09-phase-6-3-qwen-reliability.md`
- `run_benchmark(model: str, calls_per_agent: int, output_path: Path | None, summary_path: Path | None) -> list[BenchmarkRecord]` runs agents in this exact order: bull, bear, aggressive, neutral; each sequentially for the requested count.

- [ ] **Step 1: Write failing CLI contract tests**

Assert the runner uses the existing production quick kwargs and factory order with monkeypatched `create_llm_client`, does not construct `TradingAgentsGraph`, performs no parallel scheduling, and serializes JSONL rows with no prompt/completion/reasoning text keys.

- [ ] **Step 2: Run the CLI tests and verify RED**

Run the focused test file and confirm the new CLI contract fails before implementation.

- [ ] **Step 3: Implement the production-equivalent client construction**

Create a bare `TradingAgentsGraph` only for `_get_provider_kwargs(role="quick")` (no constructor, graph, tool nodes, MT5, or LLM graph resources), then call:

```python
client = create_llm_client(
    provider=config["llm_provider"],
    model=model,
    base_url=config.get("backend_url"),
    **quick_kwargs,
)
llm = client.get_llm()
```

Wrap that existing client only to capture scalar response metadata. Invoke the existing `create_bull_researcher`, `create_bear_researcher`, `create_aggressive_debator`, and `create_neutral_debator` closures one at a time with the fixed state. Do not retry errors.

- [ ] **Step 4: Implement metadata-only JSONL and Markdown output**

Write one JSON object per call with the exact record fields; write summaries by agent and overall. Do not write prompts, completions, exception messages, or reasoning text. Flush each JSONL row so an interrupted benchmark leaves a recoverable partial sample.

- [ ] **Step 5: Run CLI contract tests and verify GREEN**

Run the focused test file and require all tests to pass.

- [ ] **Step 6: Commit the CLI**

```powershell
git add scripts/benchmark_qwen_prose.py tests/test_qwen_prose_benchmark.py
git commit -m "feat: add sequential qwen prose benchmark cli"
```

### Task 4: Run the authoritative 80-call 2B benchmark

**Files:**
- Create: `data_cache/phase6-3-qwen3.5-2b.jsonl` (metadata-only, ignored diagnostic artifact)
- Create: `docs/superpowers/reports/2026-09-09-phase-6-3-qwen-reliability.md`

**Interfaces:**
- Consumes: the standalone CLI and local Ollama qwen3.5:2b.
- Produces: 20 records each for Bull, Bear, Aggressive Risk, and Neutral Risk, plus per-agent/overall summaries.

- [ ] **Step 1: Verify provider/model availability without MT5**

Query Ollama `/api/version`, `/api/tags`, and `/v1/models` status only. Fail clearly if qwen3.5:2b is unavailable; do not switch models.

- [ ] **Step 2: Run the sequential benchmark**

Run exactly:

```powershell
.venv\\Scripts\\python.exe scripts/benchmark_qwen_prose.py --calls-per-agent 20 --model qwen3.5:2b --output data_cache/phase6-3-qwen3.5-2b.jsonl --summary docs/superpowers/reports/2026-09-09-phase-6-3-qwen-reliability.md
```

Do not interrupt to inspect model text. The terminal may show only scalar progress rows. No MT5 or graph resources may be created.

- [ ] **Step 3: Review only scalar output**

Check that there are exactly 80 rows, four 20-call groups, no text-bearing keys, and all runtime/token/reasoning fields are scalar. Compute GOOD/EMPTY/LABEL_ONLY/ERROR counts, reliability, content-length and runtime statistics, and reasoning-presence counts.

### Task 5: Apply decision rules without changing production

**Files:**
- Modify: `docs/superpowers/reports/2026-09-09-phase-6-3-qwen-reliability.md`
- Optional create: `data_cache/phase6-3-qwen3.5-4b.jsonl` only when comparison is required

**Interfaces:**
- Consumes: the 2B summary and records.
- Produces: a recommendation and, only for material failures, a 4B comparison.

- [ ] **Step 1: Apply strict thresholds**

If every agent is 20/20 GOOD, recommend retaining 2B and the existing INCOMPLETE quarantine. For rare failures (1–2/20), report reliability and recommend a future bounded-retry design without implementing retries. For material failures, run the comparison in Step 2. Do not label truncation without explicit finish/token evidence.

- [ ] **Step 2: Conditionally run the 4B comparison**

Only if one or more 2B agents have material failures, run exactly 10 sequential calls per failing agent with `qwen3.5:4b` and otherwise identical fixture/configuration. Report reliability and runtime side by side; do not alter production routing.

- [ ] **Step 3: Record whether the Phase 6 symptom reproduced**

State whether any affected agent produced EMPTY/LABEL_ONLY output and include only iteration numbers/counts, never prompt/completion/reasoning text.

### Task 6: Verify and close Phase 6.3

**Files:**
- Inspect: all changed files and generated metadata/report artifacts.

- [ ] **Step 1: Run focused tests and existing regression suites**

Run:

```powershell
.venv\\Scripts\\python.exe -m pytest tests/test_qwen_prose_benchmark.py tests/test_ollama_base_url.py tests/test_openai_reasoning_effort.py tests/test_forex_phase42_telemetry.py tests/test_forex_graph_mode.py tests/test_forex_phase42_context.py tests/test_forex_phase42_prompts.py tests/test_forex_phase43_context_integrity.py tests/test_forex_shadow_cli.py tests/test_forex_shadow_runner.py tests/test_forex_shadow_evaluation.py tests/test_forex_watch_cli.py tests/test_forex_watch_store.py tests/test_forex_watcher.py -q
.venv\\Scripts\\python.exe -m pytest -q
```

- [ ] **Step 2: Run static checks**

```powershell
.venv\\Scripts\\python.exe -m ruff check tradingagents tests cli scripts
.venv\\Scripts\\python.exe -m compileall -q tradingagents cli scripts tests
git diff --check
```

- [ ] **Step 3: Review safety and artifact contents**

Confirm the diff contains no production routing/prompt/model changes, no MT5 imports or calls, no graph construction, no retries, no execution API, no Phase 5/7 work, and no text-bearing benchmark fields. Confirm the real Phase 6 database is untouched and stock CLI files are unchanged.

- [ ] **Step 4: Commit the report and plan**

```powershell
git add scripts/benchmark_qwen_prose.py tests/test_qwen_prose_benchmark.py docs/superpowers/plans/2026-09-09-phase-6-3-qwen-reliability-benchmark.md
git add -f docs/superpowers/reports/2026-09-09-phase-6-3-qwen-reliability.md
git commit -m "docs: record phase6.3 qwen reliability benchmark"
```

Stop after this report. Do not start Phase 7.
