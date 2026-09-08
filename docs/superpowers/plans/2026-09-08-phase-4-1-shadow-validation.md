# Phase 4.1 Shadow Validation and Runtime Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one evidence-backed EURUSD shadow decision through the complete Phase 4 graph when an existing configured LLM provider is available, while measuring runtime/calls and preserving the read-only boundary.

**Architecture:** Reuse `ForexShadowRunner` and the existing `TradingAgentsGraph`; inject the existing callback-based statistics handler into the runner so the already-configured provider/model pair is observed without changing provider selection or secrets. The runner will keep ownership of one MT5 snapshot and one provider lifecycle, persist the genuine structured Portfolio Manager result, and return metrics for the CLI/report. Hosted providers and Ollama remain selected solely through the existing environment/config precedence.

**Tech Stack:** Python 3.10+, LangGraph, LangChain callbacks, existing TradingAgents LLM provider registry, MetaTrader5 read-only provider, SQLite shadow store, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-08-mt5-forex-shadow-design.md`

## Global Constraints

- Do not modify `tradingagents.dataflows.mt5.provider` or MT5 symbol/snapshot behavior.
- Do not add `order_send`, order lifecycle, live mode, RiskGovernor, RAG, training, ONNX, or Phase 5 evaluation.
- Do not bypass Market/News, Bull/Bear, Trader, Aggressive/Conservative/Neutral Risk, or Portfolio Manager graph nodes.
- Keep the single cached `ForexMarketSnapshot` per run and persist `executed=False`.
- Read provider/model/API-key configuration from the existing `.env`/`TRADINGAGENTS_*` system; never add a secret CLI argument or hard-code a key.
- Fail non-zero and preserve an explicit error when the configured provider is unavailable or structured Portfolio Manager output cannot be normalized.

---

### Task 1: Add runner callback telemetry without changing graph semantics

**Files:**
- Modify: `tradingagents/forex/runner.py:run`
- Modify: `tests/test_forex_shadow_runner.py`

**Interfaces:**
- Consumes: existing `ForexShadowRunner.run(symbol, count, analysis_date, terminal_path, analysts, db_path)` and callback-compatible objects.
- Produces: optional `callbacks` keyword on `run`; `ForexShadowRunResult.metrics` keys `llm_calls`, `tool_calls`, `tokens_in`, `tokens_out`, `elapsed_seconds`, and `provider_snapshot_calls`.

- [ ] **Step 1: Write the failing test**

```python
def test_runner_passes_callbacks_and_reports_llm_metrics(tmp_path):
    callback = type("Callback", (), {"get_stats": lambda self: {
        "llm_calls": 17, "tool_calls": 4, "tokens_in": 100, "tokens_out": 50,
    }})()
    runner, provider, graph, _ = _make_runner(tmp_path, {
        "final_trade_decision": {"rating": "Hold"},
        "portfolio_manager_raw_result": {"rating": "Hold"},
        "investment_debate_state": {"bull_history": "bull", "bear_history": "bear"},
        "risk_debate_state": {},
    })

    result = runner.run(symbol="EURUSD", analysis_date="2026-09-08", callbacks=[callback])

    assert graph.invocations[0][1]["config"]["callbacks"] == [callback]
    assert result.metrics["llm_calls"] == 17
    assert result.metrics["tool_calls"] == 4
    assert result.metrics["tokens_in"] == 100
    assert result.metrics["tokens_out"] == 50
    assert result.elapsed_seconds >= 0
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_forex_shadow_runner.py::test_runner_passes_callbacks_and_reports_llm_metrics -q`

Expected: FAIL because `ForexShadowRunner.run` does not accept `callbacks` and currently always passes an empty callback list.

- [ ] **Step 3: Write the minimal implementation**

Add `callbacks: Sequence[Any] | None = None` as a keyword-only `run` argument. Normalize it to a list once, pass it to both `graph_factory(..., callbacks=...)` and `graph.propagator.get_graph_args(callbacks=...)`, then merge the first callback exposing `get_stats()` into the result metrics. Keep zero-valued defaults when no callback is supplied. Do not alter snapshot acquisition or graph nodes.

- [ ] **Step 4: Run the focused and regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_forex_shadow_runner.py -q`

Expected: all runner tests pass, including the new callback propagation/metrics assertion.

- [ ] **Step 5: Commit**

```text
git add tradingagents/forex/runner.py tests/test_forex_shadow_runner.py
git commit -m "feat: instrument forex shadow runtime metrics"
```

### Task 2: Wire the existing CLI stats handler and expose complete-run evidence

**Files:**
- Modify: `cli/forex_shadow.py`
- Modify: `tests/test_forex_shadow_cli.py`

**Interfaces:**
- Consumes: `cli.stats_handler.StatsCallbackHandler`, `ForexShadowRunResult`, and environment-resolved `DEFAULT_CONFIG` values already merged by the runner.
- Produces: CLI output containing provider, quick/deep model, resolved symbol, snapshot timestamp, LLM/tool/token counts, runtime, raw PM result, normalized action/status, decision ID, DB path, and `EXECUTED: FALSE`.

- [ ] **Step 1: Write the failing test**

```python
def test_cli_passes_stats_callback_and_prints_run_evidence(capsys, monkeypatch, tmp_path):
    captured = {}

    class FakeStats:
        def get_stats(self):
            return {"llm_calls": 9, "tool_calls": 3, "tokens_in": 10, "tokens_out": 20}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

        def run(self, **kwargs):
            captured["run"] = kwargs
            decision = type("Decision", (), {
                "decision_id": "decision-001", "action": "HOLD",
                "normalization_status": "NORMALIZED", "executed": False,
                "resolved_symbol": "EURUSD", "snapshot_timestamp": "2026-09-08T00:00:00Z",
                "raw_portfolio_manager_result_json": '{"rating":"Hold"}',
                "llm_provider": "openai", "quick_model": "gpt-5.6-luna",
                "deep_model": "gpt-5.6", "reference_bid": 1.1,
                "reference_ask": 1.2, "spread": 0.1, "spread_points": 10000,
            })()
            return type("Result", (), {
                "decision": decision, "elapsed_seconds": 1.25,
                "metrics": {"llm_calls": 9, "tool_calls": 3, "tokens_in": 10, "tokens_out": 20},
            })()

    monkeypatch.setattr("cli.forex_shadow.ForexShadowRunner", FakeRunner)
    monkeypatch.setattr("cli.forex_shadow.StatsCallbackHandler", FakeStats)

    assert main(["--db-path", str(tmp_path / "shadow.db")]) == 0
    assert len(captured["run"]["callbacks"]) == 1
    output = capsys.readouterr().out
    assert "LLM PROVIDER: openai" in output
    assert "LLM CALLS: 9" in output
    assert "NORMALIZED ACTION: HOLD" in output
    assert "EXECUTED: FALSE" in output
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_forex_shadow_cli.py::test_cli_passes_stats_callback_and_prints_run_evidence -q`

Expected: FAIL because the CLI currently constructs no stats callback, passes no callback to the runner, and prints only the decision ID/action/status.

- [ ] **Step 3: Write the minimal implementation**

Import `StatsCallbackHandler`, construct one per CLI invocation after printing the safety banner, pass it as `callbacks=[stats_handler]` to `runner.run`, and print only non-secret run evidence from the returned decision/metrics. Serialize the raw PM JSON already stored on the decision; never print API keys or infer an action from prose. Keep existing parser flags and stock entry point unchanged.

- [ ] **Step 4: Run CLI tests and help smoke**

Run: `.venv\Scripts\python.exe -m pytest tests/test_forex_shadow_cli.py -q` and `.venv\Scripts\python.exe -m cli.forex_shadow --help`

Expected: all CLI tests pass; help remains a separate forex command with no execution flags.

- [ ] **Step 5: Commit**

```text
git add cli/forex_shadow.py tests/test_forex_shadow_cli.py
git commit -m "feat: report forex shadow run evidence"
```

### Task 3: Validate configured providers and execute one real EURUSD shadow run

**Files:**
- Create: `tests/test_forex_shadow_validation.py`
- Modify: `docs/forex-shadow.md`
- Create: `docs/superpowers/reports/2026-09-08-phase-4-1-shadow-validation.md`

**Interfaces:**
- Consumes: CLI environment/config precedence, real `MT5Provider`, cached `MT5ToolAdapter`, and the unchanged full forex graph.
- Produces: one persisted `ShadowTradeDecision` or a clear non-zero provider-unavailable result; a report with exact provider/models, runtime/call metrics, raw/normalized PM evidence, decision ID/database path, and before/after MT5 orders/positions.

- [ ] **Step 1: Write the failing validation tests**

Add tests that assert (a) a callback-bearing runner result persists the raw structured Portfolio Manager mapping and `executed is False`, (b) a provider construction/configuration error returns a non-zero CLI result with `FOREX SHADOW ERROR`, and (c) the integration test remains guarded by `RUN_MT5_INTEGRATION=1` and compares positions/orders before and after.

- [ ] **Step 2: Run the focused tests to verify the new assertions fail where behavior is missing**

Run: `.venv\Scripts\python.exe -m pytest tests/test_forex_shadow_validation.py -q`

Expected: the new evidence assertions fail until Tasks 1 and 2 expose callback metrics and CLI fields.

- [ ] **Step 3: Run the real read-only preflight and provider smoke**

Run `.venv\Scripts\python.exe scripts/test_mt5_connection.py --symbol EURUSD --count 2` and a guarded provider/adapter capture. Record resolved symbol, UTC timestamp, bid/ask/spread, digits/point, cached snapshot call count, and unchanged positions/orders. Do not call any mutation API.

- [ ] **Step 4: Select the existing configured LLM provider without adding secrets**

Read the non-secret provider/model values through the project’s `.env`/`TRADINGAGENTS_*` overlay. If a hosted key is present, run with that configured provider (for example `openai` with `gpt-5.6-luna` quick and the configured deep model); otherwise retain Ollama/local support and use the configured local models with bounded count/tokens/rounds. If no provider can serve requests, stop with the explicit error and do not fabricate a decision.

- [ ] **Step 5: Run one complete EURUSD shadow analysis**

Run `forex-shadow --symbol EURUSD --count 1 --db-path <validated-path>` with only safe runtime overrides needed for the configured provider. Let every required analyst/research/risk/PM node execute. Capture stdout/stderr, total runtime, LLM calls, the persisted SQLite row, and before/after MT5 positions/orders.

- [ ] **Step 6: Verify the persisted row and scope**

Query the decision store and assert the row has a real raw PM result, `normalization_status` of `NORMALIZED` with `BUY`, `SELL`, or `HOLD` (or an explicit failed normalization if the provider returned malformed output), the resolved symbol and snapshot fields, identifiers, and `executed=False`. Confirm the git diff against the Phase 4 baseline contains no MT5-provider or execution changes and no stock CLI changes.

- [ ] **Step 7: Document evidence and commit**

Write the exact results and any provider limitation to `docs/superpowers/reports/2026-09-08-phase-4-1-shadow-validation.md`, update the local verification section in `docs/forex-shadow.md`, run the full test/lint/compile gates, then commit:

```text
git add tests/test_forex_shadow_validation.py docs/forex-shadow.md docs/superpowers/reports/2026-09-08-phase-4-1-shadow-validation.md
git commit -m "test: validate complete forex shadow decision"
```

## Verification Gate

Before reporting completion, run:

```text
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m compileall -q tradingagents cli
git diff --check
git status --short
```

The final report must distinguish a completed persisted shadow decision from an unavailable provider. It must never claim BUY/SELL/HOLD, normalization success, or end-to-end completion without the corresponding persisted evidence.
