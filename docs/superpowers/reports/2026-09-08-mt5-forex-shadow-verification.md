# Phase 4 MT5 Forex Shadow Verification

Date: 2026-09-08  
Repository: `C:\AITrading\TradingAgents`  
Baseline commit: `d4f4a49aab5958c11f9b4f693f66477e19f1d194`  
Implementation tip before this report artifact: `5ac2b42`

## Scope

Phase 4 adds the standalone `forex-shadow` command and reusable forex-mode
graph plumbing. It remains read-only and shadow-only. The existing
`tradingagents` stock CLI and `cli/main.py` were not changed. No order,
execution, live mode, RiskGovernor, RAG, training, ONNX, or automated-trading
path was added.

## Automated verification

Commands run with the repository `.venv`:

```text
.venv\Scripts\python.exe -m pytest -q
739 passed, 4 skipped, 71 subtests passed, 18 warnings
```

The skipped tests are the optional Bedrock dependency, an opt-in DeepSeek API
test, the guarded Phase 4 MT5 integration test, and the existing guarded MT5
integration test.

```text
.venv\Scripts\ruff.exe check .
All checks passed!
```

```text
.venv\Scripts\python.exe -m pytest tests/test_forex_shadow_integration.py -q
2 passed, 1 skipped
```

The skipped case is the real-terminal case when `RUN_MT5_INTEGRATION` is not
set. With explicit opt-in, the same test passed:

```text
$env:RUN_MT5_INTEGRATION='1'
.venv\Scripts\python.exe -m pytest tests/test_forex_shadow_integration.py -m integration -q
1 passed, 2 deselected
```

The existing Phase 3 read-only integration test also passed with the guard:

```text
.venv\Scripts\python.exe -m pytest tests/test_mt5_integration.py -m integration -q
1 passed
```

The standalone CLI help was checked with `python -m cli.forex_shadow --help`.
It shows only the separate forex options and the stock entry-point string
remains `tradingagents = "cli.main:app"`.

## Real MT5 read-only evidence

The local MetaTrader 5 demo terminal and Python package were available. The
connection smoke command:

```text
.venv\Scripts\python.exe scripts/test_mt5_connection.py --symbol EURUSD --count 2
```

returned `MT5 CONNECTED`, server `MetaQuotes-Demo`, no open positions, and no
orders. A second direct provider/adapter capture returned:

```json
{
  "symbol": "EURUSD",
  "timestamp": "2026-09-08T16:27:17.585000+00:00",
  "bid": 1.16245,
  "ask": 1.16246,
  "spread": 0.000010000000000065512,
  "spread_points": 1.0000000000065512,
  "digits": 5,
  "point": 0.00001,
  "bars": {"m1": 2, "m5": 2, "m15": 2, "h1": 2},
  "cached_symbol": "EURUSD",
  "before_positions": [],
  "after_positions": [],
  "before_orders": [],
  "after_orders": []
}
```

This proves the Phase 3 provider resolved EURUSD, returned timezone-aware UTC
data, included `digits`/`point`, and the Phase 4 adapter consumed the captured
snapshot without a second snapshot request. The before/after account checks
showed no broker mutation.

## Full shadow smoke status

The full graph-backed command was attempted twice against the local MT5 demo
and local Ollama Qwen models. The first used Qwen `qwen3.5:2b` for quick work
and `qwen3.5:4b` for deep work with `--count 2` and `--max-tokens 256`. The
second used `qwen3.5:2b` for both models, `--count 2`, `--max-tokens 64`, and
zero debate/risk repeat rounds (each downstream stage still remained in the
graph). Both attempts printed the safety banner and reached the graph, but
the local model requests exceeded the bounded verification wait window before
the Portfolio Manager result could be persisted. No shadow decision database
row was written by either attempt. The result is therefore **unverified for
the LLM-backed end-to-end recommendation**, rather than a fabricated or
prose-parsed action.

The local Ollama probe itself was healthy and reported both configured Qwen
models at `http://localhost:11434`; the limitation was runtime latency in the
multi-agent chain.

## Safety checks

- `MT5Provider`, `MT5ToolAdapter`, and `ForexShadowRunner` expose no public
  mutation/execution methods such as `order_send`, buy/sell, close-position, or
  modify-position APIs.
- Injected forex graph tools are name-validated against execution tokens.
- Forex market/news tool nodes are isolated from stock tools; News binds only
  `get_global_news` and prompts prohibit ticker/company, earnings,
  fundamentals, StockTwits, Reddit, P/E, EPS, and dividend context.
- Portfolio Manager normalization consumes only structured `PortfolioDecision`
  output. Missing/malformed/ambiguous output is stored with
  `normalization_status=FAILED` and a null action.
- Every persisted decision has `executed=False`; SQLite also enforces this
  invariant. The runner owns one provider lifecycle and one cached snapshot.

## Scope diff

The final scope comparison is anchored to the recorded baseline rather than a
relative `HEAD~N` assumption:

```text
git diff --stat d4f4a49aab5958c11f9b4f693f66477e19f1d194..HEAD
git diff --name-status d4f4a49aab5958c11f9b4f693f66477e19f1d194..HEAD
```

The diff contains the Phase 4 specification/plan, dedicated forex package,
optional graph/state/prompt branches, runner, CLI, tests, and verification
documentation. It does not modify `cli/main.py` or register an MT5 execution
tool.
