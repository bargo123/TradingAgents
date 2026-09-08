# `forex-shadow`: read-only MT5 forex analysis

`forex-shadow` is the Phase 4 forex entry point. It is deliberately separate
from the existing `tradingagents` stock CLI and runs in shadow mode only:

```
MT5 FOREX — SHADOW MODE
NO ORDER WILL BE SENT
```

The command never calls an MT5 mutation or execution API. It does not submit,
modify, close, or cancel orders, and it does not expose execution credentials.
The persisted decision is always `executed=False`.

## Run

Install the optional MT5 dependency and make sure a local demo terminal is
running and logged in:

```powershell
pip install -e ".[mt5]"
forex-shadow --symbol EURUSD --count 100
```

The module form is equivalent:

```powershell
python -m cli.forex_shadow --symbol EURUSD --db-path data_cache/shadow.db
```

Useful options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--symbol` | `EURUSD` | Requested six-letter forex pair; broker prefix/suffix resolution is exact-first and ambiguity-safe. |
| `--count` | `100` | Positive number of bars fetched for each M1/M5/M15/H1 timeframe. |
| `--analysis-date` | UTC today | ISO analysis date. |
| `--terminal-path` | terminal default | Optional path to `terminal64.exe`. |
| `--db-path` | configured cache DB | SQLite file for shadow decisions. |
| `--analysts` | `market,news` | Comma-separated forex-safe analyst allow-list. |
| `--llm-provider` | configured default | Non-secret LLM provider override. |
| `--quick-model` / `--deep-model` | configured defaults | Non-secret model overrides. |
| `--backend-url` | configured default | Non-secret OpenAI-compatible backend URL. |
| `--temperature` / `--max-tokens` | configured defaults | Optional model runtime limits. |

Only `market` and `news` are accepted by default. The forex News analyst uses
`get_global_news` only, limited to broad/global/macro context such as rates,
inflation, employment, central-bank policy, and geopolitics. It does not use
ticker/company news, earnings, fundamentals, StockTwits, Reddit, P/E, EPS,
dividends, or equivalent stock-specific context. Bull/Bear researchers, Trader,
risk debaters, and Portfolio Manager are reused downstream from the existing
TradingAgents graph with forex-safe prompts.

## Run flow and evidence

Each invocation owns one MT5 provider lifecycle and performs this sequence:

1. Validate the forex-safe analyst list and connect to the local MT5 terminal.
2. Resolve the requested symbol (case-insensitive exact match first; a single
   controlled broker prefix/suffix variant may be used; multiple candidates
   fail with an ambiguity error).
3. Capture exactly one normalized `ForexMarketSnapshot`. The read-only adapter
   caches that snapshot for all graph tools, so downstream agents cannot cause
   a second snapshot fetch.
4. Inject the bounded MT5 context into the existing graph and run the selected
   analysts plus the fixed downstream research, trading, risk, and portfolio
   stages.
5. Normalize only the structured `PortfolioDecision`/`PortfolioRating` result
   from Portfolio Manager. Free-form prose is never guessed into BUY/SELL/HOLD.
   A missing, malformed, or ambiguous structured result is persisted as
   `normalization_status=FAILED` with a null action.
6. Persist the decision and shut down the provider in all paths.

The SQLite record includes the raw Portfolio Manager result, normalized action
and status/error, requested and resolved symbols, UTC snapshot timestamp,
bid/ask/spread/spread-points, normalized snapshot JSON, provider/model
identifiers, summaries, and `executed=0` (represented as `False` by the Python
model). Failed normalization is retained as evidence rather than relabeled.

The current implementation intentionally does not add MT5 to the stock CLI,
TradingAgents tool registry, LangGraph registration, or any live/demo execution
workflow. A future `forex-demo` or `forex-live` command can be added as a
separate boundary without changing this command or weakening its shadow-only
contract.

## Verification

Run the unit/integration guard suite without an external terminal:

```powershell
pytest tests/test_forex_shadow_integration.py -q
```

The real-terminal check is opt-in and read-only:

```powershell
$env:RUN_MT5_INTEGRATION = "1"
pytest tests/test_forex_shadow_integration.py -m integration -q
```

If the MetaTrader 5 package, terminal, account, or configured LLM is not
available, the run should be reported as unverified rather than replaced with
Yahoo Finance data or a fabricated recommendation.
