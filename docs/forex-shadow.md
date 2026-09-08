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

## Local verification snapshot (2026-09-08)

The local demo terminal was available during Phase 4 verification. A direct
read-only snapshot of `EURUSD` returned UTC timestamp
`2026-09-08T16:27:17.585Z`, bid `1.16245`, ask `1.16246`, 1-point spread,
`digits=5`, `point=0.00001`, and two bars in each M1/M5/M15/H1 series. The
positions and orders lists were empty before and after the read. The guarded
provider/adapter integration test passed.

The full LLM-backed `forex-shadow` smoke was attempted twice with local Ollama
Qwen models. Both attempts reached the graph but exceeded the bounded local
model wait window before persistence; no decision row was written. This is
recorded as unverified for the LLM-backed decision path, not treated as a
successful recommendation.

## Phase 4.1 complete shadow validation (2026-09-08)

A complete real validation run was then executed through `ForexShadowRunner`,
the production runner invoked by this CLI, with the same one-snapshot and
forex-safe graph path. The runtime-only optimization used one global-news
query/article and `count=1`; the required Market, News, Bull, Bear, Research
Manager, Trader, three Risk Analysts, and strict Portfolio Manager stages all
remained enabled. Existing environment/config precedence selected Ollama with
Qwen 2B for both models (no hosted API key was available in the project
configuration).

Evidence from the persisted row:

```text
provider=ollama
quick_model=qwen3.5:2b
deep_model=qwen3.5:2b
requested_symbol=EURUSD
resolved_symbol=EURUSD
snapshot_timestamp=2026-09-08T18:03:18.145Z
bid=1.16304 ask=1.16305 spread=0.00001 spread_points=1
llm_calls=14 tool_calls=3 runtime_seconds=2595.45
portfolio_manager_rating=Hold
normalized_action=HOLD
normalization_status=NORMALIZED
executed=False
decision_id=b8104eac-d1b7-40cd-91fb-2055f1ecd45f
database=data_cache/phase41-probe-20260908-180317.db
positions_before=[] positions_after=[]
orders_before=[] orders_after=[]
```

The raw structured Portfolio Manager result is retained in SQLite, including
its executive summary, investment thesis, price target, rating, and time
horizon. The before/after account checks remained empty and no mutation API is
available in the provider, adapter, runner, or graph tool set.

## Phase 4.2 calibration validation — superseded v1 attempt (2026-09-08)

The Phase 4.2 run completed the full graph with the production default history
(`count=100` on M1/M5/M15/H1), one cached MT5 snapshot, and the default
forex-safe analyst set. No hosted key was configured, so the existing local
Ollama configuration was used. Quick nodes used `qwen3.5:2b` with the
forex-only `think=false` control; deep Research Manager and Portfolio Manager
nodes used `qwen3.5:4b` with `think=true`.

The complete evidence is persisted at
`data_cache/phase42-validation-20260908.db`:

```text
provider=ollama
quick_model=qwen3.5:2b
deep_model=qwen3.5:4b
requested_symbol=EURUSD
resolved_symbol=EURUSD
snapshot_timestamp=2026-09-08T20:30:12.304000Z
bid=1.16261 ask=1.16262 spread=0.0000100000000000655 spread_points=1.00000000000655
analysis_profile=INTRADAY horizon=minutes to hours valid_for_seconds=3600
bars=M1:100 M5:100 M15:100 H1:100
macro_event_status=MACRO/EVENT DATA UNAVAILABLE
llm_calls=13 tool_calls=3 tokens_in=32790 tokens_out=14128 reasoning_tokens=0
runtime_seconds=2268.3846525
portfolio_manager_rating=Hold
raw_time_horizon=minutes to hours
normalized_action=HOLD
normalization_status=NORMALIZED
decision_id=181a14d3-213d-4c20-98dc-1aac980ea58f
executed=False
positions_before=[] positions_after=[]
orders_before=[] orders_after=[]
```

The raw Portfolio Manager JSON is retained in the row, including the
intraday executive summary, thesis, `rating=Hold`, and `valid_for_seconds=3600`.
The deterministic feature payload records latest OHLC, direction, return,
high/low/range, ATR, range percentage, and close position for every timeframe;
the prompt receives those compact summaries rather than a raw bar dump.

Compared with the Phase 4.1 local run (`2595.45s`, 14 calls), this complete
Phase 4.2 run took `2268.38s` (about `327.07s`, or `12.6%`, faster). The
positions/orders before and after reconnect were both empty, and the command
printed `NO ORDER WILL BE SENT`; no execution API was called.

## Phase 4.2 final-code validation (v3 authoritative, 2026-09-08)

After the v1 run, the final code was tightened so the LLM-facing MT5 tool emits
only the compact quote/feature context (no raw candle arrays), the forex Trader
is explicitly routed through the deep model, and the forex Portfolio Manager
schema does not inherit a stock-style long-horizon description. The following
run was then executed from that final code revision through the complete graph.

The local configuration did not contain a hosted API key, so the existing
Ollama provider was used. The default forex-safe allow-list remained
`market,news`; Market, News, Bull, Bear, Research Manager, Trader, all three
Risk Analysts, and Portfolio Manager each ran, with one cached MT5 snapshot and
100 bars for every M1/M5/M15/H1 timeframe.

Authoritative persisted/CLI evidence:

```text
provider=ollama
quick_model=qwen3.5:2b (forex think=false)
deep_model=qwen3.5:4b (forex think=true)
requested_symbol=EURUSD
resolved_symbol=EURUSD
snapshot_timestamp=2026-09-08T21:55:51.236000Z
bid=1.16268 ask=1.16268 spread=0.0 spread_points=0.0
analysis_profile=INTRADAY horizon=minutes to hours valid_for_seconds=3600
valid_until=2026-09-08T22:55:51.236000Z
bars=M1:100 M5:100 M15:100 H1:100
macro_event_status=MACRO/EVENT DATA UNAVAILABLE
llm_calls=12 tool_calls=2 tokens_in=29015 tokens_out=14081 reasoning_tokens=0
runtime_seconds=2468.540167500032
portfolio_manager_rating=Hold
raw_time_horizon=null
normalized_action=HOLD
normalization_status=NORMALIZED
decision_id=5a8b4266-cd36-46c5-a898-188463d86cde
executed=False
database=data_cache/phase42-validation-20260908-v3.db
positions_before=[] positions_after=[]
orders_before=[] orders_after=[]
```

The raw structured Portfolio Manager result is retained in the SQLite row. It
reported `rating=Hold`, omitted `time_horizon`, and included
`valid_for_seconds=3600`; the validity window therefore comes from the
declared `INTRADAY` profile rather than prose. The decision remains
`future_evaluation_status=PENDING` and was not evaluated or relabeled.

Final per-agent metrics (numeric telemetry only; no prompts or private
chain-of-thought are stored):

```text
Market Analyst       model=qwen3.5:2b calls=2 input=4550 output=1400 elapsed=245.075s
News Analyst         model=qwen3.5:2b calls=2 input=6371 output=1259 elapsed=228.127s
Bull Researcher      model=qwen3.5:2b calls=1 input=2148 output=1948 elapsed=295.907s
Bear Researcher      model=qwen3.5:2b calls=1 input=2140 output=1956 elapsed=289.750s
Research Manager     model=qwen3.5:4b calls=1 input=1949 output=738  elapsed=199.636s
Trader               model=qwen3.5:4b calls=1 input=2658 output=670  elapsed=199.769s
Aggressive Analyst   model=qwen3.5:2b calls=1 input=2279 output=1817 elapsed=270.098s
Conservative Analyst model=qwen3.5:2b calls=1 input=2260 output=1836 elapsed=257.102s
Neutral Analyst      model=qwen3.5:2b calls=1 input=2259 output=1837 elapsed=282.347s
Portfolio Manager    model=qwen3.5:4b calls=1 input=2401 output=620  elapsed=193.779s
```

The final runtime was `2468.54s`, `126.91s` (approximately `4.9%`) faster
than the Phase 4.1 baseline of `2595.45s`, while using the required 100-bar
history. A pre-run MT5 read captured empty positions and orders; a fresh
post-run connection again returned empty positions and orders. The command
printed `MT5 FOREX — SHADOW MODE` and `NO ORDER WILL BE SENT`, and no execution
API was added or called.

Final verification gate for this implementation: `772 passed, 4 skipped, 71
subtests passed`; Ruff, `compileall`, and `git diff --check` passed; the opt-in
MT5 integration guard passed (`1 passed, 2 deselected`).
