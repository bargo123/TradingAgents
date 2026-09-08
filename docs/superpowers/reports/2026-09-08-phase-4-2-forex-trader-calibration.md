# Phase 4.2 Forex Trader Calibration — Validation Report

> The first evidence block in this historical report is the superseded v1
> attempt. The **Final-code v3 validation** section at the end is authoritative
> for the completed implementation.

Date: 2026-09-08
Scope baseline: `e334b88a92c1a59bc69b7a84c22bb839ede86c5a` (accepted Phase 4.1)
Mode: standalone `forex-shadow`, read-only MT5, no Phase 5 evaluation

## Result

The complete EURUSD path finished and persisted a genuine Portfolio Manager
decision. The graph retained Market, News, Bull, Bear, Research Manager,
Trader, Aggressive Risk, Conservative Risk, Neutral Risk, and Portfolio Manager
stages. No order execution path was added or invoked.

## Run evidence

| Field | Evidence |
| --- | --- |
| LLM provider | `ollama` |
| Quick model | `qwen3.5:2b` (`think=false`, forex-only) |
| Deep model | `qwen3.5:4b` (`think=true`, forex-only) |
| Requested symbol | `EURUSD` |
| Resolved broker symbol | `EURUSD` |
| MT5 snapshot timestamp | `2026-09-08T20:30:12.304000Z` |
| Bid / ask | `1.16261 / 1.16262` |
| Spread / points | `0.0000100000000000655 / 1.00000000000655` |
| Profile / horizon | `INTRADAY` / `minutes to hours` |
| Validity | `3600s`, until `2026-09-08T21:30:12.304000Z` |
| History | `100` bars each for M1, M5, M15, H1 |
| Macro status | `MACRO/EVENT DATA UNAVAILABLE` |
| Total LLM calls | `13` |
| Tool calls | `3` |
| Input / output tokens | `32790 / 14128` |
| Reasoning tokens reported | `0` |
| Total runtime | `2268.3846525s` |
| Raw Portfolio Manager rating | `Hold` |
| Raw Portfolio Manager horizon | `minutes to hours` |
| Normalized action | `HOLD` |
| Normalization status | `NORMALIZED` |
| Shadow decision ID | `181a14d3-213d-4c20-98dc-1aac980ea58f` |
| Shadow database | `data_cache/phase42-validation-20260908.db` |
| Executed | `False` |

## Raw Portfolio Manager result

The persisted raw structured result contains:

```json
{
  "analysis_profile": "INTRADAY",
  "rating": "Hold",
  "time_horizon": "minutes to hours",
  "valid_for_seconds": 3600,
  "price_target": null,
  "executive_summary": "EURUSD remains in consolidation with conflicting signals across timeframes: M1 and M15 show downward pressure while M5 and H1 indicate upward momentum. Low volatility (ATR 7.67e-05 to 0.0028) and tight spread (~0.0000065) suggest ranging conditions without clear directional bias. Hold until macro event data or stronger confirmation emerges from analyst debate.",
  "investment_thesis": "The intraday analysis for EURUSD reveals materially conflicting signals across timeframes that preclude a directional call. M1 (DOWN -0.069%) and M15 (DOWN -0.029%) indicate immediate downward pressure, while M5 (UP +0.123%) and H1 (UP +0.186%) show upward momentum. This divergence suggests ranging conditions rather than a clear trend. Low volatility across all timeframes (ATR 7.67e-05 to 0.0028) combined with MACRO/EVENT DATA UNAVAILABLE removes key context for directional positioning. Without analyst debate content or macro event confirmation, premature exposure risks capital erosion in this volatile market environment. The Research Manager's Hold recommendation aligns with the trader's proposal to avoid entry until clearer directional signal emerges."
}
```

The JSON above is the persisted structured Portfolio Manager payload (with
whitespace normalized for readability); the SQLite row contains the same
fields and exact text returned by the model.

## Per-agent metrics

The callback records numeric usage and elapsed time only. It does not retain
prompts, completions, or chain-of-thought text.

```text
Market Analyst       model=qwen3.5:2b calls=2  input=3717 output=1064 elapsed=199.133s
News Analyst         model=qwen3.5:2b calls=3  input=9869 output=1683 elapsed=264.500s
Bull Researcher      model=qwen3.5:2b calls=1  input=2151 output=1945 elapsed=287.234s
Bear Researcher      model=qwen3.5:2b calls=1  input=2143 output=1953 elapsed=288.421s
Research Manager     model=qwen3.5:4b calls=1  input=2006 output=550  elapsed=152.380s
Trader               model=qwen3.5:2b calls=1  input=2743 output=717  elapsed=125.292s
Aggressive Analyst   model=qwen3.5:2b calls=1  input=2343 output=1753 elapsed=233.939s
Conservative Analyst model=qwen3.5:2b calls=1  input=2324 output=1772 elapsed=232.977s
Neutral Analyst      model=qwen3.5:2b calls=1  input=2323 output=1773 elapsed=236.951s
Portfolio Manager    model=qwen3.5:4b calls=1  input=3171 output=918  elapsed=238.743s
```

## MT5 read-only before/after proof

Before the run, a fresh connected provider read returned:

```text
connected=True
positions=[]
orders=[]
```

After the runner shut down, a fresh connected provider read returned:

```text
connected=True
resolved=EURUSD
positions=[]
orders=[]
```

No provider, adapter, runner, or forex CLI mutation method exists. The run
printed `MT5 FOREX — SHADOW MODE` and `NO ORDER WILL BE SENT`.

## Runtime comparison

Phase 4.1 local validation measured `2595.45s` and `14` LLM calls with a
single-candle probe. Phase 4.2 measured `2268.3846525s` and `13` calls while
using the required 100-bar history per timeframe: `327.0653475s` (approximately
`12.6%`) faster. The quick/deep thinking controls and compact deterministic
features are now available for further provider-specific calibration.

## Verification commands

```text
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m compileall -q tradingagents cli
git diff --check
```

Phase 5 evaluation, outcome labeling, training, RAG, ONNX, order execution,
and live/demo promotion remain out of scope.

## Final-code v3 validation (authoritative)

The final-code EURUSD run completed the full shadow path and persisted a genuine
Portfolio Manager result. Market, News, Bull, Bear, Research Manager, Trader,
Aggressive Risk, Conservative Risk, Neutral Risk, and Portfolio Manager all ran.
No order execution path was added or invoked.

| Field | Evidence |
| --- | --- |
| LLM provider | `ollama` |
| Quick model | `qwen3.5:2b` (`think=false`, forex-only) |
| Deep model | `qwen3.5:4b` (`think=true`, forex-only) |
| Requested symbol | `EURUSD` |
| Resolved broker symbol | `EURUSD` |
| MT5 snapshot timestamp | `2026-09-08T21:55:51.236000Z` |
| Bid / ask | `1.16268 / 1.16268` |
| Spread / points | `0.0 / 0.0` |
| Profile / horizon | `INTRADAY` / `minutes to hours` |
| Validity | `3600s`, until `2026-09-08T22:55:51.236000Z` |
| History | `100` bars each for M1, M5, M15, H1 |
| Macro status | `MACRO/EVENT DATA UNAVAILABLE` |
| Total LLM calls | `12` |
| Tool calls | `2` |
| Input / output tokens | `29015 / 14081` |
| Reasoning tokens reported | `0` |
| Total runtime | `2468.540167500032s` |
| Raw Portfolio Manager rating | `Hold` |
| Raw Portfolio Manager horizon | `null` (profile validity supplied) |
| Normalized action | `HOLD` |
| Normalization status | `NORMALIZED` |
| Shadow decision ID | `5a8b4266-cd36-46c5-a898-188463d86cde` |
| Shadow database | `data_cache/phase42-validation-20260908-v3.db` |
| Executed | `False` |

The raw structured Portfolio Manager JSON is persisted unchanged. It contains
`analysis_profile=INTRADAY`, `rating=Hold`, `time_horizon=null`, and
`valid_for_seconds=3600`. Normalization used that structured rating only; no
free-form BUY/SELL/HOLD guessing occurred. The row remains
`future_evaluation_status=PENDING`.

Exact raw structured payload:

```json
{
  "analysis_profile": "INTRADAY",
  "executive_summary": "Maintain current EURUSD exposure with no directional action required. Wait for complete analyst debate content before executing trades, as zero spread conditions and missing Bull/Bear arguments preclude confident directional calls. Monitor price action and volatility within the intraday session.",
  "investment_thesis": "The research manager's Hold recommendation is appropriate given the absence of Risk Analysts Debate History content. While technical indicators show bullish direction across all timeframes (M1, M5, M15, H1), no analyst arguments are provided to evaluate the bull/bear debate. The zero spread and demo account conditions further limit confidence in any trade execution. When evidence is missing or ambiguous, Hold is the appropriate stance rather than forcing a direction.",
  "price_target": null,
  "rating": "Hold",
  "time_horizon": null,
  "valid_for_seconds": 3600
}
```

Final numeric per-agent telemetry (no prompts, completions, or private
chain-of-thought text retained):

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

The forex Trader now uses the deep model while stock-mode assignments remain
unchanged; a regression test covers this split. The LLM-facing MT5 tool emits
compact quote/feature context without raw candle arrays, while persisted
snapshot evidence remains available in SQLite.

## Final MT5 safety proof

Pre-run connected read:

```text
connected=True
positions=[]
orders=[]
```

Fresh post-run connected read:

```text
connected=True
resolved=EURUSD
positions=[]
orders=[]
```

The provider, adapter, runner, and forex CLI expose no order-send, buy/sell,
close-position, modify-position, or other mutation API. The command printed
`MT5 FOREX — SHADOW MODE` and `NO ORDER WILL BE SENT`.

## Final runtime comparison

The accepted Phase 4.1 local validation measured `2595.45s` and 14 calls with
a single-candle probe. Final Phase 4.2 measured `2468.5401675s` and 12 calls
with the required 100-bar history per timeframe: `126.9098325s`
(approximately `4.9%`) faster. This is runtime/calibration evidence only, not
a profitability or trading-performance claim.

## Verification results

```text
772 passed, 4 skipped, 71 subtests passed
Ruff: All checks passed!
compileall: passed
git diff --check: passed
MT5 integration guard: 1 passed, 2 deselected
```

Phase 5 outcome evaluation, labeling, training, RAG, ONNX, order execution,
and live/demo promotion remain out of scope.
