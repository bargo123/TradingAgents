# Phase 4.2 Forex Trader Calibration — Validation Report

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
