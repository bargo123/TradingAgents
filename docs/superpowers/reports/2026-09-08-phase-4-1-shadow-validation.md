# Phase 4.1 Complete EURUSD Shadow Validation Report

Date: 2026-09-08  
Repository: `C:\AITrading\TradingAgents`  
Phase 4 baseline: `d4f4a49aab5958c11f9b4f693f66477e19f1d194`

## Scope

This validation stays inside the Phase 4 standalone shadow architecture. It
does not modify the MT5 provider, add execution, change the stock CLI, add
Phase 5 evaluation, or bypass any forex graph stage.

## Configuration

The project `.env` resolved to Ollama locally; hosted provider keys were empty,
so no hosted request was attempted and no key was added. The complete run used
these non-secret runtime values:

```text
llm_provider=ollama
quick_think_llm=qwen3.5:2b
deep_think_llm=qwen3.5:2b
backend_url=http://localhost:11434/v1
count=1
max_tokens=64
max_debate_rounds=1
max_risk_discuss_rounds=1
checkpoint_enabled=false
global_news_article_limit=1
global_news_lookback_days=1
global_news_queries=["Federal Reserve interest rates inflation"]
```

The one-query/news limit is a validation-time runtime optimization only. It
does not remove the News analyst or any downstream node. Bull/Bear research,
Research Manager, Trader, Aggressive/Conservative/Neutral Risk, and Portfolio
Manager all executed.

The run was invoked through `ForexShadowRunner`, which is the production path
used by `forex-shadow`; the CLI wrapper only constructs the runner, supplies
the stats callback, and prints its result. A prior wrapper-level CLI attempt
was stopped after more than 17 minutes without persistence; the direct runner
run below was allowed to finish rather than fabricating a result.

## Complete decision evidence

```json
{
  "llm_provider": "ollama",
  "quick_model": "qwen3.5:2b",
  "deep_model": "qwen3.5:2b",
  "requested_symbol": "EURUSD",
  "resolved_symbol": "EURUSD",
  "snapshot_timestamp": "2026-09-08T18:03:18.145000+00:00",
  "bid": 1.16304,
  "ask": 1.16305,
  "spread": 0.000009999999999843467,
  "spread_points": 0.9999999999843466,
  "total_llm_calls": 14,
  "total_tool_calls": 3,
  "tokens_in": 29860,
  "tokens_out": 17816,
  "total_runtime_seconds": 2595.45,
  "raw_portfolio_manager_recommendation": {
    "executive_summary": "Maintain current EURUSD exposure at 0% leverage. Technical evidence is mixed with short-term bullish momentum (M1) conflicting against medium/long-term bearish pressure across H1, M5, and M15 charts. No macro data available for September 8, 2026 to confirm trend direction. Ambiguity warrants maintaining current position rather than taking directional stance. Monitor for clearer signals on higher timeframes (H4, H8) or wait for macro catalysts to confirm direction.",
    "investment_thesis": "The EURUSD technical analysis presents a short-term bullish divergence against medium/long-term bearish pressure. M1 chart shows upward momentum with price closing at 1.16304, while H1, M5, and M15 charts show downward pressure across all relevant timeframes, all closing at 1.16304. This creates a contradictory signal structure where the immediate timeframe suggests bullishness but higher timeframes indicate bearish dominance. Without macro data available for September 8, 2026 (no central bank announcements, inflation reports, or geopolitical events), technical evidence is insufficient to justify changing exposure. The current price action shows some upward movement on M1, but higher timeframes show downward momentum across all relevant timeframes. This ambiguity warrants maintaining the current position rather than taking a directional stance.",
    "price_target": 1.16305,
    "rating": "Hold",
    "time_horizon": "3-6 months"
  },
  "normalized_action": "HOLD",
  "normalization_status": "NORMALIZED",
  "normalization_error": null,
  "shadow_decision_id": "b8104eac-d1b7-40cd-91fb-2055f1ecd45f",
  "shadow_database": "C:\\AITrading\\TradingAgents\\data_cache\\phase41-probe-20260908-180317.db",
  "executed": false
}
```

## Broker mutation check

The normalized MT5 snapshot contained no positions. The read-only preflight
check recorded before the validation sequence (and again after the earlier
stopped wrapper attempt) returned no positions/orders; the post-run provider
check returned:

```text
positions_before=[]
positions_after=[]
orders_before=[]
orders_after=[]
```

The provider, adapter, runner, and injected graph tools expose no order or
position mutation API, and no `order_send` call was made.

## Verification

The complete decision row was reloaded from SQLite and verified for the exact
raw Portfolio Manager mapping, `HOLD`, `NORMALIZED`, resolved symbol,
timezone-aware timestamp, quote/spread fields, model/provider identifiers, and
`executed=False`. The Phase 4 unit/integration/safety suites, Ruff, compileall,
and git diff checks were run after the instrumentation change.

The local Qwen path is proven complete but slow (43m15s). A configured hosted
provider such as OpenAI/GPT-5.6 Luna can be selected later through the existing
environment/config system without changing this architecture or adding a
secret CLI argument.
