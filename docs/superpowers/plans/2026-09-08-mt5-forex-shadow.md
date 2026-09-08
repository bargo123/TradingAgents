# MT5 Forex Shadow Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ( - [ ] ) syntax for tracking.

**Goal:** Add a separate, read-only 'forex-shadow' command that obtains one normalized MT5 forex snapshot, reuses the existing TradingAgents graph with forex-safe context, and persists a strictly normalized shadow decision with 'executed=False'.

**Architecture:** Add a focused 'tradingagents.forex' package for context formatting, an injected read-only MT5 adapter, strict structured-result normalization, SQLite persistence, and orchestration. Extend existing graph/state interfaces with optional 'market_data_mode="stock"' and injected tools; keep stock defaults and 'cli/main.py' unchanged. The runner owns one provider lifecycle and one snapshot cache.

**Tech Stack:** Python 3.10+, existing LangGraph/LangChain graph and Pydantic schemas, Phase 3 'MT5Provider' and frozen models, stdlib 'sqlite3', argparse, pytest, Ruff.

**Spec:** 'docs/superpowers/specs/2026-09-08-mt5-forex-shadow-design.md'

## Global Constraints

- Phase 4 is read-only: no 'order_send', buy/sell/open/close/modify/pending-order/SL/TP mutation API, execution credential, live mode, or automated trading path.
- 'forex-shadow' is a separate command; do not modify 'cli/main.py' or the existing 'tradingagents' stock entry point.
- The default forex analyst allow-list is exactly 'market,news'; reject 'social' and 'fundamentals' in forex mode.
- Forex news may use only broad/global/macro-relevant news; do not inject ticker/company news, earnings/fundamentals, StockTwits, Reddit, P/E, EPS, dividends, or equivalent stock-specific context.
- Reuse existing TradingAgents graph/components; do not duplicate the agent framework or register MT5 as a normal stock vendor.
- Initialize one Phase 3 'MT5Provider', resolve symbols exact-first and fail-closed, fetch one 'ForexMarketSnapshot', cache it for the run, and always shut down in 'finally'.
- Tick timestamps prefer 'time_msc' and all timestamps are timezone-aware UTC; broker prefix/suffix variants are controlled and ambiguity raises 'Mt5SymbolAmbiguousError'.
- Forex action normalization is strict and structured: use the 'PortfolioDecision' result or its serialized structured mapping only; never infer BUY/SELL/HOLD from free-form prose. Missing, malformed, or ambiguous results remain 'normalization_status=FAILED' with null action.
- Persist the raw final Portfolio Manager result, normalized action (or null), normalization status/error, requested/resolved symbols, snapshot timestamp, bid/ask/spread, provider/model identifiers, normalized snapshot JSON, and 'executed=False'.
- The existing stock graph/tool behavior and stock tests must remain compatible.
- Run each task's failing test before its implementation, then the focused test and relevant regression tests after implementation.

---

## File map

### New production files

- 'tradingagents/forex/__init__.py': public Phase 4 exports.
- 'tradingagents/forex/context.py': bounded deterministic snapshot serializer/context formatter.
- 'tradingagents/forex/tools.py': injected read-only 'MT5ToolAdapter' and graph tool wrappers.
- 'tradingagents/forex/shadow.py': strict normalization value objects and SQLite repository.
- 'tradingagents/forex/runner.py': one-session/one-snapshot forex shadow orchestration.
- 'cli/forex_shadow.py': standalone argparse command and prominent safety output.

### Modified production files

- 'tradingagents/agents/utils/structured.py': strict structured invocation helper while preserving stock fallback.
- 'tradingagents/agents/utils/agent_states.py': optional forex mode/context/raw-normalization fields.
- 'tradingagents/graph/propagation.py': initialize the new state fields with stock-compatible defaults.
- 'tradingagents/agents/utils/agent_utils.py': append shared forex market context without a network lookup.
- 'tradingagents/graph/trading_graph.py': optional mode/adapter constructor, conditional read-only ToolNodes, mode-aware checkpoint signature.
- 'tradingagents/graph/setup.py': pass mode/adapter to factories and reject disallowed forex analysts.
- 'tradingagents/agents/analysts/market_analyst.py': forex MT5 snapshot prompt/tool branch.
- 'tradingagents/agents/analysts/news_analyst.py': forex global-news-only prompt/tool branch.
- 'tradingagents/agents/researchers/bull_researcher.py' and 'bear_researcher.py': forex-safe debate prompts.
- 'tradingagents/agents/risk_mgmt/aggressive_debator.py', 'conservative_debator.py', 'neutral_debator.py': forex-safe risk prompts.
- 'tradingagents/agents/trader/trader.py': forex-specific price/spread instruction while preserving structured trader output.
- 'tradingagents/agents/managers/portfolio_manager.py': retain raw structured 'PortfolioDecision' and strict forex failure marker.
- 'pyproject.toml': add 'forex-shadow = "cli.forex_shadow:main"' only.

### New tests

- 'tests/test_forex_shadow_contract.py'
- 'tests/test_forex_shadow_context_tools.py'
- 'tests/test_forex_graph_mode.py'
- 'tests/test_forex_prompts.py'
- 'tests/test_forex_shadow_runner.py'
- 'tests/test_forex_shadow_cli.py'
- 'tests/test_forex_shadow_integration.py'

---

### Task 1: Strict structured decision contract and shadow SQLite store

**Files:**

- Create: 'tradingagents/forex/__init__.py'
- Create: 'tradingagents/forex/shadow.py'
- Modify: 'tradingagents/agents/utils/structured.py'
- Test: 'tests/test_forex_shadow_contract.py'

**Interfaces:**

- 'invoke_structured_only(structured_llm: Any | None, prompt: Any, agent_name: str) -> BaseModel' raises 'StructuredOutputRequiredError' when the binding is unavailable, returns 'None', or invocation fails; it never calls the plain LLM.
- 'normalize_portfolio_manager_result(raw: PortfolioDecision | Mapping[str, Any] | None) -> ShadowNormalization' validates only the exact 'PortfolioRating' vocabulary and maps Buy/Overweight to BUY, Hold to HOLD, and Underweight/Sell to SELL. A string/prose result, missing rating, unknown rating, or conflicting mapping returns status FAILED and action None.
- 'ShadowNormalization' is frozen and contains 'action: str | None', 'normalization_status: Literal["NORMALIZED","FAILED"]', 'normalization_error: str | None', and a JSON-safe 'raw_result: dict[str, Any]'.
- 'ShadowTradeDecision' is frozen/slotted and contains the normalized action (nullable on failure), raw structured PM result, normalization status/error, UTC timestamps, symbols, bid/ask/spread/spread-points, summaries, provider/model identifiers, normalized snapshot payload, future-evaluation fields, and 'executed: bool = False'.
- 'ShadowDecisionStore(path: str | Path)' provides 'initialize()', 'record(decision)', 'get(decision_id)', 'list_pending(resolved_symbol: str | None = None)', and 'update_outcome(decision_id, outcome_raw, outcome_alpha, outcome_resolved_at, reflection)'. It uses only parameterized stdlib SQLite statements and cannot update 'executed'.

- [ ] **Step 1: Write the failing tests**

~~~python
def test_normalization_uses_structured_rating_only():
    result = normalize_portfolio_manager_result(
        PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="x",
            investment_thesis="y",
        )
    )
    assert result.action == "BUY"
    assert result.normalization_status == "NORMALIZED"
    assert result.raw_result["rating"] == "Overweight"


def test_normalization_rejects_prose_and_preserves_failure():
    result = normalize_portfolio_manager_result("BUY now; HOLD if uncertain")
    assert result.action is None
    assert result.normalization_status == "FAILED"
    assert "structured" in result.normalization_error.lower()


def test_shadow_decision_rejects_executed_true_and_invalid_status():
    with pytest.raises(ValueError):
        make_decision(executed=True)
    with pytest.raises(ValueError):
        make_decision(action="HOLD", normalization_status="FAILED")


def test_store_round_trip_is_idempotent_and_preserves_failed_action(tmp_path):
    store = ShadowDecisionStore(tmp_path / "shadow.db")
    failed = make_decision(action=None, normalization_status="FAILED")
    store.record(failed)
    store.record(failed)
    restored = store.get(failed.decision_id)
    assert restored.action is None
    assert restored.normalization_status == "FAILED"
    assert store.list_pending() == [restored]
~~~

- [ ] **Step 2: Run the focused tests and verify the expected RED failure**

Run:

~~~text
pytest tests/test_forex_shadow_contract.py -q
~~~

Expected: collection or assertion failures because the new module, strict helper, and store do not exist.

- [ ] **Step 3: Implement the minimal strict contract and schema**

Add a structured-only helper that does not invoke 'plain_llm'. Serialize Pydantic results with 'model_dump(mode="json")'; accept a mapping only when it contains one exact 'rating' value from 'PortfolioRating'; reject all strings and conflicting/unknown values. Define the SQLite table with nullable 'action', 'normalization_status' in NORMALIZED/FAILED, raw result JSON, a consistency CHECK requiring action for NORMALIZED and null action for FAILED, and 'executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0)'. Use 'INSERT ... ON CONFLICT DO NOTHING' so duplicate decision IDs are idempotent.

- [ ] **Step 4: Run focused and schema tests to verify GREEN**

Run:

~~~text
pytest tests/test_forex_shadow_contract.py -q
~~~

Expected: all new contract/store tests pass with no warnings.

- [ ] **Step 5: Commit the task**

~~~text
git add tradingagents/forex/__init__.py tradingagents/forex/shadow.py tradingagents/agents/utils/structured.py tests/test_forex_shadow_contract.py
git commit -m "feat: add strict forex shadow decision store"
~~~

---

### Task 2: Deterministic forex context and read-only MT5 adapter

**Files:**

- Create: 'tradingagents/forex/context.py'
- Create: 'tradingagents/forex/tools.py'
- Modify: 'tradingagents/forex/__init__.py'
- Test: 'tests/test_forex_shadow_context_tools.py'

**Interfaces:**

- 'snapshot_to_dict(snapshot: ForexMarketSnapshot) -> dict[str, Any]' emits explicit JSON-safe fields for timestamp, symbol, quote, spread, symbol metadata, account summary, positions, and bounded M1/M5/M15/H1 bars.
- 'build_forex_market_context(snapshot: ForexMarketSnapshot) -> str' begins with 'SOURCE: LIVE MT5 BROKER DATA (read-only; not Yahoo Finance)' and includes only bounded latest/basic OHLC-derived values and quote metadata.
- 'MT5ToolAdapter(provider: MT5Provider, snapshot: ForexMarketSnapshot | None = None)' has 'cache_snapshot(snapshot)', read-only methods 'get_mt5_market_snapshot', 'get_mt5_tick', 'get_mt5_bars', 'get_mt5_account_context', 'get_mt5_positions', 'get_mt5_spread', and 'as_tools()'. It never initializes/shuts down the provider and returns primitives or normalized mappings.

- [ ] **Step 1: Write failing formatter, cache, and adapter tests**

~~~python
def test_context_is_bounded_and_source_labelled(fake_snapshot):
    context = build_forex_market_context(fake_snapshot)
    assert context.startswith("SOURCE: LIVE MT5 BROKER DATA")
    assert "EURUSD" in context
    assert "bid=" in context and "ask=" in context and "spread_points=" in context
    assert "object at 0x" not in context
    assert len(context) < 12000


def test_snapshot_tool_uses_cache_without_second_provider_snapshot(fake_provider, fake_snapshot):
    adapter = MT5ToolAdapter(fake_provider, fake_snapshot)
    first = adapter.get_mt5_market_snapshot("EURUSD")
    second = adapter.get_mt5_market_snapshot("EURUSD")
    assert first == second
    assert fake_provider.market_snapshot_calls == 0


def test_adapter_methods_return_json_safe_values(fake_provider, fake_snapshot):
    adapter = MT5ToolAdapter(fake_provider, fake_snapshot)
    payload = adapter.get_mt5_tick("EURUSD")
    json.dumps(payload)
    assert not any(name.startswith("order") for name in dir(adapter))
~~~

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

~~~text
pytest tests/test_forex_shadow_context_tools.py -q
~~~

Expected: import or assertion failures because context and adapter are absent.

- [ ] **Step 3: Implement explicit serialization and adapter tools**

Serialize datetimes with UTC ISO-8601 strings, cap each timeframe to the latest bar plus count/high/low/range/direction/return, and include only non-sensitive account fields needed by read-only risk context. Build LangChain 'StructuredTool' wrappers from the bound adapter methods with stable names. If no cached snapshot exists, 'get_mt5_market_snapshot' must raise a clear cache error rather than silently fetching a second snapshot.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

~~~text
pytest tests/test_forex_shadow_context_tools.py -q
~~~

Expected: all formatter/cache/primitive-output tests pass.

- [ ] **Step 5: Commit the task**

~~~text
git add tradingagents/forex/__init__.py tradingagents/forex/context.py tradingagents/forex/tools.py tests/test_forex_shadow_context_tools.py
git commit -m "feat: add cached forex context and read-only adapter"
~~~

---

### Task 3: State and graph forex-mode plumbing with stock compatibility

**Files:**

- Modify: 'tradingagents/agents/utils/agent_states.py'
- Modify: 'tradingagents/graph/propagation.py'
- Modify: 'tradingagents/agents/utils/agent_utils.py'
- Modify: 'tradingagents/graph/trading_graph.py'
- Modify: 'tradingagents/graph/setup.py'
- Test: 'tests/test_forex_graph_mode.py'

**Interfaces:**

- 'AgentState' adds 'market_data_mode', 'market_context', 'portfolio_manager_raw_result', 'normalization_status', and 'normalization_error'.
- 'Propagator.create_initial_state(..., market_data_mode: str = "stock", market_context: str = "")' carries the new fields and initializes empty raw/normalization values without changing stock defaults.
- 'TradingAgentsGraph(..., market_data_mode: str = "stock", mt5_tools: MT5ToolAdapter | None = None)' validates 'stock' or explicit 'forex_mt5', stores the mode, and preserves the current stock ToolNodes for default mode.
- 'GraphSetup(..., market_data_mode: str = "stock", mt5_tools: MT5ToolAdapter | None = None)' passes mode/adapter to market/news factories and rejects 'social'/'fundamentals' before building a forex graph.
- '_run_signature' includes 'market_data_mode'.
- 'get_instrument_context_from_state' returns the existing identity context plus one bounded 'market_context' block when present; no network access is introduced.

- [ ] **Step 1: Write failing mode/state/stock-regression tests**

~~~python
def test_stock_graph_toolnode_contract_is_unchanged():
    nodes = TradingAgentsGraph._create_tool_nodes(None)
    assert {"get_stock_data", "get_indicators", "get_verified_market_snapshot"} <= set(
        nodes["market"].tools_by_name
    )


def test_forex_initial_state_carries_mode_and_context():
    state = Propagator().create_initial_state(
        "EURUSD", "2026-09-08", asset_type="forex",
        market_data_mode="forex_mt5", market_context="SOURCE: LIVE MT5 BROKER DATA",
    )
    assert state["market_data_mode"] == "forex_mt5"
    assert "LIVE MT5" in state["market_context"]


def test_forex_graph_rejects_stock_specific_analysts(fake_mt5_tools):
    with pytest.raises(ValueError, match="social|fundamentals"):
        TradingAgentsGraph(
            selected_analysts=("market", "fundamentals"),
            market_data_mode="forex_mt5",
            mt5_tools=fake_mt5_tools,
        )


def test_shared_context_appends_market_context_without_yahoo_lookup():
    state = {
        "company_of_interest": "EURUSD",
        "asset_type": "forex",
        "instrument_context": "The pair is EURUSD.",
        "market_context": "SOURCE: LIVE MT5 BROKER DATA",
    }
    context = get_instrument_context_from_state(state)
    assert context.count("SOURCE: LIVE MT5 BROKER DATA") == 1
~~~

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

~~~text
pytest tests/test_forex_graph_mode.py -q
~~~

Expected: failures for missing state fields, constructor arguments, mode-specific tools, and context behavior.

- [ ] **Step 3: Implement optional state and graph plumbing**

Keep the stock branch byte-for-byte behaviorally equivalent. In forex mode create the market ToolNode from 'mt5_tools.as_tools()' and the news ToolNode from only 'get_global_news'; do not route MT5 through 'get_stock_data' or the vendor registry. Pass mode and adapter arguments through 'GraphSetup' factory lambdas, validate the selected list before accessing tool nodes, and include the mode in the checkpoint signature.

- [ ] **Step 4: Run focused tests plus existing market ToolNode regression**

Run:

~~~text
pytest tests/test_forex_graph_mode.py tests/test_market_toolnode.py -q
~~~

Expected: all mode tests and the unchanged stock ToolNode regression pass.

- [ ] **Step 5: Commit the task**

~~~text
git add tradingagents/agents/utils/agent_states.py tradingagents/graph/propagation.py tradingagents/agents/utils/agent_utils.py tradingagents/graph/trading_graph.py tradingagents/graph/setup.py tests/test_forex_graph_mode.py
git commit -m "feat: add optional forex graph mode"
~~~

---

### Task 4: Forex-safe analyst prompts and strict Portfolio Manager output

**Files:**

- Modify: 'tradingagents/agents/analysts/market_analyst.py'
- Modify: 'tradingagents/agents/analysts/news_analyst.py'
- Modify: 'tradingagents/agents/researchers/bull_researcher.py'
- Modify: 'tradingagents/agents/researchers/bear_researcher.py'
- Modify: 'tradingagents/agents/risk_mgmt/aggressive_debator.py'
- Modify: 'tradingagents/agents/risk_mgmt/conservative_debator.py'
- Modify: 'tradingagents/agents/risk_mgmt/neutral_debator.py'
- Modify: 'tradingagents/agents/trader/trader.py'
- Modify: 'tradingagents/agents/managers/portfolio_manager.py'
- Test: 'tests/test_forex_prompts.py'

**Interfaces:**

- 'create_market_analyst(llm, market_data_mode: str = "stock", mt5_tools: MT5ToolAdapter | None = None)' binds stock tools in stock mode and only the cached MT5 snapshot tool in forex mode.
- 'create_news_analyst(llm, market_data_mode: str = "stock")' binds the existing stock news tools in stock mode and only 'get_global_news' in forex mode.
- Downstream factories retain their existing signatures and branch on 'state["asset_type"] == "forex"' to remove company/fundamentals/social instructions while consuming 'get_instrument_context_from_state'.
- Forex Portfolio Manager invokes 'invoke_structured_only', returns rendered 'final_trade_decision' plus JSON-safe 'portfolio_manager_raw_result', and returns a deterministic failed marker when structured output is unavailable or invalid. Stock mode keeps the current graceful prose fallback.

- [ ] **Step 1: Write failing prompt and raw-output tests**

~~~python
def test_forex_market_prompt_uses_mt5_snapshot_only(captured_llm, fake_mt5_tools):
    create_market_analyst(captured_llm, "forex_mt5", fake_mt5_tools)({
        "trade_date": "2026-09-08",
        "asset_type": "forex",
        "company_of_interest": "EURUSD",
        "instrument_context": "EURUSD pair",
        "market_context": "SOURCE: LIVE MT5 BROKER DATA",
        "messages": [("human", "analyze EURUSD")],
    })
    prompt_text = captured_llm.last_prompt_text
    assert "get_mt5_market_snapshot" in prompt_text
    assert "get_stock_data" not in prompt_text
    assert "P/E" not in prompt_text and "EPS" not in prompt_text


def test_forex_news_binds_global_news_only(captured_llm):
    create_news_analyst(captured_llm, "forex_mt5")({
        "trade_date": "2026-09-08",
        "asset_type": "forex",
        "company_of_interest": "EURUSD",
        "instrument_context": "EURUSD pair",
        "market_context": "SOURCE: LIVE MT5 BROKER DATA",
        "messages": [("human", "analyze EURUSD")],
    })
    assert captured_llm.bound_tool_names == ["get_global_news"]
    assert "ticker-specific" not in captured_llm.last_prompt_text.lower()


def test_forex_portfolio_manager_keeps_structured_raw_result(captured_llm):
    raw = PortfolioDecision(
        rating=PortfolioRating.SELL,
        executive_summary="x",
        investment_thesis="y",
    )
    result = create_portfolio_manager(captured_llm)({
        "asset_type": "forex",
        "instrument_context": "EURUSD pair",
        "risk_debate_state": empty_risk_state(),
        "investment_plan": "plan",
        "trader_investment_plan": "trade",
        "past_context": "",
    })
    assert result["portfolio_manager_raw_result"]["rating"] == "Sell"
    assert result["final_trade_decision"].startswith("**Rating**: Sell")


def test_forex_prompts_contain_no_stock_specific_sections(fake_llm):
    for node in make_forex_downstream_nodes(fake_llm):
        text = invoke_and_capture(node, forex_state())
        assert "Company Fundamentals Report" not in text
        assert "StockTwits" not in text
~~~

- [ ] **Step 2: Run the focused prompt tests and verify RED**

Run:

~~~text
pytest tests/test_forex_prompts.py -q
~~~

Expected: failures because factories still bind stock tools/prompts and PM does not preserve raw structured output.

- [ ] **Step 3: Implement mode branches and strict PM behavior**

Keep all stock prompt strings and tool lists in the stock branch. In forex branches describe only pair price action, bid/ask/spread, volatility, MT5 snapshot evidence, and lightweight broad/global news. Do not mention ticker/company earnings or stock metrics in the forex branch. For bull/bear/risk prompts replace company fundamentals sections with an explicit unavailable/no-inference guard. Add a forex trader instruction that price levels are observational only and no order is sent. In the PM node call the strict helper and store 'model_dump(mode="json")'; on failure set raw result to an explicit error mapping, normalization status to FAILED, and action remains absent for the runner/store to preserve.

- [ ] **Step 4: Run focused tests plus existing structured-agent tests**

Run:

~~~text
pytest tests/test_forex_prompts.py tests/test_structured_agents.py -q
~~~

Expected: new forex prompt/raw-output tests and all existing structured-agent tests pass.

- [ ] **Step 5: Commit the task**

~~~text
git add tradingagents/agents/analysts/market_analyst.py tradingagents/agents/analysts/news_analyst.py tradingagents/agents/researchers/bull_researcher.py tradingagents/agents/researchers/bear_researcher.py tradingagents/agents/risk_mgmt/aggressive_debator.py tradingagents/agents/risk_mgmt/conservative_debator.py tradingagents/agents/risk_mgmt/neutral_debator.py tradingagents/agents/trader/trader.py tradingagents/agents/managers/portfolio_manager.py tests/test_forex_prompts.py
git commit -m "feat: add forex-safe prompts and strict PM output"
~~~

---

### Task 5: One-session, one-snapshot ForexShadowRunner

**Files:**

- Create: 'tradingagents/forex/runner.py'
- Modify: 'tradingagents/forex/__init__.py'
- Test: 'tests/test_forex_shadow_runner.py'

**Interfaces:**

- 'ForexShadowRunResult' is frozen and contains 'decision: ShadowTradeDecision', 'final_state: Mapping[str, Any]', and metrics.
- 'ForexShadowRunner(provider_factory=MT5Provider, graph_factory=TradingAgentsGraph, store: ShadowDecisionStore | None = None, config: Mapping[str, Any] | None = None)' owns lifecycle.
- 'run(symbol: str = "EURUSD", count: int = 100, analysis_date: date | str | None = None, terminal_path: str | None = None, analysts: Sequence[str] = ("market", "news")) -> ForexShadowRunResult' validates inputs, initializes provider, resolves symbol, fetches one snapshot, builds state/context, invokes the existing compiled graph directly, persists the decision, and shuts down in 'finally'.

- [ ] **Step 1: Write failing runner tests**

~~~python
def test_runner_fetches_one_snapshot_and_persists_normalized_decision(fake_provider, fake_graph, tmp_path):
    result = ForexShadowRunner(
        provider_factory=lambda terminal_path=None: fake_provider,
        graph_factory=lambda **kwargs: fake_graph,
        store=ShadowDecisionStore(tmp_path / "shadow.db"),
    ).run()
    assert fake_provider.market_snapshot_calls == 1
    assert fake_provider.shutdown_called is True
    assert result.decision.action in {"BUY", "SELL", "HOLD"}
    assert result.decision.executed is False


def test_runner_persists_failed_normalization_without_guessing(fake_provider, graph_with_ambiguous_pm, tmp_path):
    result = make_runner(fake_provider, graph_with_ambiguous_pm, tmp_path).run()
    assert result.decision.action is None
    assert result.decision.normalization_status == "FAILED"
    assert result.decision.normalization_error
    assert result.decision.raw_portfolio_manager_result


def test_runner_shuts_down_when_graph_fails(fake_provider, failing_graph):
    with pytest.raises(RuntimeError):
        make_runner(fake_provider, failing_graph).run()
    assert fake_provider.shutdown_called is True
~~~

- [ ] **Step 2: Run the focused runner tests and verify RED**

Run:

~~~text
pytest tests/test_forex_shadow_runner.py -q
~~~

Expected: import or assertion failures because the runner does not exist.

- [ ] **Step 3: Implement the runner**

Use 'provider.initialize()' once and 'provider.get_market_snapshot(symbol, count)' once. Build 'instrument_context' with 'build_instrument_context(symbol, "forex", {})', create the cached adapter, and call 'Propagator.create_initial_state' with 'asset_type="forex"', 'market_data_mode="forex_mt5"', and compact context. Invoke 'graph.graph.invoke(initial_state, graph.propagator.get_graph_args(...))' without 'propagate', Yahoo identity resolution, or stock Markdown memory. Normalize the raw PM result only through 'normalize_portfolio_manager_result'; create a FAILED decision with null action when strict normalization fails; store raw and normalized fields; return metrics; always call shutdown.

- [ ] **Step 4: Run focused runner tests and graph regressions**

Run:

~~~text
pytest tests/test_forex_shadow_runner.py tests/test_forex_graph_mode.py tests/test_market_toolnode.py -q
~~~

Expected: one-snapshot, shutdown, strict-failure, and stock compatibility tests pass.

- [ ] **Step 5: Commit the task**

~~~text
git add tradingagents/forex/__init__.py tradingagents/forex/runner.py tests/test_forex_shadow_runner.py
git commit -m "feat: orchestrate one-snapshot forex shadow runs"
~~~

---

### Task 6: Standalone 'forex-shadow' CLI and documentation

**Files:**

- Create: 'cli/forex_shadow.py'
- Modify: 'pyproject.toml'
- Create: 'docs/forex-shadow.md'
- Test: 'tests/test_forex_shadow_cli.py'

**Interfaces:**

- 'build_parser() -> argparse.ArgumentParser' defines '--symbol EURUSD', positive '--count', UTC '--analysis-date', optional '--terminal-path', '--db-path', '--analysts market,news', and non-secret provider/model/backend overrides.
- 'main(argv: Sequence[str] | None = None) -> int' prints 'MT5 FOREX — SHADOW MODE' and 'NO ORDER WILL BE SENT' before work; on success prints 'SHADOW DECISION RECORDED', 'EXECUTED: FALSE'; errors return non-zero and do not fall back to Yahoo.

- [ ] **Step 1: Write failing CLI and entry-point tests**

~~~python
def test_cli_parser_defaults_and_positive_count():
    args = build_parser().parse_args([])
    assert args.symbol == "EURUSD"
    assert args.analysts == "market,news"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--count", "0"])


def test_cli_prints_shadow_banner_and_never_execution(capsys, monkeypatch):
    monkeypatch.setattr("cli.forex_shadow.ForexShadowRunner.run", fake_successful_run)
    assert main(["--db-path", "shadow.db"]) == 0
    out = capsys.readouterr().out
    assert "MT5 FOREX — SHADOW MODE" in out
    assert "NO ORDER WILL BE SENT" in out
    assert "SHADOW DECISION RECORDED" in out
    assert "EXECUTED: FALSE" in out


def test_stock_cli_entry_point_is_untouched():
    assert read_text("cli/main.py") == stock_cli_baseline_text
~~~

- [ ] **Step 2: Run focused CLI tests and verify RED**

Run:

~~~text
pytest tests/test_forex_shadow_cli.py -q
~~~

Expected: import/entry-point failures because the command and documentation do not exist.

- [ ] **Step 3: Implement CLI and entry point**

Use argparse without interactive questions and reject non-positive counts or invalid analyst names before creating the provider. Pass only the explicit MT5/model/backend options to 'ForexShadowRunner'; do not add credentials or execution flags. Add exactly one '[project.scripts]' line and leave 'tradingagents = "cli.main:app"' unchanged.

- [ ] **Step 4: Run focused CLI/packaging tests**

Run:

~~~text
pytest tests/test_forex_shadow_cli.py -q
python -m cli.forex_shadow --help
~~~

Expected: tests pass, help lists only forex-shadow options, and the stock script line remains present.

- [ ] **Step 5: Commit the task**

~~~text
git add cli/forex_shadow.py pyproject.toml docs/forex-shadow.md tests/test_forex_shadow_cli.py
git commit -m "feat: add standalone forex-shadow command"
~~~

---

### Task 7: Guarded integration, safety scan, full verification, and report

**Files:**

- Create: 'tests/test_forex_shadow_integration.py'
- Modify: 'docs/forex-shadow.md'
- Create: 'docs/superpowers/reports/2026-09-08-mt5-forex-shadow-verification.md'

**Interfaces:**

- The guarded test runs only when 'RUN_MT5_INTEGRATION=1'; it uses the Phase 3 provider and adapter to resolve EURUSD, read one snapshot, and assert UTC/non-empty read-only data without LLM or mutation.
- The verification report records test/lint commands, baseline/scope diff, one-snapshot evidence, provider/model/runtime metrics, decision id/action/status, database path, and positions/orders before/after. Unavailable terminal/Qwen evidence is marked unverified.

- [ ] **Step 1: Write the guarded integration and source-scan tests**

~~~python
@pytest.mark.integration
def test_real_mt5_snapshot_is_read_only():
    if os.getenv("RUN_MT5_INTEGRATION") != "1":
        pytest.skip("set RUN_MT5_INTEGRATION=1")
    provider = MT5Provider()
    provider.initialize()
    try:
        snapshot = provider.get_market_snapshot("EURUSD", count=2)
        assert snapshot.timestamp.tzinfo is timezone.utc
        assert snapshot.bid > 0 and snapshot.ask >= snapshot.bid
    finally:
        provider.shutdown()


def test_phase4_production_sources_expose_no_mutation_api():
    source = "\n".join(read_phase4_production_sources())
    for forbidden in (
        "order_send",
        "send_order",
        "place_order",
        "cancel_order",
        "close_position",
        "modify_position",
    ):
        assert forbidden not in source.lower()
    assert not hasattr(MT5Provider, "order_send")
~~~

- [ ] **Step 2: Run the guarded tests without external services and verify the skip/RED behavior**

Run:

~~~text
pytest tests/test_forex_shadow_integration.py -q
~~~

Expected: the real-terminal test is skipped unless explicitly enabled; the source scan fails until all Phase 4 production files are safe.

- [ ] **Step 3: Implement the integration guard, safety scan, and verification report**

Keep the source scan restricted to production files so test fixture strings do not create false positives. Run the real test only with explicit opt-in. For the smoke run, capture positions and orders before and after, execute 'forex-shadow --symbol EURUSD', and record the raw PM result, normalized result/status, snapshot evidence, provider/model identifiers, and 'executed=False'.

- [ ] **Step 4: Run the complete verification gate**

Run:

~~~text
pytest -q
ruff check .
pytest tests/test_forex_shadow_integration.py -q
~~~

Expected: full suite passes, Ruff is clean, and the guarded integration test is skipped or passes with explicit opt-in.

- [ ] **Step 5: Run one real EURUSD shadow smoke when local MT5 and Qwen are available**

Run the command with the local configuration, capture the required before/after read-only account snapshots, and append the result to 'docs/superpowers/reports/2026-09-08-mt5-forex-shadow-verification.md'. If either dependency is unavailable, document the exact failure and mark the real-run criterion unverified.

- [ ] **Step 6: Commit verification artifacts**

~~~text
git add tests/test_forex_shadow_integration.py docs/forex-shadow.md docs/superpowers/reports/2026-09-08-mt5-forex-shadow-verification.md
git commit -m "test: verify forex shadow safety boundary"
~~~

---

## Plan self-review checklist

- Spec amendments are covered: strict structured-only normalization and explicit FAILED/null-action persistence are Task 1, Task 4, and Task 5; global forex-only news and stock-term exclusion are Task 3 and Task 4; raw-plus-normalized evidence fields are Task 1 and Task 5.
- One-snapshot caching is covered by Task 2 adapter tests and Task 5 runner tests.
- Exact-first prefix/suffix symbol resolution and UTC 'time_msc' behavior remain in the Phase 3 provider and are exercised by Task 7 integration plus existing Phase 3 tests; no provider rewrite is planned.
- Stock CLI compatibility is guarded by Task 3 and Task 6 and no task modifies 'cli/main.py'.
- Phase 5 execution, workflow registration, macro expansion, RAG, training, ONNX, and automated trading are excluded from every task.
- No placeholder work items remain; every task names concrete files, interfaces, failing tests, commands, and commits.
