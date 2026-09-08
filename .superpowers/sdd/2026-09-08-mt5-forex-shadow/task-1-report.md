# Task 1 Report: Strict Structured Decision Contract and Shadow SQLite Store

Baseline SHA: `1a3a3f8671aa874619c3e925ea443599177ba98f`

## What changed

- Added `invoke_structured_only()` and `StructuredOutputRequiredError` to `tradingagents/agents/utils/structured.py`.
- Created `tradingagents/forex/__init__.py`.
- Created `tradingagents/forex/shadow.py` with:
  - `ShadowNormalization`
  - `ShadowTradeDecision`
  - `normalize_portfolio_manager_result()`
  - `ShadowDecisionStore`
- Added `tests/test_forex_shadow_contract.py` to cover:
  - structured-only invocation failure
  - strict portfolio-manager normalization
  - failed normalization preservation
  - `executed=False` enforcement
  - SQLite round-trip, idempotent insert, and pending lookup

## Verification

Commands run:

```text
pytest C:\AITrading\TradingAgents\tests\test_forex_shadow_contract.py -q
```

Result:

```text
5 passed in 0.37s
```

Additional verification:

```text
python -m py_compile C:\AITrading\TradingAgents\tradingagents\agents\utils\structured.py C:\AITrading\TradingAgents\tradingagents\forex\shadow.py C:\AITrading\TradingAgents\tests\test_forex_shadow_contract.py
```

Result: passed.

Attempted lint check:

```text
ruff check C:\AITrading\TradingAgents\tradingagents\agents\utils\structured.py C:\AITrading\TradingAgents\tradingagents\forex\shadow.py C:\AITrading\TradingAgents\tests\test_forex_shadow_contract.py
```

Result: unavailable in this shell (`ruff` not recognized).

## Notes / concerns

- The local environment does not have `ruff` on PATH, so lint verification could not be completed here.
- The contract tests deliberately load `tradingagents/agents/utils/structured.py` directly by file path to avoid unrelated package-import side effects from optional agent dependencies.
- No graph, runner, CLI, or MT5 execution-path changes were made in this task.

## Fix round evidence

Reviewer findings addressed:

- Canonical schema identity now comes from `tradingagents.agents.schemas` in `tradingagents/forex/shadow.py`, rather than a private module alias.
- `ShadowTradeDecision` now requires and round-trips `snapshot_timestamp` as a UTC datetime.
- Non-UTC offsets are rejected instead of being normalized silently.
- Exact PortfolioRating strings are accepted without trimming; whitespace and case variants fail closed.
- `future_evaluation_status` is validated.
- Unsupported raw values now fail normalization explicitly instead of deferring a `TypeError`.

Verification after the fix:

```text
pytest C:\AITrading\TradingAgents\tests\test_forex_shadow_contract.py -q
```

Result:

```text
13 passed in 0.35s
```

```text
python -m py_compile C:\AITrading\TradingAgents\tradingagents\agents\utils\structured.py C:\AITrading\TradingAgents\tradingagents\forex\shadow.py C:\AITrading\TradingAgents\tests\test_forex_shadow_contract.py
```

Result: passed.
