# Task 10 — knowledge retrieval quality and isolation evidence

## Scope

- Added a local ten-case HFT/FX benchmark fixture and deterministic metric
  helper for Recall@1/5/10, MRR, exact-term hits, content-type precision,
  provenance correctness, duplicate-result rate, and repeated-query ordering.
- Added real-ingestion source snapshots that compare relative paths, bytes,
  size, mode, and nanosecond modification time before and after indexing.
- Added action/trade-field checks for public `KnowledgeHit` output and AST /
  console-entrypoint isolation checks for the knowledge package.
- Updated operator documentation with the operational-evidence boundaries.
- Did not provision models, contact a network service, run Ollama/MT5/CUDA, or
  modify stock, forex, trading, or execution code.

## TDD evidence

The initial focused command was intentionally red because the required
dependency-free benchmark module did not exist:

```powershell
C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m pytest tests/test_knowledge_quality.py tests/test_knowledge_isolation.py -q
# ModuleNotFoundError: No module named 'tradingagents.knowledge.quality'
```

The minimal implementation then made the same focused suite green.

## Verification

```powershell
C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m pytest tests/test_knowledge_quality.py tests/test_knowledge_isolation.py -q
# 8 passed

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m pytest <all tests/test_knowledge_*.py> -m "not integration" -q
# 160 passed, 1 deselected

C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m compileall -q tradingagents/knowledge tests/test_knowledge_quality.py tests/test_knowledge_isolation.py
C:\AITrading\TradingAgents\.venv\Scripts\python.exe -m ruff check tradingagents/knowledge/quality.py tests/test_knowledge_quality.py tests/test_knowledge_isolation.py
git diff --check
```

Compilation, scoped Ruff, and whitespace validation exited zero. A full
knowledge-package Ruff invocation still reports pre-existing unrelated style
findings in earlier Task 1–9 files; Task 10's new/changed Python files are
clean.
