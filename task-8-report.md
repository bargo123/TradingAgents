# Task 8 implementation report

## Scope

- Added fail-closed provenance validation for returned knowledge hits.
- Added dense/lexical candidate contracts, reciprocal-rank fusion, and a
  deterministic CPU feature reranker.
- Added a read-only `KnowledgeQueryService` that validates the active matched
  generation, compares the complete `EmbeddingSpec` before dense retrieval,
  applies metadata/catalog filters, and returns provenance-complete hits.
- Query code contains no parser, ingestor, source scanner, writer, trading, or
  MT5 integration.

## TDD and verification

The focused query tests were written first and failed at collection because
the Task 8 modules were absent. After implementation:

- `python -m pytest tests/test_knowledge_query.py -q` — 16 passed.
- Tasks 1–8 knowledge suite — 141 passed, 1 skipped because optional `docling`
  is not installed.
- Ruff check via the repository virtual environment — passed.
- `python -m py_compile` for all Task 8 modules and tests — passed.
