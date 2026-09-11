# Task 6 review-fix report

## Scope

- Validated every lexical provenance row against the paired vector row,
  generation ID, source identity, and population identity during both build
  and active-generation resolution.
- Enforced canonical, generation-scoped vector and lexical artifact locations.
- Persisted and validated all `EmbeddingSpec` fields as direct vector-row
  columns, including `embedding_runtime` and
  `embedding_special_token_budget`; retained canonical
  `embedding_spec_json` validation.

## TDD evidence

The new index tests were run before implementation and failed for the expected
missing direct embedding columns and missing row/location integrity checks.

After implementation:

- `pytest tests/test_knowledge_indexes.py -q` — 27 passed.
- Tasks 1–6 suite — 113 passed, 1 skipped because optional `docling` is not
  installed.
- `python -m compileall -q tradingagents/knowledge` — passed.
- `git diff --check` — passed.

`ruff` is not installed in the active Python environment (`python -m ruff`
reports `No module named ruff`).
