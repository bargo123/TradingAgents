# Task 6 — model/tokenizer provenance

Implemented `tradingagents/finetuning/provenance.py` with an offline,
deterministic `ModelProvenance.inspect` API. Local snapshots are inspected
without Transformers, torch, or network access; mutable/unpinned remote
identifiers and missing snapshots fail closed. Provenance includes config,
tokenizer/chat-template, weight/index fingerprints, bounded file metadata,
architecture, dtype, and parameter count (including bounded safetensors-header
inspection when config metadata is absent).

Verification:

- `ruff check tradingagents/finetuning/provenance.py tests/test_phase11_provenance.py` — passed
- `python -m pytest -q tests/test_phase11_provenance.py` — 7 passed
- `git diff --check` — passed

Commit: `3fc3cdd`
