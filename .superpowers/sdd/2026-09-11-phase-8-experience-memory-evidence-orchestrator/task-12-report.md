# Phase 8 Task 12 report

Implemented the bounded local acceptance smoke in `scripts/experience_phase8_smoke.py` and its focused harness tests in `tests/test_experience_task12_scripts.py`.

The smoke requires explicit source, fresh Phase 8 artifact, Phase 7 artifact, local embedding model, and `--offline` paths. It rejects any existing/non-empty Phase 8 root, installs the offline socket/URL guard before service construction, sets the Phase 7 offline environment flags, reads the Phase 5/6 SQLite source through `ReadonlySourceReader`, imports into a new Phase 8 projection, extracts features, builds normalization cohorts, runs exact experience/statistics services, and composes a mandatory Phase 7 knowledge plus Phase 8 evidence query through the Phase 7 read contracts. Reports include source/WAL integrity, counts, tiers, aliases/quarantine, generation and embedding identity, hit/evidence status, exclusions, latency, and network attempts. No analysis, MT5, Qwen/Ollama, or source writer is constructed; `analysis_invocations` is explicitly zero.

## Verification

- RED: initial `pytest tests/test_experience_task12_scripts.py -q` failed during collection because the smoke module did not exist.
- GREEN: `pytest tests/test_experience_task12_scripts.py tests/test_experience_importer.py tests/test_experience_orchestrator.py -q` → **13 passed**.
- Review follow-up RED: added coverage for the missing `experience_artifact_root` report field and missing Phase 7 preflight; both initially failed.
- Review follow-up GREEN: focused/regression command → **15 passed** after adding the report key and requiring an existing Phase 7 `catalog.sqlite3` before constructing `KnowledgeCatalog`.
- Review round 2 RED: an existing empty Phase 8 directory produced raw `FileExistsError`.
- Review round 2 GREEN: `_ensure_fresh()` now rejects every existing path, including empty directories, with `ExperienceArtifactNotEmptyError`; focused/regression command → **16 passed**.
- Compile check: `python -m compileall -q scripts/experience_phase8_smoke.py` → passed.
- `git diff --check` → passed.
- Ruff was unavailable in the environment (`ruff` command not found).
- Full repository collection is blocked by pre-existing missing optional dependencies including `requests`, `langchain_core`, `langgraph`, `typer`, `questionary`, and `yfinance`; no Task 12 failure was observed in the focused/regression run.

No real Phase 5/6 + Phase 7 acceptance run was possible in this checkout because those external artifacts/model paths were not provided.
