# Tasks 12–14 — Phase 11 offline fixture and acceptance report

Implemented the test-only Phase 11 training fixture, positive tiny CPU LoRA
smoke, and opt-in real Phase 10 empty-generation acceptance.

## Delivered

- `tests/fixtures/phase11_training.py` builds a deterministic GPT-2-style
  decoder and `PreTrainedTokenizerFast` with an explicit native chat template,
  and publishes six valid Phase 10 rows covering BUY, SELL, and HOLD with a
  test row kept outside training. Model/tokenizer files are created only under
  pytest temporary directories.
- `tests/test_phase11_training_fixture.py` proves save/reload of the generated
  model and tokenizer with local-files-only loading and no download seam.
- `tests/test_phase11_training_smoke.py` runs two CPU optimizer steps through
  PEFT LoRA, checks split/action counts, adapter reload, frozen-base and
  trainable-parameter invariants, manifests/metrics, secret/CoT exclusion,
  test-split immutability, and a network guard.
- `scripts/phase11_real_empty_acceptance.py` performs a bounded read-only
  check of `generation-d78d143c760b021f48200255`, validating source
  fingerprints and short-circuiting to `EMPTY_ELIGIBLE_SET` /
  `NO_TRAINING_ATTEMPTED` before tokenizer or training setup.
- `tests/test_phase11_real_empty_acceptance.py` keeps the real check opt-in via
  `PHASE11_REAL_ACCEPTANCE=1`, forbids the training stack seam, and verifies
  source artifact bytes are unchanged. Missing real paths return a bounded
  skip/status.

## Verification

```text
.venv\Scripts\python.exe -m pytest -q tests/test_phase11_training_fixture.py tests/test_phase11_training_smoke.py tests/test_phase11_real_empty_acceptance.py
3 passed, 1 skipped

PHASE11_REAL_ACCEPTANCE=1 .venv\Scripts\python.exe -m pytest -q tests/test_phase11_real_empty_acceptance.py
2 passed

python -m pytest -q tests/test_phase11_training_fixture.py tests/test_phase11_training_smoke.py tests/test_phase11_real_empty_acceptance.py
2 passed, 2 skipped

.venv\Scripts\python.exe -m ruff check tests/fixtures/phase11_training.py tests/test_phase11_training_fixture.py tests/test_phase11_training_smoke.py tests/test_phase11_real_empty_acceptance.py scripts/phase11_real_empty_acceptance.py
All checks passed
```

The system interpreter skips the positive PEFT smoke because `peft` is absent;
the skip message names the missing optional training extra. No Phase 5–10
source artifact was modified and no binary model asset is tracked.
