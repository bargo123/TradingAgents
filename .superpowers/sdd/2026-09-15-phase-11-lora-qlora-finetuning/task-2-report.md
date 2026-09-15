# Task 2 report — read-only Phase 10 adapter

Implemented `tradingagents/finetuning/phase10.py` and focused tests in
`tests/test_phase11_phase10.py`.

The adapter calls the public generation validator before reading its own
projection, then reads only `manifest.json`, `train.jsonl`, and
`validation.jsonl`. It reconstructs public `CanonicalExampleV1` rows, checks
generation identity, schema and policy versions, split counts, file hashes,
and row-to-manifest source fingerprints, and exposes immutable binding
metadata. Empty eligible generations fail closed with `EmptyEligibleSetError`;
`test.jsonl` is available only via the explicit count-only audit method (and a
non-empty test split does not enter the training rows).

Verification: `pytest -q tests/test_phase11_phase10.py` — 4 passed.
