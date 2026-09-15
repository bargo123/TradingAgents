# Phase 11 Tasks 8–9 report

## Task 8 — QLoRA capability guard

Implemented `tradingagents/finetuning/qlora.py` with lazy optional-dependency
checks and fail-closed behavior. The guard requires CUDA and an importable,
compatible `bitsandbytes` installation; CPU execution returns
`GPU_REQUIRED`, missing or incompatible QLoRA support returns
`QLORA_UNAVAILABLE`, and no full-precision fallback is attempted. The
quantization contract is fixed to 4-bit NF4 with double quantization and
`bfloat16`/`float16` compute. Explicit serialization is exposed through
`quantization_kwargs`.

## Task 9 — immutable artifacts and fingerprints

The artifact and fingerprint surfaces are covered by the companion focused
tests. Request/file hashes are deterministic, environment metadata is bounded
and redacted, and publication preserves immutable run artifacts.

## Verification

```text
python -m pytest -q tests/test_phase11_qlora.py tests/test_phase11_task6_10_seams.py
8 passed

python -m ruff check tradingagents/finetuning/qlora.py tests/test_phase11_qlora.py
All checks passed!
```
