# Task 10 — adapter validation and reload

Implemented `tradingagents.finetuning.validation.validate_run` as a local,
read-only, fail-closed validator. It checks immutable run status and artifact
hash manifests, Phase 10 generation identity/count/policy/source-fingerprint
bindings, prepared formatter/tokenizer metadata and split isolation, adapter
configuration/weights/base-model compatibility, and (when Torch, Transformers,
and PEFT are installed) a local-only adapter reload, forward pass, and frozen
base-weight invariant. Missing optional dependencies produce a bounded warning;
reload failures produce `ADAPTER_INVALID`. No adapter files are modified.

Tests cover a valid tiny fixture, adapter tampering, config absence, prepared
test leakage, and reload failure. No downloads, network, MT5, or Ollama calls
are used.

Evidence:

```
python -m pytest tests/test_phase11_validation.py -q
5 passed
python -m ruff check tradingagents/finetuning/validation.py tests/test_phase11_validation.py
All checks passed!
```
