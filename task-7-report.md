# Task 7 — PEFT LoRA attach and tiny trainer

Implemented the Phase 11 LoRA attachment and deterministic CPU trainer in
`tradingagents/finetuning/lora.py` and `tradingagents/finetuning/training.py`.

## Implementation

- Optional `torch`, `transformers`, and `peft` imports are lazy. Empty prepared
  input returns `EMPTY_ELIGIBLE_SET` before optional-stack loading.
- Requested module suffixes are inspected before PEFT `LoraConfig` creation;
  missing or malformed targets raise `TargetModuleMissingError`.
- Standard PEFT LoRA attachment freezes base parameters, verifies that only
  adapter parameters are trainable, and records total/trainable counts.
- The trainer consumes only prepared train and validation rows, uses the
  existing `TokenizationPolicy` assistant-only labels, fixed seeds, explicit
  batch/accumulation/step bounds, AdamW, linear scheduling, gradient clipping,
  and validation without optimizer updates.
- Adapters are saved through PEFT `save_pretrained`; bounded loss, runtime,
  token, step, and parameter metrics are returned.
- Missing/broken optional training dependencies return
  `TRAINING_DEPENDENCY_MISSING` without a traceback.

## Verification

- `.venv\Scripts\python.exe -m pytest -q tests/test_phase11_lora.py` — 5 passed
- `.venv\Scripts\python.exe -m pytest -q tests/test_phase11_models.py tests/test_phase11_phase10.py tests/test_phase11_formatting.py tests/test_phase11_tokenization.py tests/test_phase11_preparation.py tests/test_phase11_provenance.py tests/test_phase11_lora.py` — 45 passed
- `ruff check tradingagents\finetuning\lora.py tradingagents\finetuning\training.py tests\test_phase11_lora.py` — passed

The positive tiny-model test uses an in-memory GPT-2 configuration with PEFT
and a deterministic tokenizer; it performs one optimizer step and verifies an
adapter config is written. No model download, network, Ollama, or MT5 access is
used.
