# Phase 11 LoRA/QLoRA Fine-tuning — Implementation Plan

**Baseline:** `b9e102c88f185cf0f1d500104872f5fc6a91f82d`  
**Branch:** `codex/phase-11-finetuning`  
**Scope:** implement only Phase 11; no Phase 12, deployment, execution, or source-artifact edits.

## Working rules

Use TDD for every behavior: write a deterministic RED test, run only that
test, implement the smallest change, rerun it GREEN, then run the affected
package tests. Keep optional training imports lazy. Use the SDD progress ledger
at `.superpowers/sdd/2026-09-15-phase-11-lora-qlora-finetuning/progress.md`.
No test may call Ollama, MT5, a hosted API, download a model, or require CUDA.

The real Phase 10 acceptance is a separate bounded check after the positive
fixture smoke. It must return `EMPTY_ELIGIBLE_SET` and
`NO_TRAINING_ATTEMPTED` before any model/tokenizer/GPU/optimizer construction.

## Task 1 — contracts, statuses, and configuration (RED → GREEN)

Create `tradingagents/finetuning/{__init__,models,errors}.py` with lazy public
exports and frozen JSON-safe contracts:

* version constants (`phase11-sft-format.v1`, `phase11-training-policy.v1`,
  `phase11-run-manifest.v1`, `phase11-adapter-package.v1`);
* closed `Phase11Status` values from the design;
* `TrainingConfig`, `LoraConfig`, `QloraConfig`, `DatasetBinding`,
  `SFTExample`, `PreparedManifest`, `RunManifest`, `Metrics`, and bounded
  `TrainingReport`/`ValidationReport`;
* canonical JSON/hash helpers that reject sensitive keys/values and non-finite
  numbers.

Tests in `tests/test_phase11_models.py` cover defaults, serialization,
unknown statuses/versions, bounds, and secret/CoT rejection. Do not import
torch or Transformers from these modules.

## Task 2 — read-only Phase 10 adapter (RED → GREEN)

Create `tradingagents/finetuning/phase10.py`.

`Phase10Generation.open(path)` first calls public
`tradingagents.datasets.writer.validate_generation`; invalid reports raise
`Phase10InvalidError` with bounded reasons. It then reads only
`manifest.json`, `train.jsonl`, and `validation.jsonl`, reconstructing public
`CanonicalExampleV1` rows. It binds the generation directory ID, manifest/file
hashes, policy versions, and source fingerprints. `test.jsonl` is deliberately
not opened by training; a separate audit method can report its count without
returning rows.

Tests in `tests/test_phase11_phase10.py` cover valid binding, tampered manifest/
row/hash, incompatible policy/schema/generation identity, empty generation,
and a spy proving no writer/source adapter and no test-row consumption. The
fixture uses the existing Phase 10 fixture builder; it never calls
`DatasetFactory.build`.

## Task 3 — deterministic formatter and strict target (RED → GREEN)

Create `tradingagents/finetuning/formatting.py`.

Implement `SFTFormatter(policy_version=...)` with an explicit field allow-list.
It projects bounded pre-decision `decision`, `market`, `research`, and safe
provenance fields into canonical SYSTEM/USER messages. `outcome`, future
objective/evaluation fields, trust outcome details, prompt/completion/
reasoning/scratch fields, and unknown fields are excluded or rejected.
The assistant target is the strict source-derived action/evidence projection;
no field is fabricated. Stable JSON ordering and deterministic IDs are
required. A `parse_target()` function rejects missing/extra keys and invalid
BUY/SELL/HOLD or evidence-reference values.

Tests in `tests/test_phase11_formatting.py` prove deterministic bytes, BUY/
SELL/HOLD targets, strict schema, outcome/no-CoT exclusion, sensitive-value
rejection, bounded evidence/provenance, and row-order independence.

## Task 4 — tokenizer/template and sequence guards (RED → GREEN)

Add `tradingagents/finetuning/tokenization.py` with a lazy tokenizer seam and
`TokenizationPolicy`. Require a native chat template or an explicitly named
`fallback-v1`; record tokenizer/model-max-length/config fingerprints. Validate
actual tokenizer lengths including template/special tokens. Do not truncate.

Implement structural reduction in `sequence.py`: preserve current state,
target, used evidence, equation/definition and table header/caption bundles,
then rejected evidence. Split large table/equation bundles with deterministic
parent/part IDs; if the target cannot fit, raise `SequenceTooLongError`.
Mask SYSTEM/USER labels to `-100` and supervise assistant tokens only.

Tests in `tests/test_phase11_tokenization.py` cover chat template policy,
allowed maximum, over-capacity rejection, structural table splitting,
equation/definition preservation, assistant-only masks, and a spy tokenizer
proving no input was silently truncated.

## Task 5 — prepared JSONL writer and split/leakage guards (RED → GREEN)

Create `tradingagents/finetuning/preparation.py`.

`prepare_generation(generation, output_root, formatter, tokenizer_policy)`
validates Phase 10 first, formats only train/validation, sorts by
`example_id`, and atomically writes `prepared/train.sft.jsonl` and
`prepared/validation.sft.jsonl`. It writes a hash-bound prepared manifest with
generation ID, source fingerprints, formatter/policy/tokenizer versions,
counts, and file hashes. No test rows are opened or copied. Empty valid input
returns `EMPTY_ELIGIBLE_SET` before tokenizer construction.

Tests in `tests/test_phase11_preparation.py` cover first preparation,
unchanged skip/idempotent bytes, reordered input, split preservation, test
isolation, no future outcome/CoT/secrets, interrupted staging recovery, and
atomic publication.

## Task 6 — model/tokenizer provenance (RED → GREEN)

Create `tradingagents/finetuning/provenance.py` with lazy Transformers loading
and `ModelProvenance.inspect(base_model, revision, tokenizer_path)`. Require a
local path or immutable revision; reject mutable `latest`/unresolved remote
IDs. Fingerprint config, tokenizer files/template, weight index and bounded
immutable file metadata; record architecture, dtype, and parameter count.

Tests in `tests/test_phase11_provenance.py` use a tiny generated local model and
tokenizer, prove stable fingerprints, reject missing/unpinned/mutable models,
and verify no network loader is called.

## Task 7 — PEFT LoRA attach and tiny trainer (RED → GREEN)

Create `tradingagents/finetuning/lora.py` and `training.py`. Imports of torch,
Transformers, and PEFT occur only inside functions after non-empty preparation.
Inspect requested target modules before constructing `LoraConfig`; missing
modules raise `TargetModuleMissingError`. Attach standard PEFT LoRA, freeze
base parameters, assert adapter parameters are trainable, and record counts.

Implement a small deterministic causal-LM loop with fixed seed, explicit
batch/accumulation/steps, AdamW/linear scheduling, gradient clipping,
assistant-only labels, and validation on validation rows only. Save via PEFT
`save_pretrained`; return bounded losses/runtime/throughput metrics. Fail as
`TRAINING_DEPENDENCY_MISSING` when the optional stack is absent.

Tests in `tests/test_phase11_lora.py` cover module inspection, attach/freeze,
one optimizer step, metrics, malformed model, and dependency failure. Use a
programmatic tiny Transformers model, never a real Qwen checkpoint.

## Task 8 — QLoRA capability guard (RED → GREEN)

Create `tradingagents/finetuning/qlora.py`. Validate 4-bit NF4,
double-quantization, and bf16/fp16 compute settings. Detect CUDA and
bitsandbytes without importing them on LoRA/normal paths. Return
`GPU_REQUIRED` on CPU and `QLORA_UNAVAILABLE` for missing/incompatible
bitsandbytes/CUDA; never fall back to full precision.

Tests in `tests/test_phase11_qlora.py` cover configuration serialization,
CPU rejection, mocked-capability success, and no fallback.

## Task 9 — immutable artifacts, fingerprints, metrics, environment (RED → GREEN)

Create `tradingagents/finetuning/artifacts.py` and `fingerprints.py`.

Stage and publish `<root>/<run-id>/` with the exact design layout. Persist
resolved config, dataset manifest binding, environment versions, metrics,
request fingerprint, adapter hashes, and prepared hashes. Use atomic create-
only publication; retain failed staging for diagnosis. Redact secrets and
never write prompts/completions/reasoning.

Tests in `tests/test_phase11_artifacts.py` cover deterministic request hashes,
unique run IDs, hash tamper detection, immutable destination, bounded metrics,
environment redaction, and no hidden text.

## Task 10 — adapter validation and reload (RED → GREEN)

Add `tradingagents/finetuning/validation.py`.

`validate_run(path)` verifies status/manifests/files, Phase 10 generation and
source fingerprints, formatter/tokenizer/base compatibility, adapter config
and hashes, PEFT reload/forward pass, frozen-base invariant, and test-split
isolation. It emits `ADAPTER_INVALID` on any mismatch and never merges or
modifies the adapter.

Tests in `tests/test_phase11_validation.py` cover a valid tiny adapter,
tampered adapter/config/hash/base, reload failure, and prepared test leakage.

## Task 11 — standalone CLI (RED → GREEN)

Create `tradingagents/finetuning/cli.py`, add the `finetune` entry point only,
and leave `tradingagents`/stock CLI untouched. Implement bounded JSON output
for `prepare`, `train`, `validate`, and optional metadata-only `inspect`.
Preparation validates before dependencies; empty real generations do not load
models. Train requires explicit base-model/config and never accepts a test
path. Missing dependencies/statuses return non-zero without traceback text.

Tests in `tests/test_phase11_cli.py` prove command routing, machine-readable
statuses, no heavy imports on status/inspect, empty short-circuit before model
load, invalid generation fail-closed, and stock CLI import/regression.

## Task 12 — deterministic fixture model/tokenizer (RED → GREEN)

Add test-only `tests/fixtures/phase11_training.py` that builds a tiny
decoder-only model/config and `PreTrainedTokenizerFast` vocabulary with an
explicit chat template, entirely in memory/temp paths. It includes deterministic
Phase 10 fixture rows with BUY, SELL, and HOLD. No binary model assets are
committed.

Tests prove save/reload of the fixture and no network/model download calls.

## Task 13 — positive tiny offline LoRA smoke (RED → GREEN)

Add `tests/test_phase11_training_smoke.py` (marked `smoke`) that prepares the
fixture, runs one or two CPU optimizer steps, and validates adapter save/reload
and manifests/metrics. Assert train/validation counts, all three actions,
frozen base/trainable adapter counts, no leakage/CoT/secrets, zero network/
Ollama/MT5 calls, and test split untouched. Skip only if the optional training
extra is genuinely absent, with the missing dependency reported.

## Task 14 — real Phase 10 empty acceptance (RED → GREEN)

Add `tests/test_phase11_real_empty_acceptance.py` and a bounded script/helper
that points at the existing real generation
`generation-d78d143c760b021f48200255`. Validate source fingerprints and assert
`EMPTY_ELIGIBLE_SET`/`NO_TRAINING_ATTEMPTED`; monkeypatch model/tokenizer/
optimizer/network/LLM/MT5 constructors to fail if called. Do not write to
Phase 10/5/6/7/8 artifacts. This test must be opt-in for local environments
where the real path is unavailable.

## Task 15 — packaging and operator documentation (RED → GREEN)

Add `[training]` optional dependencies for Transformers, torch, PEFT,
Accelerate, and safetensors; keep bitsandbytes optional for QLoRA and do not
import these from normal runtime modules. Do not add TRL/datasets unless a
test demonstrates a necessary contract.

Write `docs/phase11-finetuning.md` with offline installation, `finetune`
examples, model/revision requirements, LoRA/QLoRA capability behavior,
artifact layout, no-CoT/leakage guarantees, real-empty and tiny-smoke
acceptance, and the explicit Phase 12 handoff/no-deployment boundary.

## Task 16 — whole-branch verification and review

Run focused Phase 11 tests, affected Phase 9/10 tests, then the full suite,
Ruff, compileall, and `git diff --check`. Existing baseline Phase 9 Windows
spawn-timeout failures (1603 passed, 6 skipped, 6 failed) must be reported
exactly if unchanged. Review the diff for source-artifact immutability, lazy
imports, no hidden reasoning, no test leakage, no network/MT5/Ollama calls,
stock CLI compatibility, and no Phase 12 scope. Commit only Phase 11 docs/code/
tests/packaging on `codex/phase-11-finetuning`; do not merge or push.

## Acceptance evidence

The final report must include baseline/final SHA and branch; design/plan/
operator-doc paths; versions; real generation ID/fingerprints/status/eligible
count/training attempted; tiny model architecture and train/validation counts;
optimizer steps; LoRA parameters and base/trainable counts; adapter path/hash;
request fingerprint/run ID/loss/runtime; split/leakage/no-CoT/secret checks;
QLoRA capability result; downloads/network/Ollama/MT5 counters; focused and
affected tests; full suite and six known baseline failures; Ruff, compileall,
diff-check, review, and worktree status. Do not claim profitability,
superiority, or Phase 12 readiness beyond an auditable adapter artifact.
