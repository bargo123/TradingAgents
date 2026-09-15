# Phase 11 — LoRA/QLoRA Fine-tuning Pipeline Design

**Status:** approved implementation design  
**Date:** 2026-09-15  
**Baseline:** `b9e102c88f185cf0f1d500104872f5fc6a91f82d`

## 1. Purpose, goals, and non-goals

Phase 11 turns a *validated, immutable Phase 10 generation* into auditable
supervised fine-tuning (SFT) adapter candidates. It does not decide whether a
candidate is better than its base model; that is a Phase 12 responsibility.

Goals:

* deterministic preparation of structured, evidence-grounded chat examples;
* standard PEFT LoRA and capability-checked QLoRA training modes;
* immutable run manifests, provenance, hashes, metrics, and reload validation;
* an offline tiny-model CPU smoke proving a real optimizer step;
* a fail-closed empty-real-generation path.

Non-goals are RL/DPO/PPO/online learning, strategy generation, execution,
MT5/TradingAgents integration, Ollama replacement, deployment/promotion,
profitability claims, Phase 5/6/7/8 source changes, and Phase 12 evaluation.

## 2. Alternatives and recommendation

1. **TRL `SFTTrainer` + PEFT:** convenient production features, but adds a
   broad dependency surface and makes exact assistant masking/version behavior
   less explicit.
2. **Transformers + PEFT with a small local training loop:** a narrower,
   inspectable dependency surface; more code owns batching, masking, metrics,
   and validation.
3. **Custom adapter implementation:** smallest install, but would not provide
   standard PEFT interoperability and would create unnecessary maintenance risk.

Option 2 is selected. Transformers supplies the model/tokenizer and PEFT owns
LoRA serialization/reload. A tiny loop keeps the safety gates and no-CoT
contract visible. TRL/datasets are not required by V1.

## 3. Architecture and boundaries

`tradingagents.finetuning` is a lazy-import package. Importing normal
TradingAgents, Phase 5–10, or the stock CLI must not import torch,
Transformers, PEFT, bitsandbytes, or download anything.

The flow is:

```text
validated Phase 10 generation
  -> read-only adapter (train/validation only)
  -> leakage-safe canonical formatter
  -> deterministic prepared JSONL + manifest
  -> base-model/tokenizer provenance
  -> LoRA or QLoRA trainer
  -> immutable adapter candidate + metrics/manifest
  -> reload/compatibility validation
```

Suggested modules:

* `tradingagents/finetuning/models.py` — version constants, enums, frozen
  contracts (`TrainingConfig`, `DatasetBinding`, `SFTExample`, manifests,
  reports).
* `errors.py` — typed fail-closed errors/statuses.
* `phase10.py` — read-only generation validation and row adapter using
  `tradingagents.datasets.writer.validate_generation`; never reads Phase 5–9
  sources directly.
* `formatting.py` — deterministic safe projection and formatter policy.
* `preparation.py` — split-preserving writer, hashes, and preparation report.
* `provenance.py` — local model/tokenizer/config/index fingerprints.
* `lora.py` — lazy PEFT LoRA configuration, module inspection, freeze checks,
  and the small Transformers training loop.
* `qlora.py` — 4-bit NF4/double-quant configuration and capability guard.
* `artifacts.py` — immutable run layout, manifests, metrics, environment, and
  hash validation.
* `cli.py` — `prepare`, `train`, and `validate` commands; registered as
  `finetune` only, leaving `tradingagents` unchanged.

## 4. Versioned contracts

The initial versions are `phase11-sft-format.v1`,
`phase11-training-policy.v1`, `phase11-run-manifest.v1`, and
`phase11-adapter-package.v1`. Every persisted contract carries its version;
unknown versions fail closed.

`SFTExample` contains an `example_id`, source split, two or more chat messages
(SYSTEM and USER), and one structured assistant target. The target is a
strict closed projection of fields actually present in the Phase 10 canonical
row: normalized `action`, `evidence_use_status`, used evidence references, and
rejected references/reasons where present. No field is invented to fill a
missing source value. Target JSON has no extra keys and must parse before it is
written.

The system message is fixed and versioned. The user message is deterministic,
bounded, and contains only pre-decision fields from `decision`, `market`, and
the safe evidence/provenance subset of `research`. Ordering is canonical JSON
with sorted keys and compact UTF-8 serialization. The assistant message is
the target JSON and is never a hidden-reasoning transcript.

## 5. Phase 10 authority and validation

Preparation first calls `validate_generation(path)`. The adapter requires a
valid dataset schema, eligibility/canonicalization/split policy versions,
generation ID, complete file hashes, and complete Phase 5/8/9 source
fingerprints. It binds the generation identity and manifest hash into the run
manifest. Invalid or incompatible generations return `PHASE10_INVALID`.

Only `train.jsonl` and `validation.jsonl` are read. `test.jsonl` is never
opened by the training path; an explicit test-row/path guard rejects attempts
to bind it. Phase 10 eligibility is not recreated or weakened. An empty valid
generation returns `EMPTY_ELIGIBLE_SET` and `NO_TRAINING_ATTEMPTED` before
tokenizer/model/GPU/optimizer/LLM/MT5 initialization.

## 6. Leakage and privacy policy

The formatter has an allow-list, not a deny-list. `outcome` and all future
objective/evaluation values are metadata-only and are absent from system,
user, and assistant target. Hidden reasoning, Qwen thinking tokens, raw full
prompts/completions, credentials, and private scratch fields are rejected by
recursive sensitive-key/value checks. Only bounded explicit market state,
analysis profile, evidence IDs/provenance, risk/decision fields present in the
canonical row, and Phase 10 identity metadata may be emitted.

Prepared files are intentional model-input artifacts; they contain no secrets,
CoT, or future labels. No cloud service, Ollama, MT5, or external network is
used by preparation or training.

## 7. Formatter and sequence policy

The configured tokenizer is authoritative. A native chat template is required;
otherwise a versioned, explicit fallback template is selected and recorded as
`fallback-v1` (never silently inferred). SYSTEM/USER/ASSISTANT roles and
assistant-target boundaries are preserved. Padding is left-side-independent
and deterministic; pad IDs and attention masks are recorded.

Before encoding, the formatter validates every input with the actual tokenizer
and reserves room for required special/template tokens. No provider may
silently truncate. The effective maximum is
`min(config.max_sequence_length, tokenizer.model_max_length)` after the
template/tokenizer contract is applied. For BAAI/bge-small-en-v1.5 in the
fixture/embedding-adjacent policy, inputs over 512 tokenizer tokens are split,
never truncated.

Oversized prose, tables, or equation bundles are structurally reduced in this
order: preserve current market state, target/action, and used evidence;
preserve equation/definition and table header/caption bundles; then include
rejected evidence in canonical order. If the target cannot fit, preparation
returns `SEQUENCE_TOO_LONG`. Structural split metadata records parent and part
IDs, so an equation is not detached from variable definitions and a table is
not detached from its caption/header.

Loss labels mask SYSTEM and USER tokens with `-100`; only assistant target
tokens contribute. Tests assert no prompt token is accidentally supervised and
no target token is masked.

## 8. Training configuration and defaults

`TrainingConfig` is versioned and persists every effective value: mode
(`lora|qlora`), base model/revision, seed, max sequence length, learning rate,
epochs/max steps, batch size, gradient accumulation, warmup, weight decay,
optimizer, scheduler, gradient clipping, gradient checkpointing, LoRA fields,
QLoRA fields, logging/checkpoint/validation intervals, and early-stopping
policy.

Conservative production defaults are learning rate `2e-4`, two epochs,
`max_sequence_length=4096`, batch size 1, gradient accumulation 1, warmup 0,
weight decay 0, AdamW, linear scheduler, clip 1.0, gradient checkpointing
enabled, validation every epoch, and no implicit early stopping. Tiny smoke
overrides are explicit and persisted.

## 9. Base-model and tokenizer provenance

The operator supplies `--base-model` as a local path or Hugging Face-compatible
identifier and may supply `--base-model-revision`. Ollama/GGUF tags are not
translated into training checkpoints. Mutable `latest`/unresolved remote
revisions are rejected; local snapshots are fingerprinted by immutable config,
tokenizer, weight-index, and selected file metadata. Normal acceptance uses
local-files-only behavior and fails `BASE_MODEL_NOT_FOUND` rather than
downloading.

The manifest records model ID/path, immutable revision, architecture, config
hash, tokenizer hash/fingerprint, weight-index hash, dtype, parameter count,
tokenizer model-max-length, chat-template fingerprint, and embedding-independent
training identity. These fields are part of compatibility checks.

## 10. LoRA and QLoRA

LoRA defaults are `r=16`, `alpha=32`, dropout `0.05`, and target modules
`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. The actual model is
inspected first; a missing requested target fails `TARGET_MODULE_MISSING`.
Base parameters are frozen, adapter parameters trainable, and total/trainable
counts and percentage are persisted. PEFT `save_pretrained` is the adapter
package format; base weights are never copied into the adapter directory.

QLoRA means 4-bit NF4 with double quantization and bf16 compute when supported,
otherwise fp16. It requires CUDA and bitsandbytes. A missing capability returns
`GPU_REQUIRED` or `QLORA_UNAVAILABLE`; it never falls back to full-precision
LoRA. CPU tests cover config construction and deterministic rejection. The
optional QLoRA dependency is not imported on normal paths.

## 11. Tiny offline smoke

Tests construct a tiny decoder-only Transformers model and a deterministic
`PreTrainedTokenizerFast` vocabulary programmatically; no binary model asset
or download is committed. Fixture Phase 10 rows include BUY, SELL, and HOLD.
On CPU, a fixed seed and one or two optimizer steps prove formatting,
tokenization, LoRA attachment, frozen base/trainable adapter parameters,
non-zero optimization, adapter save/reload, and valid manifests/metrics.

## 12. Artifact layout and identity

Each execution is immutable under `<phase11-root>/<run-id>/`:

```text
run_manifest.json       resolved_config.json       dataset_manifest.json
metrics.json             environment.json
prepared/train.sft.jsonl prepared/validation.sft.jsonl
checkpoints/             adapter/                  logs/
```

`prepared/` is an explicit, hash-bound artifact. A deterministic request
fingerprint covers Phase 10 generation ID, formatter/policy versions, base
identity/revision, resolved config, and LoRA/QLoRA settings. It is distinct
from the unique run ID and adapter file hashes. Hardware-dependent weight
bytes are not claimed to be cross-platform deterministic.

Environment metadata records Python/platform, torch/Transformers/PEFT/
Accelerate/bitsandbytes/CUDA/GPU versions, seed, dtype, and quantization; no
secrets or prompt/completion text.

## 13. Metrics and adapter validation

Metrics are losses, steps, epochs, examples/tokens, runtime, throughput, peak
CUDA memory (when applicable), and parameter counts. No ROI, win-rate,
profitability, or candidate-superiority metric is produced.

`validate` checks all manifest/file hashes, dataset generation/policy/source
fingerprints, formatter/policy/base compatibility, adapter config and files,
reloadability, forward pass, frozen-base invariant, and test-split isolation.
There is no merge, deployment, Ollama replacement, or trading promotion.

## 14. CLI

The standalone entry point is `finetune` with:

* `finetune prepare --generation PATH --output-root PATH [config flags]`;
* `finetune train --prepared PATH --base-model PATH_OR_ID --output-root PATH`;
* `finetune validate --run PATH`;
* optional `finetune inspect --generation PATH` for bounded metadata only.

Status/list/inspect paths do not load training dependencies. Prepare may load a
tokenizer only after Phase 10 validation and only for non-empty data. Train
never accepts a test path. All errors are typed, bounded, machine-readable,
and non-zero; no normal command automatically starts training.

## 15. Closed statuses and recovery

The status set is:
`EMPTY_ELIGIBLE_SET`, `PHASE10_INVALID`, `TRAINING_DEPENDENCY_MISSING`,
`BASE_MODEL_NOT_FOUND`, `BASE_MODEL_REVISION_UNPINNED`, `TOKENIZER_INVALID`,
`CHAT_TEMPLATE_UNAVAILABLE`, `TARGET_MODULE_MISSING`, `GPU_REQUIRED`,
`QLORA_UNAVAILABLE`, `SEQUENCE_TOO_LONG`, `TRAINING_FAILED`, `ADAPTER_INVALID`,
and `COMPLETE`.

Preparation is per-generation and atomic; a failed row is reported rather than
silently omitted. Training stages into a unique run directory and publishes
only after manifest/file validation. Interrupted staging is retained for
diagnosis and is never treated as a complete adapter. Existing Phase 10/source
artifacts are never modified.

## 16. Testing and acceptance

Deterministic tests cover Phase 10 validation/empty short-circuit, ordering and
hash stability, split preservation and test inaccessibility, no-outcome/no-CoT
and strict target checks, tokenizer/template and sequence limits, structural
table/equation splitting, model provenance/revision guards, LoRA attach/freeze/
step/save/reload, malformed module/model/manifest failures, QLoRA CPU rejection,
adapter validation, CLI boundaries, and no external calls.

CI uses tiny fixtures only: no real books, Ollama, MT5, CUDA, downloads, or
network. Retrieval/trading outcomes are not Phase 11 metrics.

Real acceptance uses the existing Phase 10 generation
`generation-d78d143c760b021f48200255`, validates its source fingerprints, finds
zero eligible rows, and stops as `EMPTY_ELIGIBLE_SET`/
`NO_TRAINING_ATTEMPTED` before model/GPU/optimizer/network initialization.
Before/after fingerprints prove Phase 5–10 artifacts are unchanged. Positive
acceptance uses the tiny local fixture and proves BUY/SELL/HOLD, an optimizer
step, reload, manifests, no leakage/CoT, and zero external calls.

## 17. Performance, Windows, and security

The Ryzen 5/16 GB Windows machine is a development/fixture environment, not a
4B QLoRA host. CPU smoke uses tiny dimensions, one worker, short sequences,
and no multiprocessing. Production QLoRA is cloud-GPU ready but vendor-neutral.
Normal imports remain light; optional dependencies are isolated in a
`[training]` extra and are installed only in the worktree virtual environment.

All files and manifests remain local. Credentials are neither required nor
stored. Model downloads are disabled for acceptance and are never implicit in
the empty-data path.

## 18. Future seams and Phase 12 handoff

Interfaces deliberately permit stronger tokenizers/embedding-independent
formatters, rerunnable trainers, alternate PEFT adapters, CUDA QLoRA, and
future evaluation without changing Phase 10. A future corpus builder may
select eligible examples from immutable manifests; Phase 12 alone may compare
base versus adapter on sealed test data and decide promotion. No Phase 12
evaluation, experience memory, RAG injection, trading command, or execution
path is introduced here.

## 19. Self-review

There are no TODO/TBD placeholders. Ownership is explicit: Phase 10 owns
eligibility and splits; Phase 11 owns formatting/training artifacts; Phase 12
owns evaluation/promotion. Outcome and hidden-reasoning data cannot enter model
messages. The design assumes neither CUDA nor cloud access for acceptance and
does not modify MT5, TradingAgents decisions, Phase 5–10 sources, or the stock
CLI.
