# Phase 11 fine-tuning

Phase 11 is an offline, local-only LoRA/QLoRA adapter pipeline. It does not
call MT5, Ollama, an LLM provider, or a model registry, and it never deploys an
adapter or changes the stock `tradingagents` CLI.

## Installation

The base package remains usable without ML frameworks. On an offline machine,
install wheels from a local wheelhouse, for example:

```text
python -m pip install --no-index --find-links C:\wheelhouse "tradingagents[training]"
```

QLoRA additionally requires the optional `bitsandbytes` wheel and a working
CUDA installation. No dependency is imported on normal runtime or metadata
commands.

## Commands

```text
finetune inspect --generation PATH
finetune prepare --generation PATH --output-root PATH --base-model LOCAL_MODEL
finetune train --prepared PATH --base-model LOCAL_MODEL --config config.json --output-root PATH
finetune validate --run PATH
finetune status --output-root PATH
```

All commands emit one bounded JSON object. A non-zero exit code means the
closed status is not acceptable. Preparation validates the Phase 10
generation before loading a tokenizer; an empty generation returns
`EMPTY_ELIGIBLE_SET` without constructing a model. Training requires an
explicit base model and JSON `TrainingConfig`; it consumes only train and
validation SFT files and has no test-path option.

Base models must be local snapshots with `config.json`, tokenizer files, and an
immutable revision when a revision is supplied. `local_files_only=True` is
always used, so missing snapshots fail closed as `BASE_MODEL_NOT_FOUND` rather
than downloading. Native chat templates are required unless the operator
explicitly selects `fallback-v1` during preparation.

## LoRA and QLoRA behavior

LoRA uses PEFT, freezes base weights, and saves only adapter weights. Requested
target modules are inspected before attachment; missing modules return
`TARGET_MODULE_MISSING`. QLoRA means 4-bit NF4 with double quantization and
requires CUDA plus bitsandbytes. Missing capability returns `GPU_REQUIRED` or
`QLORA_UNAVAILABLE`; it never silently falls back to LoRA.

## Artifacts and guarantees

Published runs contain `run_manifest.json`, resolved configuration, dataset
manifest, metrics, environment metadata, hash-bound `prepared/` rows,
`checkpoints/`, `adapter/`, and `logs/`. Metrics contain training quantities
only; no ROI, win-rate, or profitability claim is generated. Formatting
projects only point-in-time decision, market, research, and provenance fields.
Outcome data, prompts, completions, chain-of-thought, secrets, and test rows
are rejected or omitted, and validation checks for leakage and hash changes.

Acceptance includes a real empty generation (no model load and no training
attempt) and an explicitly tiny offline smoke with a generated tokenizer and
decoder-only model. These acceptance tests are not a production checkpoint.

Phase 12 owns fixture and smoke expansion. Phase 11 has no merge, deployment,
Ollama replacement, live trading, or promotion boundary; any adapter remains a
research artifact until separately reviewed and governed.

## Bounded shadow-data collection

When the real Phase 10 generation has no eligible examples, use the bounded
EURUSD lifecycle below to collect one auditable shadow decision. It reuses the
existing collector/evaluator and never sends an order:

```powershell
python scripts/phase11_shadow_cycle.py collect `
  --db-path C:\AITrading\TradingAgents\data_cache\phase11-shadow.db `
  --phase7-root C:\p7fast `
  --embedding-model-path C:\path\to\local\bge-small `
  --phase8-root C:\AITrading\TradingAgents\data_cache\phase11-phase8 `
  --phase10-output-root C:\AITrading\TradingAgents\data_cache\phase11-phase10-next `
  --runtime-cache-dir C:\AITrading\TradingAgents\data_cache\phase11-runtime `
  --phase9-audit C:\AITrading\TradingAgents\data_cache\phase11-runtime\evidence_runtime\evidence_audit.sqlite3
```

The command prints `SHADOW ONLY — NO ORDER WILL BE SENT` and emits bounded JSON
metadata. After the approved shortest horizon (300 seconds plus the existing
30-second observation tolerance) has elapsed, run the evaluation stage once:

```powershell
python scripts/phase11_shadow_cycle.py evaluate `
  --db-path C:\AITrading\TradingAgents\data_cache\phase11-shadow.db `
  --phase7-root C:\p7fast `
  --embedding-model-path C:\path\to\local\bge-small `
  --phase8-root C:\AITrading\TradingAgents\data_cache\phase11-phase8 `
  --phase10-output-root C:\AITrading\TradingAgents\data_cache\phase11-phase10-next `
  --runtime-cache-dir C:\AITrading\TradingAgents\data_cache\phase11-runtime `
  --phase9-audit C:\AITrading\TradingAgents\data_cache\phase11-runtime\evidence_runtime\evidence_audit.sqlite3
```

The Phase 7 root and embedding model are read-only and offline. Phase 8 is
append-only; use a dedicated operator root rather than an acceptance fixture.
The Phase 10 output root must be new or empty; an existing non-empty root is
rejected without deletion or overwrite. The JSON report includes candidate and
eligible counts, reason counts, and a metadata-only `exclusion_audit` mapping
each Phase 10 reason to its upstream status/field values. It never includes
prompts, completions, reasoning, credentials, or rendered evidence text.

If the evaluation is not mature, the command returns `PENDING` and does not
call MT5. If the resulting eligible count is zero, stop and diagnose the
reported upstream reasons; do not weaken Phase 10 gates, start bulk collection,
or attempt real training. Phase 7 remains frozen and Phase 12 remains out of
scope.
