# Phase 14C corpus-scale book discovery

Phase 14C is an offline research workflow over the already-published Phase 7
knowledge generation. It can draft source-grounded strategy specifications
for review and run only candidates accepted by the existing finite registry
through the unchanged causal replay gates. It does not change a live strategy,
TradingAgents decision, watcher, risk policy, MT5 state, or execution path.

Published book knowledge remains separate from Phase 5/6 trade experience and
Phase 8/9/10 evidence. Phase 7 stays frozen and read-only. Local Qwen use is an
explicit `discover` operation only; no command contacts a hosted model. The
teacher output is limited to controlled classifications/IDs and is validated
against exact source evidence before a claim can enter a complete spec. No
prompt, completion, hidden reasoning, credential, or unvalidated prose is an
artifact.

## Commands

Use either the installed `phase14c` entry point or the module form shown below.
All paths are explicit. Commands do not discover a “latest” database or
generation automatically.

### Inspect a run

```powershell
phase14c status --artifact-root C:\phase14c\runs\run-001 --json
```

`status` reads only `run-manifest.json` and `phase14c-report.json`. It does not
open Phase 7, Phase 14A, HFT, or DEMO databases and does not import or construct
an embedder or teacher. A missing root is reported as `MISSING`; an existing
root without a manifest is `EMPTY`; malformed metadata is `INVALID`.

### Interpret discovery outcomes

`discover` runs the pinned stages in order: Phase 7 inventory and read-only
source preflight, local hybrid retrieval, deterministic evidence selection,
presence classification, actionable-only extraction, strict source-backed
assembly, suitability/registry checks, and strategy mapping. It records the
coverage and deferral counts before returning. A deferred selection or
classification/provider failure is `INCOMPLETE`; do not treat it as exhaustive
corpus discovery. Complete runs with no complete specification or no
HFT-suitable specification use explicit closed outcomes. Discovery does not
evaluate candidates or change strategy behavior.

Before evaluation, the report's artifact manifest is SHA-256 checked against
every listed artifact. Reports contain provenance-bearing source evidence and
scalar/code-only model telemetry, never prompts, completions, hidden reasoning,
or credentials. A changed, missing, or path-escaping artifact fails closed.

### Plan without a teacher or run artifacts

```powershell
phase14c plan-only `
  --knowledge-root C:\p7fast `
  --expected-generation gen_607de64268a04a6ab09ffa1e160fc280 `
  --expected-fingerprint <approved-generation-sha256> `
  --expected-population-hash <approved-population-hash> `
  --embedding-model-path "C:\Users\Zaid barghouthi\AppData\Local\Temp\phase7-final-artifacts\embeddings\BAAI--bge-small-en-v1.5" `
  --timeout-seconds 300
```

`plan-only` verifies the pinned generation and inventory, opens the preinstalled
local embedding model in offline mode, performs deterministic hybrid
retrieval/selection, and reports raw/unique hit, selected/deferred, and
unclassified counts. It makes zero Ollama calls and creates no Phase 14C run
directory. The runtime estimate is deliberately conservative:
`selected groups × 15 maximum local extraction attempts per group × timeout`.
The 15-call ceiling accounts for the existing bounded split retry at each
strict extraction stage and the schema's maximum concept fan-out. It is a
planning ceiling, not a measured runtime; validated cache hits can reduce
actual calls.

### Discover with the local teacher

Discovery requires a fresh Phase 14C directory and all approved source pins.
The exact generation fingerprint, population hash, embedding path, three
read-only Phase 14A/HFT/DEMO source paths, external atomic cache, local model and
immutable local model version are required. The endpoint must be a native
loopback Ollama base URL such as `http://localhost:11434` (not `/v1`); there is
no cloud-provider fallback. Temperature is fixed at zero. Timeout, output
tokens and context are bounded.

```powershell
phase14c discover `
  --knowledge-root C:\p7fast `
  --expected-generation gen_607de64268a04a6ab09ffa1e160fc280 `
  --expected-fingerprint <approved-generation-sha256> `
  --expected-population-hash <approved-population-hash> `
  --phase14c-artifact-root C:\phase14c\runs\run-20261004-a `
  --phase14a-db C:\AITrading\TradingAgents\data_cache\phase14a.sqlite3 `
  --hft-db C:\AITrading\TradingAgents\data_cache\hft.sqlite3 `
  --demo-db C:\AITrading\TradingAgents\data_cache\demo.sqlite3 `
  --embedding-model-path "C:\Users\Zaid barghouthi\AppData\Local\Temp\phase7-final-artifacts\embeddings\BAAI--bge-small-en-v1.5" `
  --atomic-cache-path C:\phase14b-cache\atomic.sqlite3 `
  --model qwen3.5:2b `
  --model-version <local-model-digest> `
  --ollama-endpoint http://localhost:11434 `
  --timeout-seconds 300 `
  --max-output-tokens 512 `
  --context-tokens 8192
```

Before creating any run artifact or constructing the teacher, discovery checks
the pinned Phase 7 generation, validates all three source databases through
the existing read-only Phase 14 preflight, validates local/offline model
settings, and rejects path overlap. The Phase 14C output root must be new and
disjoint from Phase 7, the local embedding artifacts, the Phase 14 source
directories, and the atomic cache. The cache must itself be external to those
protected paths. Existing files and directories are never removed or
overwritten.

`--resume` is explicit and is accepted only for an existing incomplete run.
The stored run identity must exactly match the generation/fingerprint and
population, embedding specification, query bank/version and settings,
selection version/config, assembly/schema versions, Ollama endpoint/model
version/settings, atomic-cache path/schema, and paths/fingerprints of all three
source databases. Any mismatch rejects resume; it does not create a new run in
the existing directory.

### Evaluate complete validated specifications

```powershell
phase14c evaluate `
  --artifact-root C:\phase14c\runs\run-20261004-a `
  --phase14a-db C:\AITrading\TradingAgents\data_cache\phase14a.sqlite3 `
  --hft-db C:\AITrading\TradingAgents\data_cache\hft.sqlite3 `
  --demo-db C:\AITrading\TradingAgents\data_cache\demo.sqlite3 `
  --source-commit <reviewed-source-commit> `
  --max-candidate-runs 5
```

Evaluation accepts only a `COMPLETE` run with a valid report and a SHA-256
verified, strictly parsed `validated-specs.json`. It does not construct a
teacher or embedder. Candidate evaluation uses the existing causal replay and
promotion contract, defaults to at most five candidates, and is capped at ten.
Anything incomplete, conflicted, unsuitable, unimplemented, or beyond the
candidate cap is not promoted; it is reported with a closed reason.
The evaluation report is written under
`<phase14c-artifact-root>/evaluations/phase14c-evaluation-report.json`.
No eligible candidate is `NOT_RUN`; blocked preflight is `BLOCKED`; a completed
batch with no candidate passing remains
`CANDIDATES_EVALUATED_NONE_PASSED`. The maximum candidate state remains
`SHADOW_CHALLENGER`.

## Artifacts and statuses

A completed run is expected to contain a safe manifest, aggregate report,
inventory/query/retrieval/selection summaries, validated classifications,
source-grounded concepts/conflicts/specifications, strategy mappings,
bounded resume records and, only when explicitly requested, candidate replay
artifacts. Full source prompt/completion text and hidden reasoning are never
stored. Resume JSONL accepts only bounded status/ID/count metadata and
validated scalar/code-only call telemetry; an unterminated malformed final
line is recoverable, while corruption in any completed line fails closed.

The run manifest is the authority for resumability. `IN_PROGRESS` and
`INTERRUPTED` may resume under exact identity. `COMPLETE` and `FAILED` are not
resumed in place. Create a new explicitly named root for a different model,
generation, setting, source fingerprint, or schema.

Interpret these counts as research workflow metrics only. A candidate marked
`SHADOW_CHALLENGER` has not been promoted to production and has no execution
authority. Phase 14C creates no trades and never starts the Forex supervisor.
