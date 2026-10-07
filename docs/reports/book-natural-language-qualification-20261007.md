# Native natural-language book-rule qualification

Date: 2026-10-07. Scope: offline research only; no broker connection, order,
runtime restart, deployment, risk change or concurrency change.

## Result

The V2 path is implemented alongside the unchanged default canonical V1 path.
The model selects addressed source spans and proposes stage/family labels;
the deterministic resolver alone reconstructs values, operators, units and
field proofs. Artifact reload independently revalidates these proofs.

The genuine bounded local-model run completed but produced **no supported
rule and no eligible candidate**. This is not a successful strategy discovery
or evidence of profitable HFT.

Artifact root: `C:/phase14c/runs/run-20261007-natural-language-v2`.
Artifacts retain selected source addresses, safe call metadata, all rejection
counts and before/after pinned-source hashes. Raw model reasoning is not stored.

| Gate | Actual count |
| --- | ---: |
| Genuine addressed source spans | 4 |
| Local-model calls | 4 |
| Structurally valid model responses | 4 |
| Supported normalized rules | 0 |
| Rejected proposals | 4 |
| Eligible candidates | 0 |
| Replay runs | 0 |
| Broker sends | 0 |

Rejections: `FAMILY_NOT_ESTABLISHED` 1, `STAGE_MISMATCH` 1,
`GRAMMAR_UNSUPPORTED` 2. Replay reports `NOT_RUN / NO_ELIGIBLE_CANDIDATES`.
No manual relabeling, fallback extraction or second model batch was used to
manufacture a positive result.

## Local teacher and source pin

Native endpoint `http://127.0.0.1:11434`, `httpx` environment proxies disabled.
Model `qwen3.5:2b`; installed digest verified before invocation:
`324d162be6ca5629ae4517c8710434d0bd2d665bc94dbad46e9af8fbf8a2f0df`.
Context 8192, requested output ceiling 512, stage budget 256, temperature 0,
timeout 300 seconds, existing bounded retries (none needed).

Generation `gen_607de64268a04a6ab09ffa1e160fc280`;
fingerprint `428929b8060d96dd59be8e10ca338ded9534c12265a657aff5fad5cc97ffe3d1`;
population `sha256:fd3ee0c846f2969747aca70576f58e55ffe1c07e8190570d2f9f7e2cb20b9e17`.

All four pinned files had identical hashes before and after qualification:

| Source | SHA256 |
| --- | --- |
| Phase 7 catalog | `bae95b6fde8e206f6825ddd69d1c6977fe4d13941c8556b5dbe2829734f86447` |
| Frozen Phase 14A | `36fae45ceeed0e847b4541c75e3f051249a96612c820882af42cdda425a0e823` |
| Frozen HFT | `1031e12a2f4a82d6f5ee0a5f5e55c1e5a94d800f4c18378822fc6676ede44bd0` |
| Frozen DEMO | `a00ec289c857883d085a4b10212f093358c0d4da74e1392ca69e2586dfb4d064` |

## Engineering evidence and remaining limits

Focused regression: 287 passed, 2 Windows symlink-privilege skips. Genuine
catalog-address tests ran without skips. Tests independently prove that the
reviewed 10-pip target and 20-pip stop constructions normalize when correctly
selected; both remain `SUPPORTED_NONEXECUTABLE`, including after reload and
mapping. Model selection did not correctly select both labels in this run.

Only the reviewed numeric-distance construction is enabled. Numeric-feature
comparison and holding-limit patterns have no reviewed genuine positive case
and stay disabled. Pip distances are not broker points or momentum thresholds;
missing direction, horizon, full eight-stage completeness, independent-source
support and reviewed primitive bindings cannot be supplied by defaults.

No current V2 rule can bind to a replay strategy. Future bindings need genuine
complete source evidence and causal costed validation. Unknown commission and
latency, and unresolved historical quote-quality gaps remain blockers, not
inputs silently assumed favorable. This bounded four-span check is not a new
corpus-scale scan. No profit or zero-loss guarantee is made.

Runtime-change audit: implementation diff from plan base `9756b46` contains
only offline book research, its CLI dispatch, tests and documentation. No
supervisor, live HFT strategy, DEMO gateway, lease or risk implementation changed.
`real_money=false` throughout qualification.
