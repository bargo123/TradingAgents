# SDD ledger — plan: docs/superpowers/plans/2026-09-12-phase-9-evidence-augmented-trading-reasoning.md

## Controller preflight

- Approved implementation baseline (before worktree setup): `19363fb4fdbe49e03082f0526bd91f5aee8596bd`.
- Worktree setup commit on `main` and implementation branch: `8b326d152873786ac47258e8de4e8d6ee45dedd9` (adds only `.worktrees/` ignore).
- Implementation branch: `codex/phase-9-evidence-reasoning`.
- Worktree: `C:\AITrading\TradingAgents\.worktrees\phase-9-evidence-reasoning`.
- Approved design SHA: `1f31428c62b0c5223468784b3057e33463660992`.
- Approved plan SHA: `49e1f2502492fa36b1fe4a6d3f247862ec2f433c`.
- Controller model policy: Luna only (`gpt-5.6-luna` for every implementer/reviewer).
- Worktree is clean at preflight; no Phase 9 production code has been changed.

## Mandatory task-pair conflict scan

The scan covers shared files, public interfaces, state, runner seams, databases, CLI/configuration, and generated artifacts. The approved plan's order is the serialization rule; no parallel implementation is permitted.

| Pair / self-check | Shared surface | Ordering / ruling |
|---|---|---|
| T1↔T2 | `EvidenceContext` contracts, IDs, budgets | T1 types first; T2 consumes only frozen T1 interfaces. |
| T1↔T3 | query-policy and snapshot-adapter contracts | T1 defines value objects; T3 implements policy/adapter. |
| T1↔T4 | `EvidenceBundle`, statuses, typed errors | T1 vocabulary is authoritative; T4 maps runtime outcomes only. |
| T1↔T5 | audit enums/records | T5 persists only T1-shaped metadata. |
| T1↔T6 | context/trace metadata types | T6 adds one context and metadata-only trace fields. |
| T1↔T7 | prompt/runtime status vocabularies | T7 renders common block; no contract drift. |
| T1↔T8 | final validation and transient fields | T8 consumes T1 validation contracts; no source-schema changes. |
| T1↔T9 | runner/analyze result and audit types | T9 uses T1 types and preserves existing persistence. |
| T1↔T11 | replay config/report and evidence context | T11 uses frozen contracts and never writes source DB. |
| T1↔T12 | config/CLI evidence settings | T12 exposes only T1-compatible configuration. |
| T1↔T14 | smoke report and status vocabularies | T14 reports T1 statuses without redefining them. |
| T2↔T4 | context builder output consumed by integration | T2 establishes deterministic caps/order/hash; T4 must preserve it. |
| T2↔T5 | context IDs referenced by audit | T5 stores refs emitted by T2 only. |
| T2↔T7 | bounded evidence block | T7 receives already-capped T2 context, never re-ranks. |
| T2↔T8 | refs/manifest validation | T8 validates T2 IDs and source metadata. |
| T2↔T9 | runner query result/audit | T9 calls the builder once through integration. |
| T2↔T11 | replay equivalence | T11 compares deterministic T2 context from saved snapshots. |
| T3↔T4 | canonical query/request and adapter state | T3 owns query construction; T4 invokes read-only interfaces. |
| T3↔T9 | snapshot-derived query in runner | T9 supplies resolved symbol/profile/timeframe only. |
| T3↔T11 | replay query determinism | T11 reuses T3 policy, no alternate query logic. |
| T4↔T5 | integration status and audit append | T4 returns status; T5 records it append-only. |
| T4↔T9 | lazy one-call integration | T9 constructs/invokes service only after ordinary source persistence. |
| T4↔T11 | timeout/read-only replay | T11 uses same bounded integration behavior without writes. |
| T4↔T13 | offline/privacy isolation | T13 verifies process/network/source-read boundaries. |
| T4↔T14 | real smoke wiring | T14 uses read-only adapters and one bounded call. |
| T5↔T8 | validation evidence audit | T8 output is audit metadata only. |
| T5↔T9 | audit location/lifecycle | T9 appends after existing decision persistence. |
| T5↔T11 | replay audit separation | T11 must not append to Phase 5/6 source DB. |
| T5↔T14 | smoke audit inspection | T14 reads Phase 9 audit artifacts only. |
| T6↔T7 | agent state evidence context | T7 reads one immutable context; no hidden reasoning. |
| T6↔T9 | graph initialization | T9 injects context without replacing debate histories. |
| T6↔T10 | checkpoint identity/context | T10 hashes forex context identity without changing stock semantics. |
| T6↔T15 | state isolation regression | T15 checks stock graph and checkpoint compatibility. |
| T7↔T8 | PM instruction and refs | T8 validates only PM structured refs; all other agents remain unchanged. |
| T7↔T9 | runner prompt integration | T9 passes evidence block through existing forex graph boundary. |
| T7↔T15 | prompt/privacy tests | T15 confirms no prompts/reasoning persisted. |
| T8↔T9 | normal persistence vs transient metadata | T9 strips transient fields before source-row persistence and audits afterward. |
| T8↔T11 | replay result validation | T11 validates without writing source rows. |
| T8↔T14 | smoke normalization/citation status | T14 reports validator outcome and quarantines failures. |
| T9↔T10 | runner/checkpoint contract | T10 preserves existing runner identity and forex-only additions. |
| T9↔T11 | analyze-only replay seam | T11 calls `analyze`, never `run`. |
| T9↔T12 | CLI config/runner construction | T12 guards forex CLI only; stock CLI unchanged. |
| T9↔T14 | smoke runner path | T14 exercises the approved non-persisting/isolated path. |
| T9↔T15 | end-to-end regression | T15 checks ordinary `run()` row shape and analyze no-write behavior. |
| T10↔T11 | checkpoint and replay identity | T11 proves same snapshot identity is deterministic. |
| T10↔T15 | resume/stock regressions | T15 runs checkpoint lifecycle/resume suites. |
| T11↔T14 | replay/smoke scripts | Separate source fingerprints and no mutation. |
| T11↔T15 | replay test coverage | T15 verifies A/B determinism and timeout behavior. |
| T12↔T13 | CLI offline/privacy flags | T13 checks no cloud/MT5/stock registration. |
| T12↔T14 | acceptance command | T14 is the only real-smoke CLI addition. |
| T12↔T15 | CLI regression | Stock `tradingagents` invocation remains byte/behavior compatible. |
| T13↔T14 | isolation gate | Smoke must fail closed when prerequisites/network/immutability gates fail. |
| T13↔T15 | whole-branch boundary review | Final review covers no MT5/execution/training/experience mixing. |
| T14↔T15 | acceptance report | T15 verifies mandatory real smoke or reports the exact prerequisite blocker. |
| T1 self-check | contracts | UTC, immutability, canonical bytes, closed vocabularies, no strategy fields. |
| T2 self-check | builder | source order, caps 4/4/2, total 6000, deterministic IDs/hash, empty/failure handling. |
| T3 self-check | policy/adapter | exact snapshot fields, canonical Phase 7/8 queries, no ranking duplication. |
| T4 self-check | runtime | lazy one call, read-only SQLite, complete embedding-spec match, process timeout, zero workers. |
| T5 self-check | audit DB | append-only/idempotent schema, no source DB mutation, metadata-only payload. |
| T6 self-check | graph state | exactly one immutable context, metadata-only trace, stock state unchanged. |
| T7 self-check | prompts | common bounded block; PM-only instruction; no chain-of-thought or prompt persistence. |
| T8 self-check | PM validation | strict refs/status, sanitizer strips recursively, action unaffected by invalid refs. |
| T9 self-check | runner | existing row schema/signature preserved, analyze no-write, one retrieval, audit after persist. |
| T10 self-check | checkpoints | forex-only identity/context; stock checkpoint compatibility. |
| T11 self-check | replay | saved snapshot, A/B, source bytes/schema/rows unchanged, no transient fields. |
| T12 self-check | config/CLI | forex-only flags, strict offline paths, no stock CLI changes. |
| T13 self-check | isolation | no network, MT5, execution, training, RAG-to-decision leakage; Tier C diagnostic-only. |
| T14 self-check | smoke | fresh verified Phase 8 root, local Qwen/Ollama, immutable sources, report gate. |
| T15 self-check | verification | focused/full tests, Ruff, compileall, diff check, final whole-branch review. |

## Task ledger

| Task | Status | Implementer commit | Review | Fix rounds |
|---|---|---|---|---|
| 1 | COMPLETE | `53f3943d230e1cd0012dee8c76804c05e663f517` (initial `266d1a4`, fixes `3d58d43`) | APPROVE; 21 focused/regression tests, Ruff, diff check pass | 2 |
| 2 | COMPLETE | `dd71c4c` (initial `1b5c554`) | APPROVE; 34 focused/regression tests, Ruff, diff check pass | 1 |
| 3 | COMPLETE | `1c948d5` | APPROVE; 24 focused/regression tests, Ruff, diff check pass | 0 |
| 4 | COMPLETE | `a3c18ff` (initial `235be50`; fixes `b0ef91d`, `4d96790`, `2a98de6`, `4dbc67f`, `946b01b`) | APPROVE; 72 focused/regression tests, Ruff, compileall, diff check pass | 6 |
| 5 | COMPLETE | `27967ed` (initial `63e0736`) | APPROVE; 7 focused tests, Ruff, compileall, diff check pass | 1 |
| 6 | COMPLETE | `86cc452` (initial `b5cdb6e`) | APPROVE; 18 focused tests, Ruff, compileall, diff check pass | 1 |
| 7 | COMPLETE | `9b84b36` (initial `d9c1763`) | APPROVE; 35 focused tests, Ruff, compileall, diff check pass | 1 |
| 8 | COMPLETE | `ebfa0d7` (initial `d3256a2`) | APPROVE; 64 focused tests, 153 regression tests, Ruff, compileall, diff check pass | 1 |
| 9 | COMPLETE | `22e92ab` (prior `7e12da8`) | APPROVE; 66 focused tests, 1 skipped; Phase 9/shadow suite 197 passed, 2 skipped; Ruff, compileall, diff check pass | 1 |
| 10 | COMPLETE | `c0285a3` (prior `c4a9057`) | APPROVE; 57 focused tests, 153 broader Phase 9/checkpoint tests; Ruff and diff checks pass | 1 |
| 11 | COMPLETE | `51ed1b3` (prior `223ab79`) | APPROVE; replay/runner/checkpoint 62 tests, Phase 9 focused 115 tests; Windows handle regression, Ruff, compileall, diff pass | 3 |
| 12 | COMPLETE | `4ee1069` (prior `9fbd1e1`) | APPROVE; 47 focused + 30 Phase 9 integration tests; Ruff, compileall, diff pass | 1 |
| 13 | COMPLETE | `02c8173` (prior `95c612b`) | APPROVE; 41 focused + 91 integration/replay/runner/graph tests; Ruff, compileall, diff pass | 1 |
| 14 | COMPLETE | `d303f1c` (prior `6a4f038`, `26fa905`) | APPROVE; full Phase 8 preflight, replay telemetry/gates, strict validation, network isolation, and recursive privacy checks verified | 4 |
| 15 | NOT COMPLETE (real smoke prerequisite failed) | — | REPORT | 0 |

## Decisions / rulings

- Approved plan corrections are authoritative: Phase 5/6 source schemas remain unchanged; evidence metadata belongs in a separate Phase 9 audit store.
- All implementation and review agents are Luna-only and must not spawn additional agents.
- Tasks are serialized because the preflight identifies shared interfaces and state boundaries.

## Baseline focused verification (before implementation)

All required baseline commands passed; no unrelated failures were present:

- `python -m pytest -q tests/test_forex_shadow_runner.py` — **18 passed** (9.98s)
- `python -m pytest -q tests/test_forex_graph_mode.py` — **5 passed** (8.22s)
- `python -m pytest -q tests/test_checkpoint_lifecycle.py tests/test_checkpoint_resume.py` — **11 passed** (8.42s)
- `python -m pytest -q tests/test_experience_orchestrator.py` — **5 passed** (0.79s)
- `python -m pytest -q tests/test_knowledge_query.py` — **16 passed** (0.85s)
