# Phase 7 Local HFT/FX Knowledge RAG Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, read-only, provenance-complete knowledge retriever for the approved PDF/EPUB library, with incremental ingestion, structure-aware chunks, local embeddings, paired LanceDB/SQLite FTS5 indexes, hybrid search, and deterministic quality evidence.

**Architecture:** A new `tradingagents.knowledge` package owns source discovery, Docling-first parsing, scan quarantine, structured chunking, local embedding, catalog/state, and versioned index projections. A read-only query service selects one validated vector/lexical generation, fuses dense and BM25 candidates with reciprocal-rank fusion, applies a deterministic reranker, and returns provenance-complete `KnowledgeHit` values. The package has no imports into MT5, `tradingagents.forex`, LangGraph, agent decisions, or Phase 5/6 stores.

**Tech Stack:** Python >=3.10; standard-library `dataclasses`, `enum`, `hashlib`, `json`, `pathlib`, `sqlite3`, `zipfile`, and `xml.etree`; Docling for PDF/EPUB structure normalization; FastEmbed with ONNX Runtime for local CPU embeddings; LanceDB for dense storage; SQLite FTS5 for BM25; `argparse`; `pytest`; Ruff; and the existing `setuptools` console-script layout. Docling/FastEmbed/LanceDB are isolated in an optional knowledge extra and are imported lazily so the base TradingAgents install and stock CLI remain usable without them.

**Spec:** `docs/superpowers/specs/2026-09-10-phase-7-hft-knowledge-rag-design.md`

## Global Constraints

- The only approved source is `C:\Users\Zaid barghouthi\Downloads\new books`; every source operation is read-only, and generated artifacts live outside that tree.
- Recursively enumerate all regular files in deterministic relative-path order. PDF and EPUB are the only supported formats; every other regular file is visible as `UNSUPPORTED` and is never parsed, embedded, or indexed. Directories, reparse targets escaping the root, and internal filesystem metadata are not resources.
- V1 detects likely image-only/scanned documents as `NEEDS_OCR`; it never performs OCR and never indexes those documents as trusted text. Docling PDF options must explicitly set `do_ocr=False`; local Docling artifacts are resolved through a configurable path in offline mode, with no automatic download.
- Formula enrichment is disabled by default. It may be enabled only when the required local formula artifacts are present; otherwise preserve parser-native equation structure and record the limitation without inventing LaTeX or enabling OCR.
- Preserve title, authors, type, chapter/section hierarchy, page or EPUB location, reading order, equations, tables, captions, references, and structured metadata whenever the parser exposes them.
- Chunk by semantic structure using the actual configured embedding tokenizer with truncation disabled. For BGE-small V1, use a 448-token soft target, a 510-token effective corpus-content limit, and a 512-token model-input maximum (including special tokens); no indexed input may exceed the loaded model's real limit.
- `source_hash` is SHA-256 of bytes; exact byte duplicates share one `document_id` and separate aliases. A document remains active while at least one current non-removed alias references its hash.
- Failed changed-file ingestion keeps the previous relationship as `RETAINED_PREVIOUS` for diagnostics; another current duplicate alias remains unaffected, and stale content is excluded from the default view when no current alias remains.
- FastEmbed/ONNX embeddings are local and CPU-only. The v1 default is a configurable local `BAAI/bge-small-en-v1.5`-compatible 384-dimensional model; the complete `EmbeddingSpec` (resolved version, artifact hash, dimensions, normalization, tokenizer fingerprint, model/effective limits, truncation flag, and corpus/query instruction policies) is persisted and compared before dense retrieval.
- LanceDB and SQLite FTS5 are local files. Vector and lexical projections are generation-scoped, carry identical population/version identity, and are activated as one matched generation. A query must reject incompatible generations.
- Rebuilds are explicit, side-by-side, validated, and atomically activated; the previous complete generation remains available for rollback/diagnostics. Incremental documents are visible only after both projections are ready in the active generation.
- The query API returns only provenance-complete knowledge evidence. It has no BUY/SELL/HOLD, `PortfolioDecision`, MT5, `forex-watch`, TradingAgents graph, Phase 5 label, or experience-memory field.
- No source content is sent to hosted LLMs, Ollama, cloud embeddings, external vector databases, telemetry services, or a network service. No credentials are accepted or stored.
- CI tests use tiny fixtures and fakes. They do not parse the full library, download models, call Ollama or MT5, require network access, require CUDA, or start a long indexing job.
- Phase 7 closure additionally requires one bounded real local integration smoke over 1–3 approved resources. Model/dependency provisioning is a separate explicit setup action; normal `knowledge index` and `knowledge search` runtime must remain offline and must never auto-download.
- Do not modify `cli/main.py`, the existing `tradingagents` stock entry point, `tradingagents.forex`, MT5 provider code, shadow/evaluation stores, or agent prompts/registration.

## File Map

### New package files

- `tradingagents/knowledge/__init__.py` — public knowledge-only exports.
- `tradingagents/knowledge/models.py` — enums and immutable contracts for resources, parsed blocks, chunks, embeddings, generations, queries, hits, and run summaries.
- `tradingagents/knowledge/config.py` — validated source/artifact paths, local Docling artifact/formula settings, embedding context policy, and CPU/offline limits.
- `tradingagents/knowledge/identity.py` — canonical relative paths, SHA-256 identity, duplicate aliases, and source-integrity checks.
- `tradingagents/knowledge/discovery.py` — deterministic recursive scanner including unsupported regular files and excluding directories/internal metadata.
- `tradingagents/knowledge/catalog.py` — SQLite catalog, alias currentness, ingestion events, projection readiness, and active-generation registry.
- `tradingagents/knowledge/parser.py` — parser protocol and normalized IR helpers.
- `tradingagents/knowledge/docling_parser.py` — PDF and native-EPUB Docling adapters, explicit OCR/formula/offline controls, and provenance-safe OPF/spine fallback.
- `tradingagents/knowledge/scanned.py` — deterministic image-only detection and `NEEDS_OCR` evidence.
- `tradingagents/knowledge/chunking.py` — semantic-unit assembly, size policy, and deterministic chunk IDs.
- `tradingagents/knowledge/embeddings.py` — embedding protocol, FastEmbed/ONNX adapter, model validation, and artifact cache.
- `tradingagents/knowledge/index_generation.py` — paired LanceDB/FTS5 generation metadata, validation, activation, and rollback-safe swaps.
- `tradingagents/knowledge/vector_index.py` — LanceDB writer/reader adapter.
- `tradingagents/knowledge/lexical_index.py` — generation-scoped SQLite FTS5/BM25 writer/reader adapter.
- `tradingagents/knowledge/ingestion.py` — incremental/rebuild coordinator, per-resource transactions, and interrupted-run recovery.
- `tradingagents/knowledge/fusion.py` — dense/BM25 candidate model and reciprocal-rank fusion.
- `tradingagents/knowledge/reranking.py` — reranker protocol and deterministic CPU feature reranker.
- `tradingagents/knowledge/provenance.py` — completeness validation and source rendering.
- `tradingagents/knowledge/query.py` — read-only `KnowledgeQuery` service.
- `tradingagents/knowledge/diagnostics.py` — bounded quarantine records and run summaries.
- `tradingagents/knowledge/cli.py` — separate `knowledge` console command and subcommands.

### Existing files with minimal changes

- `pyproject.toml` — add only the optional knowledge dependency extra and the separate `knowledge` console entry point; keep the `tradingagents` script unchanged.

### New documentation and tests

- `docs/knowledge-rag.md` — operator guide for explicit local indexing/search commands, generated layout, statuses, offline behavior, and provenance.
- `tests/test_knowledge_models.py`
- `tests/test_knowledge_config.py`
- `tests/test_knowledge_identity.py`
- `tests/test_knowledge_discovery.py`
- `tests/test_knowledge_catalog.py`
- `tests/test_knowledge_parser.py`
- `tests/test_knowledge_scanned.py`
- `tests/test_knowledge_chunking.py`
- `tests/test_knowledge_embeddings.py`
- `tests/test_knowledge_indexes.py`
- `tests/test_knowledge_ingestion.py`
- `tests/test_knowledge_query.py`
- `tests/test_knowledge_cli.py`
- `tests/test_knowledge_quality.py`
- `tests/test_knowledge_isolation.py`
- `tests/fixtures/knowledge/` — tiny deterministic source/IR fixtures and benchmark labels.
- `scripts/provision_knowledge_models.py` — explicit setup-only provisioning of local Docling and BGE-small artifacts; never called by normal indexing/search.
- `scripts/knowledge_phase7_smoke.py` — bounded real local parser/embedding/index/query smoke with source-integrity and network guards.

## Deterministic fixture and fake conventions

All normal tests use a `tmp_path / "source"` tree and a separate
`tmp_path / "artifacts"` tree. The source tree contains `book.pdf`,
`chapter.epub`, `notes.txt`, `desktop.ini`, and a nested directory. The test
helper creates PDF/EPUB bytes with fixed content and a fixed timestamp; it
never writes generated artifacts below the source tree.

The test suite defines these reusable fakes before the first dependent test:

- `FakeParser` implements `DocumentParser`, returns a supplied `ParsedDocument`, and records `parse_calls`.
- `FakeTokenizer` exposes deterministic special-token-aware corpus/query lengths, a configurable model limit, and a stable tokenizer fingerprint.
- `FakeEmbeddingProvider` exposes a supplied `EmbeddingSpec` and `FakeTokenizer`, returns deterministic vectors derived from `sha256(text)`, records `embed_calls`, and fails if the service attempts truncation or an over-limit input; `RecordingFakeEmbeddingProvider` additionally records `seen_content_tokens`, `seen_model_input_tokens`, and the truncation flag.
- `FakeDoclingPipeline` records every constructed option, exposes `do_ocr`,
  formula-enrichment, artifact-path, and offline values, and can return an
  image-only document with an empty text inventory.
- `FakeVectorBackend` and `FakeLexicalBackend` implement the index protocols and expose failure switches for vector-build, lexical-build, and activation tests; `FakeVectorReader(fail_if_called=True)` proves spec mismatches stop before dense retrieval.
- `FakeReranker` records candidate metadata and returns a deterministic order without reading hidden text.
- `make_parsed_document(...)` creates blocks for prose, equation/variable definitions, table/caption/notes, figure caption, and reference locations.
- `make_indexed_catalog(...)` creates two duplicate aliases for one hash and a second unique document, with a known active generation ID.
- `fixture_resource(name)`, `structured_fixture()`, `make_structured_document()`, `make_long_section_document()`, `make_chunk(text)`, and `make_source_tree(tmp_path)` return the fixed resources/documents/chunks used by parser, chunking, query, and integrity tests.
- `max_overlap_tokens(chunks)`, `snapshot_tree(path)`, `imported_modules_under(path)`, `candidate(label, rank, chunk_id)`, `make_hit(**overrides)`, `make_embedding_spec(**overrides)`, `make_docling_options(**overrides)`, and `forbidden_constructor()` are pure test assertions/builders.
- `fake_docling_parser(...)`, `fake_docling_parser_with_image_only_pdf(...)`,
  `fake_docling_document_with_native_equation(...)`,
  `make_large_table_document()`, `make_large_equation_definition_document()`,
  and `forbid_parser_ingestor_source_scanner_and_writers(...)` provide the
  parser/chunker/CLI boundary fixtures without external dependencies.
- `bge_chunker_fixture()` returns a `FakeTokenizer`, `EmbeddingSpec`, and
  `StructureAwareChunker` wired together so no test can construct a chunker
  without its tokenizer/spec.
- `ingestion_harness(tmp_path, duplicate=False, interrupt_after=None, source=None)` returns `.source`, `.catalog`, `.parser`, `.embedder`, `.ingestor`, and `.shared_document_id`; its aliases are keyed by `resource_id_for("book.pdf")` and `resource_id_for("copy.pdf")`.
- `make_generation_manager(tmp_path, vector=None, lexical=None, embedding_spec=None)`, `make_chunks()`, `make_vectors()`, and `write_fake_lexical_metadata(path, generation_id)` construct the Task 6 generation fakes.
- `query_harness(dense=(), lexical=(), vector_generation=None, lexical_generation=None, index_embedding_spec=None, query_embedding_spec=None, dense_reader=None)`, `seed_query_fixture(path)`, `fixture_query_service()`, `load_cases(path)`, and `run_benchmark(cases, service)` construct the Task 8–10 query/benchmark fixtures; the harness exposes a spy query embedder and forbidden writer/parser/ingestor constructors.

Every test that checks text checks only fixture text or scalar lengths. No test prints or stores prompts, model reasoning, or unrelated source copies.

---

### Task 1: Establish knowledge contracts, configuration, identity, and discovery

**Files:**
- Create: `tradingagents/knowledge/models.py`
- Create: `tradingagents/knowledge/config.py`
- Create: `tradingagents/knowledge/identity.py`
- Create: `tradingagents/knowledge/discovery.py`
- Create: `tradingagents/knowledge/__init__.py`
- Create: `tests/test_knowledge_models.py`
- Create: `tests/test_knowledge_config.py`
- Create: `tests/test_knowledge_identity.py`
- Create: `tests/test_knowledge_discovery.py`

**Interfaces:**
- Produces `IngestionState`, `AliasRelation`, `ContentType`, `DiscoveredResource`, `DocumentMetadata`, `ParsedBlock`, `ParsedDocument`, `ChunkRecord`, `EmbeddingSpec`, `IndexGeneration`, `KnowledgeQuery`, `KnowledgeHit`, `IngestionRunSummary`, and `KnowledgeConfig` for every later task.
- `KnowledgeConfig(source_root, artifact_root, docling_artifacts_path, docling_offline, docling_do_ocr=False, formula_enrichment_enabled, embedding_model_id, embedding_model_path, embedding_dimensions, offline, worker_count, embedding_batch_size, parser/chunker/index settings)` is immutable and serializable; its validated paths and component fingerprints drive all later reuse checks. V1 rejects `docling_offline=False` and any request to set `docling_do_ocr=True`.
- Produces `canonical_relative_path(path, source_root) -> str`, `resource_id_for(relative_path) -> str`, `document_id_for(source_hash) -> str`, and `sha256_file(path) -> tuple[str, int]`.
- Produces `SourceScanner(config).discover() -> tuple[DiscoveredResource, ...]`; it includes all regular non-metadata files, marks unsupported extensions before hashing, and sorts by case-folded relative path then display path.

- [ ] **Step 1: Write the failing contract and discovery tests**

~~~python
def test_discovery_exposes_unsupported_regular_file_without_parser_or_embed(tmp_path):
    source = tmp_path / "new books"
    (source / "nested").mkdir(parents=True)
    (source / "book.pdf").write_bytes(b"pdf-fixture")
    (source / "notes.txt").write_text("not a book", encoding="utf-8")
    (source / "desktop.ini").write_text("[.ShellClassInfo]", encoding="utf-8")
    (source / "nested" / "paper.epub").write_bytes(b"epub-fixture")

    resources = SourceScanner(KnowledgeConfig(source_root=source, artifact_root=tmp_path / "artifacts")).discover()

    assert [(item.relative_path, item.state) for item in resources] == [
        ("book.pdf", IngestionState.HASHED),
        ("nested/paper.epub", IngestionState.HASHED),
        ("notes.txt", IngestionState.UNSUPPORTED),
    ]
    assert all(item.source_hash is None for item in resources if item.state is IngestionState.UNSUPPORTED)


def test_identity_is_stable_and_exact_duplicates_share_document_id():
    assert canonical_relative_path(Path("root/A/B.pdf"), Path("root")) == "a/b.pdf"
    assert resource_id_for("a/b.pdf") == resource_id_for("A/B.pdf")
    assert document_id_for("ab" * 32) == document_id_for("ab" * 32)
    assert document_id_for("ab" * 32) != document_id_for("cd" * 32)
~~~

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

~~~powershell
pytest tests/test_knowledge_models.py tests/test_knowledge_config.py tests/test_knowledge_identity.py tests/test_knowledge_discovery.py -q
~~~

Expected: FAIL because the knowledge package and its contracts do not yet exist.

- [ ] **Step 3: Implement the contracts and source scanner**

Use string-valued enums so SQLite/JSON values are stable. `KnowledgeConfig`
validates that `source_root` is a directory, `artifact_root` is outside it,
`docling_artifacts_path` is a local path outside the source tree (defaulting to
`artifact_root / "docling"`), `docling_offline` is true, and OCR is disabled;
worker count is 1–6, embedding batch size is positive, and general offline mode
is true by default. Formula enrichment defaults to false and may be true only
with an explicitly configured local formula artifact path. `SourceScanner` uses
`os.scandir` recursion, rejects symlinks or reparse points that escape the root,
skips `desktop.ini`, `Thumbs.db`, and filesystem metadata directories, and
records every other regular file.
Unsupported extensions terminate at `UNSUPPORTED` before a content hash is
requested.

~~~python
class IngestionState(str, Enum):
    DISCOVERED = "DISCOVERED"
    UNSUPPORTED = "UNSUPPORTED"
    HASHED = "HASHED"
    UNCHANGED = "UNCHANGED"
    DUPLICATE = "DUPLICATE"
    PARSING = "PARSING"
    PARSED = "PARSED"
    CHUNKED = "CHUNKED"
    EMBEDDING = "EMBEDDING"
    EMBED_FAILED = "EMBED_FAILED"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    NEEDS_OCR = "NEEDS_OCR"
    PARSE_FAILED = "PARSE_FAILED"
    INDEX_FAILED = "INDEX_FAILED"
    REMOVED = "REMOVED"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    INTERRUPTED = "INTERRUPTED"
    RETAINED_PREVIOUS = "RETAINED_PREVIOUS"
~~~

- [ ] **Step 4: Run the focused tests to verify they pass**

~~~powershell
pytest tests/test_knowledge_models.py tests/test_knowledge_config.py tests/test_knowledge_identity.py tests/test_knowledge_discovery.py -q
~~~

Expected: PASS, including proof that `notes.txt` is visible and no parser or embedding seam is invoked.

- [ ] **Step 5: Commit**

~~~powershell
git add tradingagents/knowledge/__init__.py tradingagents/knowledge/models.py tradingagents/knowledge/config.py tradingagents/knowledge/identity.py tradingagents/knowledge/discovery.py tests/test_knowledge_models.py tests/test_knowledge_config.py tests/test_knowledge_identity.py tests/test_knowledge_discovery.py
git commit -m "feat: add knowledge contracts and source discovery"
~~~

### Task 2: Add the transactional catalog, alias currentness, and generation registry

**Files:**
- Create: `tradingagents/knowledge/catalog.py`
- Create: `tradingagents/knowledge/diagnostics.py`
- Create: `tests/test_knowledge_catalog.py`

**Interfaces:**
- Consumes `DiscoveredResource`, `ParsedDocument`, `ChunkRecord`, `IndexGeneration`, `IngestionState`, and `AliasRelation` from Task 1.
- Produces `KnowledgeCatalog(path)`, `initialize()`, `upsert_discovered(resource)`, `register_document(document)`, `get_document(document_id)`, `get_alias(resource_id)`, `commit_alias(resource_id, document_id, source_hash, relation)`, `retain_previous(resource_id, attempted_hash, previous_document_id)`, `mark_removed(resource_id)`, `current_alias_count(document_id)`, `document_is_active(document_id)`, `record_projection_ready(document_id, generation_id, vector_ready, lexical_ready)`, `set_active_generation(generation)`, `active_generation()`, `list_resources(state=None)`, `list_runs()`, and `list_quarantine()`.
- Produces catalog tables named `knowledge_resources`, `knowledge_documents`, `knowledge_aliases`, `knowledge_chunks`, `knowledge_ingestion_runs`, `knowledge_ingestion_events`, and `knowledge_index_generations`. The catalog contains no trading-decision or experience-memory columns.

- [ ] **Step 1: Write failing catalog and alias tests**

~~~python
def test_remove_one_duplicate_alias_keeps_document_active(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("11" * 32)
    catalog.register_document(make_parsed_document(document_id=document_id, source_hash="11" * 32))
    catalog.commit_alias("res_a", document_id, "11" * 32, AliasRelation.CURRENT)
    catalog.commit_alias("res_b", document_id, "11" * 32, AliasRelation.CURRENT)

    catalog.mark_removed("res_a")

    assert catalog.current_alias_count(document_id) == 1
    assert catalog.document_is_active(document_id) is True


def test_remove_last_alias_deactivates_default_view_but_keeps_rows(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    document_id = document_id_for("22" * 32)
    catalog.register_document(make_parsed_document(document_id=document_id, source_hash="22" * 32))
    catalog.commit_alias("res_only", document_id, "22" * 32, AliasRelation.CURRENT)

    catalog.mark_removed("res_only")

    assert catalog.current_alias_count(document_id) == 0
    assert catalog.document_is_active(document_id) is False
    assert catalog.get_document(document_id) is not None


def test_changed_duplicate_and_failed_change_preserve_other_alias(tmp_path):
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    old_id = document_id_for("33" * 32)
    catalog.register_document(make_parsed_document(document_id=old_id, source_hash="33" * 32))
    catalog.commit_alias("res_a", old_id, "33" * 32, AliasRelation.CURRENT)
    catalog.commit_alias("res_b", old_id, "33" * 32, AliasRelation.CURRENT)

    catalog.retain_previous("res_a", attempted_hash="44" * 32, previous_document_id=old_id)

    assert catalog.current_alias_count(old_id) == 1
    assert catalog.get_alias("res_b").relation is AliasRelation.CURRENT
    assert catalog.get_alias("res_a").relation is AliasRelation.RETAINED_PREVIOUS
~~~

- [ ] **Step 2: Run the focused catalog tests to verify they fail**

~~~powershell
pytest tests/test_knowledge_catalog.py -q
~~~

Expected: FAIL because the catalog schema and alias transitions are not implemented.

- [ ] **Step 3: Implement SQLite transactions and derived currentness**

Open a fresh SQLite connection per operation, enable foreign keys, use a busy
timeout, and wrap alias/document/projection changes in `BEGIN IMMEDIATE`.
`knowledge_aliases.relation_status = CURRENT` is the only relation counted by
`current_alias_count`. A removed alias changes only its own row. A successful
changed-file commit updates the resource's relation and document count in one
transaction; a failed attempt writes an event and `RETAINED_PREVIOUS` without
touching another resource's relation.

Store `vector_ready`, `lexical_ready`, `projection_generation`, and
`projection_population_hash` on document/chunk projection records. The
default-active query predicate is:

~~~sql
current_alias_count > 0
AND vector_ready = 1
AND lexical_ready = 1
AND projection_generation = active_generation_id
AND active = 1
~~~

`set_active_generation` validates that vector and lexical locations, component
versions, and population hashes agree before committing the registry row.
Quarantine events store IDs, status, stage, bounded error metadata, and scalar
counters only.

- [ ] **Step 4: Run the focused catalog tests to verify they pass**

~~~powershell
pytest tests/test_knowledge_catalog.py -q
~~~

Expected: PASS for one-alias removal, last-alias removal, duplicate change,
failed changed-file retention, idempotent schema creation, and generation
registry persistence.

- [ ] **Step 5: Commit**

~~~powershell
git add tradingagents/knowledge/catalog.py tradingagents/knowledge/diagnostics.py tests/test_knowledge_catalog.py
git commit -m "feat: add knowledge catalog and alias currentness"
~~~

### Task 3: Add Docling-first parsing, EPUB adaptation, and scan quarantine

**Files:**
- Create: `tradingagents/knowledge/parser.py`
- Create: `tradingagents/knowledge/docling_parser.py`
- Create: `tradingagents/knowledge/scanned.py`
- Create: `tests/test_knowledge_parser.py`
- Create: `tests/test_knowledge_scanned.py`
- Create: `tests/fixtures/knowledge/structured_ir.json`

**Interfaces:**
- Consumes `DiscoveredResource` and `KnowledgeConfig` from Tasks 1–2.
- Produces `DocumentParser` with `parse(resource: DiscoveredResource, staging_dir: Path) -> ParsedDocument`.
- Produces `DoclingDocumentParser`, which routes PDF bytes through an explicitly configured offline Docling pipeline (`PdfPipelineOptions(do_ocr=False, ...)`) and tries native Docling EPUB conversion first. An OPF/spine adapter is a provenance-checked fallback only when native EPUB output cannot preserve deterministic spine/anchor locations.
- Produces `ScanDecision` and `ScannedDetector.classify(document: ParsedDocument) -> ScanDecision`.
- Produces typed `ParseFailure`, `ParserDependencyUnavailable`, and `DoclingArtifactsUnavailable` exceptions; the ingestion coordinator records them per resource without aborting unrelated documents.

- [ ] **Step 1: Write failing parser/scan tests**

~~~python
def test_parser_preserves_equation_table_caption_and_locations():
    document = FakeParser(structured_fixture()).parse(
        fixture_resource("paper.pdf"), Path("staging")
    )

    assert document.title == "Microstructure Fixture"
    assert document.blocks[1].content_type is ContentType.EQUATION
    assert document.blocks[1].equation.latex == r"p_t = m_t + \lambda I_t"
    assert document.blocks[2].content_type is ContentType.TABLE
    assert document.blocks[2].table.headers == ("Horizon", "Accuracy")
    assert document.blocks[2].page_start == 4
    assert document.blocks[2].section_path == ("Results", "Prediction")


def test_scan_detector_marks_image_only_pdf_needs_ocr():
    decision = ScannedDetector().classify(
        make_parsed_document(page_text_chars=(0, 2, 0, 1), image_pages=(True, True, True, True))
    )

    assert decision.state is IngestionState.NEEDS_OCR
    assert decision.text_bearing_pages == 0
    assert decision.image_bearing_pages == 4


def test_docling_pdf_options_disable_ocr_and_image_only_text_is_not_ocr_generated(monkeypatch):
    options = make_docling_options(do_ocr=False, do_formula_enrichment=False)
    assert options.do_ocr is False
    parser = fake_docling_parser_with_image_only_pdf(options, monkeypatch)

    document = parser.parse(fixture_resource("scanned.pdf"), Path("staging"))
    assert not document.text_blocks
    assert ScannedDetector().classify(document).state is IngestionState.NEEDS_OCR


@pytest.mark.integration
def test_installed_docling_pdf_pipeline_options_are_explicitly_ocr_disabled(tmp_path):
    pytest.importorskip("docling")
    parser = DoclingDocumentParser(
        KnowledgeConfig(
            artifact_root=tmp_path / "artifacts",
            docling_artifacts_path=tmp_path / "docling-artifacts",
        )
    )
    options = parser.build_pdf_pipeline_options()
    assert options.do_ocr is False
    assert parser.config.docling_offline is True


def test_missing_docling_artifacts_fails_offline_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network"))

    with pytest.raises(DoclingArtifactsUnavailable):
        DoclingDocumentParser(
            KnowledgeConfig(
                artifact_root=tmp_path / "artifacts",
                docling_artifacts_path=tmp_path / "missing-docling",
            )
        ).parse(fixture_resource("paper.pdf"), tmp_path / "staging")


def test_formula_enrichment_is_local_optional_and_preserves_native_equation():
    disabled = make_docling_options(formula_enrichment_enabled=False)
    assert disabled.do_formula_enrichment is False
    document = fake_docling_document_with_native_equation(formula_status="UNAVAILABLE_NATIVE_PRESERVED")
    assert document.blocks[0].equation.parser_native
    assert document.parser_provenance["formula_enrichment_status"] == "UNAVAILABLE_NATIVE_PRESERVED"


def test_requested_formula_enrichment_requires_local_artifact_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network"))
    with pytest.raises(DoclingArtifactsUnavailable):
        DoclingDocumentParser(
            KnowledgeConfig(
                artifact_root=tmp_path / "artifacts",
                docling_artifacts_path=tmp_path / "missing-docling",
                formula_enrichment_enabled=True,
            )
        ).build_pdf_pipeline_options()


def test_native_epub_path_is_preferred_and_fallback_records_spine_anchor():
    parser = fake_docling_parser(native_epub=True)
    document = parser.parse(fixture_resource("chapter.epub"), Path("staging"))
    assert document.parser_provenance["adapter_path"] == "NATIVE_DOCLING_EPUB"
    assert document.blocks[0].epub_spine_item and document.blocks[0].anchor


def test_epub_opf_fallback_is_used_only_when_native_provenance_fails():
    parser = fake_docling_parser(native_epub=False, native_missing_locations=True)
    document = parser.parse(fixture_resource("chapter.epub"), Path("staging"))
    assert document.parser_provenance["adapter_path"] == "OPF_SPINE_FALLBACK"
    assert document.blocks[0].epub_spine_item and document.blocks[0].anchor


def test_parser_exception_is_parse_failed_not_scanned():
    with pytest.raises(ParseFailure):
        FailingParser().parse(fixture_resource("broken.pdf"), Path("staging"))
~~~

- [ ] **Step 2: Run the focused parser tests to verify they fail**

~~~powershell
pytest tests/test_knowledge_parser.py tests/test_knowledge_scanned.py -q
~~~

Expected: FAIL because the normalized IR, parser adapters, and scan detector are absent.

- [ ] **Step 3: Implement the normalized IR and adapters**

Define `ParsedBlock` fields for block ID, content type, text, reading order,
page/spine location, chapter, section path, equation data, table data, and
figure/reference metadata. The PDF adapter maps Docling document items without
flattening equations/tables. Construct the pinned PDF options explicitly with
`do_ocr=False`, `do_formula_enrichment=config.formula_enrichment_enabled`, and
the configured `docling_artifacts_path`; assert OCR remains false immediately
before conversion. Resolve all Docling artifacts locally with offline mode and
raise `DoclingArtifactsUnavailable` rather than download when required files
are absent. Formula enrichment is disabled by default; when requested it must
use local formula artifacts and persist model/version/hash. If enrichment is
disabled or unavailable without a requested artifact, retain the parser-native
equation representation and a bounded limitation status; never invent LaTeX
or enable OCR as a side effect.

Attempt native Docling EPUB conversion first, validate deterministic spine and
anchor provenance, and use an explicit OPF/spine adapter only for that failed
contract. Both paths read source bytes read-only and write temporary parser
inputs only under staging; the selected adapter path is persisted in parser
provenance.

The detector applies the fixed 64 non-whitespace character page/item rule:
`NEEDS_OCR` requires fewer than 20% text-bearing pages/items and at least 80%
image-bearing pages/items, or zero extractable text with page images. A parser
exception remains `PARSE_FAILED`. Detection records parser/scanner versions
and scalar counts in `ScanDecision`; it never emits OCR text.

Use lazy imports for Docling/EPUB libraries. Missing optional packages raise a
typed `ParserDependencyUnavailable`; missing local Docling/formula artifacts
raise `DoclingArtifactsUnavailable`. Both produce visible per-resource
`PARSE_FAILED` diagnostics without affecting already indexed documents. Add an
integration-marked test that inspects the actual constructed Docling options,
proves `do_ocr is False`, and verifies an image-only PDF produces no parser
text before `NEEDS_OCR` classification.

- [ ] **Step 4: Run the focused parser tests to verify they pass**

~~~powershell
pytest tests/test_knowledge_parser.py tests/test_knowledge_scanned.py -q
~~~

Expected: PASS for structure preservation, native-EPUB preference and fallback
locations, explicit OCR-off/offline artifact behavior, formula policy, scan
classification thresholds, and parser-failure isolation.

- [ ] **Step 5: Commit**

~~~powershell
git add tradingagents/knowledge/parser.py tradingagents/knowledge/docling_parser.py tradingagents/knowledge/scanned.py tests/test_knowledge_parser.py tests/test_knowledge_scanned.py tests/fixtures/knowledge/structured_ir.json
git commit -m "feat: add structured document parsing and scan quarantine"
~~~

### Task 4: Add structure-aware chunking and deterministic chunk IDs

**Files:**
- Create: `tradingagents/knowledge/chunking.py`
- Create: `tests/test_knowledge_chunking.py`

**Interfaces:**
- Consumes `ParsedDocument`, `ParsedBlock`, `ContentType`, and `KnowledgeConfig` from Tasks 1 and 3.
- Consumes the active `EmbeddingSpec`/`EmbeddingTokenizer`; chunk limits are
  derived from the actual model tokenizer, not a character/word estimate.
- Produces `ChunkPolicy(soft_token_target=448, hard_content_token_limit=510, overlap_tokens=64, version="structure-v2")` for the BGE-small V1 profile, plus a factory that derives the hard content limit from any loaded `EmbeddingSpec`.
- Produces `StructureAwareChunker(policy, tokenizer).chunk(document) -> tuple[ChunkRecord, ...]`; the constructor requires both arguments and has no implicit tokenizer or default policy. Production wiring must derive `policy = ChunkPolicy.for_embedding(embedder.spec)` and pass the same provider's actual tokenizer.
- Produces `deterministic_chunk_id(document_id, source_hash, policy_version, content_type, section_path, block_range, ordinal, normalized_text) -> str`.

- [ ] **Step 1: Write failing chunking tests**

~~~python
def test_equation_keeps_variable_definition_and_table_keeps_header():
    _, _, chunker = bge_chunker_fixture()
    chunks = chunker.chunk(make_structured_document())

    equation = next(item for item in chunks if item.content_type is ContentType.EQUATION)
    table = next(item for item in chunks if item.content_type is ContentType.TABLE)
    assert "inventory" in equation.text.lower()
    assert table.table_metadata["headers"] == ["Horizon", "Accuracy"]
    assert table.table_metadata["caption"] == "Prediction accuracy"


def test_chunk_ids_and_order_are_repeatable():
    _, _, chunker = bge_chunker_fixture()
    first = chunker.chunk(make_structured_document())
    second = chunker.chunk(make_structured_document())

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert [item.chunk_ordinal for item in first] == list(range(len(first)))


def test_long_prose_splits_only_inside_section_and_uses_bounded_overlap():
    tokenizer = FakeTokenizer(model_max_input_tokens=512, special_token_budget=2)
    policy = ChunkPolicy.for_embedding(make_embedding_spec(model_max_input_tokens=512, effective_corpus_content_token_limit=510))
    chunks = StructureAwareChunker(policy, tokenizer).chunk(make_long_section_document())

    assert all(item.embedding_input_tokens <= 512 for item in chunks)
    assert all(item.content_tokens <= 510 for item in chunks)
    assert max_overlap_tokens(chunks) <= 64
    assert all(item.section_path == ("Methods",) for item in chunks)


def test_chunk_at_model_capacity_is_allowed_but_over_capacity_fails_without_truncation():
    tokenizer = FakeTokenizer(model_max_input_tokens=512, special_token_budget=2)
    policy = ChunkPolicy.for_embedding(make_embedding_spec(model_max_input_tokens=512, effective_corpus_content_token_limit=510))
    allowed = make_long_section_document(tokens=510)
    assert StructureAwareChunker(policy, tokenizer).chunk(allowed)

    with pytest.raises(ChunkTooLargeForEmbedding):
        StructureAwareChunker(policy, tokenizer).chunk(make_long_section_document(tokens=511, indivisible=True))


def test_large_table_splits_by_complete_rows_and_repeats_context():
    tokenizer = FakeTokenizer(model_max_input_tokens=512, special_token_budget=2)
    policy = ChunkPolicy.for_embedding(make_embedding_spec(model_max_input_tokens=512, effective_corpus_content_token_limit=510))
    chunks = StructureAwareChunker(policy, tokenizer).chunk(make_large_table_document())
    assert len(chunks) > 1
    assert all(item.table_metadata["headers"] for item in chunks)
    assert all(item.table_metadata["row_group_is_complete"] for item in chunks)


def test_equation_definition_bundle_is_preserved_when_splitting():
    tokenizer = FakeTokenizer(model_max_input_tokens=128, special_token_budget=2)
    policy = ChunkPolicy.for_embedding(make_embedding_spec(model_max_input_tokens=128, effective_corpus_content_token_limit=126))
    chunks = StructureAwareChunker(policy, tokenizer).chunk(make_large_equation_definition_document())
    assert all(item.equation_metadata["variable_definitions"] for item in chunks)
~~~

- [ ] **Step 2: Run the focused chunking tests to verify they fail**

~~~powershell
pytest tests/test_knowledge_chunking.py -q
~~~

Expected: FAIL because the structure-aware chunker and deterministic ID function are absent.

- [ ] **Step 3: Implement semantic-unit chunking**

Build units in reading order: section paragraph groups, equation plus adjacent
variable definitions, table plus caption/header/units/notes, figure caption
plus nearby explanation, and reference entries. Tokenize each candidate with
the actual embedding tokenizer, `add_special_tokens=True`, the corpus
instruction policy, and `truncation=False`. Derive the effective content limit
from the model's real maximum; for BGE-small V1 this is 510 content tokens and
512 model-input tokens. Split oversized prose only at tokenizer-confirmed
paragraph boundaries or within the same paragraph using the 64-token overlap;
split oversized tables only between complete row groups while repeating
caption and headers; split equation bundles only at complete definition units.
If no legal structural split fits, raise `ChunkTooLargeForEmbedding` and let
ingestion record `EMBED_FAILED`. Never truncate or send an over-limit input.
Never place a heading in an anonymous chunk; carry it in `section_path`.

Populate every `ChunkRecord` with source hash, document metadata, page/spine
location, section path, content type, structured equation/table metadata,
parser/chunker versions, estimated token count, and the SHA-256 ID formula
from the spec. Normalize line endings and Unicode before hashing, while
retaining the display text separately.

Add a wiring assertion that the ingestion path cannot construct a chunker from
`ChunkPolicy()` alone: it must derive the policy from the active provider's
complete `EmbeddingSpec` and pass that provider's tokenizer. Tests and
production code use only `StructureAwareChunker(policy, tokenizer)`.

- [ ] **Step 4: Run the focused chunking tests to verify they pass**

~~~powershell
pytest tests/test_knowledge_chunking.py -q
~~~

Expected: PASS for equation/variable bundling, table context and structural
splits, section boundaries, tokenizer-derived limits, no-truncation checks,
bounded overlap, deterministic IDs, and stable ordering.

- [ ] **Step 5: Commit**

~~~powershell
git add tradingagents/knowledge/chunking.py tests/test_knowledge_chunking.py
git commit -m "feat: add structure-aware knowledge chunking"
~~~

### Task 5: Add local embedding abstraction and versioned embedding cache

**Files:**
- Modify: `pyproject.toml` (add the optional `knowledge` dependency group only)
- Create: `tradingagents/knowledge/embeddings.py`
- Create: `tests/test_knowledge_embeddings.py`

**Interfaces:**
- Consumes `ChunkRecord` and `KnowledgeConfig` from Tasks 1 and 4.
- Produces `EmbeddingProvider` with `spec: EmbeddingSpec`, an actual
  `EmbeddingTokenizer`, and `embed(texts: Sequence[str], *, purpose: Literal["corpus", "query"] = "corpus") -> tuple[tuple[float, ...], ...]`.
- Produces `FastEmbedProvider.from_config(config)`, with lazy imports and local-model-only validation.
- Produces `EmbeddingArtifactStore(root).load_or_compute(chunks, provider)`, keyed by document/chunk hash and the complete canonical embedding spec.
- Produces typed `EmbeddingContractError`, `LocalModelUnavailable`, `EmbeddingInputTooLong`, `EmbeddingVersionMismatch`, and `EmbeddingSpecMismatch` exceptions.

- [ ] **Step 1: Write failing embedding tests**

~~~python
def test_embedding_provider_rejects_wrong_dimension_and_non_finite_values():
    provider = FakeEmbeddingProvider(dimensions=3, vectors=((1.0, 2.0),))

    with pytest.raises(EmbeddingContractError, match="dimensions"):
        provider.embed(("OFI",))


def test_embedding_cache_skips_repeat_work_only_for_matching_model_spec(tmp_path):
    provider = FakeEmbeddingProvider(dimensions=3)
    store = EmbeddingArtifactStore(tmp_path / "embeddings")
    chunks = (make_chunk("VPIN"),)

    first = store.load_or_compute(chunks, provider)
    second = store.load_or_compute(chunks, provider)

    assert provider.embed_calls == 1
    assert first == second


def test_missing_local_model_fails_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.sessions.Session.request", lambda *args, **kwargs: pytest.fail("network"))

    with pytest.raises(LocalModelUnavailable):
        FastEmbedProvider.from_config(KnowledgeConfig(artifact_root=tmp_path, embedding_model_path=tmp_path / "missing"))


def test_embedding_rejects_over_limit_without_truncation():
    provider = FakeEmbeddingProvider(
        spec=make_embedding_spec(model_max_input_tokens=512, effective_corpus_content_token_limit=510),
        tokenizer=FakeTokenizer(model_max_input_tokens=512, special_token_budget=2),
    )
    with pytest.raises(EmbeddingInputTooLong):
        provider.embed(("token " * 511,), purpose="corpus")


def test_embedding_spec_persists_tokenizer_limit_and_instruction_policies():
    spec = make_embedding_spec(
        model_id="BAAI/bge-small-en-v1.5",
        dimensions=384,
        model_max_input_tokens=512,
        effective_corpus_content_token_limit=510,
        tokenizer_fingerprint="tok-v1",
        corpus_instruction_policy="none-v1",
        query_instruction_policy="bge-search-prefix-v1",
    )
    assert spec.model_max_input_tokens == 512
    assert spec.effective_corpus_content_token_limit == 510
    assert spec.tokenizer_fingerprint == "tok-v1"
    assert spec.query_instruction_policy == "bge-search-prefix-v1"


def test_allowed_embedding_input_reaches_model_without_truncation():
    provider = RecordingFakeEmbeddingProvider(
        spec=make_embedding_spec(model_max_input_tokens=512, effective_corpus_content_token_limit=510),
        tokenizer=FakeTokenizer(model_max_input_tokens=512, special_token_budget=2),
    )
    text = "token " * 510
    provider.embed((text,), purpose="corpus")
    assert provider.seen_content_tokens == [510]
    assert provider.seen_model_input_tokens == [512]
    assert provider.truncation_requested is False
~~~

- [ ] **Step 2: Run the focused embedding tests to verify they fail**

~~~powershell
pytest tests/test_knowledge_embeddings.py -q
~~~

Expected: FAIL because the provider contract and artifact cache do not exist.

- [ ] **Step 3: Implement local provider and cache**

Add an optional `knowledge` extra containing Docling, FastEmbed, ONNX Runtime,
and LanceDB. Keep all imports inside adapter factory functions.
`FastEmbedProvider` must require a local model path/cache, set CPU thread
limits from `KnowledgeConfig`, resolve the actual tokenizer/model limit, and
expose an `EmbeddingSpec` containing model ID, resolved version, runtime,
artifact hash, dimensions, normalization, tokenizer/config fingerprint,
special-token budget, model/effective limits, `truncation=false`, and explicit
corpus/query instruction policies. The BGE-small V1 default is 512 model-input
tokens, 510 effective corpus-content tokens, `none-v1` corpus formatting, and
the versioned `bge-search-prefix-v1` query prefix. It must not call a download
helper.

Persist the canonical `resolved_model_version` and `artifact_hash` under the
row/registry names `embedding_model_version` and `embedding_artifact_hash`
respectively; they are direct serialization mappings, not separate values.

Before every corpus or query inference, tokenize with the actual configured
tokenizer and `truncation=False` (including special tokens and the relevant
instruction policy). Raise `EmbeddingInputTooLong` before model invocation if
the effective limit would be exceeded. The provider and fake tokenizer must
prove that no input is silently truncated; a returned vector is valid only for
the exact supplied text.

Cache files live under
`embeddings/<embedding_model_id>/<model_version>/<document_id>.npy` plus a
JSON sidecar. A cache hit is accepted only when chunk IDs, source hashes,
embedding spec, and dimensions match exactly. A mismatch raises
`EmbeddingVersionMismatch` for reuse checks and causes a new generation during
ingestion; it never mixes vectors. Persist the complete spec in every cache
sidecar and generation metadata so query/index compatibility can compare model
ID, resolved version, artifact hash, dimensions, normalization, tokenizer
fingerprint, model/effective limits, truncation, and instruction policies—not
just dimensionality.

- [ ] **Step 4: Run the focused embedding tests to verify they pass**

~~~powershell
pytest tests/test_knowledge_embeddings.py -q
~~~

Expected: PASS for dimension/finite validation, cache reuse, model-version and
artifact/spec mismatch, actual tokenizer limits, no-truncation behavior,
missing-model failure, bounded batches, explicit BGE query formatting, and
no-network behavior.

- [ ] **Step 5: Commit**

~~~powershell
git add pyproject.toml tradingagents/knowledge/embeddings.py tests/test_knowledge_embeddings.py
git commit -m "feat: add local knowledge embeddings"
~~~

### Task 6: Add generation-scoped LanceDB and SQLite FTS5 projections

**Files:**
- Create: `tradingagents/knowledge/vector_index.py`
- Create: `tradingagents/knowledge/lexical_index.py`
- Create: `tradingagents/knowledge/index_generation.py`
- Create: `tests/test_knowledge_indexes.py`

**Interfaces:**
- Consumes `ChunkRecord`, `EmbeddingSpec`, `IndexGeneration`, catalog registry methods, and embedding vectors from Tasks 1–5.
- Produces `VectorIndexWriter`, `VectorIndexReader`, `LexicalIndexWriter`, `LexicalIndexReader`, and `IndexGenerationManager`.
- `IndexGenerationManager.build_generation(chunks, vectors, generation_id) -> IndexGeneration` writes `vector/lancedb/<generation_id>/` and `keyword/<generation_id>/bm25.sqlite3`.
- `IndexGenerationManager.validate_generation(generation) -> None`, `build_and_activate(chunks, vectors, generation_id) -> IndexGeneration`, `activate_generation(generation) -> None`, and `resolve_active_generation() -> IndexGeneration` enforce matched locations, versions, complete embedding specs, dimensions, and population hashes.
- Produces typed `VectorIndexError`, `LexicalIndexError`, `IncompatibleIndexGeneration`, and `EmbeddingSpecMismatch` exceptions.

- [ ] **Step 1: Write failing generation and projection tests**

~~~python
def test_failed_vector_build_does_not_swap_active_generation(tmp_path):
    manager = make_generation_manager(tmp_path, vector=FakeVectorBackend(fail=True))
    old = manager.active_generation()

    with pytest.raises(VectorIndexError):
        manager.build_and_activate(make_chunks(), make_vectors(), "gen-002")

    assert manager.active_generation().generation_id == old.generation_id
    assert (tmp_path / "vector" / "lancedb" / old.generation_id).exists()


def test_successful_rebuild_activates_both_and_keeps_previous(tmp_path):
    manager = make_generation_manager(tmp_path)
    old = manager.active_generation()

    new = manager.build_and_activate(make_chunks(), make_vectors(), "gen-002")

    assert manager.active_generation().generation_id == "gen-002"
    assert new.lexical_location.name == "bm25.sqlite3"
    assert old.vector_location.exists()
    assert old.lexical_location.exists()


def test_query_rejects_vector_lexical_generation_mismatch(tmp_path):
    manager = make_generation_manager(tmp_path)
    generation = manager.active_generation()
    write_fake_lexical_metadata(generation.lexical_location, generation_id="other")

    with pytest.raises(IncompatibleIndexGeneration):
        manager.resolve_active_generation()


def test_generation_persists_complete_embedding_spec_not_dimensions_only(tmp_path):
    spec = make_embedding_spec(
        model_id="BAAI/bge-small-en-v1.5",
        resolved_model_version="2026-01",
        artifact_hash="sha256:local-model",
        dimensions=384,
        tokenizer_fingerprint="tok-v1",
        query_instruction_policy="bge-search-prefix-v1",
    )
    generation = make_generation_manager(tmp_path, embedding_spec=spec).active_generation()
    assert generation.embedding_spec == spec
    assert generation.embedding_spec.artifact_hash == "sha256:local-model"
    assert generation.embedding_spec.query_instruction_policy == "bge-search-prefix-v1"
~~~

- [ ] **Step 2: Run the focused index tests to verify they fail**

~~~powershell
pytest tests/test_knowledge_indexes.py -q
~~~

Expected: FAIL because generation directories, LanceDB/FTS5 adapters, and atomic activation are absent.

- [ ] **Step 3: Implement paired generation storage**

Use the exact layout:

~~~text
vector/lancedb/<generation_id>/
keyword/<generation_id>/bm25.sqlite3
state/active-index.json
~~~

The LanceDB row contains `chunk_id`, `document_id`, `source_hash`, text,
vector, content type, all source location fields, component versions,
the complete `EmbeddingSpec` fields (including artifact hash, tokenizer
fingerprint, model/effective limits, truncation flag, and corpus/query
instruction policies), `projection_generation`, and `active`. The lexical
database stores FTS5 text, `chunk_id`, provenance, generation ID, tokenizer
settings, schema version, and population hashes. The FTS5 tokenizer uses Unicode
normalization, case folding, diacritic removal, and `_`/`-` token characters.

`build_generation` writes into a staging directory, validates vector dimensions,
finite values, FTS5 availability, row counts, sorted document/chunk population
hashes, and sampled provenance. `activate_generation` commits the catalog
registry first, then atomically replaces `active-index.json`; startup rebuilds
the JSON pointer from the catalog if it is stale. The old generation is never
deleted. Incremental projection writes set `vector_ready` and `lexical_ready`
only after both rows carry the same generation and population transaction.

- [ ] **Step 4: Run the focused index tests to verify they pass**

~~~powershell
pytest tests/test_knowledge_indexes.py -q
~~~

Expected: PASS for failed vector/lexical builds, successful paired swap,
previous-generation retention, FTS5 exact terms, population validation,
complete embedding-spec metadata storage, and incompatible-generation/spec
rejection.

- [ ] **Step 5: Commit**

~~~powershell
git add tradingagents/knowledge/vector_index.py tradingagents/knowledge/lexical_index.py tradingagents/knowledge/index_generation.py tests/test_knowledge_indexes.py
git commit -m "feat: add paired local knowledge indexes"
~~~

### Task 7: Add incremental ingestion, removal, retained-previous behavior, and crash recovery

**Files:**
- Create: `tradingagents/knowledge/ingestion.py`
- Modify: `tradingagents/knowledge/catalog.py`
- Create: `tests/test_knowledge_ingestion.py`

**Interfaces:**
- Consumes `KnowledgeConfig`, `SourceScanner`, `KnowledgeCatalog`, `DocumentParser`, `ScannedDetector`, `StructureAwareChunker`, `EmbeddingProvider`, and `IndexGenerationManager` from Tasks 1–6.
- Produces `IngestionMode` with `INCREMENTAL` and `REBUILD`, `KnowledgeIngestor.run(mode) -> IngestionRunSummary`, and `recover_interrupted_runs() -> tuple[str, ...]`.
- Produces per-resource events for `UNSUPPORTED`, `UNCHANGED`, `DUPLICATE`, `NEEDS_OCR`, `PARSE_FAILED`, `EMBED_FAILED`, `INDEX_FAILED`, `INDEXED`, `REMOVED`, `SOURCE_CHANGED`, and `INTERRUPTED`.

- [ ] **Step 1: Write failing ingestion tests**

~~~python
def test_incremental_run_skips_unchanged_and_deduplicates_bytes(tmp_path):
    harness = ingestion_harness(tmp_path)
    first = harness.ingestor.run(IngestionMode.INCREMENTAL)
    harness.parser.parse_calls.clear()
    harness.embedder.embed_calls = 0

    second = harness.ingestor.run(IngestionMode.INCREMENTAL)

    assert first.counts[IngestionState.INDEXED] == 2
    assert second.counts[IngestionState.UNCHANGED] == 2
    assert second.counts[IngestionState.DUPLICATE] == 1
    assert harness.parser.parse_calls == []
    assert harness.embedder.embed_calls == 0


def test_remove_one_alias_then_last_alias_updates_default_activity(tmp_path):
    harness = ingestion_harness(tmp_path, duplicate=True)
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    (harness.source / "book.pdf").unlink()

    one_removed = harness.ingestor.run(IngestionMode.INCREMENTAL)
    assert one_removed.counts[IngestionState.REMOVED] == 1
    assert harness.catalog.document_is_active(harness.shared_document_id) is True

    (harness.source / "copy.pdf").unlink()
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    assert harness.catalog.document_is_active(harness.shared_document_id) is False


def test_failed_changed_alias_does_not_deactivate_other_duplicate(tmp_path):
    harness = ingestion_harness(tmp_path, duplicate=True)
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    (harness.source / "book.pdf").write_bytes(b"changed bytes")
    harness.parser.fail_for.add("book.pdf")

    result = harness.ingestor.run(IngestionMode.INCREMENTAL)

    assert result.counts[IngestionState.PARSE_FAILED] == 1
    assert harness.catalog.get_alias(resource_id_for("copy.pdf")).relation is AliasRelation.CURRENT
    assert harness.catalog.document_is_active(harness.shared_document_id) is True
    assert harness.catalog.get_alias(resource_id_for("book.pdf")).relation is AliasRelation.RETAINED_PREVIOUS


def test_interrupted_run_recovery_removes_partial_projection_and_keeps_active_pair(tmp_path):
    harness = ingestion_harness(tmp_path, interrupt_after="lexical")
    old_generation = harness.catalog.active_generation().generation_id
    harness.ingestor.run(IngestionMode.INCREMENTAL)

    harness.ingestor.recover_interrupted_runs()

    assert harness.catalog.active_generation().generation_id == old_generation
    assert harness.catalog.list_runs()[-1].state is IngestionState.INTERRUPTED
~~~

Add `test_ingestion_derives_chunk_policy_from_provider_spec_and_tokenizer`.
It spies on the chunker constructor and asserts that the production
coordinator passes `ChunkPolicy.for_embedding(harness.embedder.spec)` and
`harness.embedder.tokenizer`, with no character/word-limit fallback. This
keeps every test and production call site on the required
`StructureAwareChunker(policy, tokenizer)` contract.

- [ ] **Step 2: Run the focused ingestion tests to verify they fail**

~~~powershell
pytest tests/test_knowledge_ingestion.py -q
~~~

Expected: FAIL because the coordinator, per-document transactions, removal reconciliation, and recovery do not exist.

- [ ] **Step 3: Implement the coordinator and failure isolation**

The incremental coordinator scans all regular files, records unsupported files,
hashes supported files, and compares hash plus component fingerprints before
parsing. Exact duplicate hashes attach aliases without parsing or embedding.
Removed resource IDs are marked `REMOVED`; only the last current alias makes a
document inactive. A changed-file attempt stages its new document and both
projections. On success, alias reassignment and readiness are one catalog
transaction. On failure, staged rows are deactivated by run ID, the previous
relation becomes `RETAINED_PREVIOUS`, and other aliases remain unchanged.

Each source is statted before and after reading. A change during read records
`SOURCE_CHANGED` and discards staging. Per-resource failures continue the run;
the aggregate is `SUCCEEDED`, `PARTIAL_FAILURE`, or `FAILED` with visible
counts. `REBUILD` builds a complete new pair and calls `activate_generation`
only after both projections validate.

Write `RUNNING` ingestion rows before work. On startup, rows left running by a
crash become `INTERRUPTED`; partial files and rows with that run ID are
removed/deactivated, and the previous active matched generation remains
queryable. A single artifact-root lock prevents two writers; queries remain
read-only.

- [ ] **Step 4: Run the focused ingestion tests to verify they pass**

~~~powershell
pytest tests/test_knowledge_ingestion.py -q
~~~

Expected: PASS for first ingestion, unchanged skip, duplicates, one/last alias
removal, successful change, failed change retention, unsupported visibility,
scan/parser/embed/index failures, source mutation detection, rebuild swap, and
interrupted recovery.

- [ ] **Step 5: Commit**

~~~powershell
git add tradingagents/knowledge/ingestion.py tradingagents/knowledge/catalog.py tests/test_knowledge_ingestion.py
git commit -m "feat: add incremental knowledge ingestion and recovery"
~~~

### Task 8: Add provenance validation, hybrid fusion, and deterministic reranking

**Files:**
- Create: `tradingagents/knowledge/provenance.py`
- Create: `tradingagents/knowledge/fusion.py`
- Create: `tradingagents/knowledge/reranking.py`
- Create: `tradingagents/knowledge/query.py`
- Create: `tests/test_knowledge_query.py`

**Interfaces:**
- Consumes active-generation readers, `KnowledgeQuery`, `KnowledgeHit`, `ChunkRecord`, and catalog filters from Tasks 1–7.
- Produces `DenseCandidate`, `LexicalCandidate`, `FusedCandidate`, `RRFConfig(k=60)`, `reciprocal_rank_fuse(dense, lexical, config)`, and `Reranker.rerank(query, candidates)`.
- Produces `KnowledgeQueryService(vector_reader, lexical_reader, catalog, embedder, reranker).search(request: KnowledgeQuery) -> tuple[KnowledgeHit, ...]` and `validate_hit_provenance(hit) -> None`.
- Produces typed `ProvenanceError`, `EmbeddingSpecMismatch`, and reuses `IncompatibleIndexGeneration` for active-pair mismatches.

- [ ] **Step 1: Write failing query and ranking tests**

~~~python
def test_exact_term_uses_lexical_candidate_and_hybrid_order_is_deterministic():
    service = query_harness(
        dense=[candidate("semantic", rank=1, chunk_id="chunk-b")],
        lexical=[candidate("VPIN", rank=1, chunk_id="chunk-a")],
    )

    result = service.search(KnowledgeQuery(text="VPIN", top_k=2))

    assert [item.chunk_id for item in result] == ["chunk-a", "chunk-b"]
    assert result[0].lexical_score is not None
    assert result[0].source_hash
    assert result[0].page is not None


def test_content_type_and_document_filters_are_applied_before_return():
    service = query_harness()

    result = service.search(
        KnowledgeQuery(text="inventory risk", content_types=(ContentType.EQUATION,), document_ids=("doc-1",))
    )

    assert result
    assert all(item.content_type is ContentType.EQUATION for item in result)
    assert all(item.document_id == "doc-1" for item in result)


def test_anonymous_hit_and_generation_mismatch_fail_closed():
    with pytest.raises(ProvenanceError):
        validate_hit_provenance(make_hit(source_hash=None))
    with pytest.raises(IncompatibleIndexGeneration):
        query_harness(vector_generation="gen-a", lexical_generation="gen-b").search(
            KnowledgeQuery(text="OFI")
        )


def test_identical_query_and_index_embedding_specs_allow_dense_search():
    spec = make_embedding_spec()
    service = query_harness(index_embedding_spec=spec, query_embedding_spec=spec)
    assert service.search(KnowledgeQuery(text="OFI"))


@pytest.mark.parametrize("field", [
    "model_id",
    "resolved_model_version",
    "artifact_hash",
    "normalization_policy",
    "tokenizer_fingerprint",
    "model_max_input_tokens",
    "query_instruction_policy",
    "query_instruction_version",
])
def test_embedding_spec_mismatch_fails_before_dense_retrieval(field):
    index_spec = make_embedding_spec()
    value = 513 if field == "model_max_input_tokens" else f"different-{field}"
    query_spec = make_embedding_spec(**{field: value})
    service = query_harness(
        index_embedding_spec=index_spec,
        query_embedding_spec=query_spec,
        dense_reader=FakeVectorReader(fail_if_called=True),
    )

    with pytest.raises(EmbeddingSpecMismatch):
        service.search(KnowledgeQuery(text="inventory risk"))


def test_same_dimensions_do_not_make_different_specs_compatible():
    index_spec = make_embedding_spec(model_id="model-a", dimensions=384)
    query_spec = make_embedding_spec(model_id="model-b", dimensions=384)
    with pytest.raises(EmbeddingSpecMismatch):
        query_harness(index_embedding_spec=index_spec, query_embedding_spec=query_spec).search(
            KnowledgeQuery(text="microprice")
        )


def test_search_constructs_read_only_query_embedder_once_and_no_writers():
    harness = query_harness()
    service = harness.service
    result = service.search(KnowledgeQuery(text="queue imbalance"))
    assert result
    assert harness.embedder.embed_calls == 1
    assert harness.forbidden_constructors.all_zero()
~~~

- [ ] **Step 2: Run the focused query tests to verify they fail**

~~~powershell
pytest tests/test_knowledge_query.py -q
~~~

Expected: FAIL because candidate fusion, provenance validation, and the query service are absent.

- [ ] **Step 3: Implement RRF, feature reranking, and the read-only service**

Fetch bounded candidate sets from both readers using the active matched
generation. Apply document/content-type filters at each reader and recheck
them against the catalog. Before the first dense reader call, compare the
query provider's complete canonical `EmbeddingSpec` with the generation spec
field-for-field (model ID, resolved version, artifact hash, dimensions,
normalization, tokenizer fingerprint, model/effective limits, truncation,
and corpus/query instruction policies/versions). Raise
`EmbeddingSpecMismatch` on any difference, including same-dimension models,
without invoking the dense reader. Encode the query through the provider's
versioned query-instruction policy with truncation disabled. Fuse with:

~~~python
score(chunk) = sum(1.0 / (60 + rank) for rank in available_ranks)
~~~

The v1 CPU reranker adds exact-token coverage, phrase coverage, title/section
match, requested content-type match, and source diversity. Final ties sort by
rerank score, fused score, semantic score, lexical score, then ascending
`chunk_id`. It never calls an LLM or uses trading/action fields.

`validate_hit_provenance` requires document ID, source filename/path, source
hash, chunk ID, content type, source location, parser/chunker/index versions,
and generation identity. `KnowledgeQueryService` rejects empty/missing model
or mismatched indexes/specs, never silently falls back to another generation,
and returns only active documents with both projections ready. The service
depends on read-only readers and a query embedder; it never constructs a
parser, ingestion coordinator, source scanner, or writer.

- [ ] **Step 4: Run the focused query tests to verify they pass**

~~~powershell
pytest tests/test_knowledge_query.py -q
~~~

Expected: PASS for exact HFT terminology, semantic/lexical fusion, filters,
RRF scores, reranker ordering, tie behavior, provenance, stale exclusion,
complete embedding-spec compatibility (including all required mismatch
variants), no dense call on mismatch, and generation mismatch rejection.

- [ ] **Step 5: Commit**

~~~powershell
git add tradingagents/knowledge/provenance.py tradingagents/knowledge/fusion.py tradingagents/knowledge/reranking.py tradingagents/knowledge/query.py tests/test_knowledge_query.py
git commit -m "feat: add provenance-safe hybrid knowledge search"
~~~

### Task 9: Add the separate knowledge CLI and operator documentation

**Files:**
- Create: `tradingagents/knowledge/cli.py`
- Modify: `pyproject.toml`
- Create: `docs/knowledge-rag.md`
- Create: `tests/test_knowledge_cli.py`

**Interfaces:**
- Consumes `KnowledgeConfig`, `KnowledgeIngestor`, `KnowledgeCatalog`, and `KnowledgeQueryService` from Tasks 1–8.
- Produces the independent console command `knowledge` with `main(argv=None) -> int`.
- Produces these commands exactly: `knowledge index`, `knowledge rebuild`, `knowledge status`, `knowledge list`, `knowledge search`, `knowledge document`, and `knowledge quarantine`.

- [ ] **Step 1: Write failing CLI and stock-isolation tests**

~~~python
def test_status_does_not_construct_parser_or_embedding_provider(tmp_path, monkeypatch, capsys):
    forbidden = forbidden_constructor()
    monkeypatch.setattr("tradingagents.knowledge.cli.KnowledgeIngestor", forbidden)
    monkeypatch.setattr("tradingagents.knowledge.cli.EmbeddingProvider", forbidden)

    assert main(["status", "--artifact-root", str(tmp_path)]) == 0
    assert "INDEXED" in capsys.readouterr().out
    assert forbidden.calls == 0


@pytest.mark.parametrize("command", ["status", "list", "document", "quarantine"])
def test_metadata_commands_construct_no_parser_embedder_ingestor_or_writers(command, tmp_path, monkeypatch):
    forbidden = forbidden_constructor()
    monkeypatch.setattr("tradingagents.knowledge.cli.DocumentParser", forbidden)
    monkeypatch.setattr("tradingagents.knowledge.cli.EmbeddingProvider", forbidden)
    monkeypatch.setattr("tradingagents.knowledge.cli.KnowledgeIngestor", forbidden)
    monkeypatch.setattr("tradingagents.knowledge.cli.VectorIndexWriter", forbidden)
    monkeypatch.setattr("tradingagents.knowledge.cli.LexicalIndexWriter", forbidden)

    argv = [command, "--artifact-root", str(tmp_path)]
    if command == "document":
        argv.insert(1, "doc-test")
    assert main(argv) == 0
    assert forbidden.calls == 0


def test_search_prints_provenance_and_supports_content_type_filter(tmp_path, capsys):
    seed_query_fixture(tmp_path)

    assert main(["search", "order flow imbalance", "--content-type", "EQUATION", "--json", "--artifact-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["chunk_id"]
    assert payload[0]["source_hash"]
    assert payload[0]["content_type"] == "EQUATION"


def test_search_constructs_only_read_only_local_query_embedder(tmp_path, monkeypatch):
    seed_query_fixture(tmp_path)
    constructors = forbid_parser_ingestor_source_scanner_and_writers(monkeypatch)
    embedder = spy_query_embedder(monkeypatch)

    assert main(["search", "order flow imbalance", "--artifact-root", str(tmp_path)]) == 0
    assert embedder.calls == 1
    assert constructors.all_zero()


def test_stock_console_entrypoint_is_unchanged():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["tradingagents"] == "cli.main:app"
~~~

- [ ] **Step 2: Run the focused CLI tests to verify they fail**

~~~powershell
pytest tests/test_knowledge_cli.py -q
~~~

Expected: FAIL because the knowledge command and documentation do not exist.

- [ ] **Step 3: Implement explicit commands without implicit indexing**

Use `argparse` and preserve machine-readable JSON output. The command
contracts are:

~~~text
knowledge index [--source-root PATH] [--artifact-root PATH] [--docling-artifacts-path PATH] [--embedding-model-path PATH] [--json]
knowledge rebuild [--source-root PATH] [--artifact-root PATH] [--docling-artifacts-path PATH] [--embedding-model-path PATH] [--json]
knowledge status [--artifact-root PATH] [--json]
knowledge list [--artifact-root PATH] [--state STATE] [--json]
knowledge search TEXT [--artifact-root PATH] [--top-k N] [--content-type TYPE] [--document-id ID] [--include-stale] [--json]
knowledge document DOCUMENT_ID [--artifact-root PATH] [--json]
knowledge quarantine [--artifact-root PATH] [--json]
~~~

`status`, `list`, `document`, and `quarantine` instantiate only catalog/
diagnostic readers; they do not instantiate a parser, embedder, ingestor,
source scanner, or vector/lexical writer. `search` may instantiate one
read-only local embedding provider for query encoding and the active index
readers, but it never instantiates the document parser, ingestion coordinator,
source scanner, vector writer, lexical writer, or any source-mutating
component. It validates the complete query/index `EmbeddingSpec` before dense
retrieval. `index` and `rebuild` refuse an artifact root inside the source root,
print per-state counts, return a nonzero code for partial/failure runs, and
leave the source untouched. Add `knowledge = "tradingagents.knowledge.cli:main"`
without changing the stock `tradingagents` script.

Document `data_cache/knowledge`, the full generated layout, status meanings,
alias/currentness examples, versioned generation activation, offline model
requirements, provenance output, and the explicit Phase 8 boundary in
`docs/knowledge-rag.md`. Do not add knowledge options to `cli/main.py`.

- [ ] **Step 4: Run the focused CLI tests to verify they pass**

~~~powershell
pytest tests/test_knowledge_cli.py -q
~~~

Expected: PASS for command parsing, no implicit indexing, metadata-command
constructor isolation, read-only search-embedder construction, query/index
spec validation, JSON provenance, state listing, source/artifact-root safety,
exit codes, and unchanged stock entry-point configuration.

- [ ] **Step 5: Commit**

~~~powershell
git add tradingagents/knowledge/cli.py pyproject.toml docs/knowledge-rag.md tests/test_knowledge_cli.py
git commit -m "feat: add standalone knowledge CLI"
~~~

### Task 10: Add retrieval-quality benchmarks, source-integrity, privacy, and package-isolation coverage

**Files:**
- Create: `tests/test_knowledge_quality.py`
- Create: `tests/test_knowledge_isolation.py`
- Create: `tests/fixtures/knowledge/benchmark_cases.json`
- Modify: `docs/knowledge-rag.md` (benchmark and operational evidence sections)

**Interfaces:**
- Consumes the public query, catalog, ingestion, and provenance contracts from Tasks 1–9.
- Produces deterministic benchmark reporting for Recall@1/5/10, MRR, exact-keyword hit rate, content-type filter precision, provenance correctness, duplicate-result rate, and deterministic ordering.
- Produces source immutability and import/path isolation assertions that run without optional dependencies.

- [ ] **Step 1: Write failing quality and boundary tests**

~~~python
def test_hft_benchmark_reports_exact_terms_and_provenance():
    report = run_benchmark(load_cases("tests/fixtures/knowledge/benchmark_cases.json"), fixture_query_service())

    assert report.exact_keyword_hit_rate == 1.0
    assert report.provenance_correctness == 1.0
    assert report.recall_at[10] >= 0.8
    assert report.mrr > 0


def test_indexing_does_not_change_source_bytes_or_metadata(tmp_path):
    source = make_source_tree(tmp_path)
    before = snapshot_tree(source)
    ingestion_harness(tmp_path, source=source).ingestor.run(IngestionMode.INCREMENTAL)

    assert snapshot_tree(source) == before


def test_knowledge_package_has_no_trading_or_external_experience_imports():
    forbidden = {"tradingagents.forex", "tradingagents.graph", "tradingagents.agents", "cli.forex_watch"}
    assert forbidden.isdisjoint(imported_modules_under("tradingagents/knowledge"))
~~~

- [ ] **Step 2: Run the focused quality tests to verify they fail**

~~~powershell
pytest tests/test_knowledge_quality.py tests/test_knowledge_isolation.py -q
~~~

Expected: FAIL because the benchmark harness, source snapshot check, and isolation assertions are not implemented.

- [ ] **Step 3: Implement the deterministic benchmark and boundary checks**

The benchmark JSON contains labeled cases for:

~~~json
[
  {"query": "order flow imbalance short horizon prediction", "documents": ["doc-ofi"], "terms": ["order flow imbalance"]},
  {"query": "market making inventory risk", "content_type": "EQUATION", "documents": ["doc-mm"]},
  {"query": "queue imbalance price movement", "documents": ["doc-queue"]},
  {"query": "FX order flow exchange rate microstructure", "documents": ["doc-fx"]},
  {"query": "adverse selection in high-frequency market making", "documents": ["doc-mm"]},
  {"query": "Hawkes process high frequency order arrivals", "documents": ["doc-hawkes"]},
  {"query": "VPIN", "terms": ["VPIN"]},
  {"query": "DeepLOB", "terms": ["DeepLOB"]},
  {"query": "Avellaneda-Stoikov", "terms": ["Avellaneda-Stoikov"]},
  {"query": "Kyle lambda", "terms": ["Kyle lambda"]}
]
~~~

Implement metric calculations with deterministic rank handling and zero-
division rules. Require zero provenance errors and 100% exact-term hit rate on
the fixture; report semantic metrics rather than making a profitability claim.
Snapshot source relative paths, bytes, and metadata before/after indexing.
AST-scan the knowledge package and pyproject paths to prove it does not import
forex/graph/agent/CLI modules, Phase 5/6 stores, or network clients. Add a
test that no query result has an action/trade field.

- [ ] **Step 4: Run the focused quality tests to verify they pass**

~~~powershell
pytest tests/test_knowledge_quality.py tests/test_knowledge_isolation.py -q
~~~

Expected: PASS for semantic/exact-term/content-type benchmark cases,
provenance correctness, source byte preservation, no duplicate results, and
knowledge-package isolation.

- [ ] **Step 5: Commit**

~~~powershell
git add tests/test_knowledge_quality.py tests/test_knowledge_isolation.py tests/fixtures/knowledge/benchmark_cases.json docs/knowledge-rag.md
git commit -m "test: add knowledge retrieval quality and isolation gates"
~~~

### Task 11: Run the complete Phase 7 verification gate and prepare the handoff

**Files:**
- Create: `scripts/provision_knowledge_models.py` (explicit setup-only model and
  artifact provisioning; never called by normal CLI/index/search)
- Create: `scripts/knowledge_phase7_smoke.py` (bounded real local end-to-end
  parser/embedding/index/query smoke with source-integrity and network guards)
- Modify: none unless a verification command identifies a concrete defect in
  the owning task; any fix must add a regression test in that task's test file.

**Interfaces:**
- Verification covers discovery, unsupported visibility, aliases, removals,
  changed-file recovery, scan quarantine, explicit Docling OCR-offline/formula
  controls, native-EPUB/fallback provenance, parser structure, tokenizer-aware
  chunk limits/no truncation, complete embedding-spec compatibility, local
  embeddings, generation pairing, incremental/rebuild behavior, hybrid
  retrieval, metadata filtering, provenance, CLI, benchmark, source integrity,
  offline operation, and stock/forex isolation.

- [ ] **Step 1: Run the focused knowledge suite**

~~~powershell
pytest tests/test_knowledge_models.py tests/test_knowledge_config.py tests/test_knowledge_identity.py tests/test_knowledge_discovery.py tests/test_knowledge_catalog.py tests/test_knowledge_parser.py tests/test_knowledge_scanned.py tests/test_knowledge_chunking.py tests/test_knowledge_embeddings.py tests/test_knowledge_indexes.py tests/test_knowledge_ingestion.py tests/test_knowledge_query.py tests/test_knowledge_cli.py tests/test_knowledge_quality.py tests/test_knowledge_isolation.py -m "not integration" -q
~~~

Expected: all deterministic knowledge tests pass without downloading models or
calling a network/MT5/Ollama service. The Docling-option tests either use a
fake adapter or an already-installed optional dependency and assert
`do_ocr=False`; they never enable OCR or download artifacts.

- [ ] **Step 2: Run the complete repository suite**

~~~powershell
pytest -q
~~~

Expected: existing Phase 3–6, forex, provider, and stock-mode tests remain
green. No test modifies `new books` or starts automatic indexing.

- [ ] **Step 3: Run lint, compilation, and scope checks**

~~~powershell
ruff check tradingagents/knowledge tests/test_knowledge_*.py
python -m compileall -q tradingagents cli
git diff --check 6bf750590ab3b5eece4a36af11d6629fde2d0885..HEAD
~~~

Expected: Ruff and compileall pass; the diff contains only the knowledge
package, its optional dependency/entry point, documentation, and tests. The
stock CLI source and forex/MT5 files are unchanged.

- [ ] **Step 4: Run the offline/source-integrity smoke**

~~~powershell
$env:KNOWLEDGE_OFFLINE = "1"
pytest tests/test_knowledge_quality.py::test_indexing_does_not_change_source_bytes_or_metadata tests/test_knowledge_isolation.py -q
~~~

Expected: PASS with no HTTP request, no model download, no CUDA requirement,
and identical source-tree snapshot.

- [ ] **Step 5: Provision optional dependencies and local model artifacts explicitly**

Use the project's existing virtual environment for this one-time setup step.
Network access is allowed only for this explicit provisioning command; it is
never allowed during normal indexing, search, or the smoke itself.

~~~powershell
$ProjectPython = "C:\\AITrading\\TradingAgents\\.venv\\Scripts\\python.exe"
& $ProjectPython -m pip install -e "C:\\AITrading\\TradingAgents-phase7-worktree[knowledge]"
& $ProjectPython scripts/provision_knowledge_models.py provision all `
    --source-root "C:\\Users\\Zaid barghouthi\\Downloads\\new books" `
    --artifact-root "$env:TEMP\\phase7-knowledge-artifacts" `
    --docling-artifacts-path "$env:TEMP\\phase7-knowledge-artifacts\\docling" `
    --embedding-model-id "BAAI/bge-small-en-v1.5" `
    --embedding-model-path "$env:TEMP\\phase7-knowledge-artifacts\\embeddings\\bge-small-en-v1.5" `
    --allow-network
~~~

`provision_knowledge_models.py` is the only command allowed to download or
populate model artifacts. It supports independently retryable `provision
docling`, `provision embedding`, `provision all`, and offline `verify` stages.
Each stage uses a `<target>.partial-<run-id>` directory and publishes a
manifest only after validation. It must record resolved versions, artifact
hashes, tokenizer fingerprints, and dimensions, and fail clearly if
provisioning cannot complete. Its required `--source-root` guard rejects every
overlapping artifact root or destination, so it cannot touch `new books`.

- [ ] **Step 6: Run one bounded real local end-to-end smoke (mandatory)**

Select one to three deterministic PDF/EPUB resources from the approved source
folder (not the entire library), snapshot their bytes and filesystem metadata,
and run the real Docling parser, FastEmbed/ONNX provider, LanceDB projection,
SQLite FTS5 projection, and hybrid query service. The smoke must set
`KNOWLEDGE_OFFLINE=1`, use a dedicated artifact root outside `new books`, and
install no models or dependencies. `scripts/knowledge_phase7_smoke.py` must
fail closed if a network request is attempted, if Docling options do not show
`do_ocr=False`, or if local artifacts are missing; it must not fall back to a
fake parser/embedder or cloud service.

~~~powershell
$env:KNOWLEDGE_OFFLINE = "1"
& $ProjectPython scripts/knowledge_phase7_smoke.py `
    --source-root "C:\\Users\\Zaid barghouthi\\Downloads\\new books" `
    --artifact-root "$env:TEMP\\phase7-knowledge-smoke" `
    --docling-artifacts-path "$env:TEMP\\phase7-knowledge-artifacts\\docling" `
    --embedding-model-path "$env:TEMP\\phase7-knowledge-artifacts\\embeddings\\bge-small-en-v1.5" `
    --resource-limit 3 `
    --offline
~~~

The bounded smoke is a closure gate, not an optional test. Its scalar report
must include: selected resource paths and pre/post source hashes plus metadata;
parser and Docling versions; `do_ocr=False`; formula-enrichment enabled/disabled
and local model/version status; scan classifications; chunker version and
soft/effective/model token limits; tokenizer fingerprint; complete embedding
spec and generation ID; vector/lexical row counts and matched population hash;
hybrid/RRF query count, query latency, and provenance-complete result fields;
and a network-attempt count of zero. It must prove originals are byte- and
metadata-identical after the run. A missing artifact, parser/index failure, or
provenance/spec mismatch makes the Phase 7 result `NOT COMPLETE` and preserves
the typed diagnostic; do not claim closure from deterministic tests alone.

- [ ] **Step 7: Inspect final scope and report evidence**

~~~powershell
git status --short
git diff --stat 6bf750590ab3b5eece4a36af11d6629fde2d0885..HEAD
git log --oneline --decorate -15
~~~

The handoff must state the baseline/final SHAs, files changed, dependency and
CLI isolation, per-state ingestion counts, active generation identity,
duplicate/removal behavior, scan quarantine counts, benchmark metrics,
provenance correctness, source-preservation result, offline result, and the
explicit absence of MT5, TradingAgents decision integration, execution,
training, fine-tuning, and experience-memory mixing. This plan stops at Phase
7 verification and does not authorize Phase 8.

## Plan self-review

The plan maps every approved design section to an implementation task: goals
and boundaries (header/global constraints), architecture and package layout
(file map), storage/identity/state (Tasks 1–2 and 6–7), parsing, explicit
Docling OCR-offline/formula controls, native EPUB preference and scan detection
(Task 3), tokenizer-aware chunk/equation/table handling (Task 4), complete
local embedding specification and no-truncation validation (Task 5),
LanceDB/FTS5/fusion/reranking/query compatibility (Tasks 6 and 8), CLI and
query-embedder construction boundaries (Task 9), incremental/rebuild/recovery
(Task 7), versioning/provenance/privacy and performance (Tasks 2, 5–8, 10–11),
benchmark/acceptance (Task 10), and Phase 8 handoff isolation (global
constraints and Tasks 9–11).

The task interfaces use the same names and field meanings throughout. The
unsupported-file branch is terminal before hashing/parsing; alias activity is
derived from all current aliases; vector/lexical generation activation is
paired; Docling OCR is explicitly false and offline; formula enrichment is
local-only and optional; chunks are bounded by the actual tokenizer; and query
dense retrieval requires complete embedding-spec equality. Each task has a
failing test, a bounded implementation action, a focused passing command, and
its own commit. No production implementation, implicit indexing, cloud
service, trading integration, or experience-memory mixing is included.
