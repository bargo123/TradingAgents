# Phase 7 — Local HFT/FX Knowledge RAG Foundation

**Status:** Design only; implementation is not authorized by this document
**Date:** 2026-09-10
**Repository:** `C:\AITrading\TradingAgents`
**Baseline SHA:** `3cfae70a6bd07124273d7471c16423da8627191f`

## 1. Goals and non-goals

### Goals

Phase 7 creates a local, read-only knowledge-retrieval foundation over every
approved digitally extractable PDF or EPUB below:

```text
C:\Users\Zaid barghouthi\Downloads\new books
```

The system will:

1. discover resources recursively without category or usefulness filtering;
2. preserve document structure while parsing;
3. identify likely scanned/image-only resources and quarantine them as
   `NEEDS_OCR`;
4. create deterministic, structure-aware chunks with first-class equations,
   tables, definitions, captions, and references;
5. embed locally through an ONNX-backed FastEmbed adapter;
6. maintain a local LanceDB dense index and a local SQLite FTS5 BM25 index;
7. combine dense and lexical candidates with deterministic fusion and a
   replaceable reranker;
8. return only auditable, provenance-complete evidence;
9. ingest incrementally using content hashes and recover safely after
   interruption; and
10. expose a small read-only query API and explicit local CLI operations.

The success artifact is an evidence retriever, not a trading system:

```text
approved books/papers
  -> structured parse
  -> semantic chunks
  -> local embeddings + local lexical index
  -> hybrid retrieval
  -> provenance-complete evidence
```

### Non-goals and hard boundaries

Phase 7 does not:

- generate, select, backtest, deploy, or optimize trading strategies;
- change any TradingAgents decision or prompt;
- modify `forex-watch`, `forex-shadow`, `forex-evaluate`, LangGraph
  registration, or the stock `tradingagents` CLI;
- feed retrieved knowledge into any live or shadow decision;
- use an LLM, Ollama, hosted model, cloud embedding service, or cloud vector
  database;
- fine-tune Qwen, train a model, export ONNX weights, or run an evaluation
  pipeline;
- execute trades, call MT5, inspect positions/orders, or change broker state;
- mix published books/papers with Phase 5/6 decisions or outcomes; or
- treat a retrieval result as a BUY, SELL, HOLD, `PortfolioDecision`, or order.

The source folder is strictly read-only. No command may rewrite, rename, move,
delete, or create files below it. All generated material is outside that
folder. Normal indexing and querying must not require network access once the
software and local model artifacts have been installed.

## 2. Existing repository context

The repository is a Python package using `setuptools`, `pytest`, Ruff, and
separate console entry points. The stock command is `tradingagents`; the
read-only forex commands are separate entry points under `cli/` and the
`tradingagents.forex` package. The existing Phase 4–6 stores live below the
ignored `data_cache/` directory and must remain independent.

There is no existing RAG, Docling, FastEmbed, LanceDB, or BM25 package in the
repository or virtual environment at design time. The base dependency set is
therefore not expanded by this design. A future implementation may add a
clearly isolated optional knowledge extra and pin its versions without
changing the base TradingAgents install. The normal stock and forex commands
must continue to work when that extra is absent.

The source inventory was inspected read-only. It currently contains 102 PDF
files and no EPUB files; that observation does not change the rule that all
future supported PDFs and EPUBs are approved and must be considered.

## 3. Design alternatives and recommendation

### Approach A — embedded hybrid indexes (recommended)

Use Docling-first parsing, FastEmbed/ONNX local embeddings, LanceDB for dense
retrieval, SQLite FTS5 for BM25, and a SQLite catalog for state and provenance.
The vector and lexical stores remain local files; no server is required. A
deterministic reciprocal-rank fusion layer combines candidates, and a small
feature-based reranker sits behind an interface.

**Advantages:** exact HFT terms such as `VPIN`, `OFI`, and `Kyle lambda` are
handled by the lexical path; semantic paraphrases are handled by embeddings;
SQLite provides transactional manifests and recovery; all components are
practical on a six-core/16 GB Windows machine; and each component can be
replaced behind a stable boundary.

**Trade-offs:** two search indexes must be kept consistent, and LanceDB is an
additional local dependency. The implementation therefore needs staged writes,
active-index metadata, and crash reconciliation.

### Approach B — LanceDB-only hybrid emulation

Store vectors and text in LanceDB and implement lexical scoring in Python over
the candidate set.

**Advantages:** one search storage technology and a simpler deployment shape.

**Trade-offs:** Python lexical scoring is slower and less predictable as the
library grows, full-corpus BM25 statistics are harder to maintain, and exact
term behavior would depend on a custom tokenizer. It is not the preferred v1
quality or reproducibility trade-off.

### Approach C — external search service

Use Elasticsearch, OpenSearch, Qdrant, or another separately managed service
for dense and lexical retrieval.

**Advantages:** mature ranking and operational tooling.

**Trade-offs:** a server, credentials, network lifecycle, and backup/upgrade
burden would violate the local/no-server/privacy goals and add failure modes
that are unnecessary for this machine. This approach is rejected for Phase 7.

### Selected approach

Approach A is the Phase 7 design. It provides the strongest v1 retrieval
behavior without coupling the knowledge subsystem to TradingAgents or to an
external service. The catalog is the authority for document state and
provenance; LanceDB and FTS5 are replaceable projections of successfully
chunked documents.

## 4. Architecture and data flow

The runtime is split into an ingestion plane and a read-only query plane.

```text
source folder (read-only)
        |
        v
  deterministic discovery + SHA-256 identity
        |
        v
  parser adapter (Docling-first PDF/native EPUB, OCR-offline)
        |
        v
  normalized structured document IR
        |
        v
  scanned detector + structure-aware chunker
        |
        +--------------------+
        v                    v
  local embeddings       chunk metadata/text
        |                    |
        v                    v
  LanceDB vector table   SQLite FTS5 BM25 table
        \                    /
         \                  /
          v                v
             hybrid query + RRF
                    |
                    v
             deterministic reranker
                    |
                    v
       provenance-complete KnowledgeHit
```

Ingestion is explicit. No TradingAgents command starts it implicitly. A
single index writer owns an artifact root at a time; read-only queries may run
concurrently against the last active index. Rebuilds are staged beside the
active index and become visible only after validation and an atomic active
index pointer update.

The query plane imports only knowledge contracts. It does not import
`tradingagents.forex`, `tradingagents.graph`, `cli.forex_watch`, or any
Phase 5/6 store. A future orchestration layer may combine published evidence
and proprietary experience, but that combination is outside this phase.

## 5. Package and module boundaries

The future implementation should add a self-contained package:

```text
tradingagents/knowledge/
  __init__.py
  config.py              # source/artifact roots and component settings
  models.py              # IR, manifest, query, hit, and status contracts
  identity.py            # path identity, content hashes, duplicate handling
  catalog.py             # transactional metadata/state catalog interface
  parser.py              # parser protocol and normalized document IR
  docling_parser.py      # PDF/EPUB adapters around Docling
  scanned.py             # deterministic image-only detection
  chunking.py            # structure-aware chunker and deterministic IDs
  embeddings.py          # local embedding protocol and FastEmbed adapter
  vector_index.py        # LanceDB projection and active-index handling
  lexical_index.py       # SQLite FTS5/BM25 projection
  fusion.py              # candidate fusion and deterministic ordering
  reranking.py           # reranker protocol and v1 feature reranker
  provenance.py          # completeness validation and source rendering
  ingestion.py           # incremental/rebuild coordinator and recovery
  query.py               # read-only KnowledgeQuery service
  diagnostics.py         # quarantine, counters, and safe error records
  cli.py                 # knowledge-only command dispatch
```

The names describe ownership, not a requirement to create one file per class.
The important dependency direction is:

```text
CLI -> ingestion/query services -> adapters/indexes/catalog
parser/chunker/models -> no forex/graph/LLM imports
query API -> provenance/models only
```

The knowledge package must not add exports to the stock CLI or alter existing
agent state types. There is no shared vector table or shared catalog with
experience memory.

## 6. Configuration and generated storage layout

### Configuration

`KnowledgeConfig` is a serializable, validated value object containing:

- `source_root`, defaulting to the approved `new books` path;
- `artifact_root`, defaulting to `data_cache/knowledge` in the repository;
- parser ID/version and parser configuration hash;
- a configurable local `docling_artifacts_path` (defaulting to
  `<artifact_root>/docling`), `docling_offline=true`, and the V1 parser safety
  setting `docling_do_ocr=false` (there is no V1 override that enables OCR);
- `formula_enrichment_enabled=false` by default, plus the optional local
  formula-model ID/version/artifact identity when enrichment is enabled;
- chunker version and size policy;
- embedding model ID, local model path/cache, runtime, dimensions, and
  normalization policy, resolved model version, artifact hash, tokenizer
  fingerprint, actual model input limit, effective corpus-content limit, and
  corpus/query instruction policies;
- lexical tokenizer/index version;
- active vector/index version;
- query candidate, fusion, and reranker settings; and
- CPU worker/batch limits.

Environment variables and CLI flags may override paths and non-secret tuning,
but credentials are neither accepted nor stored. The artifact root must be
resolved before use and must not be equal to, or contained by, the source
root. A configuration error is reported before parsing starts.

### Generated layout

The recommended default layout is:

```text
data_cache/knowledge/
  catalog.sqlite3                         # authoritative manifest/state
  docling/                                  # local prefetched Docling artifacts only
  manifests/
    resources.jsonl                       # deterministic exported catalog
    documents.jsonl
    index.json
  parsed/
    <document_id>/
      normalized.json                     # normalized structured IR
      parser_output.json                  # parser cache when serializable
      metadata.json
  chunks/
    <chunker_version>/
      <document_id>.jsonl                 # deterministic chunk artifacts
  embeddings/
    <embedding_model_id>/<model_version>/
      <document_id>.npy
      <document_id>.json
  vector/
    lancedb/<index_version>/              # staged and active LanceDB tables
  keyword/
    <index_version>/
      bm25.sqlite3                        # SQLite FTS5 projection
  state/
    active-index.json                     # atomic active-index pointer
    ingestion-runs.jsonl                  # append-only run summaries
  quarantine/
    <run_id>/<resource_id>.json            # safe diagnostics only
  locks/
    index.lock
```

The catalog is authoritative for whether a document/version is current and
retrievable. JSONL manifests and `active-index.json` are deterministic,
human-inspectable projections and are regenerated atomically; they are not a
second source of truth. Parser caches, chunks, and embedding files are
content-addressed or versioned and may be reused only when their component
versions match the catalog. `active-index.json` and the catalog's
`index_registry` row identify one complete generation, including the matching
vector location, lexical location, embedding specification, lexical tokenizer
settings, schema/index version, and document/chunk population identity. The
catalog registry is the transactional authority; the JSON pointer is written
with temp-file replacement and reconciled from the registry on startup if a
crash leaves it stale. A vector table from one generation is never paired with
an FTS5 database from another generation.

## 7. Document identity, duplicates, editions, and aliases

Identity has two levels:

- `resource_id = res_<sha256(canonical_relative_path)>` identifies a path in
  the source tree. Windows separators are normalized to `/`, and the path is
  case-folded for identity while the display spelling is retained.
- `source_hash = sha256(raw_bytes)` identifies exact content. The stable
  `document_id` is `doc_<source_hash>`.

Exact byte duplicates therefore share one parsed document, one chunk set, and
one index projection even when filenames differ. Each path is retained as a
source alias with its own filename and relative path in the manifest.

Different bytes always produce different document IDs. Revised editions,
preprints, and published versions are never merged solely because titles or
authors look similar. Any future edition grouping is metadata only and cannot
collapse provenance or retrieval rows.

The source hash is computed from bytes, not modification time. Size and mtime
may be used as a pre-hash fast path but never as proof of unchanged content.
The file is statted before and after reading. If size, mtime identity, or a
second hash check indicates that a source changed during ingestion, staged
artifacts are discarded and the attempt is recorded as `SOURCE_CHANGED`; the
source is never written.

### Alias currentness invariant

The catalog keeps the resource-to-document relationship separate from the
latest ingestion attempt. An alias relation is `CURRENT` only when that path's
latest successfully ingested hash is the relation's `source_hash` and the
resource has not been removed. Exact duplicate paths therefore create several
`CURRENT` aliases for one document.

A document and all of its projections are `ACTIVE` in the default retrieval
view while at least one current, non-removed alias references its source hash.
The active flag is derived from this alias relation (or a transactionally
maintained equivalent), not from one selected filename.

For example, if `A.pdf` and `B.pdf` both reference hash X, removing A marks
only A `REMOVED`; B remains `CURRENT`, so document X and its chunks/vectors/
BM25 rows remain active and searchable. X becomes inactive in the default view
only after the last current alias disappears. Physical rows remain available
for diagnostics and explicit rebuild cleanup.

When A changes from X to Y, the staged Y attempt does not alter X's relation.
After successful ingestion, A is atomically reassigned to Y (or marked a
duplicate of an already indexed Y), X loses only A's alias, and B continues to
keep X active. Y becomes active only after both projections for Y are ready.

If changed-file ingestion fails, A's previous X relationship is retained as a
`RETAINED_PREVIOUS` diagnostic relation and the failed Y attempt is recorded
separately. That retained relation is not counted as a current alias for the
default view when A is the only path, so stale content cannot silently appear
as current. If B still references X, X remains active through B exactly as
before. A later successful attempt replaces the retained relation atomically.

## 8. Ingestion state machine and incremental behavior

Each resource has a latest-attempt state and each content identity has a
document state. The supported terminal and diagnostic states are:

```text
DISCOVERED
  -> UNSUPPORTED
  -> HASHED
       -> UNCHANGED
       -> DUPLICATE
       -> PARSING
            -> NEEDS_OCR
            -> PARSE_FAILED
            -> PARSED
                 -> CHUNKED
                      -> EMBEDDING
                           -> EMBED_FAILED
                           -> INDEXING
                                -> INDEX_FAILED
                                -> INDEXED
```

Additional resource-level states are `REMOVED`, `SOURCE_CHANGED`,
`INTERRUPTED`, and `RETAINED_PREVIOUS`. They describe a failed or superseded
attempt and are never presented as successful indexing.

The `UNSUPPORTED` branch is terminal for that resource attempt: it is recorded
in the catalog/status view and stops before hashing, parsing, embedding, or
index projection. Only supported PDF/EPUB resources continue to `HASHED`.

For one incremental run:

1. recursively enumerate **all regular files** under the source root in
   deterministic relative-path order. `.pdf` and `.epub` (case-insensitive)
   are the supported Phase 7 formats; every other regular file is recorded as
   `UNSUPPORTED` and is never parsed, embedded, or indexed. Directories,
   directory entries, and internal filesystem metadata are not resources;
2. hash each supported resource and compare the current hash plus all
   component versions with the catalog;
3. record `UNCHANGED` when the hash and compatible index artifacts already
   exist;
4. attach a new path as `DUPLICATE` when its hash already has a successfully
   indexed document;
5. parse, detect scans, normalize structure, chunk, embed, and project one
   changed/new document at a time;
6. compare the observed resource IDs with the previous scan and mark missing
   resources `REMOVED`; and
7. write run counts and diagnostics without requiring an LLM.

When a changed file fails, its previous successfully indexed version is kept
physically and the resource's previous relation is marked `RETAINED_PREVIOUS`
with the failed latest attempt. The old content is excluded from the default
query view when that resource is the only alias, but remains available for
explicit diagnostics and can be restored by a later successful re-ingestion.
If another current alias references the same old hash, that other alias keeps
the document active and searchable. This prevents a bad document from
corrupting unrelated documents or destroying the last known-good bytes.

`knowledge-index --rebuild` is an explicit maintenance operation. It forces
compatible resources through parsing/chunking as needed and constructs a new
vector/lexical index version side by side. It is never triggered by
`knowledge-search`, `knowledge-status`, or any TradingAgents command.

## 9. Parsing and normalized document IR

### Parser contract

`DocumentParser.parse(source: ReadOnlyResource) -> ParsedDocument` returns a
normalized intermediate representation (IR) with ordered blocks and document
metadata. The parser contract includes a parser ID, parser version, config
hash, warnings, and a deterministic source-location mapping. Parser failures
are caught per resource and become `PARSE_FAILED` with a quarantine record.

### Docling-first adapters and explicit offline controls

PDF parsing uses a pinned Docling conversion path and retains its reading
order, headings, page boundaries, tables, figures, equations, and references
where the source exposes them. The pipeline is configured explicitly rather
than relying on Docling defaults. Conceptually the PDF options are:

```python
PdfPipelineOptions(
    do_ocr=False,
    do_formula_enrichment=config.formula_enrichment_enabled,
    artifacts_path=config.docling_artifacts_path,
    # other pinned, non-network parser options
)
```

The implementation binds these names to the installed Docling version and
asserts the resulting options still have `do_ocr is False` before conversion.
V1 accepts only local, prefetched Docling artifacts. `docling_offline` is
always true; model/artifact resolution is restricted to
`docling_artifacts_path`, and no missing artifact may trigger a download or
other network access. A missing Docling dependency or required artifact raises
the typed `ParserDependencyUnavailable` or `DoclingArtifactsUnavailable`
error, which the ingestion coordinator records as a visible
`PARSE_FAILED` diagnostic while leaving other resources independent.

Formula enrichment is optional and disabled by default. When it is enabled,
the configured local formula artifacts must exist and their model ID/version
and artifact hash are recorded in parser provenance. If enrichment is disabled,
unsupported by the installed Docling version, or otherwise unavailable without
a requested local formula artifact, the parser preserves the best
parser-native equation representation and records
`formula_enrichment_status=DISABLED` or
`UNAVAILABLE_NATIVE_PRESERVED`; it never invents LaTeX and never enables OCR
as a side effect. If enrichment was explicitly requested but its required
local artifacts are missing, the typed artifact error is surfaced instead of
silently changing the requested mode.

For EPUB, the parser first attempts Docling's native EPUB conversion. It then
validates that the normalized IR preserves deterministic spine-item and anchor
locations. Only when native output cannot satisfy that provenance contract is
an explicit OPF/spine adapter used as a fallback; this fallback is not a second
default parser. Both paths read the EPUB container from bytes, never unpacking
into the source folder, and use deterministic spine order/anchors. The selected
path (`NATIVE_DOCLING_EPUB` or `OPF_SPINE_FALLBACK`) is persisted in parser
provenance so a future Docling change is auditable.

### Metadata and reading order

The IR records, where available:

- title and alternate title;
- ordered authors and detected publication year;
- document type and format;
- chapter, section, and subsection hierarchy;
- page number/page range for PDF, or EPUB spine item/anchor location;
- paragraph/list text and reading order;
- equation representation and variable definitions;
- table caption, headers, cells, units, and notes;
- figure captions and nearby explanatory text; and
- references/citations with their source location.

Parser provenance also records `docling_offline=true`, `docling_do_ocr=false`,
the resolved `docling_artifacts_path` identity, formula-enrichment requested/
status/model version (when applicable), and the native-EPUB versus fallback
adapter path. These values participate in the parser configuration hash.

Absent metadata is represented as `null` or an explicit empty list. It is not
invented from unrelated documents. A filename-derived title may be recorded
as `title_source=FILENAME_FALLBACK` so consumers can distinguish it.

## 10. Scanned/image-only detection and quarantine

V1 does not OCR. A likely scanned document is not indexed as trustworthy text.

The detector runs after the parser's text/image inventory is available and
records its version and scalar evidence. The PDF parser must have been built
with `PdfPipelineOptions(do_ocr=False)`; the detector is a classification step,
never an OCR fallback. For an image-only fixture, the parser therefore leaves
the text inventory empty, adds no OCR-generated blocks, and the detector emits
`NEEDS_OCR`. For PDFs, the default deterministic rule is:

- a page is text-bearing when it has at least 64 Unicode non-whitespace
  characters or a non-empty equation/table text block;
- a page is image-bearing when it contains a raster image or scan-like full
  page image; and
- classify `NEEDS_OCR` when fewer than 20% of pages are text-bearing and at
  least 80% are image-bearing, or when every page has no extractable text and
  the document contains page images.

For EPUBs, the same 64-character threshold is applied to spine items. If all
text-bearing items are below the threshold and the package is image-dominant,
the state is `NEEDS_OCR`. An empty or malformed package without scan evidence
is `PARSE_FAILED`, not `NEEDS_OCR`.

The quarantine record contains only safe diagnostics: resource/document IDs,
source hash, format, page/spine counts, text/image counts, detector version,
timestamps, and a bounded exception fingerprint where applicable. It does not
contain prompts, LLM output, or hidden reasoning. The original file remains
untouched and a status command lists every `NEEDS_OCR` resource.

## 11. Structure-aware chunking policy

Chunking consumes normalized blocks, not flattened arbitrary character
windows. The chunker version and policy are persisted with every chunk.

### Semantic units

The chunker first forms semantic units:

- a heading establishes a section path and is carried as metadata;
- adjacent paragraphs in one section are grouped until the size budget;
- an equation is bundled with its immediately surrounding variable-definition
  text when that text is in the same section;
- a table is bundled with its caption, header, units, notes, and nearby
  explanation; large tables split by complete row groups while repeating the
  caption/header context; and
- a figure caption is bundled with nearby explanatory prose, while the image
  itself is not embedded as an opaque text surrogate.

### Size and overlap

Chunking receives the active embedding provider's actual tokenizer and
`EmbeddingSpec`; a character/word estimate is never authoritative. For every
candidate chunk the tokenizer is called with `truncation=False` and the
purpose `corpus`, including any corpus instruction/formatting and required
special tokens. The chunker computes:

```text
effective_content_token_limit =
    max_input_tokens - special_token_budget - corpus_instruction_tokens
```

and rejects a non-positive limit. The final model input must be no longer than
`max_input_tokens`; no embedding provider may silently truncate an indexed
chunk.

The default `BAAI/bge-small-en-v1.5` profile has a 512-token model limit and
384 dimensions. With the V1 corpus policy (no query prefix on corpus chunks)
and a two-token special-token budget, its effective content limit is 510
tokens. The default structure policy therefore uses a 448-token soft target,
510-token hard content limit, and at most 64 content-token overlap. The hard
model-input limit, not an independent 768-token estimate, is the invariant;
other models derive their limits from their loaded tokenizer.

Adjacent paragraphs may be split at tokenizer-confirmed boundaries within a
section. Large tables split between complete row groups while repeating
caption/header context; equation bundles split only at complete definition
units. If an indivisible structure still cannot fit, the chunker raises the
typed `ChunkTooLargeForEmbedding` error and the resource is recorded as
`EMBED_FAILED`; it is never truncated or indexed with an over-limit vector.
Overlap is never used across a section boundary, equation-variable bundle,
table row group, or figure-caption bundle. There is no per-book tuning.

### Deterministic IDs

For ordinal `n`, block range `b`, and normalized chunk text `t`:

```text
chunk_id = chk_<sha256(
  schema_version | document_id | source_hash | chunker_version |
  content_type | section_path | b | n | normalized_text(t)
)>
```

The full hash is retained. Ordinal and block range aid inspection; the hash
prevents accidental collisions. Re-ingesting identical bytes with identical
component versions produces identical chunk IDs and content.

## 12. Equation, table, figure, and reference handling

`content_type` is a first-class indexed field. V1 defines:

```text
PROSE
EQUATION
TABLE
FIGURE_CAPTION
DEFINITION
REFERENCE
LIST
```

`HEADING` is retained in section metadata rather than emitted as an anonymous
retrieval unit. Additional types require a schema/version change and an
explicit test.

### Equations

An equation chunk stores the best available structured form (`latex`, MathML,
or parser-native representation), a plain-text rendering for lexical search,
variable definitions, and the surrounding explanatory text. The representation
format and parser confidence are metadata. If LaTeX is unavailable, the
original structured representation is preserved rather than replaced by a
guessed transcription. Formula-enrichment requested, enabled, model-version,
artifact-hash, and fallback status are retained in the equation metadata and
parser provenance.

### Tables

A table chunk stores caption, column headers, units, row/column counts, cells,
footnotes, and nearby explanation. The lexical text serialization includes
headers before cell values so a hit such as a prediction-accuracy table is
understandable. Large tables split only between complete rows and repeat the
caption/header context in each chunk.

### Figures and references

Figure captions and their adjacent explanatory text are searchable as
`FIGURE_CAPTION`; pixels are not sent to an embedding model in v1. References
retain citation text, identifiers, and page/section location as
`REFERENCE`. These units remain traceable even when the cited external paper
is not present in `new books`.

## 13. Metadata and strict provenance schema

Every indexed chunk must carry all of the following, either as a column or in
the authoritative catalog row:

```text
document_id
source_filename
source_relative_path
source_hash
title
authors
publication_year
document_type
format
page_start / page_end (nullable for EPUB)
chapter
section_path
epub_spine_item / anchor (nullable for PDF)
content_type
chunk_id
chunk_ordinal
parser_id / parser_version / parser_config_hash
chunker_version
embedding_model_id / embedding_model_version / embedding_artifact_hash
embedding_dimensions / embedding_max_input_tokens /
embedding_effective_content_token_limit / embedding_tokenizer_fingerprint /
embedding_normalization / embedding_truncation
embedding_corpus_instruction_policy /
embedding_query_instruction_policy / embedding_query_instruction_version
lexical_index_version
index_version
```

Additional fields include `text`, `content_hash`, `reading_order`,
`table_metadata`, `equation_metadata`, `active`, `vector_ready`,
`lexical_ready`, `projection_generation`, and a source-currentness marker. A
result lacking `document_id`, `source_hash`, `chunk_id`, source
filename, content type, or a source location is invalid and must be dropped or
raise a provenance error; it must never be returned anonymously.

The catalog conceptually contains these records (the implementation may use
normalized SQLite tables with the same semantics):

- `resources`: one row per source path, current hash, state, timestamps, and
  latest attempt;
- `documents`: one row per unique source hash, document metadata, parser
  artifact, and currentness;
- `document_aliases`: every filename/path that has the same content hash,
  relation status (`CURRENT`, `RETAINED_PREVIOUS`, or `REMOVED`), and the
  transaction that last changed that relation;
- `chunks`: one row per deterministic chunk and all provenance fields;
- `ingestion_runs` and `ingestion_events`: run/stage outcomes and bounded
  diagnostics; and
- `index_registry`: component versions, one matched vector/lexical generation
  pointer, dimensions, population identity, and build status.

The catalog exposes a derived `current_alias_count` for each document. A
document is eligible for default retrieval only when that count is greater
than zero **and** its vector and lexical projections are both ready in the
same active generation. Updating an alias relation and its derived count is a
single catalog transaction; removing one duplicate alias cannot deactivate
the shared document while another current alias remains.

The catalog contains no portfolio decisions, outcome labels, MT5 fields, or
experience-memory rows.

## 14. Local embedding abstraction

The query and ingestion services depend on an interface, not directly on
FastEmbed:

```python
EmbeddingProvider.spec -> EmbeddingSpec
EmbeddingProvider.tokenizer -> EmbeddingTokenizer
EmbeddingProvider.embed(
    texts: Sequence[str], *, purpose: Literal["corpus", "query"]
) -> Sequence[Vector]
```

`EmbeddingTokenizer` exposes the resolved tokenizer's special-token-aware
length, without truncation, for both corpus and query formatting.
`EmbeddingSpec` persists the complete vector-semantic contract:

```text
model_id
resolved_model_version
runtime
artifact_hash
dimensions
normalization_policy
tokenizer_fingerprint
model_max_input_tokens
special_token_budget
effective_corpus_content_token_limit
corpus_instruction_policy/version
query_instruction_policy/version
truncation=false
```

The provider validates finite values, exact dimensionality, and the actual
tokenized input length before every batch. It calls the model with truncation
disabled (or the equivalent version-pinned setting) and raises the typed
`EmbeddingInputTooLong` error before inference if a chunk exceeds the
effective limit. A provider must never return a vector for silently truncated
text.

The recommended v1 adapter is FastEmbed using a CPU-friendly ONNX model such
as `BAAI/bge-small-en-v1.5` (384 dimensions), pinned and configured through a
local model/cache path. BGE V1 uses `corpus_instruction_policy=none-v1` and
`query_instruction_policy=bge-search-prefix-v1`, whose exact query prefix is
`Represent this sentence for searching relevant passages: `; the policy and
version are part of the persisted spec. Corpus and query encoding may therefore
have different, explicit formatting while remaining reproducible. The exact
installed package/model versions, tokenizer fingerprint, model limit, and
artifact hash are recorded in the index registry. A stronger HFT/financial
model can later implement the same interface and build a new index version.

Indexing never silently downloads a model. If the configured local model is
unavailable, the affected resource is `EMBED_FAILED` with a visible diagnostic
and the existing active index remains intact. Normal query operation is fully
offline. CPU thread count and batch size are bounded by configuration; CUDA is
not assumed.

## 15. LanceDB vector schema and paired index lifecycle

The active LanceDB table has one row per active chunk with fields equivalent to:

```text
chunk_id: string (primary logical key)
document_id: string
source_hash: string
text: string
vector: fixed-length float32 array
content_type: string
source_filename: string
source_relative_path: string
title: string
authors_json: string
publication_year: integer nullable
page_start: integer nullable
page_end: integer nullable
chapter: string nullable
section_path_json: string
epub_location_json: string nullable
parser_version: string
chunker_version: string
embedding_model_id: string
embedding_model_version: string
embedding_artifact_hash: string
embedding_dimensions: integer
embedding_max_input_tokens: integer
embedding_effective_content_token_limit: integer
embedding_tokenizer_fingerprint: string
embedding_normalization: string
embedding_truncation: boolean
embedding_corpus_instruction_policy: string
embedding_query_instruction_policy: string
embedding_query_instruction_version: string
lexical_index_version: string
index_version: string
active: boolean
```

The vector column dimension is fixed by `EmbeddingSpec`. Any incompatible
embedding-semantic field (not only model ID, dimensions, or normalization) or
index schema is never mixed in one active table. A change creates a new
versioned table and updates the registry only after validation.

### Generation contract

An `index_version` names a **matched generation**, not just a vector table. Its
registry entry and `state/active-index.json` contain at least:

```text
generation_id / index_version
vector_location = vector/lancedb/<index_version>/
lexical_location = keyword/<index_version>/bm25.sqlite3
embedding_spec: complete canonical `EmbeddingSpec` (including model ID,
  resolved version, artifact hash, dimensions, normalization, tokenizer
  fingerprint, model/effective limits, truncation flag, and corpus/query
  instruction policies)
lexical_index_version / tokenizer_settings
schema_version
document_population_hash
chunk_population_hash
fusion_version / reranker_version
build_status
```

The population hashes are computed from the sorted active document/chunk IDs
and source hashes. Query startup loads this record first, verifies that both
locations exist and advertise the same generation, and refuses to combine
incompatible projections. Before any dense query, the query provider's
complete canonical `EmbeddingSpec` is compared field-for-field with the
generation's stored spec; equal dimensions alone are insufficient. A missing
or mismatched lexical projection or embedding spec is a typed index/
compatibility error, not permission to fall back to an older database silently.

### Rebuild atomicity

An explicit rebuild creates both projections under a new generation directory,
validates row counts, dimensions, population hashes, FTS5 availability, and
sampled provenance, then commits the matched generation in the catalog
registry and atomically replaces the derived active-index pointer. The old
complete generation remains intact for rollback/diagnostics. Activation never
publishes a new vector table with an old lexical database, or the reverse; a
startup reconciliation regenerates the pointer from the catalog if a crash
occurs between the database commit and pointer replacement.

### Incremental projection consistency

Incremental ingestion may append to the current generation for a single
document, but a document is not made active until its vector and lexical rows
are both present, checksummed, and tagged with the same generation and
population transaction. The catalog stores `vector_ready`, `lexical_ready`,
and `projection_generation`; query filters require both readiness flags and a
generation equal to the active registry entry. If either projection fails,
the document remains `INDEX_FAILED` (or its previous version remains in
service), staged rows are deactivated by run ID, and no partial document is
visible. When an incremental document is committed, the active generation's
population identity is advanced in the same catalog transaction after both
projections are verified. An implementation that cannot safely update the
active generation uses a side-by-side replacement generation for that
incremental batch instead.

For a small corpus, an exact/flat search is acceptable. LanceDB index type and
build parameters are recorded in `index.json`; a future implementation may
select an approximate index when corpus size warrants it without changing the
query contract. Metadata filters are applied in the vector query where
supported and rechecked against the catalog before returning a hit.

Deletes and replacements use `active=false` projection/tombstone semantics
until a rebuild compacts the table. A query never returns inactive rows or a
row whose projection generation is not the active matched generation.

## 16. Lexical/BM25 strategy

Each index generation has its own local SQLite database at
`keyword/<index_version>/bm25.sqlite3`. It contains an FTS5 virtual table and
a provenance side table keyed by `chunk_id`. FTS5's built-in `bm25()` rank
supplies corpus-aware lexical scores without a server or a separate Python
ranking dependency. The database stores the same generation ID, population
hashes, schema version, and lexical settings as the paired LanceDB table.

The tokenizer is versioned and uses Unicode normalization, case folding,
diacritic removal, and token characters that preserve `_` and `-` where they
are meaningful in terms such as `DeepLOB` and `order-flow`. The index stores
the structured text serialization, including equation representations, table
headers, captions, section headings, and definitions. Query construction
escapes FTS syntax and preserves quoted multiword phrases.

Exact-term requests are not reduced to an embedding search. A query containing
`VPIN`, `DeepLOB`, `Avellaneda-Stoikov`, or `Kyle lambda` receives a lexical
candidate set even when the dense model has no close neighbor. FTS5 metadata
filters are applied through the side table and rechecked before a hit is
returned. Lexical index version/settings are stored per row and in the active
registry. Query startup opens the lexical location named by the active matched
generation; it never reuses a fixed or older `bm25.sqlite3` path.

If FTS5 is unavailable in the Python SQLite build, indexing fails visibly with
`INDEX_FAILED`; it does not silently substitute an unmeasured tokenizer.

## 17. Hybrid fusion and reranking

For a `KnowledgeQuery`, the service obtains up to `candidate_k` results from
both dense and lexical search after applying safe metadata filters. V1 uses
reciprocal-rank fusion:

```text
RRF(chunk) = sum over available lists of 1 / (60 + rank_in_list)
```

The constant `60` is part of the fusion version. A chunk present in only one
list remains eligible; its provenance and source scores identify that fact.

The reranker is an interface:

```python
Reranker.rerank(query, candidates) -> Sequence[RankedCandidate]
```

V1 uses a deterministic CPU feature reranker, not an LLM or cross-encoder. It
combines normalized RRF, exact-term coverage, phrase coverage, section/title
match, content-type match, and source diversity. It never uses BUY/SELL/HOLD,
portfolio fields, or trading outcomes. A future local cross-encoder may
replace it behind the same interface and a new reranker version.

Final ties are resolved deterministically by fused score, rerank score,
semantic score, lexical score, then ascending `chunk_id`. The final result
contains both component scores and the version of fusion/reranking used.

## 18. Read-only query API

The public boundary is deliberately knowledge-only:

```python
KnowledgeQuery(
    text: str,
    top_k: int = 10,
    content_types: tuple[str, ...] = (),
    document_ids: tuple[str, ...] = (),
    include_stale: bool = False,
)
```

The service returns `KnowledgeHit` values containing:

```python
KnowledgeHit(
    chunk_id,
    document_id,
    content_type,
    text,
    score,
    semantic_score,
    lexical_score,
    fused_score,
    rerank_score,
    source_filename,
    source_relative_path,
    source_hash,
    title,
    authors,
    publication_year,
    page,
    page_start,
    page_end,
    chapter,
    section,
    parser_version,
    chunker_version,
    embedding_model_id,
    embedding_model_version,
    index_version,
)
```

`page` is a convenience alias for a single PDF page; EPUB results use spine
item/anchor provenance when page numbers do not exist. The API may expose
structured equation/table metadata in an extension field, but every returned
hit remains self-contained and auditable.

The API validates `top_k`, content types, document IDs, and provenance. It has
no knowledge of `forex-watch`, MT5, portfolio decisions, action labels, Phase 5
evaluation, or TradingAgents graph state. Empty indexes, missing model files,
incompatible index versions, embedding-spec mismatches, and provenance
violations raise typed, deterministic errors rather than returning fabricated
evidence.

### Query/index embedding compatibility

Hybrid search may instantiate one read-only local embedding provider to encode
the user's query. Before invoking the dense reader, it compares the provider's
complete canonical `EmbeddingSpec` with the active generation's stored spec:
model ID, resolved model version, local artifact hash, dimensions, normalization
policy, tokenizer/config fingerprint, model/effective limits, truncation flag,
and corpus/query instruction policy and versions (plus any other setting that
changes vector semantics). Same dimensionality is not compatibility. A mismatch
raises the typed `EmbeddingSpecMismatch` error and no dense retrieval call is
made. The lexical reader may still be used only through an explicitly selected
lexical-only mode; the default hybrid command fails closed rather than silently
mixing semantic spaces.

For the BGE V1 profile, corpus chunks use `none-v1` and queries use the exact
versioned `bge-search-prefix-v1` instruction policy. The query provider applies
that policy before token validation and embedding; the policy is persisted with
the index generation so a later query can reproduce and verify it.

## 19. CLI design

Use one separate `knowledge` console entry point so the existing stock CLI and
forex commands remain untouched. Its explicit subcommands are:

```text
knowledge index       # incremental ingestion; never implicit
knowledge rebuild     # explicit side-by-side full index rebuild
knowledge status      # counts and component/version status, no parsing
knowledge list        # indexed, failed, removed, or NEEDS_OCR resources
knowledge search      # hybrid search with content/document filters
knowledge document    # show document/chunk provenance by ID
knowledge quarantine  # list/show bounded diagnostic records
```

Required behavior:

- `index` accepts source/artifact overrides and reports per-state counts;
- `rebuild` requires an explicit flag/command and reports the active-index
  swap only after all requested validation passes;
- `status` answers approved-resource, indexed, `UNSUPPORTED`, `NEEDS_OCR`,
  failed, changed, removed, and version questions from the catalog alone;
- `list --state NEEDS_OCR` and equivalent state filters expose every
  quarantined resource;
- `search` prints provenance for every hit and supports `--content-type`,
   `--document-id`, `--top-k`, and machine-readable JSON output; and
- `document` shows title, aliases, hash, parser/chunker/embedding/index
   versions, locations, and chunk IDs without requiring an LLM.

Component construction is intentionally explicit: `status`, `list`,
`document`, and `quarantine` instantiate only catalog/diagnostic readers. They
do not instantiate a parser, embedder, ingestion coordinator, source scanner,
or vector/lexical writer. `search` may instantiate the read-only local
embedding provider needed for query encoding and the active vector/lexical
readers, but it never instantiates the document parser, ingestion coordinator,
source scanner, vector writer, lexical writer, or any source-mutating
component. Embedding-spec compatibility is checked before the dense reader is
called.

No command starts indexing because another TradingAgents command was launched.
The CLI opens source files read-only, refuses an artifact root inside the
source root, and returns a nonzero status for a run containing failed
resources while still preserving successful resources.

## 20. Incremental, rebuild, and removal behavior

The incremental coordinator uses source hashes and component fingerprints:

| Source/index condition | Result |
| --- | --- |
| regular file with unsupported extension | `UNSUPPORTED`; visible in manifest/status; no parse/embed/index |
| same path, same hash, compatible versions, indexed | `UNCHANGED`; no parse/embed/index |
| new path, hash already indexed | `DUPLICATE`; attach alias only |
| new path, new hash | parse and index one document |
| existing path, changed hash | ingest new version; retain old until replacement succeeds |
| parser/chunker version changed | reparse/rechunk affected document |
| embedding model/dimensions changed | build a new matched vector+lexical generation; lexical chunk text may be reused |
| lexical settings changed | build a new matched generation with a rebuilt FTS5 projection |
| source path removed while another current alias remains | mark only that alias `REMOVED`; keep the shared document/projections active |
| source path removed and it was the last current alias | mark `REMOVED`; deactivate the document in the default view; retain physical rows |

Physical tombstone cleanup is explicit rebuild/maintenance work. Query filters
use the catalog's alias-current, document-active, projection-ready, and
generation fields, so removing one duplicate alias cannot deactivate a shared
document and removed/superseded chunks cannot leak through a stale vector row.
A rebuild swaps the complete vector and lexical projections atomically and
leaves the previous matched generation intact for diagnostic rollback until an
explicit cleanup operation.

For an incremental update, the catalog changes a resource's alias relation and
the document's current-alias count only after both projections are ready in
the active generation. If the vector or lexical half fails, the relation and
active count remain at their prior committed values; staged rows are
deactivated by run ID. This prevents the active catalog from pointing at a
vector-only or lexical-only document.

## 21. Error, transaction, and recovery semantics

Errors are per-resource and stage-specific. The following typed diagnostics are
recorded with their corresponding terminal state (artifact/input/spec errors
are not additional successful states):

```text
UNSUPPORTED
PARSE_FAILED
NEEDS_OCR
EMBED_FAILED
INDEX_FAILED
SOURCE_CHANGED
INTERRUPTED
```

`DoclingArtifactsUnavailable` is recorded under `PARSE_FAILED`, while
`EmbeddingInputTooLong` and `EmbeddingSpecMismatch` are recorded under
`EMBED_FAILED` or query-time compatibility diagnostics respectively.

Successful outcomes are `INDEXED`, `UNCHANGED`, and `DUPLICATE`. A run may be
`SUCCEEDED`, `PARTIAL_FAILURE`, or `FAILED` based on counts; it never hides a
failed resource behind a successful aggregate message.

Each document is processed in a staging directory named by run/resource IDs.
Files are written to temporary names and atomically replaced only after a
complete checksum. Catalog updates use SQLite transactions. Lexical rows and
vector rows carry the run/generation version so orphaned rows can be removed
or deactivated during reconciliation. A resource alias reassignment, its
current-alias count, and the document/projection readiness flags are committed
only after both projections report the same generation.

For a rebuild, the catalog remains pointed at the previous complete generation
until vector and lexical validation both pass. Activation records the matched
locations and population hashes in one transaction; a crash before activation
therefore leaves the previous generation queryable. A crash after activation
can only expose the newly validated pair, never a mixed pair.

At startup, an index writer acquires `locks/index.lock` and checks for stale
`RUNNING` ingestion records. An interrupted run is marked `INTERRUPTED`; its
partial artifacts and projections are deleted/deactivated by run ID, while
the last active matched generation remains queryable. Recovery is idempotent
and safe to repeat. A crash cannot turn a parsed-but-unembedded,
embedded-but-unindexed, vector-only, or lexical-only resource into `INDEXED`.

Quarantine diagnostics are bounded and machine-readable. They include stage,
exception type/message fingerprint, resource/document hash, parser/index
versions, and scalar counters; they do not contain secrets or unnecessary
source copies. One malformed PDF cannot roll back or corrupt other documents.

## 22. Versioning and reproducibility

Every run and indexed document persists:

```text
schema_version
parser_id
parser_version
parser_config_hash
chunker_version
embedding_model_id
embedding_model_version
embedding_artifact_hash
embedding_dimensions
embedding_normalization
embedding_tokenizer_fingerprint
embedding_max_input_tokens
embedding_effective_content_token_limit
embedding_truncation
embedding_corpus_instruction_policy/version
embedding_query_instruction_policy/version
lexical_index_version
lexical_tokenizer_settings
fusion_version
reranker_version
index_version
application_version
```

The index registry also stores component package versions, model artifact
hash, LanceDB/SQLite feature versions, build timestamps, source root identity,
and document/chunk counts. A component mismatch is detected before query or
incremental reuse:

- parser/chunker mismatch invalidates affected parsed/chunk artifacts;
- any embedding-semantic mismatch (ID, resolved version, artifact hash,
  dimensions, normalization, tokenizer/config fingerprint, model/effective
  limits, truncation setting, or corpus/query instruction policy/version)
  requires a new matched vector+lexical generation and forbids mixed vectors;
- lexical tokenizer/settings mismatch requires a new matched generation with
  an FTS5 rebuild;
- fusion/reranker changes affect query ranking but not stored chunks; and
- schema changes require an explicit migration or new index version.

An index can therefore be reproduced from source hashes, parser configuration,
chunk artifacts, model artifact identity, and the recorded component versions.

## 23. Testing strategy and deterministic test matrix

CI tests use tiny checked-in PDF/EPUB fixtures, fake parser/embedding/index
adapters, and the local SQLite implementation. They do not parse the full
library, download models, call Ollama or MT5, call a hosted service, require
network access, or require CUDA.

The test matrix must include:

1. first ingestion of a PDF and EPUB;
2. unchanged hash skip with zero parser/embed calls;
3. changed-file re-ingestion and old-version retention;
4. exact-byte duplicate detection and alias provenance;
5. unsupported regular-file visibility in the manifest/status with zero parser
   or embedding calls;
6. removed-resource/tombstone handling;
7. removing one of two duplicate aliases while the shared document remains
   active and searchable;
8. removing the last alias and deactivating only that document in the default
   view;
9. changing one duplicate alias to new bytes while the other alias keeps the
   old document active, with the new document independently indexed after
   success;
10. failed changed-file ingestion retaining the prior relationship while a
    second duplicate alias remains current and unaffected;
11. explicit Docling PDF options with `do_ocr=False`, offline artifact-path
    resolution, and a missing-artifact typed failure with no network call;
12. image-only PDF classification as `NEEDS_OCR` with zero OCR-generated text;
13. formula enrichment disabled by default, local-only when enabled, and
    parser-native equation preservation/status when unavailable;
14. native Docling EPUB conversion preferred, deterministic spine/anchor
    provenance, and OPF/spine fallback only when native provenance fails;
15. parser failure isolation while other resources index;
16. deterministic chunk IDs and stable ordering;
17. title/author/year/page/chapter/section provenance;
18. equation metadata and variable-definition bundling;
19. table caption/header/cell metadata and row-safe structural splitting;
20. tokenizer-aware chunk at the allowed maximum and rejection of an
    over-capacity chunk without truncation;
21. large-table structural splitting and equation/definition preservation
    under the embedding limit;
22. embedding input length validation proving the model saw no truncated text;
23. complete embedding-spec mismatch rejection, including same-dimension
    different model ID, artifact/version, normalization, tokenizer, and query
    instruction policy;
24. identical query/index embedding specs succeeding before dense retrieval;
25. FTS5 exact-term and phrase retrieval;
26. dense, lexical, and hybrid/RRF retrieval;
27. content-type and document metadata filters;
28. deterministic score/tie ordering;
29. reranker interface and v1 feature-reranker ordering;
30. failed vector generation build not swapping the active pair;
31. failed lexical generation build not swapping the active pair;
32. successful rebuild swapping vector and lexical projections together;
33. previous complete generation remaining intact after a successful swap;
34. query rejection when vector and lexical generation IDs or population
    identities do not match;
35. incremental partial-projection failure not becoming visible;
36. interrupted ingestion cleanup and restart recovery;
37. provenance validation rejecting anonymous hits;
38. source-folder hash and file-byte preservation before/after indexing;
39. status/list/document/quarantine constructing no parser, embedder,
    ingestor, source scanner, or writer;
40. search constructing only a read-only local query embedder/readers and no
    parser, ingestor, source scanner, or writers;
41. offline/no-network behavior when local model artifacts are present; and
42. isolation checks proving no imports or storage paths reference forex,
    MT5, TradingAgents decisions, or experience memory.

Optional dependency tests may run when the knowledge extra is installed, but
the core contract tests remain runnable with fakes in the base development
environment.

## 24. Retrieval-quality benchmark

Retrieval quality is evaluated with a deterministic labeled benchmark, not by
whether any result happens to be returned and not by future trading
profitability.

The benchmark fixture contains small documents with known locations and
content types. A manually curated real-library benchmark may live outside the
source folder under `docs/knowledge/benchmarks/` and records only labels and
provenance, not modified book resources.

Required query families include:

```text
order flow imbalance short horizon prediction
market making inventory risk
queue imbalance price movement
FX order flow exchange rate microstructure
adverse selection in high-frequency market making
Hawkes process high frequency order arrivals
```

Exact-term cases include:

```text
VPIN
DeepLOB
Avellaneda-Stoikov
Kyle lambda
```

Content-type cases include:

```text
equations about inventory risk
tables reporting prediction accuracy
definitions of order-flow imbalance
```

Each labeled query records expected document ID(s), page/section where known,
content type, and expected terminology. The harness reports:

- Recall@1, Recall@5, and Recall@10;
- reciprocal rank/MRR;
- exact-keyword hit rate;
- content-type filter precision;
- provenance correctness rate and missing-field count;
- duplicate-result rate; and
- deterministic ordering/tie results.

The fixture acceptance gate is zero provenance errors, zero anonymous hits,
100% exact-keyword hit rate in the configured top-five exact-term cases, and
the labeled semantic recall/MRR values printed in the test report. Real-book
benchmarks establish a recorded baseline for later retrieval improvements;
they do not authorize a trading or profitability claim.

## 25. Windows and 16 GB performance design

The target machine is Windows 10, Ryzen 5 7430U (6 cores/12 threads), 16 GB
RAM, integrated Radeon graphics, and local SSD. The implementation should:

- use CPU-only ONNX execution with a bounded thread pool (default at most six
  worker threads);
- parse one or a small bounded number of documents at a time;
- stream page/block normalization rather than holding the entire library in
  memory;
- embed in batches of approximately 16–32 chunks and flush per document;
- cache embeddings by document/chunk/model fingerprint;
- keep LanceDB and FTS5 writes transactional per document/run;
- use candidate limits such as 50–200 for interactive retrieval;
- avoid loading all vectors or all text into a long-lived Python list; and
- report elapsed time, document/chunk counts, and memory-sensitive batch
  settings in run summaries.

No CUDA runtime, GPU driver, cloud GPU, or external server is assumed. The
configuration can lower batch size and workers for a smaller memory budget.
Long indexing is an explicit foreground operation; interactive search uses
the last validated active index.

## 26. Security, privacy, and source integrity

All book/paper bytes, parsed text, embeddings, indexes, and metadata remain on
the local machine. No source content is sent to hosted LLMs, cloud embedding
APIs, external vector stores, telemetry endpoints, or analytics services.
Phase 7 requires no credentials and stores none.

The source root is opened read-only. The implementation validates that every
resolved source path stays within the configured root, refuses symlink/reparse
targets that escape it, and never follows an artifact path back into the
source. A before/after source manifest can prove that names, bytes, and
timestamps were not changed by indexing.

CLI output treats document text as data and supports JSON encoding without
executing markup. Diagnostic records are bounded and do not copy entire source
documents. Paths in provenance are relative to the approved source root so a
manifest can be moved without exposing unrelated local paths.

## 27. Future extension seams and explicit Phase 8 handoff

The following seams are intentional and remain unimplemented in Phase 7:

- alternate parsers behind `DocumentParser`;
- stronger local embedding models behind `EmbeddingProvider`;
- cross-encoder or other local rerankers behind `Reranker`;
- additional content-type extractors for reliable structured evidence;
- vector/lexical backend replacements behind index protocols; and
- a later orchestration layer that can request published knowledge and
  proprietary experience separately.

Phase 8 may define a higher-level evidence orchestrator that combines:

```text
published books/papers -> Knowledge RAG (this phase)
shadow decisions/outcomes -> separate Experience Memory (future)
```

That layer must keep separate stores, provenance, trust labels, and retention
rules. It may decide how evidence is presented to an agent, but it must not
retroactively make Phase 7 know about BUY/SELL/HOLD, PortfolioDecision, MT5,
or `forex-watch`. No Phase 8 integration is part of this design or its
implementation plan.

## 28. Phase 7 acceptance criteria

Phase 7 will be considered implemented only when the following can be
demonstrated with deterministic evidence:

```text
new books/
  -> incremental discovery and hashing
  -> Docling-first structured parsing (offline, explicitly OCR-disabled)
  -> scan detection and NEEDS_OCR quarantine
  -> structure-aware HFT chunks
  -> local versioned embeddings
  -> local LanceDB + SQLite FTS5 indexes
  -> hybrid fusion and deterministic reranking
  -> exact source/page/section/chunk provenance
```

The evidence must also show:

- originals remain byte-for-byte unchanged;
- scanned resources are quarantined rather than indexed as trusted text;
- Docling PDF options prove `do_ocr=False`, no model download occurs, and a
  missing local artifact is a typed visible failure;
- formula enrichment is disabled by default, local-only when enabled, and
  parser-native equations remain available when enrichment is unavailable;
- new and changed resources are handled incrementally;
- exact-byte duplicates do not duplicate chunks or vectors;
- removed resources cannot appear in default retrieval;
- equations, tables, captions, and surrounding context survive sufficiently;
- exact HFT terminology and semantic HFT concepts are both retrievable;
- index/version mismatches fail closed or rebuild explicitly;
- chunks are validated against the loaded tokenizer/model limit (BGE-small V1:
  448-token soft target, 510-token effective content limit, 512-token model
  input maximum) with no silent truncation;
- query dense retrieval compares the complete embedding specification with the
  active generation before issuing a vector search;
- interrupted ingestion leaves the last active index usable;
- no MT5, TradingAgents, stock CLI, execution, or shadow decision behavior
  changes;
- no RAG context is injected into trading decisions;
- no training, fine-tuning, or ONNX model training occurs; and
- Phase 5/6 experience is absent from every Phase 7 table, file, and index.

## 29. Design self-review

This document was reviewed against the requested boundaries:

- no unresolved placeholders or implementation tasks are hidden in the
  architecture;
- catalog ownership, index projection ownership, and active-index swaps are
  explicit;
- discovery covers every regular file, with an explicit terminal
  `UNSUPPORTED` state before hashing/parsing;
- shared-document currentness is derived from all current aliases, so removing
  or changing one duplicate path cannot deactivate another path's knowledge;
- vector and lexical projections are generation-scoped, validated as a pair,
  and activated together with the previous complete generation retained;
- incremental failure, duplicate, removal, and interruption semantics do not
  contradict one another;
- PDF/EPUB support and `NEEDS_OCR` behavior are explicit, with Docling OCR
  disabled in pipeline options and local/offline artifact resolution;
- formula enrichment is explicitly optional/local-only and cannot enable OCR or
  replace a parser-native equation representation with invented LaTeX;
- native Docling EPUB conversion is preferred, with an auditable OPF/spine
  fallback only when native provenance cannot satisfy the contract;
- chunk limits are derived from the actual embedding tokenizer and model
  context (including special tokens/instructions), and over-limit inputs fail
  rather than truncate;
- the complete embedding specification, including artifact, tokenizer, and
  query-instruction identity, is checked before dense query retrieval;
- search may construct only a read-only local query embedder/readers, while
  status/list/document/quarantine construct no parser, embedder, ingestor, or
  writer;
- equations/tables retain structure instead of being flattened blindly;
- every query result requires provenance;
- published knowledge and future experience memory use different stores and
  trust semantics;
- no section adds MT5, LangGraph, TradingAgents, execution, RAG injection,
  training, or fine-tuning behavior; and
- the design assumes CPU/local operation and does not require CUDA, a cloud
  GPU, a hosted model, or a network service.

There are no genuinely blocking design questions. Exact patch-level dependency
versions and the final FastEmbed model artifact hash are implementation-time
provenance values; the interfaces, offline behavior, versioning rules, and
failure semantics above are fixed by this design.
