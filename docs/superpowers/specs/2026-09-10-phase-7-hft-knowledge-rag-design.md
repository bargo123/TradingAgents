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
  parser adapter (Docling-first PDF/EPUB)
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
- chunker version and size policy;
- embedding model ID, local model path/cache, runtime, dimensions, and
  normalization policy;
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
    bm25.sqlite3                          # SQLite FTS5 projection
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
versions match the catalog.

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

For one incremental run:

1. recursively enumerate regular `.pdf` and `.epub` files in deterministic
   relative-path order; unsupported files receive `UNSUPPORTED` and are not
   silently ignored;
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
physically and marked `RETAINED_PREVIOUS` with the failed latest attempt. The
old content is excluded from the default query view once the resource is
known to have changed, but remains available for explicit diagnostics and can
be restored by a later successful re-ingestion. This prevents a bad document
from corrupting unrelated documents or destroying the last known-good bytes.

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

### Docling-first adapters

PDF parsing uses Docling's document conversion path and retains its reading
order, headings, page boundaries, tables, figures, equations, and references
where the source exposes them.

EPUB parsing uses an EPUB container adapter that reads the OPF metadata and
spine in order, then passes each XHTML/HTML spine item through the same
Docling-first structure normalizer. The adapter preserves chapter names,
section headings, anchors, MathML/equation blocks, tables, captions, and
references. The EPUB package is read from bytes; it is never unpacked into the
source folder. Generated intermediate files, if needed by the parser library,
are placed under the artifact staging directory.

If a future Docling release changes its native EPUB support, the adapter
boundary remains stable: only the source-format adapter changes, and the
normalized IR and provenance contract do not.

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

Absent metadata is represented as `null` or an explicit empty list. It is not
invented from unrelated documents. A filename-derived title may be recorded
as `title_source=FILENAME_FALLBACK` so consumers can distinguish it.

## 10. Scanned/image-only detection and quarantine

V1 does not OCR. A likely scanned document is not indexed as trustworthy text.

The detector runs after the parser's text/image inventory is available and
records its version and scalar evidence. For PDFs, the default deterministic
rule is:

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

The v1 policy uses a shared tokenizer/word estimator only for size accounting,
with these bounds:

- soft target: 512 estimated tokens;
- hard maximum: 768 estimated tokens for a chunk;
- overlap: at most 64 estimated tokens, only when a long prose unit must be
  split within the same section; and
- no overlap across a section boundary, equation-variable bundle, table row
  group, or figure-caption bundle.

The estimator is deterministic and versioned. A single atomic equation/table
bundle may exceed the soft target so its meaning is not destroyed; if it
exceeds the hard maximum, it is split only at defined row/definition units,
with the required caption/header/variable context repeated. There is no
per-book tuning.

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
guessed transcription.

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
embedding_model_id / embedding_model_version / embedding_dimensions
lexical_index_version
index_version
```

Additional fields include `text`, `content_hash`, `reading_order`,
`table_metadata`, `equation_metadata`, `active`, and a source-currentness
marker. A result lacking `document_id`, `source_hash`, `chunk_id`, source
filename, content type, or a source location is invalid and must be dropped or
raise a provenance error; it must never be returned anonymously.

The catalog conceptually contains these records (the implementation may use
normalized SQLite tables with the same semantics):

- `resources`: one row per source path, current hash, state, timestamps, and
  latest attempt;
- `documents`: one row per unique source hash, document metadata, parser
  artifact, and currentness;
- `document_aliases`: every filename/path that has the same content hash;
- `chunks`: one row per deterministic chunk and all provenance fields;
- `ingestion_runs` and `ingestion_events`: run/stage outcomes and bounded
  diagnostics; and
- `index_registry`: component versions, active index pointer, dimensions, and
  build status.

The catalog contains no portfolio decisions, outcome labels, MT5 fields, or
experience-memory rows.

## 14. Local embedding abstraction

The query and ingestion services depend on an interface, not directly on
FastEmbed:

```python
EmbeddingProvider.spec -> EmbeddingSpec
EmbeddingProvider.embed(texts: Sequence[str]) -> Sequence[Vector]
```

`EmbeddingSpec` persists model ID, model artifact/version, runtime (`onnx`),
dimensions, normalization, tokenizer/config hash, and local artifact hash.
The provider validates finite values and exact dimensionality for every batch.

The recommended v1 adapter is FastEmbed using a CPU-friendly ONNX model such
as `BAAI/bge-small-en-v1.5` (384 dimensions), pinned and configured through a
local model/cache path. The exact installed package/model artifact versions
are recorded in the index registry. A stronger HFT/financial model can later
implement the same interface and build a new index version.

Indexing never silently downloads a model. If the configured local model is
unavailable, the affected resource is `EMBED_FAILED` with a visible diagnostic
and the existing active index remains intact. Normal query operation is fully
offline. CPU thread count and batch size are bounded by configuration; CUDA is
not assumed.

## 15. LanceDB vector schema and index lifecycle

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
embedding_dimensions: integer
lexical_index_version: string
index_version: string
active: boolean
```

The vector column dimension is fixed by `EmbeddingSpec`. Incompatible model
IDs, dimensions, normalization policies, or index schemas are never mixed in
one active table. A change creates a new versioned table and updates the
registry only after validation.

For a small corpus, an exact/flat search is acceptable. LanceDB index type and
build parameters are recorded in `index.json`; a future implementation may
select an approximate index when corpus size warrants it without changing the
query contract. Metadata filters are applied in the vector query where
supported and rechecked against the catalog before returning a hit.

Deletes and replacements use `active=false` projection/tombstone semantics
until a rebuild compacts the table. A query never returns inactive rows.

## 16. Lexical/BM25 strategy

The keyword projection is a local SQLite database containing an FTS5 virtual
table and a provenance side table keyed by `chunk_id`. FTS5's built-in
`bm25()` rank supplies corpus-aware lexical scores without a server or a
separate Python ranking dependency.

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
registry.

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
incompatible index versions, and provenance violations raise typed,
deterministic errors rather than returning fabricated evidence.

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
- `status` answers approved-resource, indexed, `NEEDS_OCR`, failed, changed,
  and version questions from the catalog alone;
- `list --state NEEDS_OCR` and equivalent state filters expose every
  quarantined resource;
- `search` prints provenance for every hit and supports `--content-type`,
  `--document-id`, `--top-k`, and machine-readable JSON output; and
- `document` shows title, aliases, hash, parser/chunker/embedding/index
  versions, locations, and chunk IDs without requiring an LLM.

No command starts indexing because another TradingAgents command was launched.
The CLI opens source files read-only, refuses an artifact root inside the
source root, and returns a nonzero status for a run containing failed
resources while still preserving successful resources.

## 20. Incremental, rebuild, and removal behavior

The incremental coordinator uses source hashes and component fingerprints:

| Source/index condition | Result |
| --- | --- |
| same path, same hash, compatible versions, indexed | `UNCHANGED`; no parse/embed/index |
| new path, hash already indexed | `DUPLICATE`; attach alias only |
| new path, new hash | parse and index one document |
| existing path, changed hash | ingest new version; retain old until replacement succeeds |
| parser/chunker version changed | reparse/rechunk affected document |
| embedding model/dimensions changed | build a new vector index; lexical chunks may be reused |
| lexical settings changed | rebuild the FTS5 projection |
| source path removed | mark `REMOVED`, deactivate its active projection safely |

Physical tombstone cleanup is explicit rebuild/maintenance work. Query filters
use the catalog's active/current flags, so removed or superseded chunks cannot
leak through a stale vector row. A rebuild swaps the complete vector and
lexical projections atomically and leaves the previous active version for
diagnostic rollback until an explicit cleanup operation.

## 21. Error, transaction, and recovery semantics

Errors are per-resource and stage-specific:

```text
UNSUPPORTED
PARSE_FAILED
NEEDS_OCR
EMBED_FAILED
INDEX_FAILED
SOURCE_CHANGED
INTERRUPTED
```

Successful outcomes are `INDEXED`, `UNCHANGED`, and `DUPLICATE`. A run may be
`SUCCEEDED`, `PARTIAL_FAILURE`, or `FAILED` based on counts; it never hides a
failed resource behind a successful aggregate message.

Each document is processed in a staging directory named by run/resource IDs.
Files are written to temporary names and atomically replaced only after a
complete checksum. Catalog updates use SQLite transactions. Lexical rows and
vector rows carry the run/index version so orphaned rows can be removed or
deactivated during reconciliation.

At startup, an index writer acquires `locks/index.lock` and checks for stale
`RUNNING` ingestion records. An interrupted run is marked `INTERRUPTED`; its
partial artifacts and projections are deleted/deactivated by run ID, while
the last active index remains queryable. Recovery is idempotent and safe to
repeat. A crash cannot turn a parsed-but-unembedded or embedded-but-unindexed
resource into `INDEXED`.

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
embedding_dimensions
embedding_normalization
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
- embedding ID/version/dimensions/normalization mismatch requires a new vector
  index and forbids mixed vectors;
- lexical tokenizer/settings mismatch requires an FTS5 rebuild;
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
5. removed-resource/tombstone handling;
6. unsupported extension classification;
7. scanned/image-only quarantine as `NEEDS_OCR`;
8. parser failure isolation while other resources index;
9. deterministic chunk IDs and stable ordering;
10. title/author/year/page/chapter/section provenance;
11. equation metadata and variable-definition bundling;
12. table caption/header/cell metadata and row-safe splitting;
13. embedding dimension/model mismatch rejection;
14. FTS5 exact-term and phrase retrieval;
15. dense, lexical, and hybrid/RRF retrieval;
16. content-type and document metadata filters;
17. deterministic score/tie ordering;
18. reranker interface and v1 feature-reranker ordering;
19. explicit rebuild and atomic active-index swap;
20. interrupted ingestion cleanup and restart recovery;
21. provenance validation rejecting anonymous hits;
22. source-folder hash and file-byte preservation before/after indexing;
23. offline/no-network behavior when local model artifacts are present; and
24. isolation checks proving no imports or storage paths reference forex,
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
  -> Docling-first structured parsing
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
- new and changed resources are handled incrementally;
- exact-byte duplicates do not duplicate chunks or vectors;
- removed resources cannot appear in default retrieval;
- equations, tables, captions, and surrounding context survive sufficiently;
- exact HFT terminology and semantic HFT concepts are both retrievable;
- index/version mismatches fail closed or rebuild explicitly;
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
- incremental failure, duplicate, removal, and interruption semantics do not
  contradict one another;
- PDF/EPUB support and `NEEDS_OCR` behavior are explicit;
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
