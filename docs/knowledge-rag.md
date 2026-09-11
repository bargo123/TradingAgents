# Local knowledge RAG operations

Phase 7 adds a deliberately separate local operator command: `knowledge`.
It does not change the stock `tradingagents` command, the forex tools, MT5,
or any execution behavior. The knowledge index is research infrastructure,
not a trading signal or an order path.

## Local artifacts and source safety

The default artifact root is `data_cache/knowledge`. It must be outside the
read-only source root. Indexing writes only beneath that artifact root and
never modifies source documents. The generated layout is:

```text
data_cache/knowledge/
  catalog.sqlite3                         # authoritative resource, alias, generation, and diagnostic state
  docling/                                  # explicitly prefetched Docling assets only
  manifests/
    resources.jsonl                       # deterministic exported catalog
    documents.jsonl
    index.json
  parsed/
    <document-id>/
      normalized.json                     # normalized structured IR
      parser_output.json                  # serializable parser cache when available
      metadata.json
  chunks/
    <chunker-version>/<document-id>.jsonl # deterministic chunk artifacts
  embeddings/
    <embedding-model-id>/<model-version>/
      <document-id>.npy
      <document-id>.json
  vector/lancedb/<generation-id>/          # dense projection and index metadata
  keyword/<generation-id>/bm25.sqlite3     # FTS5 lexical projection and provenance
  state/
    active-index.json                      # atomic, replaceable active-generation pointer
    ingestion-runs.jsonl                   # append-only run summaries
  quarantine/<run-id>/<resource-id>.json   # bounded safe diagnostics only
  locks/index.lock                         # one writer ownership lock
  .index-staging/                          # recoverable unpublished generations
  .ingestion-staging/                      # recoverable unpublished resources
```

Use only explicit commands; neither status, search, nor another TradingAgents
command starts ingestion:

```powershell
knowledge index --source-root "D:\books" --artifact-root data_cache/knowledge
knowledge rebuild --source-root "D:\books" --artifact-root data_cache/knowledge
knowledge status --artifact-root data_cache/knowledge --json
knowledge list --state NEEDS_OCR --artifact-root data_cache/knowledge
knowledge document <document-id> --artifact-root data_cache/knowledge --json
knowledge quarantine --artifact-root data_cache/knowledge --json
knowledge search "order flow imbalance" --content-type EQUATION --top-k 10 --artifact-root data_cache/knowledge --json
```

`index` is incremental. `rebuild` is the only explicit full, side-by-side
generation rebuild. Both print per-state counts and return nonzero when a run
is partial or failed. An artifact root inside the source root is rejected.

## States, aliases, and generations

`INDEXED` means a current resource is present in the active matched dense and
lexical generation. `UNCHANGED` reuses compatible indexed bytes;
`DUPLICATE` adds an alias to already-indexed bytes. `UNSUPPORTED`,
`NEEDS_OCR`, `PARSE_FAILED`, `EMBED_FAILED`, `INDEX_FAILED`, and
`SOURCE_CHANGED` are visible in `status`, `list`, and `quarantine` without
starting a parser or model.

Two source paths with the same source hash are aliases of one document. If one
alias disappears, the document remains current while another `CURRENT` alias
exists. If the last alias is removed, the document remains auditable but is
not returned by ordinary search. `RETAINED_PREVIOUS` records a failed change
without making an old document silently current.

`knowledge document <document-id>` reads the catalog only and returns the
title/metadata, aliases, source hash, parser and component fingerprints, and
the full provenance for every chunk (including locations and chunk IDs).

Every successful publication activates one versioned generation only after
both dense and lexical projections validate as the same population and
embedding specification. The previous active generation stays queryable until
that swap. The active pointer is repairable metadata; the catalog generation
record is authoritative.

## Offline requirements and provenance

PDF/EPUB parsing requires locally prefetched Docling assets at
`--docling-artifacts-path` (default `<artifact-root>/docling`), runs offline,
and keeps OCR disabled. Querying needs a local FastEmbed/ONNX model; set
`KNOWLEDGE_EMBEDDING_MODEL_PATH` to its pre-provisioned directory, or place it
at `<artifact-root>/models/embedding`. No command downloads models or accepts
credentials.

Search creates only the local query embedder and read-only active index
readers. Before dense retrieval it compares the full query provider
`EmbeddingSpec` with the active index: model/version/artifact, dimensions,
runtime, normalization, tokenizer, token limits, truncation, and corpus/query
instruction policies. A mismatch fails closed.

Each search result includes provenance such as `chunk_id`, `document_id`,
`source_hash`, source filename and relative path, content type, location,
parser/chunker/index versions, and ranking scores. Prefer `--json` for tools;
the output is a JSON array of those provenance-complete results.

## Boundary after Phase 7

Phase 7 ends at offline ingestion, catalog diagnostics, and read-only local
retrieval. Retrieval quality benchmarks, source-integrity/privacy validation,
and package-isolation evidence belong to Phase 8. This command does not
authorize model downloads, network access, LLM calls, broker access, MT5
actions, strategy promotion, or order submission.
