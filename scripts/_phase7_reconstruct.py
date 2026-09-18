from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.distillation.phase7 import Phase7KnowledgeSource
from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.chunking import ChunkPolicy, ChunkTooLargeForEmbedding, StructureAwareChunker
from tradingagents.knowledge.config import KnowledgeConfig
from tradingagents.knowledge.discovery import SourceScanner
from tradingagents.knowledge.docling_parser import DoclingDocumentParser
from tradingagents.knowledge.embeddings import FastEmbedProvider
from tradingagents.knowledge.ingestion import IngestionMode, KnowledgeIngestor
from tradingagents.knowledge.models import ParsedBlock
from tradingagents.knowledge.query import KnowledgeQueryService
from tradingagents.knowledge.vector_index import VectorIndexReader
from tradingagents.knowledge.lexical_index import LexicalIndexReader
from tradingagents.knowledge.models import KnowledgeQuery


SOURCE_ROOT = Path(r"C:\Users\Zaid barghouthi\Downloads\new books")
ARTIFACT_ROOT = Path(r"C:\p7r3")
VECTOR_CACHE_ROOT = Path(r"C:\Users\Zaid barghouthi\AppData\Local\Temp\phase7-full-corpus-20260915-r3")
DOCLING_ROOT = Path(r"C:\Users\Zaid barghouthi\AppData\Local\Temp\phase7-final-artifacts\docling")
EMBEDDING_ROOT = Path(
    r"C:\Users\Zaid barghouthi\AppData\Local\Temp\phase7-final-artifacts\embeddings\BAAI--bge-small-en-v1.5"
)
LINUX_MARKER = "the linux programming interface"
LOCAL_SPLIT_MARKERS = (
    "marcel link - high probability trading",
    "dter and dtter as high-frequency trading",
)
REPORT = Path(r"C:\Users\Zaid barghouthi\AppData\Local\Temp\phase7-reconstruction-report.json")


def snapshot(path: Path, relative_path: str) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "relative_path": relative_path,
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def options_factory(**kwargs):
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.models.inference_engines.object_detection.onnxruntime_engine import (
        OnnxRuntimeObjectDetectionEngineOptions,
    )

    supported = set(inspect.signature(PdfPipelineOptions).parameters)
    options = PdfPipelineOptions(**{key: value for key, value in kwargs.items() if key in supported})
    options.do_ocr = False
    options.do_table_structure = False
    options.accelerator_options = AcceleratorOptions(num_threads=1, device="cpu")
    options.layout_batch_size = 1
    options.table_batch_size = 1
    options.queue_max_size = 4
    options.layout_options.engine_options = OnnxRuntimeObjectDetectionEngineOptions()
    return options


class FilteredScanner:
    def __init__(self, scanner: SourceScanner) -> None:
        self.scanner = scanner
        self.all_resources = ()

    def discover(self):
        self.all_resources = self.scanner.discover()
        return tuple(
            resource
            for resource in self.all_resources
            if LINUX_MARKER not in resource.relative_path.casefold()
        )


class KnownOversizedChunker:
    """Document-local fallback for the known 512-token source blocks."""

    def __init__(self, base: StructureAwareChunker, tokenizer) -> None:
        self.base = base
        self.tokenizer = tokenizer
        self.policy = base.policy

    def chunk(self, document):
        try:
            return self.base.chunk(document)
        except ChunkTooLargeForEmbedding as error:
            source_path = (document.metadata.source_relative_path or "").casefold()
            if not any(marker in source_path for marker in LOCAL_SPLIT_MARKERS):
                raise
            blocks = list(document.blocks)
            target_index = next(
                (
                    index
                    for index, block in enumerate(blocks)
                    if error.block_id is not None and block.block_id == error.block_id
                ),
                None,
            )
            if target_index is None:
                target_index = self._first_oversized_block(document)
            if target_index is None:
                raise
            split_blocks = self._split_block(blocks[target_index])
            blocks[target_index : target_index + 1] = split_blocks
            rebuilt = replace(
                document,
                blocks=tuple(blocks),
                warnings=tuple(document.warnings) + ("marcel_embedding_limit_split",),
            )
            chunks = self.base.chunk(rebuilt)
            if any(chunk.embedding_input_tokens > self.policy.model_max_input_tokens for chunk in chunks):
                raise ChunkTooLargeForEmbedding("local split still exceeds the active embedding limit")
            return chunks

    def _first_oversized_block(self, document):
        for index, block in enumerate(document.blocks):
            text = self._block_text(block)
            if not self._fits(text):
                return index
        return None

    def _fits(self, text: str) -> bool:
        encoded = self.tokenizer.encode(text, add_special_tokens=True, purpose="corpus", truncation=False)
        tokens = encoded.get("input_ids") if isinstance(encoded, dict) else encoded
        return len(tokens) <= self.policy.model_max_input_tokens

    @staticmethod
    def _block_text(block: ParsedBlock) -> str:
        values = [block.text]
        if block.table is not None:
            values.extend((block.table.caption, " | ".join(block.table.headers), "\n".join(block.table.notes)))
            values.extend(" | ".join(row) for row in block.table.cells)
        if block.equation is not None:
            values.extend(
                (
                    block.equation.latex,
                    block.equation.mathml,
                    block.equation.parser_native,
                    block.equation.plain_text,
                    *block.equation.variable_definitions,
                )
            )
        return "\n".join(str(value) for value in values if value)

    def _split_block(self, block: ParsedBlock) -> list[ParsedBlock]:
        text = self._block_text(block)
        words = text.split()
        if not words:
            raise ChunkTooLargeForEmbedding("cannot split an empty oversized block")
        parts: list[str] = []
        start = 0
        while start < len(words):
            end = start
            while end < len(words) and self._fits(" ".join(words[start : end + 1])):
                end += 1
            if end == start:
                raise ChunkTooLargeForEmbedding("source contains an indivisible token over the embedding limit")
            parts.append(" ".join(words[start:end]))
            start = end
        result: list[ParsedBlock] = []
        for ordinal, part in enumerate(parts):
            metadata = dict(block.metadata)
            metadata.update(
                {
                    "split_from_block_id": block.block_id,
                    "split_part": ordinal,
                    "split_reason": "embedding_input_limit",
                }
            )
            result.append(
                replace(
                    block,
                    block_id=f"{block.block_id}:embedding-part-{ordinal:04d}",
                    text=part,
                    table=None,
                    equation=None,
                    reading_order=(block.reading_order * 10000) + ordinal,
                    metadata=metadata,
                )
            )
        return result


class StrictCacheIngestor(KnowledgeIngestor):
    def __init__(self, *args, normal_chunker, marcel_chunker, **kwargs):
        self.reused_documents: set[str] = set()
        self.new_documents: set[str] = set()
        self.reused_vectors = 0
        self.new_vectors = 0
        self.cache_mismatch_documents: set[str] = set()
        self._normal_chunker = normal_chunker
        self._marcel_chunker = marcel_chunker
        super().__init__(*args, **kwargs)

    def _stage_resource(self, run_id, resource, previous, counts):
        source_path = resource.relative_path.casefold()
        self.chunker = (
            self._marcel_chunker
            if any(marker in source_path for marker in LOCAL_SPLIT_MARKERS)
            else self._normal_chunker
        )
        return super()._stage_resource(run_id, resource, previous, counts)

    def _vectors_for(self, document_id, chunks):
        if not chunks:
            return ()
        path = self._vector_cache_path(document_id)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("embedding_spec") != self.embedder.spec.to_dict():
                    raise ValueError("embedding specification mismatch")
                expected_ids = [chunk.chunk_id for chunk in chunks]
                if payload.get("chunk_ids") != expected_ids:
                    raise ValueError("chunk identity/content/chunker mismatch")
                vectors = tuple(tuple(float(value) for value in row) for row in payload["vectors"])
                if len(vectors) != len(chunks) or any(
                    len(vector) != self.embedder.spec.dimensions
                    or any(not math.isfinite(value) for value in vector)
                    for vector in vectors
                ):
                    raise ValueError("embedding dimension or finite-value mismatch")
                if document_id not in self.reused_documents and document_id not in self.new_documents:
                    self.reused_documents.add(document_id)
                    self.reused_vectors += len(vectors)
                return vectors
            except Exception as error:
                self.cache_mismatch_documents.add(document_id)
                raise RuntimeError(f"existing embedding cache cannot be proven compatible: {error}") from error
        vectors = tuple(
            tuple(float(value) for value in vector)
            for vector in self.embedder.embed(tuple(chunk.text for chunk in chunks), purpose="corpus")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "embedding_spec": self.embedder.spec.to_dict(),
                    "chunk_ids": [chunk.chunk_id for chunk in chunks],
                    "vectors": vectors,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        if document_id not in self.new_documents:
            self.new_documents.add(document_id)
            self.new_vectors += len(vectors)
        return vectors

    def _vector_cache_path(self, document_id):
        # Keep the completed cache in its original artifact root while the
        # publication catalog/index use a short Windows path.  LanceDB emits
        # nested generation paths, so the short root avoids MAX_PATH failures
        # without recomputing any proven vectors.
        return VECTOR_CACHE_ROOT / "embeddings" / "ingestion-v1" / f"{document_id}.json"


def main() -> int:
    started = time.perf_counter()
    config = KnowledgeConfig(
        source_root=SOURCE_ROOT,
        artifact_root=ARTIFACT_ROOT,
        docling_artifacts_path=DOCLING_ROOT,
        embedding_model_path=EMBEDDING_ROOT,
        worker_count=1,
        embedding_batch_size=16,
    )
    base_scanner = SourceScanner(config)
    all_resources = base_scanner.discover()
    included_resources = tuple(r for r in all_resources if LINUX_MARKER not in r.relative_path.casefold())
    before = tuple(snapshot(r.path, r.relative_path) for r in included_resources if r.path.suffix.casefold() in {".pdf", ".epub"})
    raw_hashes: dict[str, list[str]] = defaultdict(list)
    for resource in all_resources:
        if resource.path.suffix.casefold() in {".pdf", ".epub"}:
            raw_hashes[resource.source_hash].append(resource.relative_path)
    included_hashes: dict[str, list[str]] = defaultdict(list)
    for resource in included_resources:
        if resource.path.suffix.casefold() in {".pdf", ".epub"}:
            included_hashes[resource.source_hash].append(resource.relative_path)

    scanner = FilteredScanner(base_scanner)
    # Force discovery to return the filtered source universe on every call.
    scanner.discover = lambda: included_resources
    parser = DoclingDocumentParser(config, pdf_options_factory=options_factory)
    embedder = FastEmbedProvider.from_config(config)
    normal_chunker = StructureAwareChunker(ChunkPolicy.for_embedding(embedder.spec), embedder.tokenizer)
    safe_chunker = KnownOversizedChunker(normal_chunker, embedder.tokenizer)
    ingestor = StrictCacheIngestor(
        config,
        scanner=scanner,
        parser=parser,
        embedder=embedder,
        normal_chunker=normal_chunker,
        marcel_chunker=safe_chunker,
    )
    summary = ingestor.run(IngestionMode.REBUILD)
    catalog = KnowledgeCatalog(ARTIFACT_ROOT / "catalog.sqlite3")
    generation = catalog.active_generation()
    after = tuple(snapshot(r.path, r.relative_path) for r in included_resources if r.path.suffix.casefold() in {".pdf", ".epub"})
    current_ids = catalog.current_document_ids()
    chunk_count = sum(len(catalog.chunks_for_document(document_id)) for document_id in current_ids)
    with sqlite3.connect(ARTIFACT_ROOT / "catalog.sqlite3") as connection:
        failed_paths = [
            row[0]
            for row in connection.execute(
                "SELECT r.relative_path FROM knowledge_ingestion_events e JOIN knowledge_resources r ON r.resource_id=e.resource_id WHERE e.run_id=? AND e.state IN ('PARSE_FAILED','EMBED_FAILED','INDEX_FAILED','NEEDS_OCR')",
                (summary.run_id,),
            )
        ]
    document_paths = {
        str(catalog.get_document(doc_id).metadata.get("source_relative_path") or "")
        for doc_id in current_ids
        if catalog.get_document(doc_id) is not None
    }
    marcel_indexed = any("marcel link - high probability trading" in path.casefold() for path in document_paths)
    linux_indexed = any(LINUX_MARKER in path.casefold() for path in document_paths)
    retrieval = {"hits": 0, "nonempty_text": False, "provenance_complete": False}
    if generation is not None:
        query_config = KnowledgeConfig(
            source_root=SOURCE_ROOT,
            artifact_root=ARTIFACT_ROOT,
            embedding_model_path=EMBEDDING_ROOT,
            embedding_model_id=generation.embedding_spec.model_id,
            embedding_model_version=generation.embedding_spec.resolved_model_version,
            embedding_artifact_hash=generation.embedding_spec.artifact_hash,
            embedding_dimensions=generation.embedding_spec.dimensions,
            embedding_runtime=generation.embedding_spec.runtime,
            embedding_normalization=generation.embedding_spec.normalization_policy,
            embedding_tokenizer_fingerprint=generation.embedding_spec.tokenizer_fingerprint,
            embedding_max_input_tokens=generation.embedding_spec.model_max_input_tokens,
            embedding_special_token_budget=generation.embedding_spec.special_token_budget,
            embedding_effective_content_token_limit=generation.embedding_spec.effective_corpus_content_token_limit,
            embedding_corpus_instruction_policy=generation.embedding_spec.corpus_instruction_policy,
            embedding_corpus_instruction_version=generation.embedding_spec.corpus_instruction_version,
            embedding_query_instruction_policy=generation.embedding_spec.query_instruction_policy,
            embedding_query_instruction_version=generation.embedding_spec.query_instruction_version,
            embedding_truncation=generation.embedding_spec.truncation,
        )
        query_embedder = FastEmbedProvider.from_config(query_config)
        service = KnowledgeQueryService(
            VectorIndexReader(generation.vector_location),
            LexicalIndexReader(generation.lexical_location),
            catalog,
            query_embedder,
        )
        hits = service.search(KnowledgeQuery(text="order flow imbalance", top_k=3))
        retrieval = {
            "hits": len(hits),
            "nonempty_text": bool(hits) and all(bool(hit.text.strip()) for hit in hits),
            "provenance_complete": bool(hits)
            and all(bool(hit.chunk_id and hit.document_id and hit.source_filename and hit.source_hash) for hit in hits),
        }
        Phase7KnowledgeSource.open(ARTIFACT_ROOT, expected_generation_id=generation.generation_id)

    report = {
        "source_pdfs_discovered": len(all_resources),
        "source_unique_hashes": len(raw_hashes),
        "source_duplicate_extra_files": len(all_resources) - len(raw_hashes),
        "included_pdfs": len(included_resources),
        "included_unique_hashes": len(included_hashes),
        "linux_excluded": not linux_indexed,
        "summary": summary.to_dict(),
        "catalog_documents": len(current_ids),
        "catalog_chunks": chunk_count,
        "generation": generation.to_dict() if generation else None,
        "generation_fingerprint": hashlib.sha256(
            json.dumps(generation.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if generation
        else None,
        "reused_cache_documents": len(ingestor.reused_documents),
        "reused_cache_vectors": ingestor.reused_vectors,
        "new_embedding_documents": len(ingestor.new_documents),
        "new_embedding_vectors": ingestor.new_vectors,
        "cache_mismatch_documents": sorted(ingestor.cache_mismatch_documents),
        "failed_paths": failed_paths,
        "marcel_indexed": marcel_indexed,
        "retrieval": retrieval,
        "source_fingerprints_unchanged": before == after,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "parser_runtime": {"ocr": False, "table_structure": False, "offline": True, "linux_excluded": True},
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if generation is not None and retrieval["nonempty_text"] and retrieval["provenance_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
