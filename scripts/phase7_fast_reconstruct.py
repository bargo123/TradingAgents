"""Publish Phase 7 from a resumable, OCR-free PDFium parse cache.

This is a bounded recovery path for the Windows corpus.  It never writes to
the approved source directory and never treats an old vector cache as
compatible unless its complete embedding specification and deterministic chunk
IDs match the newly parsed text.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path

from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.config import KnowledgeConfig
from tradingagents.knowledge.discovery import SourceScanner
from tradingagents.knowledge.embeddings import FastEmbedProvider
from tradingagents.knowledge.fast_pdfium_parser import CachedPdfiumParser
from tradingagents.knowledge.ingestion import IngestionMode, KnowledgeIngestor
from tradingagents.knowledge.lexical_index import LexicalIndexReader
from tradingagents.knowledge.models import KnowledgeQuery
from tradingagents.knowledge.query import KnowledgeQueryService
from tradingagents.knowledge.vector_index import VectorIndexReader

SOURCE_ROOT = Path(r"C:\Users\Zaid barghouthi\Downloads\new books")
ARTIFACT_ROOT = Path(r"C:\p7fast")
PARSE_CACHE_ROOT = ARTIFACT_ROOT / "parse-cache"
NEW_VECTOR_CACHE_ROOT = ARTIFACT_ROOT / "embeddings" / "ingestion-v1"
OLD_VECTOR_CACHE_ROOT = Path(
    r"C:\Users\Zaid barghouthi\AppData\Local\Temp\phase7-full-corpus-20260915-r3"
) / "embeddings" / "ingestion-v1"
EMBEDDING_ROOT = Path(
    r"C:\Users\Zaid barghouthi\AppData\Local\Temp\phase7-final-artifacts\embeddings\BAAI--bge-small-en-v1.5"
)
DOCLING_ROOT = Path(r"C:\Users\Zaid barghouthi\AppData\Local\Temp\phase7-final-artifacts\docling")
LINUX_MARKER = "the linux programming interface"


class FilteredScanner:
    def __init__(self, scanner: SourceScanner) -> None:
        self.scanner = scanner

    def discover(self):
        return tuple(
            resource
            for resource in self.scanner.discover()
            if resource.path.suffix.casefold() == ".pdf"
            and LINUX_MARKER not in resource.relative_path.casefold()
        )


class CacheAwareIngestor(KnowledgeIngestor):
    def __init__(self, *args, old_cache_root: Path, new_cache_root: Path, **kwargs):
        self.old_cache_root = old_cache_root
        self.new_cache_root = new_cache_root
        self.reused_cache_documents: set[str] = set()
        self.reused_cache_vectors = 0
        self.new_embedding_documents: set[str] = set()
        self.new_embedding_vectors = 0
        self.cache_mismatches: set[str] = set()
        super().__init__(*args, **kwargs)

    def _load_cache(self, path: Path, document_id: str, chunks):
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
            return vectors
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.cache_mismatches.add(document_id)
            raise ValueError(str(exc)) from exc

    def _vectors_for(self, document_id, chunks):
        if not chunks:
            return ()
        for root in (self.new_cache_root, self.old_cache_root):
            path = root / f"{document_id}.json"
            if not path.is_file():
                continue
            try:
                vectors = self._load_cache(path, document_id, chunks)
            except ValueError:
                continue
            self.reused_cache_documents.add(document_id)
            self.reused_cache_vectors += len(vectors)
            return vectors

        vectors = tuple(
            tuple(float(value) for value in vector)
            for vector in self.embedder.embed(tuple(chunk.text for chunk in chunks), purpose="corpus")
        )
        self.new_cache_root.mkdir(parents=True, exist_ok=True)
        path = self.new_cache_root / f"{document_id}.json"
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
        temporary.replace(path)
        self.new_embedding_documents.add(document_id)
        self.new_embedding_vectors += len(vectors)
        return vectors


def _generation_fingerprint(generation) -> str:
    return hashlib.sha256(
        json.dumps(generation.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    started = time.perf_counter()
    config = KnowledgeConfig(
        source_root=SOURCE_ROOT,
        artifact_root=ARTIFACT_ROOT,
        docling_artifacts_path=DOCLING_ROOT,
        embedding_model_path=EMBEDDING_ROOT,
        parser_id="pypdfium2",
        parser_version="pypdfium2-page-v1",
        parser_config_hash="sha256:" + hashlib.sha256(
            b"pypdfium2-page-v1|ocr=false|deterministic-page-order=true"
        ).hexdigest(),
        worker_count=1,
        embedding_batch_size=16,
    )
    scanner = FilteredScanner(SourceScanner(config))
    embedder = FastEmbedProvider.from_config(config)
    parser = CachedPdfiumParser(PARSE_CACHE_ROOT)
    ingestor = CacheAwareIngestor(
        config,
        scanner=scanner,
        parser=parser,
        embedder=embedder,
        old_cache_root=OLD_VECTOR_CACHE_ROOT,
        new_cache_root=NEW_VECTOR_CACHE_ROOT,
    )
    summary = ingestor.run(IngestionMode.INCREMENTAL)
    catalog = KnowledgeCatalog(ARTIFACT_ROOT / "catalog.sqlite3")
    generation = catalog.active_generation()
    document_ids = catalog.current_document_ids()
    chunk_count = sum(len(catalog.chunks_for_document(doc_id)) for doc_id in document_ids)
    failed_states = (
        "PARSE_FAILED",
        "EMBED_FAILED",
        "INDEX_FAILED",
        "NEEDS_OCR",
        "SOURCE_CHANGED",
    )
    with sqlite3.connect(ARTIFACT_ROOT / "catalog.sqlite3") as connection:
        failed = connection.execute(
            f"""
            SELECT r.relative_path, r.state, e.error_type, e.error_message
            FROM knowledge_resources r
            LEFT JOIN knowledge_ingestion_events e ON e.resource_id = r.resource_id
            WHERE r.state IN ({','.join('?' for _ in failed_states)})
            ORDER BY r.relative_path
            """,
            failed_states,
        ).fetchall()

    retrieval = {"hits": 0, "nonempty_text": False, "provenance_complete": False}
    if generation is not None:
        query_embedder = FastEmbedProvider.from_config(config)
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
            and all(
                bool(hit.chunk_id and hit.document_id and hit.source_filename and hit.source_hash and hit.page)
                for hit in hits
            ),
        }

    result = {
        "source_files_discovered": len(scanner.scanner.discover()),
        "included_files": len(scanner.discover()),
        "linux_excluded": all(LINUX_MARKER not in r.relative_path.casefold() for r in scanner.discover()),
        "summary": summary.to_dict(),
        "catalog_documents": len(document_ids),
        "catalog_chunks": chunk_count,
        "generation_id": generation.generation_id if generation else None,
        "generation_fingerprint": _generation_fingerprint(generation) if generation else None,
        "reused_cache_documents": len(ingestor.reused_cache_documents),
        "reused_cache_vectors": ingestor.reused_cache_vectors,
        "new_embedding_documents": len(ingestor.new_embedding_documents),
        "new_embedding_vectors": ingestor.new_embedding_vectors,
        "cache_mismatch_documents": sorted(ingestor.cache_mismatches),
        "failed": [
            dict(zip(("relative_path", "state", "error_type", "error_message"), row, strict=True))
            for row in failed
        ],
        "marcel_indexed": any(
            "marcel link - high probability trading" in (catalog.get_document(doc_id).metadata.get("source_relative_path") or "").casefold()
            for doc_id in document_ids
            if catalog.get_document(doc_id) is not None
        ),
        "dter_indexed": any(
            "dter" in (catalog.get_document(doc_id).metadata.get("source_relative_path") or "").casefold()
            for doc_id in document_ids
            if catalog.get_document(doc_id) is not None
        ),
        "retrieval": retrieval,
        "source_root": str(SOURCE_ROOT),
        "artifact_root": str(ARTIFACT_ROOT),
        "parse_cache_root": str(PARSE_CACHE_ROOT),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if generation is not None and retrieval["nonempty_text"] and retrieval["provenance_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
