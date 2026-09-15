"""Read-only view over a completed Phase 7 knowledge catalog."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tradingagents.knowledge.catalog import KnowledgeCatalog

from .errors import SourceGenerationInvalidError, SourceUnavailableError
from .models import SourceBlock, SourceRef, canonical_hash

_CONTENT_TYPES = {"PROSE", "EQUATION", "TABLE", "FIGURE_CAPTION", "DEFINITION", "REFERENCE", "LIST"}


class Phase7KnowledgeSource:
    def __init__(self, root: Path, catalog: KnowledgeCatalog, generation, docs, quarantined=()):
        self.root, self._catalog, self._generation, self._docs = root, catalog, generation, docs
        self.quarantined_chunks = tuple(quarantined)
        self.generation_id = generation.generation_id
        self.source_fingerprints = {
            "catalog": _file_hash(root / "catalog.sqlite3"),
            "generation": canonical_hash(generation),
        }

    @classmethod
    def open(
        cls, root: str | Path, expected_generation_id: str | None = None
    ) -> Phase7KnowledgeSource:
        root = Path(root)
        db = root / "catalog.sqlite3"
        if not db.is_file():
            raise SourceUnavailableError(f"missing Phase 7 catalog: {db}")
        catalog = KnowledgeCatalog(db)
        generation = catalog.active_generation()
        if generation is None:
            raise SourceGenerationInvalidError("Phase 7 has no active generation")
        if (
            expected_generation_id is not None
            and generation.generation_id != expected_generation_id
        ):
            raise SourceGenerationInvalidError("Phase 7 generation does not match expectation")
        if (
            generation.status != "VALIDATED"
            or not generation.vector_ready
            or not generation.lexical_ready
        ):
            raise SourceGenerationInvalidError("Phase 7 active generation is incomplete")
        docs = []
        quarantined = []
        for doc_id in catalog.current_document_ids():
            doc = catalog.get_document(doc_id)
            if doc is None:
                raise SourceGenerationInvalidError(f"current Phase 7 document is missing: {doc_id}")
            if not doc.active:
                raise SourceGenerationInvalidError(
                    f"current Phase 7 document is inactive: {doc_id}"
                )
            chunks = catalog.chunks_for_document(doc_id)
            if not chunks:
                raise SourceGenerationInvalidError(
                    f"current Phase 7 document has no chunks: {doc_id}"
                )
            valid_chunks = []
            for chunk in chunks:
                if not str(getattr(chunk, "text", "") or "").strip():
                    quarantined.append({"document_id": doc_id, "chunk_id": chunk.chunk_id, "code": "EMPTY_CHUNK_TEXT"})
                    continue
                _validate_chunk(chunk, doc.source_hash, generation.generation_id)
                valid_chunks.append(chunk)
            if not valid_chunks:
                raise SourceGenerationInvalidError(f"current Phase 7 document has no valid text chunks: {doc_id}")
            docs.append((doc, tuple(valid_chunks)))
        return cls(root, catalog, generation, tuple(docs), quarantined)

    def documents(self):
        return tuple(doc for doc, _ in self._docs)

    def blocks(self):
        for _doc, chunks in self._docs:
            for chunk in chunks:
                yield self.block_from_chunk(chunk, self.generation_id)

    @staticmethod
    def block_from_chunk(chunk, generation_id: str) -> SourceBlock:
        _validate_chunk(chunk, chunk.source_hash, generation_id)
        ref = SourceRef(
            document_id=chunk.document_id,
            source_filename=chunk.source_filename or "",
            source_hash=chunk.source_hash,
            chunk_id=chunk.chunk_id,
            generation_id=generation_id,
            page=chunk.page,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chapter=chunk.chapter,
            section=" / ".join(chunk.section_path) or None,
            content_type=str(
                chunk.content_type.value
                if hasattr(chunk.content_type, "value")
                else chunk.content_type
            ),
        )
        metadata = {
            "title": chunk.title,
            "authors": chunk.authors,
            "table_metadata": chunk.table_metadata,
            "equation_metadata": chunk.equation_metadata,
            "source_relative_path": chunk.source_relative_path,
        }
        return SourceBlock(
            ref=ref,
            text=chunk.text,
            content_type=ref.content_type,
            reading_order=chunk.reading_order or chunk.chunk_ordinal,
            section_path=chunk.section_path,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for part in iter(lambda: f.read(1024 * 1024), b""):
                h.update(part)
    except OSError as exc:
        raise SourceUnavailableError(f"cannot read Phase 7 catalog: {path}") from exc
    return h.hexdigest()


def _validate_chunk(chunk, document_source_hash: str, generation_id: str) -> None:
    """Validate provenance before exposing any Phase 7 text downstream."""
    required = ("document_id", "chunk_id", "source_hash", "text")
    if any(not str(getattr(chunk, name, "")).strip() for name in required):
        raise SourceGenerationInvalidError("Phase 7 chunk has incomplete provenance")
    if not str(getattr(chunk, "source_filename", "") or "").strip():
        raise SourceGenerationInvalidError(f"chunk has no source filename: {chunk.chunk_id}")
    if chunk.source_hash != document_source_hash:
        raise SourceGenerationInvalidError(f"chunk source hash mismatch: {chunk.chunk_id}")
    content_type = getattr(chunk.content_type, "value", chunk.content_type)
    if str(content_type).upper() not in _CONTENT_TYPES:
        raise SourceGenerationInvalidError(f"unsupported chunk content type: {content_type}")
    if (
        getattr(chunk, "page_start", None) is not None
        and getattr(chunk, "page_end", None) is not None
        and chunk.page_end < chunk.page_start
    ):
        raise SourceGenerationInvalidError(f"invalid chunk page range: {chunk.chunk_id}")
    projection_generation = getattr(chunk, "projection_generation", None)
    if projection_generation not in (None, "", generation_id):
        raise SourceGenerationInvalidError(f"chunk generation mismatch: {chunk.chunk_id}")
