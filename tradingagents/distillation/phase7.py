"""Read-only view over a completed Phase 7 knowledge catalog."""
from __future__ import annotations
import hashlib
from pathlib import Path
from .errors import SourceGenerationInvalidError, SourceUnavailableError
from .models import SourceBlock, SourceRef, canonical_hash
from tradingagents.knowledge.catalog import KnowledgeCatalog

class Phase7KnowledgeSource:
    def __init__(self, root: Path, catalog: KnowledgeCatalog, generation, docs):
        self.root, self._catalog, self._generation, self._docs = root, catalog, generation, docs
        self.generation_id = generation.generation_id
        self.source_fingerprints = {
            "catalog": _file_hash(root / "catalog.sqlite3"),
            "generation": canonical_hash(generation),
        }
    @classmethod
    def open(cls, root: str | Path, expected_generation_id: str | None = None) -> "Phase7KnowledgeSource":
        root = Path(root)
        db = root / "catalog.sqlite3"
        if not db.is_file(): raise SourceUnavailableError(f"missing Phase 7 catalog: {db}")
        catalog = KnowledgeCatalog(db)
        generation = catalog.active_generation()
        if generation is None: raise SourceGenerationInvalidError("Phase 7 has no active generation")
        if expected_generation_id is not None and generation.generation_id != expected_generation_id:
            raise SourceGenerationInvalidError("Phase 7 generation does not match expectation")
        if generation.status != "VALIDATED" or not generation.vector_ready or not generation.lexical_ready:
            raise SourceGenerationInvalidError("Phase 7 active generation is incomplete")
        docs=[]
        for doc_id in catalog.current_document_ids():
            doc = catalog.get_document(doc_id)
            if doc is None or not doc.active: continue
            chunks = catalog.chunks_for_document(doc_id)
            if chunks: docs.append((doc,chunks))
        return cls(root,catalog,generation,tuple(docs))
    def documents(self): return tuple(doc for doc,_ in self._docs)
    def blocks(self):
        for doc,chunks in self._docs:
            for chunk in chunks:
                yield self.block_from_chunk(chunk, self.generation_id)
    @staticmethod
    def block_from_chunk(chunk, generation_id: str) -> SourceBlock:
        ref=SourceRef(document_id=chunk.document_id, source_filename=chunk.source_filename or "", source_hash=chunk.source_hash, chunk_id=chunk.chunk_id, generation_id=generation_id, page=chunk.page, page_start=chunk.page_start, page_end=chunk.page_end, chapter=chunk.chapter, section=" / ".join(chunk.section_path) or None, content_type=str(chunk.content_type.value if hasattr(chunk.content_type,"value") else chunk.content_type))
        metadata={"title":chunk.title,"authors":chunk.authors,"table_metadata":chunk.table_metadata,"equation_metadata":chunk.equation_metadata,"source_relative_path":chunk.source_relative_path}
        return SourceBlock(ref=ref,text=chunk.text,content_type=ref.content_type,reading_order=chunk.reading_order or chunk.chunk_ordinal,section_path=chunk.section_path,metadata={k:v for k,v in metadata.items() if v is not None})

def _file_hash(path: Path) -> str:
    h=hashlib.sha256()
    try:
        with path.open("rb") as f:
            for part in iter(lambda:f.read(1024*1024),b""): h.update(part)
    except OSError as exc: raise SourceUnavailableError(f"cannot read Phase 7 catalog: {path}") from exc
    return h.hexdigest()

