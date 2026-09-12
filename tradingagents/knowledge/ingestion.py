"""Explicit, local-only incremental knowledge ingestion and recovery.

The coordinator is deliberately the sole writer for one artifact root.  It
never modifies a source resource: discovery and the additional before/after
hash checks are read-only, while parsed/chunk/vector/index artifacts are
staged below ``artifact_root``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

from .catalog import CatalogAlias, KnowledgeCatalog
from .chunking import ChunkPolicy, ChunkTooLargeForEmbedding, StructureAwareChunker
from .config import KnowledgeConfig
from .discovery import SourceScanner
from .embeddings import EmbeddingProvider
from .identity import SourceChangedError, sha256_file
from .index_generation import IndexGenerationManager
from .models import AliasRelation, ChunkRecord, IngestionRunSummary, IngestionState, ParsedDocument
from .parser import DocumentParser
from .scanned import ScannedDetector


class IngestionMode(str, Enum):
    INCREMENTAL = "INCREMENTAL"
    REBUILD = "REBUILD"


class IngestionLockedError(RuntimeError):
    """Another ingestion writer owns this artifact root."""


@dataclass(frozen=True, slots=True)
class _PendingDocument:
    resource: object
    document: ParsedDocument
    chunks: tuple[ChunkRecord, ...]
    vectors: tuple[tuple[float, ...], ...]
    previous_alias: CatalogAlias | None


class KnowledgeIngestor:
    """Coordinate safe incremental/rebuild ingestion with per-resource isolation."""

    def __init__(
        self,
        config: KnowledgeConfig,
        *,
        scanner: SourceScanner | None = None,
        catalog: KnowledgeCatalog | None = None,
        parser: DocumentParser,
        scanned_detector: ScannedDetector | None = None,
        embedder: EmbeddingProvider,
        generation_manager: IndexGenerationManager | None = None,
    ) -> None:
        if not isinstance(config, KnowledgeConfig):
            raise TypeError("config must be a KnowledgeConfig")
        if embedder is None or not hasattr(embedder, "spec") or not hasattr(embedder, "tokenizer"):
            raise TypeError("embedder must expose the active spec and tokenizer")
        self.config = config
        self.scanner = scanner or SourceScanner(config)
        self.catalog = catalog or KnowledgeCatalog(config.artifact_root / "catalog.sqlite3")
        self.catalog.initialize()
        self.parser = parser
        self.scanned_detector = scanned_detector or ScannedDetector()
        self.embedder = embedder
        # This is intentionally derived from the *resolved provider*, not
        # from configurable character/word limits.  Chunking and inference
        # therefore share the exact tokenizer and no-truncation budget.
        self.chunker = StructureAwareChunker(ChunkPolicy.for_embedding(embedder.spec), embedder.tokenizer)
        self.generation_manager = generation_manager or IndexGenerationManager(
            config=config,
            catalog=self.catalog,
            embedding_spec=embedder.spec,
        )
        if self.generation_manager.catalog is not self.catalog:
            raise ValueError("generation_manager and ingestor must share one KnowledgeCatalog")
        self._fingerprints = {
            **dict(config.component_fingerprints),
            "embedding_spec": embedder.spec.to_json(),
        }
        self.recover_interrupted_runs()
        self._repair_active_pointer()

    def run(self, mode: IngestionMode | str = IngestionMode.INCREMENTAL) -> IngestionRunSummary:
        mode = IngestionMode(mode)
        with self._artifact_lock():
            self._recover_interrupted_runs_locked()
            resources = self.scanner.discover()
            run_id = uuid4().hex
            self.catalog.start_run(run_id, source_count=len(resources))
            counts = dict.fromkeys(IngestionState, 0)
            pending: dict[str, _PendingDocument] = {}
            pending_aliases: list[tuple[object, str, CatalogAlias | None]] = []
            seen_hashes: set[str] = set()
            changed = mode is IngestionMode.REBUILD
            observed_ids = {resource.resource_id for resource in resources}

            try:
                for resource in resources:
                    self.catalog.upsert_discovered(resource)
                    if resource.state is IngestionState.UNSUPPORTED:
                        self._event(run_id, resource, None, IngestionState.UNSUPPORTED, "DISCOVERY", counts)
                        continue
                    if resource.state is IngestionState.SOURCE_CHANGED or not resource.source_hash:
                        self._failure(run_id, resource, None, IngestionState.SOURCE_CHANGED, "HASH", None, counts)
                        continue

                    previous = self.catalog.get_alias(resource.resource_id)
                    document = self.catalog.find_document_by_source_hash(resource.source_hash)
                    repeated_hash = resource.source_hash in seen_hashes
                    seen_hashes.add(resource.source_hash)
                    has_chunks = bool(document and self.catalog.chunks_for_document(document.document_id))
                    compatible = bool(
                        document
                        and has_chunks
                        and self.catalog.document_is_compatible(document.document_id, self._fingerprints)
                    )
                    is_current = bool(
                        previous
                        and previous.relation is AliasRelation.CURRENT
                        and previous.source_hash == resource.source_hash
                        and previous.document_id == (document.document_id if document else None)
                    )
                    if mode is IngestionMode.INCREMENTAL and compatible and is_current:
                        state = IngestionState.DUPLICATE if repeated_hash else IngestionState.UNCHANGED
                        self.catalog.set_resource_state(resource.resource_id, state, source_hash=resource.source_hash)
                        self._event(run_id, resource, document.document_id, state, "REUSE", counts)
                        continue
                    if compatible and (
                        not is_current or repeated_hash or (document and document.document_id in pending)
                    ):
                        pending_aliases.append((resource, document.document_id, previous))
                        self._event(run_id, resource, document.document_id, IngestionState.DUPLICATE, "DEDUPLICATE", counts)
                        changed = True
                        continue
                    if resource.source_hash in pending:
                        pending_aliases.append((resource, pending[resource.source_hash].document.document_id, previous))
                        self._event(run_id, resource, None, IngestionState.DUPLICATE, "DEDUPLICATE", counts)
                        changed = True
                        continue

                    staged = self._stage_resource(run_id, resource, previous, counts)
                    if staged is not None:
                        pending[resource.source_hash] = staged
                        changed = True

                removed_resource_ids = self._reconcile_removed(run_id, observed_ids, counts)
                changed = changed or bool(removed_resource_ids)
                if not changed:
                    return self._finish(run_id, counts, source_count=len(resources))

                return self._publish(
                    run_id, pending, pending_aliases, removed_resource_ids, counts,
                    source_count=len(resources),
                )
            except InterruptedError:
                # A crash leaves RUNNING as the durable signal.  Recovery
                # cleans only this run's staging and cannot disturb a prior
                # matched, active generation.
                for summary in reversed(self.catalog.list_runs()):
                    if summary.run_id == run_id and summary.state != "RUNNING":
                        return summary
                return IngestionRunSummary(
                    run_id=run_id,
                    state="RUNNING",
                    counts=counts,
                    started_at=datetime.now(UTC).isoformat(),
                    source_count=len(resources),
                )

    def recover_interrupted_runs(self) -> tuple[str, ...]:
        with self._artifact_lock():
            return self._recover_interrupted_runs_locked()

    def _recover_interrupted_runs_locked(self) -> tuple[str, ...]:
        recovered: list[str] = []
        for run_id in self.catalog.running_run_ids():
            self._discard_unpublished_run(run_id)
            self.catalog.mark_run_interrupted(run_id)
            recovered.append(run_id)
        return tuple(recovered)

    def _stage_resource(self, run_id, resource, previous, counts) -> _PendingDocument | None:
        staging = self._resource_staging_path(run_id, resource.resource_id)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            # The scanner hash is not trusted after discovery: parser reads
            # occur only between two source identity checks.
            before_hash, _ = sha256_file(resource.path)
            if before_hash != resource.source_hash:
                raise SourceChangedError("source bytes differ from the discovery hash")
            document = self.parser.parse(resource, staging)
            after_hash, _ = sha256_file(resource.path)
            if after_hash != resource.source_hash or document.source_hash != resource.source_hash:
                raise SourceChangedError("source changed while parsing")
        except SourceChangedError as error:
            self._cleanup_resource_staging(run_id, resource.resource_id)
            self._failure(run_id, resource, previous, IngestionState.SOURCE_CHANGED, "SOURCE", error, counts)
            return None
        except Exception as error:
            self._cleanup_resource_staging(run_id, resource.resource_id)
            self._failure(run_id, resource, previous, IngestionState.PARSE_FAILED, "PARSE", error, counts)
            return None
        try:
            decision = self.scanned_detector.classify(document)
            if decision.state is IngestionState.NEEDS_OCR:
                self._cleanup_resource_staging(run_id, resource.resource_id)
                self._failure(run_id, resource, previous, IngestionState.NEEDS_OCR, "SCAN", None, counts)
                return None
            chunks = tuple(self.chunker.chunk(document))
        except ChunkTooLargeForEmbedding as error:
            self._cleanup_resource_staging(run_id, resource.resource_id)
            self._failure(run_id, resource, previous, IngestionState.INDEX_FAILED, "CHUNK", error, counts)
            return None
        except Exception as error:
            self._cleanup_resource_staging(run_id, resource.resource_id)
            self._failure(run_id, resource, previous, IngestionState.INDEX_FAILED, "CHUNK", error, counts)
            return None
        try:
            vectors = self._vectors_for(document.document_id, chunks)
        except Exception as error:
            self._cleanup_resource_staging(run_id, resource.resource_id)
            self._failure(run_id, resource, previous, IngestionState.EMBED_FAILED, "EMBED", error, counts)
            return None
        return _PendingDocument(resource, document, chunks, vectors, previous)

    def _publish(self, run_id, pending, pending_aliases, removed_resource_ids, counts, *, source_count) -> IngestionRunSummary:
        current_ids = set(self.catalog.current_document_ids())
        pending_by_id = {item.document.document_id: item for item in pending.values()}
        replacement_aliases = [item.previous_alias for item in pending.values()] + [
            previous for _, _, previous in pending_aliases
        ]
        for previous_alias in replacement_aliases:
            if (
                previous_alias
                and previous_alias.relation is AliasRelation.CURRENT
                and self.catalog.current_alias_count(previous_alias.document_id) == 1
            ):
                current_ids.discard(previous_alias.document_id)
        for resource_id in removed_resource_ids:
            previous = self.catalog.get_alias(resource_id)
            if (
                previous
                and previous.relation is AliasRelation.CURRENT
                and self.catalog.current_alias_count(previous.document_id) == 1
            ):
                current_ids.discard(previous.document_id)
        current_ids.update(pending_by_id)
        current_ids.update(document_id for _, document_id, _ in pending_aliases)
        chunks: list[ChunkRecord] = []
        vectors: list[tuple[float, ...]] = []
        for document_id in sorted(current_ids):
            pending_item = pending_by_id.get(document_id)
            if pending_item is not None:
                chunks.extend(pending_item.chunks)
                vectors.extend(pending_item.vectors)
            else:
                old_chunks = self.catalog.chunks_for_document(document_id)
                chunks.extend(old_chunks)
                vectors.extend(self._vectors_for(document_id, old_chunks))
        publication_counts = dict(counts)
        publication_counts[IngestionState.INDEXED] += len(pending)
        summary = self._summary(
            run_id, publication_counts, source_count=source_count,
            document_count=len(current_ids), chunk_count=len(chunks),
        )
        alias_updates = tuple(
            [(item.resource.resource_id, item.document.document_id, item.document.source_hash, IngestionState.INDEXED)
             for item in pending.values()]
            + [(resource.resource_id, document_id, self._source_hash_for(document_id, pending_by_id), IngestionState.DUPLICATE)
               for resource, document_id, _ in pending_aliases]
        )
        events = tuple(
            [(item.resource.resource_id, item.document.document_id, item.document.source_hash, IngestionState.INDEXED, "INDEX")
             for item in pending.values()]
            + [(resource_id, None, None, IngestionState.REMOVED, "REMOVAL") for resource_id in removed_resource_ids]
        )
        try:
            generation = None
            if current_ids:
                generation = self.generation_manager.build_generation(chunks, vectors, f"gen_{run_id}")
            self.catalog.publish_generation(
                generation,
                documents_by_id={item.document.document_id: item.document for item in pending.values()},
                component_fingerprints=self._fingerprints,
                chunks_by_document={item.document.document_id: item.chunks for item in pending.values()},
                aliases=alias_updates,
                removed_resource_ids=tuple(removed_resource_ids),
                ready_document_ids=tuple(sorted(current_ids)),
                summary=summary,
                events=events,
            )
        except InterruptedError:
            raise
        except Exception as error:
            for item in pending.values():
                self._failure(run_id, item.resource, item.previous_alias, IngestionState.INDEX_FAILED, "INDEX", error, counts)
            for resource, _, previous in pending_aliases:
                self._failure(run_id, resource, previous, IngestionState.INDEX_FAILED, "INDEX", error, counts)
            self._discard_unpublished_run(run_id)
            return self._finish(run_id, counts, source_count=source_count)
        if generation is not None:
            self.generation_manager.resolve_active_generation()
        counts.clear()
        counts.update(publication_counts)
        shutil.rmtree(self.config.artifact_root / ".ingestion-staging" / run_id, ignore_errors=True)
        return summary

    def _document_for_id(self, document_id: str):
        document = self.catalog.get_document(document_id)
        if document is None:
            raise ValueError(f"unknown duplicate document: {document_id}")
        return document

    def _source_hash_for(self, document_id: str, pending_by_id: Mapping[str, _PendingDocument]) -> str:
        pending = pending_by_id.get(document_id)
        if pending is not None:
            return pending.document.source_hash
        return self._document_for_id(document_id).source_hash

    def _reconcile_removed(self, run_id: str, observed_ids: set[str], counts: dict[IngestionState, int]) -> tuple[str, ...]:
        removed: list[str] = []
        for resource in self.catalog.list_resources():
            if resource.resource_id not in observed_ids and resource.state is not IngestionState.REMOVED:
                counts[IngestionState.REMOVED] += 1
                removed.append(resource.resource_id)
        return tuple(removed)

    def _failure(self, run_id, resource, previous, state, stage, error, counts) -> None:
        if previous and previous.document_id and previous.source_hash != resource.source_hash:
            self.catalog.retain_previous(
                resource.resource_id,
                attempted_hash=resource.source_hash or "",
                previous_document_id=previous.document_id,
            )
        else:
            self.catalog.set_resource_state(
                resource.resource_id, state, source_hash=resource.source_hash, attempted_hash=resource.source_hash
            )
        self.catalog.record_event(
            resource_id=resource.resource_id,
            document_id=previous.document_id if previous else None,
            attempted_hash=resource.source_hash,
            state=state,
            stage=stage,
            error=error,
            run_id=run_id,
        )
        counts[state] += 1

    def _cleanup_resource_staging(self, run_id: str, resource_id: str) -> None:
        shutil.rmtree(self._resource_staging_path(run_id, resource_id), ignore_errors=True)

    def _resource_staging_path(self, run_id: str, resource_id: str) -> Path:
        resource_digest = hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:16]
        return self.config.artifact_root / ".ingestion-staging" / run_id / f"res_{resource_digest}"

    def _discard_unpublished_run(self, run_id: str) -> None:
        """Discard only unpublished run-owned state; never remove an active pair."""

        active = self.catalog.active_generation()
        generation_id = f"gen_{run_id}"
        shutil.rmtree(self.config.artifact_root / ".ingestion-staging" / run_id, ignore_errors=True)
        if active is not None and active.generation_id == generation_id:
            # A legacy writer may have crashed after activation.  The catalog
            # is already authoritative for that pair, so recovery must retain
            # its files and rows rather than creating an active dangling ID.
            return
        self.catalog.discard_staged_run(run_id)
        shutil.rmtree(self.config.artifact_root / "vector" / "lancedb" / generation_id, ignore_errors=True)
        shutil.rmtree(self.config.artifact_root / "keyword" / generation_id, ignore_errors=True)

    def _repair_active_pointer(self) -> None:
        if self.catalog.active_generation() is not None:
            self.generation_manager.resolve_active_generation()

    def _event(self, run_id, resource, document_id, state, stage, counts) -> None:
        self.catalog.record_event(
            resource_id=resource.resource_id,
            document_id=document_id,
            attempted_hash=resource.source_hash,
            state=state,
            stage=stage,
            run_id=run_id,
        )
        counts[state] += 1

    def _vectors_for(self, document_id: str, chunks: Sequence[ChunkRecord]) -> tuple[tuple[float, ...], ...]:
        if not chunks:
            return ()
        path = self._vector_cache_path(document_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["embedding_spec"] != self.embedder.spec.to_dict() or payload["chunk_ids"] != [chunk.chunk_id for chunk in chunks]:
                raise ValueError("cached vectors do not match the active document/chunker/specification")
            vectors = tuple(tuple(float(value) for value in vector) for vector in payload["vectors"])
            if len(vectors) != len(chunks) or any(len(vector) != self.embedder.spec.dimensions for vector in vectors):
                raise ValueError("cached vectors are invalid")
            return vectors
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            vectors = tuple(tuple(float(value) for value in vector) for vector in self.embedder.embed(
                tuple(chunk.text for chunk in chunks), purpose="corpus"
            ))
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
            return vectors

    def _vector_cache_path(self, document_id: str) -> Path:
        return self.config.artifact_root / "embeddings" / "ingestion-v1" / f"{document_id}.json"

    def _finish(
        self,
        run_id: str,
        counts: Mapping[IngestionState, int],
        *,
        source_count: int,
    ) -> IngestionRunSummary:
        summary = self._summary(
            run_id, counts, source_count=source_count,
            document_count=len(self.catalog.current_document_ids()),
            chunk_count=sum(len(self.catalog.chunks_for_document(doc)) for doc in self.catalog.current_document_ids()),
        )
        self.catalog.record_run(summary)
        return summary

    @staticmethod
    def _summary(
        run_id: str,
        counts: Mapping[IngestionState, int],
        *,
        source_count: int,
        document_count: int,
        chunk_count: int,
    ) -> IngestionRunSummary:
        failures = sum(
            counts.get(state, 0)
            for state in (IngestionState.PARSE_FAILED, IngestionState.NEEDS_OCR, IngestionState.EMBED_FAILED, IngestionState.INDEX_FAILED, IngestionState.SOURCE_CHANGED)
        )
        successful = sum(counts.get(state, 0) for state in (IngestionState.INDEXED, IngestionState.UNCHANGED, IngestionState.DUPLICATE))
        state = "SUCCEEDED" if failures == 0 else ("PARTIAL_FAILURE" if successful else "FAILED")
        summary = IngestionRunSummary(
            run_id=run_id,
            state=state,
            counts=counts,
            started_at=datetime.now(UTC).isoformat(),
            completed_at=datetime.now(UTC).isoformat(),
            source_count=source_count,
            document_count=document_count,
            chunk_count=chunk_count,
        )
        return summary

    @contextmanager
    def _artifact_lock(self) -> Iterator[None]:
        path = self.config.artifact_root / "locks" / "index.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR)
        try:
            if os.path.getsize(path) == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - Windows is the supported deployment target.
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(descriptor)
            raise IngestionLockedError(
                f"knowledge ingestion already owns artifact root: {self.config.artifact_root}"
            ) from error
        try:
            owner = json.dumps({"pid": os.getpid()}, sort_keys=True).encode("ascii")
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, owner)
            os.fsync(descriptor)
            yield
        finally:
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - Windows is the supported deployment target.
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


__all__ = ["IngestionLockedError", "IngestionMode", "KnowledgeIngestor"]
