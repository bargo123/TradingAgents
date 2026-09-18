"""Transactional SQLite catalog for local, read-only knowledge artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .diagnostics import QuarantineRecord, bounded_error_metadata
from .models import (
    AliasRelation,
    ChunkRecord,
    DiscoveredResource,
    IndexGeneration,
    IngestionRunSummary,
    IngestionState,
    ParsedDocument,
)

_BUSY_TIMEOUT_MS = 5_000


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_mapping(value: str | None) -> dict[str, Any]:
    return dict(json.loads(value or "{}"))


@dataclass(frozen=True, slots=True)
class CatalogAlias:
    resource_id: str
    document_id: str | None
    source_hash: str | None
    relation: AliasRelation
    attempted_hash: str | None
    updated_at: str


@dataclass(frozen=True, slots=True)
class CatalogDocument:
    document_id: str
    source_hash: str
    metadata: Mapping[str, Any]
    parser_id: str
    parser_version: str
    parser_config_hash: str
    state: IngestionState
    active: bool
    vector_ready: bool
    lexical_ready: bool
    projection_generation: str | None
    projection_population_hash: str | None
    component_fingerprints: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogResource:
    resource_id: str
    relative_path: str | None
    display_path: str | None
    source_hash: str | None
    state: IngestionState
    attempted_hash: str | None


class KnowledgeCatalog:
    """A short-lived-connection SQLite boundary for knowledge metadata.

    This class stores catalog authority only.  It does not parse sources,
    build projections, query indexes, or mutate any source-root file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        return connection

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        """Yield a read connection and deterministically release its handle."""

        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the catalog schema idempotently."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_resources (
                    resource_id TEXT PRIMARY KEY,
                    relative_path TEXT,
                    display_path TEXT,
                    format TEXT,
                    source_hash TEXT,
                    size_bytes INTEGER,
                    modified_ns INTEGER,
                    state TEXT NOT NULL,
                    attempted_hash TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id TEXT PRIMARY KEY,
                    source_hash TEXT NOT NULL UNIQUE,
                    metadata_json TEXT NOT NULL,
                    parser_id TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    parser_config_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0 CHECK(active IN (0, 1)),
                    vector_ready INTEGER NOT NULL DEFAULT 0 CHECK(vector_ready IN (0, 1)),
                    lexical_ready INTEGER NOT NULL DEFAULT 0 CHECK(lexical_ready IN (0, 1)),
                    projection_generation TEXT,
                    projection_population_hash TEXT,
                    component_fingerprints_json TEXT NOT NULL DEFAULT '{}',
                    ingestion_run_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_aliases (
                    resource_id TEXT PRIMARY KEY REFERENCES knowledge_resources(resource_id),
                    document_id TEXT REFERENCES knowledge_documents(document_id),
                    source_hash TEXT,
                    relation_status TEXT NOT NULL,
                    attempted_hash TEXT,
                    changed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS knowledge_aliases_document_current_idx
                    ON knowledge_aliases(document_id, relation_status);
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES knowledge_documents(document_id),
                    source_hash TEXT NOT NULL,
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    active INTEGER NOT NULL DEFAULT 0 CHECK(active IN (0, 1)),
                    vector_ready INTEGER NOT NULL DEFAULT 0 CHECK(vector_ready IN (0, 1)),
                    lexical_ready INTEGER NOT NULL DEFAULT 0 CHECK(lexical_ready IN (0, 1)),
                    projection_generation TEXT,
                    projection_population_hash TEXT
                    ,ingestion_run_id TEXT
                );
                CREATE TABLE IF NOT EXISTS knowledge_ingestion_runs (
                    run_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    counts_json TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    source_count INTEGER,
                    document_count INTEGER,
                    chunk_count INTEGER,
                    diagnostics_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS knowledge_ingestion_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT REFERENCES knowledge_ingestion_runs(run_id),
                    resource_id TEXT REFERENCES knowledge_resources(resource_id),
                    document_id TEXT REFERENCES knowledge_documents(document_id),
                    attempted_hash TEXT,
                    state TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    error_fingerprint TEXT,
                    counters_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS knowledge_ingestion_events_quarantine_idx
                    ON knowledge_ingestion_events(state, event_id);
                CREATE TABLE IF NOT EXISTS knowledge_index_generations (
                    generation_id TEXT PRIMARY KEY,
                    vector_location TEXT NOT NULL,
                    lexical_location TEXT NOT NULL,
                    embedding_spec_json TEXT NOT NULL,
                    lexical_index_version TEXT NOT NULL,
                    lexical_tokenizer_settings_json TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    population_hash TEXT NOT NULL,
                    population_identity TEXT,
                    document_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    vector_ready INTEGER NOT NULL CHECK(vector_ready IN (0, 1)),
                    lexical_ready INTEGER NOT NULL CHECK(lexical_ready IN (0, 1)),
                    status TEXT NOT NULL,
                    created_at TEXT,
                    activated_at TEXT,
                    component_versions_json TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0 CHECK(is_active IN (0, 1))
                );
                CREATE UNIQUE INDEX IF NOT EXISTS knowledge_index_generations_one_active_idx
                    ON knowledge_index_generations(is_active) WHERE is_active = 1;
                """
            )
            # SQLite's CREATE IF NOT EXISTS does not add columns to catalogs
            # created by earlier Phase 7 tasks.  These additive migrations
            # keep existing local metadata usable without rewriting sources.
            self._add_column_if_missing(connection, "knowledge_documents", "component_fingerprints_json TEXT NOT NULL DEFAULT '{}'")
            self._add_column_if_missing(connection, "knowledge_documents", "ingestion_run_id TEXT")
            self._add_column_if_missing(connection, "knowledge_chunks", "ingestion_run_id TEXT")

    @staticmethod
    def _add_column_if_missing(connection: sqlite3.Connection, table: str, definition: str) -> None:
        name = definition.split()[0]
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if name not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    @staticmethod
    def _row_document(row: sqlite3.Row) -> CatalogDocument:
        return CatalogDocument(
            document_id=row["document_id"],
            source_hash=row["source_hash"],
            metadata=_json_mapping(row["metadata_json"]),
            parser_id=row["parser_id"],
            parser_version=row["parser_version"],
            parser_config_hash=row["parser_config_hash"],
            state=IngestionState(row["state"]),
            active=bool(row["active"]),
            vector_ready=bool(row["vector_ready"]),
            lexical_ready=bool(row["lexical_ready"]),
            projection_generation=row["projection_generation"],
            projection_population_hash=row["projection_population_hash"],
            component_fingerprints=_json_mapping(row["component_fingerprints_json"]),
        )

    @staticmethod
    def _row_alias(row: sqlite3.Row) -> CatalogAlias:
        return CatalogAlias(
            resource_id=row["resource_id"],
            document_id=row["document_id"],
            source_hash=row["source_hash"],
            relation=AliasRelation(row["relation_status"]),
            attempted_hash=row["attempted_hash"],
            updated_at=row["changed_at"],
        )

    @staticmethod
    def _row_generation(row: sqlite3.Row) -> IndexGeneration:
        return IndexGeneration.from_dict(
            {
                "generation_id": row["generation_id"],
                "vector_location": row["vector_location"],
                "lexical_location": row["lexical_location"],
                "embedding_spec": json.loads(row["embedding_spec_json"]),
                "lexical_index_version": row["lexical_index_version"],
                "lexical_tokenizer_settings": json.loads(row["lexical_tokenizer_settings_json"]),
                "index_version": row["index_version"],
                "population_hash": row["population_hash"],
                "population_identity": row["population_identity"],
                "document_count": row["document_count"],
                "chunk_count": row["chunk_count"],
                "vector_ready": bool(row["vector_ready"]),
                "lexical_ready": bool(row["lexical_ready"]),
                "status": row["status"],
                "created_at": row["created_at"],
                "activated_at": row["activated_at"],
                "component_versions": json.loads(row["component_versions_json"]),
            }
        )

    def _ensure_resource(self, connection: sqlite3.Connection, resource_id: str) -> None:
        timestamp = _now()
        connection.execute(
            """INSERT OR IGNORE INTO knowledge_resources
               (resource_id, state, created_at, updated_at)
               VALUES (?, ?, ?, ?)""",
            (resource_id, IngestionState.DISCOVERED.value, timestamp, timestamp),
        )

    def _refresh_document_currentness(self, connection: sqlite3.Connection, *document_ids: str | None) -> None:
        for document_id in {value for value in document_ids if value}:
            connection.execute(
                """UPDATE knowledge_documents AS document
                   SET active = CASE WHEN EXISTS (
                       SELECT 1 FROM knowledge_aliases AS alias
                       WHERE alias.document_id = document.document_id
                       AND alias.relation_status = ?
                   ) THEN 1 ELSE 0 END,
                   updated_at = ?
                   WHERE document_id = ?""",
                (AliasRelation.CURRENT.value, _now(), document_id),
            )

    def upsert_discovered(self, resource: DiscoveredResource) -> None:
        with self._write() as connection:
            timestamp = _now()
            connection.execute(
                """INSERT INTO knowledge_resources
                   (resource_id, relative_path, display_path, format, source_hash, size_bytes,
                    modified_ns, state, attempted_hash, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(resource_id) DO UPDATE SET
                     relative_path=excluded.relative_path, display_path=excluded.display_path,
                     format=excluded.format, source_hash=excluded.source_hash,
                     size_bytes=excluded.size_bytes, modified_ns=excluded.modified_ns,
                     state=excluded.state, attempted_hash=excluded.attempted_hash,
                     updated_at=excluded.updated_at""",
                (
                    resource.resource_id, resource.relative_path, resource.display_path,
                    resource.format, resource.source_hash, resource.size_bytes, resource.modified_ns,
                    resource.state.value, resource.source_hash, timestamp, timestamp,
                ),
            )

    def register_document(
        self,
        document: ParsedDocument,
        *,
        component_fingerprints: Mapping[str, str] | None = None,
        run_id: str | None = None,
    ) -> None:
        with self._write() as connection:
            timestamp = _now()
            connection.execute(
                """INSERT INTO knowledge_documents
                   (document_id, source_hash, metadata_json, parser_id, parser_version,
                    parser_config_hash, state, component_fingerprints_json, ingestion_run_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(document_id) DO UPDATE SET
                     source_hash=excluded.source_hash, metadata_json=excluded.metadata_json,
                     parser_id=excluded.parser_id, parser_version=excluded.parser_version,
                     parser_config_hash=excluded.parser_config_hash, state=excluded.state,
                     component_fingerprints_json=excluded.component_fingerprints_json,
                     ingestion_run_id=excluded.ingestion_run_id,
                     updated_at=excluded.updated_at""",
                (
                    document.document_id, document.source_hash, _json(document.metadata.to_dict()),
                    document.parser_id, document.parser_version, document.parser_config_hash,
                    IngestionState.PARSED.value, _json(dict(component_fingerprints or {})), run_id,
                    timestamp, timestamp,
                ),
            )

    def find_document_by_source_hash(self, source_hash: str) -> CatalogDocument | None:
        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_documents WHERE source_hash = ?", (source_hash,)
            ).fetchone()
        return None if row is None else self._row_document(row)

    def document_is_compatible(self, document_id: str, component_fingerprints: Mapping[str, str]) -> bool:
        document = self.get_document(document_id)
        return document is not None and dict(document.component_fingerprints) == dict(component_fingerprints)

    def set_resource_state(
        self,
        resource_id: str,
        state: IngestionState,
        *,
        source_hash: str | None = None,
        attempted_hash: str | None = None,
    ) -> None:
        with self._write() as connection:
            self._ensure_resource(connection, resource_id)
            connection.execute(
                """UPDATE knowledge_resources SET state = ?,
                   source_hash = COALESCE(?, source_hash), attempted_hash = COALESCE(?, attempted_hash),
                   updated_at = ? WHERE resource_id = ?""",
                (state.value, source_hash, attempted_hash, _now(), resource_id),
            )

    def current_document_ids(self) -> tuple[str, ...]:
        with self._read() as connection:
            rows = connection.execute(
                """SELECT DISTINCT document_id FROM knowledge_aliases
                   WHERE relation_status = ? AND document_id IS NOT NULL ORDER BY document_id""",
                (AliasRelation.CURRENT.value,),
            ).fetchall()
        return tuple(str(row["document_id"]) for row in rows)

    def store_chunks(
        self, document_id: str, chunks: tuple[ChunkRecord, ...], *, run_id: str | None = None
    ) -> None:
        with self._write() as connection:
            if connection.execute("SELECT 1 FROM knowledge_documents WHERE document_id = ?", (document_id,)).fetchone() is None:
                raise ValueError(f"unknown document_id: {document_id}")
            connection.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (document_id,))
            connection.executemany(
                """INSERT INTO knowledge_chunks
                   (chunk_id, document_id, source_hash, provenance_json, ingestion_run_id)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (chunk.chunk_id, document_id, chunk.source_hash, _json(chunk.to_dict()), run_id)
                    for chunk in chunks
                ],
            )

    def chunks_for_document(self, document_id: str) -> tuple[ChunkRecord, ...]:
        with self._read() as connection:
            rows = connection.execute(
                "SELECT provenance_json FROM knowledge_chunks WHERE document_id = ? ORDER BY chunk_id",
                (document_id,),
            ).fetchall()
        # Chunk caches and projection writers use the deterministic semantic
        # order represented by ``chunk_ordinal``.  The SQLite row key is the
        # hash-derived chunk id, so lexical SQL ordering is not the cache
        # identity order and would force safe cache misses on a later
        # incremental publication.
        chunks = [ChunkRecord.from_dict(json.loads(row["provenance_json"])) for row in rows]
        chunks.sort(key=lambda chunk: (chunk.chunk_ordinal, chunk.chunk_id))
        return tuple(chunks)

    def discard_staged_run(self, run_id: str) -> None:
        """Remove unaliased document/chunk rows created by an interrupted run."""

        with self._write() as connection:
            staged_document_ids = tuple(
                str(row["document_id"])
                for row in connection.execute(
                    """SELECT document_id FROM knowledge_documents
                       WHERE ingestion_run_id = ?
                       AND NOT EXISTS (
                           SELECT 1 FROM knowledge_aliases
                           WHERE knowledge_aliases.document_id = knowledge_documents.document_id
                       )""",
                    (run_id,),
                ).fetchall()
            )
            if staged_document_ids:
                placeholders = ",".join("?" for _ in staged_document_ids)
                connection.execute(
                    f"""UPDATE knowledge_ingestion_events SET document_id = NULL
                       WHERE document_id IN ({placeholders})""",
                    staged_document_ids,
                )
            connection.execute("DELETE FROM knowledge_chunks WHERE ingestion_run_id = ?", (run_id,))
            connection.execute(
                """DELETE FROM knowledge_documents WHERE ingestion_run_id = ?
                   AND NOT EXISTS (SELECT 1 FROM knowledge_aliases
                                   WHERE knowledge_aliases.document_id = knowledge_documents.document_id)""",
                (run_id,),
            )

    def get_document(self, document_id: str) -> CatalogDocument | None:
        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return None if row is None else self._row_document(row)

    def get_alias(self, resource_id: str) -> CatalogAlias | None:
        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_aliases WHERE resource_id = ?", (resource_id,)
            ).fetchone()
        return None if row is None else self._row_alias(row)

    def commit_alias(
        self,
        resource_id: str,
        document_id: str,
        source_hash: str,
        relation: AliasRelation = AliasRelation.CURRENT,
    ) -> None:
        """Atomically replace one resource alias without touching its peers."""

        relation = AliasRelation(relation)
        with self._write() as connection:
            self._ensure_resource(connection, resource_id)
            old = connection.execute(
                "SELECT document_id FROM knowledge_aliases WHERE resource_id = ?", (resource_id,)
            ).fetchone()
            if connection.execute(
                "SELECT 1 FROM knowledge_documents WHERE document_id = ?", (document_id,)
            ).fetchone() is None:
                raise ValueError(f"unknown document_id: {document_id}")
            timestamp = _now()
            connection.execute(
                """INSERT INTO knowledge_aliases
                   (resource_id, document_id, source_hash, relation_status, attempted_hash, changed_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(resource_id) DO UPDATE SET
                     document_id=excluded.document_id, source_hash=excluded.source_hash,
                     relation_status=excluded.relation_status, attempted_hash=excluded.attempted_hash,
                     changed_at=excluded.changed_at""",
                (resource_id, document_id, source_hash, relation.value, source_hash, timestamp),
            )
            resource_state = (
                IngestionState.INDEXED if relation is AliasRelation.CURRENT
                else IngestionState.RETAINED_PREVIOUS if relation is AliasRelation.RETAINED_PREVIOUS
                else IngestionState.REMOVED
            )
            connection.execute(
                """UPDATE knowledge_resources SET source_hash = ?, attempted_hash = ?, state = ?,
                   updated_at = ? WHERE resource_id = ?""",
                (source_hash, source_hash, resource_state.value, timestamp, resource_id),
            )
            self._refresh_document_currentness(
                connection, old["document_id"] if old else None, document_id
            )

    def retain_previous(self, resource_id: str, *, attempted_hash: str, previous_document_id: str) -> None:
        """Keep the prior document for diagnostics without counting it as current."""

        with self._write() as connection:
            self._ensure_resource(connection, resource_id)
            if connection.execute(
                "SELECT 1 FROM knowledge_documents WHERE document_id = ?", (previous_document_id,)
            ).fetchone() is None:
                raise ValueError(f"unknown previous_document_id: {previous_document_id}")
            old = connection.execute(
                "SELECT document_id FROM knowledge_aliases WHERE resource_id = ?", (resource_id,)
            ).fetchone()
            timestamp = _now()
            connection.execute(
                """INSERT INTO knowledge_aliases
                   (resource_id, document_id, source_hash, relation_status, attempted_hash, changed_at)
                   VALUES (?, ?, (SELECT source_hash FROM knowledge_documents WHERE document_id = ?), ?, ?, ?)
                   ON CONFLICT(resource_id) DO UPDATE SET
                     document_id=excluded.document_id, source_hash=excluded.source_hash,
                     relation_status=excluded.relation_status, attempted_hash=excluded.attempted_hash,
                     changed_at=excluded.changed_at""",
                (resource_id, previous_document_id, previous_document_id,
                 AliasRelation.RETAINED_PREVIOUS.value, attempted_hash, timestamp),
            )
            connection.execute(
                """UPDATE knowledge_resources SET state = ?, attempted_hash = ?, updated_at = ?
                   WHERE resource_id = ?""",
                (IngestionState.RETAINED_PREVIOUS.value, attempted_hash, timestamp, resource_id),
            )
            self._record_event_in_transaction(
                connection, resource_id=resource_id, document_id=previous_document_id,
                attempted_hash=attempted_hash, state=IngestionState.RETAINED_PREVIOUS,
                stage="RETAIN_PREVIOUS",
            )
            self._refresh_document_currentness(
                connection, old["document_id"] if old else None, previous_document_id
            )

    def mark_removed(self, resource_id: str) -> None:
        with self._write() as connection:
            existing = connection.execute(
                "SELECT document_id, source_hash FROM knowledge_aliases WHERE resource_id = ?", (resource_id,)
            ).fetchone()
            if existing is None:
                return
            timestamp = _now()
            connection.execute(
                """UPDATE knowledge_aliases SET relation_status = ?, attempted_hash = source_hash,
                   changed_at = ? WHERE resource_id = ?""",
                (AliasRelation.REMOVED.value, timestamp, resource_id),
            )
            connection.execute(
                """UPDATE knowledge_resources SET state = ?, updated_at = ? WHERE resource_id = ?""",
                (IngestionState.REMOVED.value, timestamp, resource_id),
            )
            self._refresh_document_currentness(connection, existing["document_id"])

    def current_alias_count(self, document_id: str) -> int:
        with self._read() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS count FROM knowledge_aliases
                   WHERE document_id = ? AND relation_status = ?""",
                (document_id, AliasRelation.CURRENT.value),
            ).fetchone()
        return int(row["count"])

    def document_is_active(self, document_id: str) -> bool:
        """Return source currentness independent of projection publication."""

        with self._read() as connection:
            row = connection.execute(
                """SELECT 1 FROM knowledge_documents AS document
                   WHERE document.document_id = ?
                     AND document.active = 1
                     AND EXISTS (
                         SELECT 1 FROM knowledge_aliases AS alias
                         WHERE alias.document_id = document.document_id
                           AND alias.relation_status = ?
                     )""",
                (document_id, AliasRelation.CURRENT.value),
            ).fetchone()
        return row is not None

    def document_is_retrieval_ready(self, document_id: str) -> bool:
        """Return the complete default-view predicate for later query readers."""

        with self._read() as connection:
            row = connection.execute(
                """SELECT 1 FROM knowledge_documents AS document
                   JOIN knowledge_index_generations AS generation
                     ON generation.generation_id = document.projection_generation
                    AND generation.is_active = 1
                   WHERE document.document_id = ?
                     AND document.active = 1
                     AND document.vector_ready = 1
                     AND document.lexical_ready = 1
                     AND EXISTS (
                         SELECT 1 FROM knowledge_aliases AS alias
                         WHERE alias.document_id = document.document_id
                           AND alias.relation_status = ?
                     )""",
                (document_id, AliasRelation.CURRENT.value),
            ).fetchone()
        return row is not None

    def record_projection_ready(
        self, document_id: str, generation_id: str, vector_ready: bool, lexical_ready: bool
    ) -> None:
        with self._write() as connection:
            if connection.execute(
                "SELECT 1 FROM knowledge_documents WHERE document_id = ?", (document_id,)
            ).fetchone() is None:
                raise ValueError(f"unknown document_id: {document_id}")
            generation = connection.execute(
                "SELECT population_hash FROM knowledge_index_generations WHERE generation_id = ?",
                (generation_id,),
            ).fetchone()
            connection.execute(
                """UPDATE knowledge_documents SET vector_ready = ?, lexical_ready = ?,
                   projection_generation = ?, projection_population_hash = ?, updated_at = ?
                   WHERE document_id = ?""",
                (
                    int(bool(vector_ready)), int(bool(lexical_ready)), generation_id,
                    generation["population_hash"] if generation else None, _now(), document_id,
                ),
            )

    def publish_generation(
        self,
        generation: IndexGeneration | None,
        *,
        documents_by_id: Mapping[str, ParsedDocument],
        component_fingerprints: Mapping[str, str],
        chunks_by_document: Mapping[str, tuple[ChunkRecord, ...]],
        aliases: tuple[tuple[str, str, str, IngestionState], ...],
        removed_resource_ids: tuple[str, ...],
        ready_document_ids: tuple[str, ...],
        summary: IngestionRunSummary,
        events: tuple[tuple[str | None, str | None, str | None, IngestionState, str], ...] = (),
    ) -> None:
        """Atomically make a matched projection and its catalog view visible.

        Projection files are built and validated before this call.  SQLite is
        then the single publication boundary for the active generation,
        aliases/currentness, chunk rows/readiness, terminal run marker, and
        the corresponding compact events.  A reader can therefore observe
        either the old committed view or the complete new view, never an
        activated generation with stale document readiness.
        """

        if generation is not None:
            self._validate_generation(generation)
        if not ready_document_ids and generation is not None:
            raise ValueError("an active generation requires its ready document population")
        with self._write() as connection:
            if generation is not None:
                self._set_active_generation_in_transaction(connection, generation)

            affected_documents: set[str] = set(ready_document_ids)
            for document_id, document in documents_by_id.items():
                if document.document_id != document_id:
                    raise ValueError(f"document map key mismatch: {document_id}")
                timestamp = _now()
                connection.execute(
                    """INSERT INTO knowledge_documents
                       (document_id, source_hash, metadata_json, parser_id, parser_version,
                        parser_config_hash, state, component_fingerprints_json, ingestion_run_id,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                       ON CONFLICT(document_id) DO UPDATE SET
                         source_hash=excluded.source_hash, metadata_json=excluded.metadata_json,
                         parser_id=excluded.parser_id, parser_version=excluded.parser_version,
                         parser_config_hash=excluded.parser_config_hash, state=excluded.state,
                         component_fingerprints_json=excluded.component_fingerprints_json,
                         ingestion_run_id=NULL, updated_at=excluded.updated_at""",
                    (
                        document.document_id, document.source_hash, _json(document.metadata.to_dict()),
                        document.parser_id, document.parser_version, document.parser_config_hash,
                        IngestionState.PARSED.value, _json(dict(component_fingerprints)),
                        timestamp, timestamp,
                    ),
                )
            for document_id, chunks in chunks_by_document.items():
                if connection.execute(
                    "SELECT 1 FROM knowledge_documents WHERE document_id = ?", (document_id,)
                ).fetchone() is None:
                    raise ValueError(f"unknown document_id: {document_id}")
                connection.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (document_id,))
                connection.executemany(
                    """INSERT INTO knowledge_chunks
                       (chunk_id, document_id, source_hash, provenance_json, ingestion_run_id)
                       VALUES (?, ?, ?, ?, NULL)""",
                    [
                        (chunk.chunk_id, document_id, chunk.source_hash, _json(chunk.to_dict()))
                        for chunk in chunks
                    ],
                )
                connection.execute(
                    "UPDATE knowledge_documents SET ingestion_run_id = NULL WHERE document_id = ?",
                    (document_id,),
                )

            for resource_id, document_id, source_hash, resource_state in aliases:
                self._ensure_resource(connection, resource_id)
                old = connection.execute(
                    "SELECT document_id FROM knowledge_aliases WHERE resource_id = ?", (resource_id,)
                ).fetchone()
                if connection.execute(
                    "SELECT 1 FROM knowledge_documents WHERE document_id = ?", (document_id,)
                ).fetchone() is None:
                    raise ValueError(f"unknown document_id: {document_id}")
                timestamp = _now()
                connection.execute(
                    """INSERT INTO knowledge_aliases
                       (resource_id, document_id, source_hash, relation_status, attempted_hash, changed_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(resource_id) DO UPDATE SET
                         document_id=excluded.document_id, source_hash=excluded.source_hash,
                         relation_status=excluded.relation_status, attempted_hash=excluded.attempted_hash,
                         changed_at=excluded.changed_at""",
                    (resource_id, document_id, source_hash, AliasRelation.CURRENT.value, source_hash, timestamp),
                )
                connection.execute(
                    """UPDATE knowledge_resources SET source_hash = ?, attempted_hash = ?, state = ?,
                       updated_at = ? WHERE resource_id = ?""",
                    (source_hash, source_hash, resource_state.value, timestamp, resource_id),
                )
                affected_documents.add(document_id)
                if old is not None:
                    affected_documents.add(old["document_id"])

            for resource_id in removed_resource_ids:
                old = connection.execute(
                    "SELECT document_id FROM knowledge_aliases WHERE resource_id = ?", (resource_id,)
                ).fetchone()
                if old is None:
                    continue
                connection.execute(
                    """UPDATE knowledge_aliases SET relation_status = ?, attempted_hash = source_hash,
                       changed_at = ? WHERE resource_id = ?""",
                    (AliasRelation.REMOVED.value, _now(), resource_id),
                )
                connection.execute(
                    "UPDATE knowledge_resources SET state = ?, updated_at = ? WHERE resource_id = ?",
                    (IngestionState.REMOVED.value, _now(), resource_id),
                )
                affected_documents.add(old["document_id"])

            if generation is not None:
                placeholders = ",".join("?" for _ in ready_document_ids)
                connection.execute(
                    f"""UPDATE knowledge_documents SET state = ?, active = 1, vector_ready = 1,
                       lexical_ready = 1, projection_generation = ?, projection_population_hash = ?,
                       ingestion_run_id = NULL, updated_at = ?
                       WHERE document_id IN ({placeholders})""",
                    (
                        IngestionState.INDEXED.value, generation.generation_id, generation.population_hash,
                        _now(), *ready_document_ids,
                    ),
                )
                connection.execute(
                    f"""UPDATE knowledge_chunks SET active = 1, vector_ready = 1, lexical_ready = 1,
                       projection_generation = ?, projection_population_hash = ?, ingestion_run_id = NULL
                       WHERE document_id IN ({placeholders})""",
                    (generation.generation_id, generation.population_hash, *ready_document_ids),
                )

            self._refresh_document_currentness(connection, *affected_documents)
            for resource_id, document_id, attempted_hash, state, stage in events:
                if resource_id is not None:
                    self._ensure_resource(connection, resource_id)
                self._record_event_in_transaction(
                    connection, resource_id=resource_id, document_id=document_id,
                    attempted_hash=attempted_hash, state=state, stage=stage, run_id=summary.run_id,
                )
            self._record_run_in_transaction(connection, summary)

    @staticmethod
    def _validate_generation(generation: IndexGeneration) -> None:
        if not generation.vector_ready or not generation.lexical_ready:
            raise ValueError("active generation requires both vector and lexical projections ready")
        if generation.status != "VALIDATED":
            raise ValueError("active generation requires VALIDATED status")
        if generation.vector_location == Path(".") or generation.lexical_location == Path("."):
            raise ValueError("active generation requires non-placeholder vector and lexical locations")
        if not generation.population_hash.strip():
            raise ValueError("active generation requires a population hash")
        components = dict(generation.component_versions)
        if not str(components.get("vector", "")).strip() or not str(components.get("lexical", "")).strip():
            raise ValueError("active generation requires explicit vector and lexical component identities")
        if (
            components.get("vector_generation_id") != generation.generation_id
            or components.get("lexical_generation_id") != generation.generation_id
        ):
            raise ValueError("active generation requires matching vector and lexical generation identities")
        if (
            components.get("vector_population_hash") != generation.population_hash
            or components.get("lexical_population_hash") != generation.population_hash
        ):
            raise ValueError("active generation requires matching vector and lexical population hashes")

    def _set_active_generation_in_transaction(
        self, connection: sqlite3.Connection, generation: IndexGeneration
    ) -> None:
        """Write the generation registry while an enclosing transaction is active."""

        payload = generation.to_dict()
        connection.execute("UPDATE knowledge_index_generations SET is_active = 0 WHERE is_active = 1")
        connection.execute(
            """INSERT INTO knowledge_index_generations
               (generation_id, vector_location, lexical_location, embedding_spec_json,
                lexical_index_version, lexical_tokenizer_settings_json, index_version,
                population_hash, population_identity, document_count, chunk_count,
                vector_ready, lexical_ready, status, created_at, activated_at,
                component_versions_json, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
               ON CONFLICT(generation_id) DO UPDATE SET
                 vector_location=excluded.vector_location, lexical_location=excluded.lexical_location,
                 embedding_spec_json=excluded.embedding_spec_json,
                 lexical_index_version=excluded.lexical_index_version,
                 lexical_tokenizer_settings_json=excluded.lexical_tokenizer_settings_json,
                 index_version=excluded.index_version, population_hash=excluded.population_hash,
                 population_identity=excluded.population_identity,
                 document_count=excluded.document_count, chunk_count=excluded.chunk_count,
                 vector_ready=excluded.vector_ready, lexical_ready=excluded.lexical_ready,
                 status=excluded.status, created_at=excluded.created_at,
                 activated_at=excluded.activated_at,
                 component_versions_json=excluded.component_versions_json, is_active=1""",
            (
                generation.generation_id, str(generation.vector_location), str(generation.lexical_location),
                _json(payload["embedding_spec"]), generation.lexical_index_version,
                _json(payload["lexical_tokenizer_settings"]), generation.index_version,
                generation.population_hash, generation.population_identity, generation.document_count,
                generation.chunk_count, int(generation.vector_ready), int(generation.lexical_ready),
                generation.status, payload["created_at"], generation.activated_at or _now(),
                _json(payload["component_versions"]),
            ),
        )

    def set_active_generation(self, generation: IndexGeneration) -> None:
        """Persist and atomically activate one complete matched generation."""

        self._validate_generation(generation)
        with self._write() as connection:
            self._set_active_generation_in_transaction(connection, generation)
            # A projection can be staged before its matched generation is
            # activated.  Publication supplies the authoritative population
            # identity to every already-ready document/chunk in that pair.
            connection.execute(
                """UPDATE knowledge_documents SET projection_population_hash = ?, updated_at = ?
                   WHERE projection_generation = ?""",
                (generation.population_hash, _now(), generation.generation_id),
            )
            connection.execute(
                """UPDATE knowledge_chunks SET projection_population_hash = ?
                   WHERE projection_generation = ?""",
                (generation.population_hash, generation.generation_id),
            )

    def active_generation(self) -> IndexGeneration | None:
        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_index_generations WHERE is_active = 1"
            ).fetchone()
        return None if row is None else self._row_generation(row)

    def list_resources(self, state: IngestionState | str | None = None) -> tuple[CatalogResource, ...]:
        query = "SELECT * FROM knowledge_resources"
        parameters: tuple[str, ...] = ()
        if state is not None:
            query += " WHERE state = ?"
            parameters = (IngestionState(state).value,)
        query += " ORDER BY COALESCE(relative_path, resource_id), resource_id"
        with self._read() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            CatalogResource(
                resource_id=row["resource_id"], relative_path=row["relative_path"],
                display_path=row["display_path"], source_hash=row["source_hash"],
                state=IngestionState(row["state"]), attempted_hash=row["attempted_hash"],
            )
            for row in rows
        )

    @staticmethod
    def _record_run_in_transaction(connection: sqlite3.Connection, summary: IngestionRunSummary) -> None:
        connection.execute(
            """INSERT INTO knowledge_ingestion_runs
               (run_id, state, counts_json, started_at, completed_at, source_count,
                document_count, chunk_count, diagnostics_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET state=excluded.state,
                 counts_json=excluded.counts_json, started_at=excluded.started_at,
                 completed_at=excluded.completed_at, source_count=excluded.source_count,
                 document_count=excluded.document_count, chunk_count=excluded.chunk_count,
                 diagnostics_json=excluded.diagnostics_json""",
            (
                summary.run_id, str(summary.state.value if isinstance(summary.state, IngestionState) else summary.state),
                _json(summary.to_dict()["counts"]), summary.started_at, summary.completed_at,
                summary.source_count, summary.document_count, summary.chunk_count,
                _json(summary.to_dict()["diagnostics"]),
            ),
        )

    def record_run(self, summary: IngestionRunSummary) -> None:
        with self._write() as connection:
            self._record_run_in_transaction(connection, summary)

    def start_run(self, run_id: str, *, source_count: int = 0) -> None:
        """Persist a RUNNING marker before any artifact can be staged."""

        self.record_run(
            IngestionRunSummary(
                run_id=run_id,
                state="RUNNING",
                counts={},
                started_at=_now(),
                source_count=source_count,
                document_count=0,
                chunk_count=0,
            )
        )

    def running_run_ids(self) -> tuple[str, ...]:
        with self._read() as connection:
            rows = connection.execute(
                "SELECT run_id FROM knowledge_ingestion_runs WHERE state = 'RUNNING' ORDER BY run_id"
            ).fetchall()
        return tuple(str(row["run_id"]) for row in rows)

    def mark_run_interrupted(self, run_id: str) -> None:
        with self._write() as connection:
            row = connection.execute(
                "SELECT counts_json, source_count, document_count, chunk_count, diagnostics_json "
                "FROM knowledge_ingestion_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None or row["counts_json"] is None:
                return
            counts = json.loads(row["counts_json"])
            counts[IngestionState.INTERRUPTED.value] = int(counts.get(IngestionState.INTERRUPTED.value, 0)) + 1
            connection.execute(
                """UPDATE knowledge_ingestion_runs SET state = ?, counts_json = ?, completed_at = ?
                   WHERE run_id = ?""",
                ("INTERRUPTED", _json(counts), _now(), run_id),
            )
            self._record_event_in_transaction(
                connection,
                resource_id=None,
                document_id=None,
                attempted_hash=None,
                state=IngestionState.INTERRUPTED,
                stage="RECOVERY",
                run_id=run_id,
            )

    def list_runs(self) -> tuple[IngestionRunSummary, ...]:
        with self._read() as connection:
            rows = connection.execute("SELECT * FROM knowledge_ingestion_runs ORDER BY COALESCE(started_at, run_id), run_id").fetchall()
        return tuple(
            IngestionRunSummary(
                run_id=row["run_id"],
                state=(
                    IngestionState(row["state"])
                    if row["state"] in {state.value for state in IngestionState}
                    else row["state"]
                ),
                counts=json.loads(row["counts_json"]),
                started_at=row["started_at"], completed_at=row["completed_at"],
                source_count=row["source_count"], document_count=row["document_count"],
                chunk_count=row["chunk_count"], diagnostics=tuple(json.loads(row["diagnostics_json"])),
            )
            for row in rows
        )

    def _record_event_in_transaction(
        self, connection: sqlite3.Connection, *, resource_id: str | None, document_id: str | None,
        attempted_hash: str | None, state: IngestionState, stage: str,
        error: BaseException | str | None = None, counters: Mapping[str, int] | None = None,
        run_id: str | None = None,
    ) -> None:
        error_type, error_message, fingerprint = bounded_error_metadata(error)
        scalar_counters = {str(key): int(value) for key, value in (counters or {}).items()}
        if any(value < 0 for value in scalar_counters.values()):
            raise ValueError("diagnostic counters must be non-negative")
        connection.execute(
            """INSERT INTO knowledge_ingestion_events
               (run_id, resource_id, document_id, attempted_hash, state, stage, error_type,
                error_message, error_fingerprint, counters_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, resource_id, document_id, attempted_hash, state.value, str(stage),
             error_type, error_message, fingerprint, _json(scalar_counters), _now()),
        )

    def record_event(
        self, *, resource_id: str | None, document_id: str | None, attempted_hash: str | None,
        state: IngestionState, stage: str, error: BaseException | str | None = None,
        counters: Mapping[str, int] | None = None, run_id: str | None = None,
    ) -> None:
        """Record a compact event for future ingestion stages without source text."""

        with self._write() as connection:
            if resource_id is not None:
                self._ensure_resource(connection, resource_id)
            self._record_event_in_transaction(
                connection, resource_id=resource_id, document_id=document_id,
                attempted_hash=attempted_hash, state=IngestionState(state), stage=stage,
                error=error, counters=counters, run_id=run_id,
            )

    def list_quarantine(self) -> tuple[QuarantineRecord, ...]:
        quarantine_states = (
            IngestionState.UNSUPPORTED.value, IngestionState.PARSE_FAILED.value,
            IngestionState.NEEDS_OCR.value, IngestionState.EMBED_FAILED.value,
            IngestionState.INDEX_FAILED.value, IngestionState.SOURCE_CHANGED.value,
            IngestionState.INTERRUPTED.value, IngestionState.RETAINED_PREVIOUS.value,
        )
        placeholders = ", ".join("?" for _ in quarantine_states)
        with self._read() as connection:
            rows = connection.execute(
                f"SELECT * FROM knowledge_ingestion_events WHERE state IN ({placeholders}) ORDER BY event_id",
                quarantine_states,
            ).fetchall()
        return tuple(
            QuarantineRecord(
                event_id=int(row["event_id"]), state=IngestionState(row["state"]),
                stage=row["stage"], resource_id=row["resource_id"], document_id=row["document_id"],
                attempted_hash=row["attempted_hash"], error_type=row["error_type"],
                error_message=row["error_message"], error_fingerprint=row["error_fingerprint"],
                counters={str(key): int(value) for key, value in json.loads(row["counters_json"]).items()},
                created_at=row["created_at"],
            )
            for row in rows
        )


__all__ = [
    "CatalogAlias",
    "CatalogDocument",
    "CatalogResource",
    "KnowledgeCatalog",
]
