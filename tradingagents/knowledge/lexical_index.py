"""SQLite FTS5 lexical projection for one matched index generation."""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from .models import ChunkRecord, EmbeddingSpec

_SCHEMA_VERSION = "lexical-projection-v1"
_METADATA_TABLE = "knowledge_index_metadata"


class LexicalIndexError(RuntimeError):
    """SQLite FTS5 is unavailable or its generation is invalid."""


def _normalize_text(value: str) -> str:
    """Canonicalize text before SQLite's unicode61 tokenizer processes it."""

    return unicodedata.normalize("NFC", str(value)).casefold()


class LexicalIndexWriter:
    """Create an isolated FTS5 database; no fallback tokenizer is permitted."""

    component_identity = "sqlite-fts5-v1"

    def write(
        self,
        location: str | Path,
        chunks: Sequence[ChunkRecord],
        *,
        generation_id: str,
        embedding_spec: EmbeddingSpec,
        lexical_index_version: str,
        lexical_tokenizer_settings: Mapping[str, Any],
        index_version: str,
        population_hash: str,
        document_population_hash: str,
        chunk_population_hash: str,
    ) -> dict[str, Any]:
        path = Path(location)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with closing(sqlite3.connect(path)) as connection:
                self._create_schema(connection)
                seen: set[str] = set()
                for chunk in chunks:
                    if not chunk.chunk_id or chunk.chunk_id in seen:
                        raise LexicalIndexError("chunk IDs must be non-empty and unique")
                    seen.add(chunk.chunk_id)
                    self._insert_chunk(
                        connection,
                        chunk,
                        generation_id,
                        population_hash,
                        lexical_index_version,
                        index_version,
                    )
                metadata = {
                    "schema_version": _SCHEMA_VERSION,
                    "generation_id": generation_id,
                    "embedding_spec": embedding_spec.to_dict(),
                    "lexical_index_version": lexical_index_version,
                    "lexical_tokenizer_settings": dict(lexical_tokenizer_settings),
                    "index_version": index_version,
                    "population_hash": population_hash,
                    "document_population_hash": document_population_hash,
                    "chunk_population_hash": chunk_population_hash,
                    "document_count": len({chunk.document_id for chunk in chunks}),
                    "chunk_count": len(chunks),
                    "component_identity": self.component_identity,
                }
                self._replace_metadata(connection, metadata)
                self._assert_fts5(connection)
                connection.commit()
                return metadata
        except LexicalIndexError:
            raise
        except sqlite3.Error as exc:
            raise LexicalIndexError(f"could not write SQLite FTS5 projection: {exc}") from exc

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE knowledge_fts USING fts5("
                "chunk_id UNINDEXED, text, tokenize = 'unicode61 remove_diacritics 2 tokenchars ''_-'''"
                ")"
            )
        except sqlite3.OperationalError as exc:
            raise LexicalIndexError("SQLite was built without required FTS5/tokenizer support") from exc
        connection.execute(
            "CREATE TABLE knowledge_chunk_provenance ("
            "chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, source_hash TEXT NOT NULL, "
            "generation_id TEXT NOT NULL, content_type TEXT NOT NULL, provenance_json TEXT NOT NULL)"
        )
        connection.execute(
            f"CREATE TABLE {_METADATA_TABLE} (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)"
        )

    @staticmethod
    def _insert_chunk(
        connection: sqlite3.Connection,
        chunk: ChunkRecord,
        generation_id: str,
        population_hash: str,
        lexical_index_version: str,
        index_version: str,
    ) -> None:
        provenance = chunk.to_dict()
        provenance["content_type"] = chunk.content_type.value
        # Lexical rows are published only as part of a matched, validated
        # vector/lexical generation; expose that readiness to query filters.
        provenance["vector_ready"] = True
        provenance["lexical_ready"] = True
        provenance["lexical_index_version"] = lexical_index_version
        provenance["index_version"] = index_version
        provenance["projection_generation"] = generation_id
        provenance["projection_population_hash"] = population_hash
        connection.execute(
            "INSERT INTO knowledge_fts(chunk_id, text) VALUES (?, ?)",
            (chunk.chunk_id, _normalize_text(chunk.text)),
        )
        connection.execute(
            "INSERT INTO knowledge_chunk_provenance "
            "(chunk_id, document_id, source_hash, generation_id, content_type, provenance_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.source_hash,
                generation_id,
                chunk.content_type.value,
                json.dumps(provenance, sort_keys=True, separators=(",", ":")),
            ),
        )

    @staticmethod
    def _replace_metadata(connection: sqlite3.Connection, metadata: Mapping[str, Any]) -> None:
        connection.executemany(
            f"INSERT OR REPLACE INTO {_METADATA_TABLE}(key, value_json) VALUES (?, ?)",
            [
                (str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
                for key, value in metadata.items()
            ],
        )

    @staticmethod
    def _assert_fts5(connection: sqlite3.Connection) -> None:
        connection.execute("SELECT count(*) FROM knowledge_fts").fetchone()

    def write_metadata(self, location: str | Path, **changes: Any) -> None:
        """Test/recovery helper that mutates only manifest metadata, never FTS rows."""

        path = Path(location)
        try:
            with closing(sqlite3.connect(path)) as connection:
                self._replace_metadata(connection, changes)
                connection.commit()
        except sqlite3.Error as exc:
            raise LexicalIndexError("lexical generation metadata is unreadable") from exc


class LexicalIndexReader:
    """Read FTS5 and provenance from the generation-specific database only."""

    def __init__(self, location: str | Path) -> None:
        self.location = Path(location)

    def metadata(self) -> dict[str, Any]:
        try:
            with closing(sqlite3.connect(self.location)) as connection:
                rows = connection.execute(
                    f"SELECT key, value_json FROM {_METADATA_TABLE} ORDER BY key"
                ).fetchall()
        except sqlite3.Error as exc:
            raise LexicalIndexError("lexical generation metadata is unreadable") from exc
        if not rows:
            raise LexicalIndexError("lexical generation metadata is missing")
        try:
            return {key: json.loads(value) for key, value in rows}
        except json.JSONDecodeError as exc:
            raise LexicalIndexError("lexical generation metadata is invalid") from exc

    def row_count(self) -> int:
        try:
            with closing(sqlite3.connect(self.location)) as connection:
                return int(connection.execute("SELECT count(*) FROM knowledge_fts").fetchone()[0])
        except sqlite3.Error as exc:
            raise LexicalIndexError("lexical FTS5 table is unavailable") from exc

    def provenance_rows(self) -> tuple[dict[str, Any], ...]:
        """Return every lexical provenance row for generation-integrity checks."""

        try:
            with closing(sqlite3.connect(self.location)) as connection:
                rows = connection.execute(
                    "SELECT chunk_id, document_id, source_hash, generation_id, content_type, provenance_json "
                    "FROM knowledge_chunk_provenance ORDER BY chunk_id"
                ).fetchall()
        except sqlite3.Error as exc:
            raise LexicalIndexError("lexical provenance rows are unavailable") from exc
        try:
            return tuple(
                {
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "source_hash": row[2],
                    "generation_id": row[3],
                    "content_type": row[4],
                    "provenance": json.loads(row[5]),
                }
                for row in rows
            )
        except json.JSONDecodeError as exc:
            raise LexicalIndexError("lexical provenance JSON is invalid") from exc

    def search(self, query: str, *, limit: int = 50) -> tuple[dict[str, Any], ...]:
        if not str(query).strip():
            return ()
        try:
            with closing(sqlite3.connect(self.location)) as connection:
                rows = connection.execute(
                    "SELECT provenance.provenance_json, bm25(knowledge_fts) AS lexical_score "
                    "FROM knowledge_fts JOIN knowledge_chunk_provenance AS provenance "
                    "ON provenance.chunk_id = knowledge_fts.chunk_id "
                    "WHERE knowledge_fts MATCH ? "
                    "ORDER BY lexical_score, provenance.chunk_id LIMIT ?",
                    (_normalize_text(query), int(limit)),
                ).fetchall()
        except sqlite3.Error as exc:
            raise LexicalIndexError(f"lexical query failed: {exc}") from exc
        result = []
        for provenance_json, score in rows:
            item = json.loads(provenance_json)
            item["lexical_score"] = float(score)
            result.append(item)
        return tuple(result)


__all__ = ["LexicalIndexError", "LexicalIndexReader", "LexicalIndexWriter"]
