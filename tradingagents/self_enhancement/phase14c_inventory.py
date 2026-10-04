"""Read-only Phase 7 inventory and pinned-generation preflight for Phase 14C."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from tradingagents.knowledge.models import AliasRelation, IndexGeneration, IngestionState

from .phase14c_models import CorpusInventory, Phase14CGenerationPin

INVENTORY_SCHEMA_VERSION = "phase14c-inventory-v1"


class Phase14CInventoryError(RuntimeError):
    """The frozen knowledge catalog cannot be inventoried safely."""


class PinnedPhase7GenerationError(Phase14CInventoryError):
    """The active Phase 7 generation differs from the explicit run pin."""


class ArtifactRootValidationError(ValueError):
    """A Phase 14C artifact root is not fresh and safely isolated."""


_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "knowledge_resources": frozenset(
        {"resource_id", "relative_path", "display_path", "format", "source_hash", "state"}
    ),
    "knowledge_documents": frozenset(
        {
            "document_id",
            "source_hash",
            "metadata_json",
            "state",
            "active",
            "vector_ready",
            "lexical_ready",
            "projection_generation",
            "projection_population_hash",
        }
    ),
    "knowledge_aliases": frozenset(
        {"resource_id", "document_id", "source_hash", "relation_status"}
    ),
    "knowledge_chunks": frozenset(
        {
            "chunk_id",
            "document_id",
            "source_hash",
            "provenance_json",
            "active",
            "vector_ready",
            "lexical_ready",
            "projection_generation",
            "projection_population_hash",
        }
    ),
    "knowledge_index_generations": frozenset(
        {
            "generation_id",
            "vector_location",
            "lexical_location",
            "embedding_spec_json",
            "lexical_index_version",
            "lexical_tokenizer_settings_json",
            "index_version",
            "population_hash",
            "population_identity",
            "document_count",
            "chunk_count",
            "vector_ready",
            "lexical_ready",
            "status",
            "created_at",
            "activated_at",
            "component_versions_json",
            "is_active",
        }
    ),
}

_KNOWN_INGESTION_STATES = frozenset(state.value for state in IngestionState)
_KNOWN_ALIAS_STATES = frozenset(state.value for state in AliasRelation)
_UNAVAILABLE_STATES = frozenset(
    {
        IngestionState.UNSUPPORTED.value,
        IngestionState.NEEDS_OCR.value,
        IngestionState.PARSE_FAILED.value,
        IngestionState.EMBED_FAILED.value,
        IngestionState.INDEX_FAILED.value,
        IngestionState.SOURCE_CHANGED.value,
        IngestionState.INTERRUPTED.value,
        IngestionState.RETAINED_PREVIOUS.value,
        IngestionState.DISCOVERED.value,
        IngestionState.HASHED.value,
        IngestionState.PARSING.value,
        IngestionState.PARSED.value,
        IngestionState.CHUNKED.value,
        IngestionState.EMBEDDING.value,
        IngestionState.INDEXING.value,
    }
)


def _canonical_generation_fingerprint(generation: IndexGeneration) -> str:
    # Match the canonical IndexGeneration representation used by the Phase 7
    # acceptance fingerprint without importing the Phase 11A package.
    payload = generation.to_json().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _open_read_only_catalog(catalog_path: Path) -> sqlite3.Connection:
    if not catalog_path.is_file():
        raise Phase14CInventoryError("missing Phase 7 catalog")
    try:
        uri = f"{catalog_path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        if int(connection.execute("PRAGMA query_only").fetchone()[0]) != 1:
            connection.close()
            raise Phase14CInventoryError("SQLite query-only mode could not be enabled")
        return connection
    except Phase14CInventoryError:
        raise
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise Phase14CInventoryError("unable to open Phase 7 catalog read-only") from exc


def _validate_schema(connection: sqlite3.Connection) -> None:
    present = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    missing_tables = sorted(set(_REQUIRED_COLUMNS) - present)
    if missing_tables:
        raise Phase14CInventoryError(
            "unknown Phase 7 catalog schema: missing required tables " + ", ".join(missing_tables)
        )
    for table, expected in _REQUIRED_COLUMNS.items():
        columns = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        missing_columns = sorted(expected - columns)
        if missing_columns:
            raise Phase14CInventoryError(
                f"unknown Phase 7 catalog schema: {table} missing required columns "
                + ", ".join(missing_columns)
            )


def _read_active_generation(connection: sqlite3.Connection) -> IndexGeneration:
    rows = connection.execute(
        "SELECT * FROM knowledge_index_generations WHERE is_active=1"
    ).fetchall()
    if len(rows) != 1:
        raise PinnedPhase7GenerationError("Phase 7 must have exactly one active generation")
    row = rows[0]
    try:
        generation = IndexGeneration.from_dict(
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
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase14CInventoryError("active Phase 7 generation metadata is malformed") from exc
    if generation.status != "VALIDATED" or not generation.vector_ready or not generation.lexical_ready:
        raise PinnedPhase7GenerationError("active Phase 7 generation is not fully validated and ready")
    if not generation.population_hash.strip():
        raise PinnedPhase7GenerationError("active Phase 7 generation has no population hash")
    if not str(generation.vector_location).strip() or not str(generation.lexical_location).strip():
        raise PinnedPhase7GenerationError("active Phase 7 generation has incomplete index locations")
    return generation


def _validate_closed_states(
    resources: Sequence[sqlite3.Row],
    documents: Sequence[sqlite3.Row],
    aliases: Sequence[sqlite3.Row],
) -> None:
    for label, rows in (("resource", resources), ("document", documents)):
        for row in rows:
            if str(row["state"]) not in _KNOWN_INGESTION_STATES:
                raise Phase14CInventoryError(f"unknown Phase 7 {label} state")
    for row in aliases:
        if str(row["relation_status"]) not in _KNOWN_ALIAS_STATES:
            raise Phase14CInventoryError("unknown Phase 7 alias relation state")


def _safe_bool(row: sqlite3.Row, field: str, *, table: str) -> bool:
    value = row[field]
    if value not in (0, 1):
        raise Phase14CInventoryError(f"invalid boolean field {table}.{field}")
    return bool(value)


def _source_manifest_fingerprint(
    generation: IndexGeneration,
    resources: Sequence[sqlite3.Row],
    documents: Sequence[sqlite3.Row],
    aliases: Sequence[sqlite3.Row],
    chunks: Sequence[sqlite3.Row],
) -> str:
    manifest = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "generation_id": generation.generation_id,
        "population_hash": generation.population_hash,
        "resources": [
            [row["resource_id"], row["relative_path"], row["source_hash"], row["state"]]
            for row in resources
        ],
        "documents": [
            [row["document_id"], row["source_hash"], row["state"], row["active"]]
            for row in documents
        ],
        "aliases": [
            [row["resource_id"], row["document_id"], row["source_hash"], row["relation_status"]]
            for row in aliases
        ],
        "chunks": [
            [
                row["chunk_id"],
                row["document_id"],
                row["source_hash"],
                row["active"],
                row["projection_generation"],
                row["projection_population_hash"],
            ]
            for row in chunks
        ],
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_phase7_inventory(
    knowledge_root: Path,
    *,
    expected_generation_id: str,
    expected_generation_fingerprint: str,
    expected_population_hash: str,
) -> tuple[Phase14CGenerationPin, CorpusInventory]:
    """Read and verify the pinned Phase 7 generation without initializing or writing it."""

    root = Path(knowledge_root)
    catalog_path = root / "catalog.sqlite3"
    connection = _open_read_only_catalog(catalog_path)
    try:
        _validate_schema(connection)
        generation = _read_active_generation(connection)
        fingerprint = _canonical_generation_fingerprint(generation)
        if (
            generation.generation_id != expected_generation_id
            or fingerprint != expected_generation_fingerprint
            or generation.population_hash != expected_population_hash
        ):
            raise PinnedPhase7GenerationError("active Phase 7 generation does not match explicit pin")

        resources = connection.execute(
            """SELECT resource_id, relative_path, display_path, format, source_hash, state
               FROM knowledge_resources ORDER BY resource_id"""
        ).fetchall()
        documents = connection.execute(
            """SELECT document_id, source_hash, metadata_json, state, active, vector_ready,
                      lexical_ready, projection_generation, projection_population_hash
               FROM knowledge_documents ORDER BY document_id"""
        ).fetchall()
        aliases = connection.execute(
            """SELECT resource_id, document_id, source_hash, relation_status
               FROM knowledge_aliases ORDER BY resource_id"""
        ).fetchall()
        chunks = connection.execute(
            """SELECT chunk_id, document_id, source_hash, provenance_json, active, vector_ready,
                      lexical_ready, projection_generation, projection_population_hash
               FROM knowledge_chunks ORDER BY document_id, chunk_id"""
        ).fetchall()
    except Phase14CInventoryError:
        raise
    except (sqlite3.Error, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase14CInventoryError("Phase 7 inventory read failed closed") from exc
    finally:
        connection.close()

    _validate_closed_states(resources, documents, aliases)
    documents_by_id = {str(row["document_id"]): row for row in documents}
    current_aliases = [
        row for row in aliases if row["relation_status"] == AliasRelation.CURRENT.value
    ]
    current_alias_counts = Counter(
        str(row["document_id"]) for row in current_aliases if row["document_id"] is not None
    )
    alias_count = sum(max(0, count - 1) for count in current_alias_counts.values())
    current_document_ids = set(current_alias_counts)

    ready_documents: set[str] = set()
    indexed_documents: set[str] = set()
    for document_id in current_document_ids:
        row = documents_by_id.get(document_id)
        if row is None:
            raise Phase14CInventoryError("current Phase 7 alias refers to a missing document")
        active = _safe_bool(row, "active", table="knowledge_documents")
        vector_ready = _safe_bool(row, "vector_ready", table="knowledge_documents")
        lexical_ready = _safe_bool(row, "lexical_ready", table="knowledge_documents")
        if (
            active
            and vector_ready
            and lexical_ready
            and row["projection_generation"] == generation.generation_id
            and row["projection_population_hash"] == generation.population_hash
        ):
            ready_documents.add(document_id)
            if row["state"] == IngestionState.INDEXED.value:
                indexed_documents.add(document_id)

    active_chunk_count = 0
    chunked_documents: set[str] = set()
    for row in chunks:
        active = _safe_bool(row, "active", table="knowledge_chunks")
        vector_ready = _safe_bool(row, "vector_ready", table="knowledge_chunks")
        lexical_ready = _safe_bool(row, "lexical_ready", table="knowledge_chunks")
        if not active or row["projection_generation"] != generation.generation_id:
            continue
        if (
            not vector_ready
            or not lexical_ready
            or row["projection_population_hash"] != generation.population_hash
        ):
            raise Phase14CInventoryError("active Phase 7 chunk has mismatched projection readiness")
        document_id = str(row["document_id"])
        document = documents_by_id.get(document_id)
        if document is None or document_id not in ready_documents:
            raise Phase14CInventoryError("active Phase 7 chunk has no ready active document")
        if str(row["source_hash"]) != str(document["source_hash"]):
            raise Phase14CInventoryError("active Phase 7 chunk source hash differs from its document")
        try:
            provenance = json.loads(row["provenance_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise Phase14CInventoryError("active Phase 7 chunk provenance is malformed") from exc
        if not isinstance(provenance, dict):
            raise Phase14CInventoryError("active Phase 7 chunk provenance is malformed")
        if (
            provenance.get("chunk_id") != row["chunk_id"]
            or provenance.get("document_id") != document_id
            or provenance.get("source_hash") != row["source_hash"]
            or not str(provenance.get("text") or "").strip()
        ):
            raise Phase14CInventoryError("active Phase 7 chunk text or provenance is incomplete")
        active_chunk_count += 1
        chunked_documents.add(document_id)

    if len(ready_documents) != generation.document_count or active_chunk_count != generation.chunk_count:
        raise PinnedPhase7GenerationError("active Phase 7 catalog population differs from generation counts")
    queryable_documents = ready_documents & chunked_documents
    if len(queryable_documents) != generation.document_count:
        raise PinnedPhase7GenerationError("active Phase 7 generation contains a non-queryable document")

    source_hashes = {
        str(row["source_hash"])
        for row in resources
        if row["source_hash"] and row["state"] != IngestionState.REMOVED.value
    }
    source_hashes.update(str(row["source_hash"]) for row in current_aliases if row["source_hash"])
    ocr_hashes = {
        str(row["source_hash"])
        for row in resources
        if row["source_hash"] and row["state"] == IngestionState.NEEDS_OCR.value
    }
    unavailable_count = sum(
        1 for row in resources if row["state"] in _UNAVAILABLE_STATES
    )

    pin = Phase14CGenerationPin(
        generation_id=generation.generation_id,
        generation_fingerprint=fingerprint,
        population_hash=generation.population_hash,
        status=generation.status,
        vector_ready=generation.vector_ready,
        lexical_ready=generation.lexical_ready,
        vector_location=generation.vector_location,
        lexical_location=generation.lexical_location,
        inventory_schema_version=INVENTORY_SCHEMA_VERSION,
        document_count=generation.document_count,
        chunk_count=generation.chunk_count,
    )
    inventory = CorpusInventory(
        unique_source_count=len(source_hashes),
        indexed_document_count=len(indexed_documents),
        queryable_document_count=len(queryable_documents),
        alias_count=alias_count,
        needs_ocr_count=len(ocr_hashes),
        unavailable_resource_count=unavailable_count,
        active_chunk_count=active_chunk_count,
        source_manifest_fingerprint=_source_manifest_fingerprint(
            generation, resources, documents, aliases, chunks
        ),
    )
    return pin, inventory


def _resolved(path: Path) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactRootValidationError("artifact/protected path cannot be resolved") from exc


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        common = Path(os.path.commonpath((str(left), str(right))))
    except ValueError:
        return False
    return common in (left, right)


def validate_fresh_artifact_root(
    artifact_root: Path,
    *,
    protected_roots: Sequence[Path],
) -> None:
    """Require a new artifact path that cannot overlap frozen/source data."""

    candidate = Path(artifact_root)
    if not candidate.is_absolute():
        raise ArtifactRootValidationError("Phase 14C artifact root must be an explicit absolute path")
    resolved_candidate = _resolved(candidate)
    for protected_path in protected_roots:
        protected = _resolved(Path(protected_path))
        if _paths_overlap(resolved_candidate, protected):
            raise ArtifactRootValidationError("Phase 14C artifact root overlaps protected data")
    if candidate.exists():
        raise ArtifactRootValidationError("Phase 14C artifact root already exists")


__all__ = [
    "ArtifactRootValidationError",
    "INVENTORY_SCHEMA_VERSION",
    "Phase14CInventoryError",
    "PinnedPhase7GenerationError",
    "read_phase7_inventory",
    "validate_fresh_artifact_root",
]
