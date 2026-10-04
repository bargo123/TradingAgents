from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.identity import document_id_for
from tradingagents.knowledge.models import (
    ChunkRecord,
    DiscoveredResource,
    DocumentMetadata,
    EmbeddingSpec,
    IndexGeneration,
    IngestionRunSummary,
    IngestionState,
    ParsedDocument,
)
from tradingagents.self_enhancement.phase14c_inventory import (
    ArtifactRootValidationError,
    Phase14CInventoryError,
    PinnedPhase7GenerationError,
    read_phase7_inventory,
    validate_fresh_artifact_root,
)

GENERATION_ID = "gen_607de64268a04a6ab09ffa1e160fc280"
POPULATION_HASH = "sha256:fd3ee0c846f2969747aca70576f58e55ffe1c07e8190570d2f9f7e2cb20b9e17"


def _generation(document_count: int = 93) -> IndexGeneration:
    return IndexGeneration(
        generation_id=GENERATION_ID,
        vector_location="vector/lancedb/gen_fixture",
        lexical_location="keyword/gen_fixture/bm25.sqlite3",
        embedding_spec=EmbeddingSpec(model_id="fixture", dimensions=3),
        lexical_index_version="fts5-fixture-v1",
        lexical_tokenizer_settings={"unicode": "NFC"},
        index_version="fixture-v1",
        population_hash=POPULATION_HASH,
        population_identity=POPULATION_HASH,
        activated_at="2026-01-01T00:00:00+00:00",
        document_count=document_count,
        chunk_count=document_count,
        vector_ready=True,
        lexical_ready=True,
        status="VALIDATED",
        component_versions={
            "vector": "fixture-vector-v1",
            "lexical": "fixture-lexical-v1",
            "vector_generation_id": GENERATION_ID,
            "lexical_generation_id": GENERATION_ID,
            "vector_population_hash": POPULATION_HASH,
            "lexical_population_hash": POPULATION_HASH,
        },
    )


def _seed_catalog(root: Path, *, document_count: int = 93) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "catalog.sqlite3"
    catalog = KnowledgeCatalog(catalog_path)
    catalog.initialize()

    documents: dict[str, ParsedDocument] = {}
    chunks_by_document: dict[str, tuple[ChunkRecord, ...]] = {}
    aliases: list[tuple[str, str, str, IngestionState]] = []
    for index in range(document_count):
        source_hash = f"{index + 1:064x}"
        document_id = document_id_for(source_hash)
        relative_path = f"fixture/book-{index:03}.pdf"
        documents[document_id] = ParsedDocument(
            document_id=document_id,
            source_hash=source_hash,
            metadata=DocumentMetadata(
                title=f"Fixture book {index}",
                source_filename=Path(relative_path).name,
                source_relative_path=relative_path,
                format="pdf",
            ),
            parser_id="fixture-parser",
            parser_version="1",
            parser_config_hash="fixture-config",
        )
        chunk_id = f"chunk-{index:03}"
        chunks_by_document[document_id] = (
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                source_hash=source_hash,
                text=f"fixture market evidence {index}",
                source_filename=Path(relative_path).name,
                source_relative_path=relative_path,
                page=1,
                chunk_ordinal=0,
            ),
        )
        resource_id = f"resource-{index:03}"
        catalog.upsert_discovered(
            DiscoveredResource(
                resource_id=resource_id,
                relative_path=relative_path,
                path=root / relative_path,
                format="pdf",
                source_hash=source_hash,
                state=IngestionState.INDEXED,
            )
        )
        aliases.append((resource_id, document_id, source_hash, IngestionState.INDEXED))

    if document_count:
        document_id = next(iter(documents))
        source_hash = documents[document_id].source_hash
        alias_id = "duplicate-alias"
        catalog.upsert_discovered(
            DiscoveredResource(
                resource_id=alias_id,
                relative_path="fixture/duplicate-copy.pdf",
                path=root / "fixture" / "duplicate-copy.pdf",
                format="pdf",
                source_hash=source_hash,
                state=IngestionState.DUPLICATE,
            )
        )
        aliases.append((alias_id, document_id, source_hash, IngestionState.DUPLICATE))

    for index in range(4):
        source_hash = f"{document_count + index + 1:064x}"
        relative_path = f"fixture/scanned-{index}.pdf"
        catalog.upsert_discovered(
            DiscoveredResource(
                resource_id=f"ocr-resource-{index}",
                relative_path=relative_path,
                path=root / relative_path,
                format="pdf",
                source_hash=source_hash,
                state=IngestionState.NEEDS_OCR,
            )
        )

    catalog.publish_generation(
        _generation(document_count),
        documents_by_id=documents,
        component_fingerprints={"fixture": "fixture-v1"},
        chunks_by_document=chunks_by_document,
        aliases=tuple(aliases),
        removed_resource_ids=(),
        ready_document_ids=tuple(documents),
        summary=IngestionRunSummary(
            run_id="fixture-ingestion",
            state="SUCCEEDED",
            source_count=document_count + 5,
            document_count=document_count,
            chunk_count=document_count,
        ),
    )
    return catalog_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_only_inventory_accounts_for_aliases_and_needs_ocr_without_writes(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path / "knowledge")
    before = _sha256(catalog_path)
    generation = _generation()

    pin, inventory = read_phase7_inventory(
        catalog_path.parent,
        expected_generation_id=GENERATION_ID,
        expected_generation_fingerprint=hashlib.sha256(generation.to_json().encode()).hexdigest(),
        expected_population_hash=POPULATION_HASH,
    )

    assert pin.generation_id == GENERATION_ID
    assert pin.population_hash == POPULATION_HASH
    assert pin.status == "VALIDATED"
    assert pin.vector_ready is True
    assert pin.lexical_ready is True
    assert inventory.unique_source_count == 97
    assert inventory.indexed_document_count == 93
    assert inventory.queryable_document_count == 93
    assert inventory.alias_count == 1
    assert inventory.needs_ocr_count == 4
    assert inventory.active_chunk_count == 93
    assert len(inventory.source_manifest_fingerprint) == 64
    assert _sha256(catalog_path) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_generation_id", "wrong-generation"),
        ("expected_generation_fingerprint", "0" * 64),
        ("expected_population_hash", "wrong-population"),
    ],
)
def test_generation_pin_mismatch_fails_closed(tmp_path: Path, field: str, value: str) -> None:
    catalog_path = _seed_catalog(tmp_path / "knowledge", document_count=1)
    generation = _generation(1)
    expected = {
        "expected_generation_id": GENERATION_ID,
        "expected_generation_fingerprint": hashlib.sha256(generation.to_json().encode()).hexdigest(),
        "expected_population_hash": POPULATION_HASH,
    }
    expected[field] = value

    with pytest.raises(PinnedPhase7GenerationError):
        read_phase7_inventory(catalog_path.parent, **expected)


def test_unknown_catalog_schema_fails_closed(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(catalog_path) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")

    with pytest.raises(Phase14CInventoryError, match="schema"):
        read_phase7_inventory(
            tmp_path,
            expected_generation_id=GENERATION_ID,
            expected_generation_fingerprint="0" * 64,
            expected_population_hash=POPULATION_HASH,
        )


def test_unknown_catalog_state_fails_closed(tmp_path: Path) -> None:
    catalog_path = _seed_catalog(tmp_path / "knowledge", document_count=1)
    with sqlite3.connect(catalog_path) as connection:
        connection.execute(
            "UPDATE knowledge_resources SET state='FUTURE_UNKNOWN' WHERE resource_id='resource-000'"
        )

    generation = _generation(1)
    with pytest.raises(Phase14CInventoryError, match="state"):
        read_phase7_inventory(
            catalog_path.parent,
            expected_generation_id=GENERATION_ID,
            expected_generation_fingerprint=hashlib.sha256(generation.to_json().encode()).hexdigest(),
            expected_population_hash=POPULATION_HASH,
        )


def test_fresh_artifact_root_rejects_existing_roots_without_modifying_them(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    before = _sha256(sentinel)

    with pytest.raises(ArtifactRootValidationError, match="already exists"):
        validate_fresh_artifact_root(existing, protected_roots=())

    assert _sha256(sentinel) == before


@pytest.mark.parametrize("target_kind", ["same", "child", "parent"])
def test_fresh_artifact_root_rejects_protected_path_overlap(tmp_path: Path, target_kind: str) -> None:
    protected_dir = tmp_path / "protected"
    protected_dir.mkdir()
    protected_file = protected_dir / "artifact.db"
    protected_file.write_bytes(b"protected")
    if target_kind == "same":
        target = protected_file
        protected_roots = (protected_file,)
    elif target_kind == "child":
        target = protected_dir / "phase14c"
        protected_roots = (protected_dir,)
    else:
        target = tmp_path
        protected_roots = (protected_file,)

    with pytest.raises(ArtifactRootValidationError, match="overlaps protected"):
        validate_fresh_artifact_root(target, protected_roots=protected_roots)

    assert protected_file.read_bytes() == b"protected"


def test_nonexistent_fresh_artifact_root_is_validated_without_creation(tmp_path: Path) -> None:
    target = tmp_path / "new-phase14c-root"

    validate_fresh_artifact_root(target, protected_roots=(tmp_path / "phase7", tmp_path / "source.db"))

    assert not target.exists()
