"""Incremental, failure-isolated ingestion tests for the local knowledge store."""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from tradingagents.knowledge.catalog import KnowledgeCatalog
from tradingagents.knowledge.config import KnowledgeConfig
from tradingagents.knowledge.discovery import SourceScanner
from tradingagents.knowledge.embeddings import EmbeddingProvider
from tradingagents.knowledge.identity import document_id_for, resource_id_for
from tradingagents.knowledge.index_generation import IndexGenerationManager
from tradingagents.knowledge.lexical_index import LexicalIndexWriter
from tradingagents.knowledge.models import (
    AliasRelation,
    ContentType,
    DocumentMetadata,
    EmbeddingSpec,
    IngestionState,
    ParsedBlock,
    ParsedDocument,
)
from tradingagents.knowledge.vector_index import VectorIndexWriter


class Tokenizer:
    model_max_input_tokens = 16

    def encode(self, text, *, add_special_tokens, purpose, truncation):
        assert truncation is False
        return list(range(len(str(text).split()) + (2 if add_special_tokens else 0)))


class FakeEmbedder(EmbeddingProvider):
    def __init__(self) -> None:
        self.embed_calls = 0
        super().__init__(
            spec=EmbeddingSpec(
                model_id="fixture/embedding",
                resolved_model_version="fixture-v1",
                runtime="fixture",
                artifact_hash="fixture-hash",
                dimensions=2,
                tokenizer_fingerprint="fixture-tokenizer",
                model_max_input_tokens=16,
                special_token_budget=2,
                effective_corpus_content_token_limit=14,
            ),
            tokenizer=Tokenizer(),
        )

    def _embed_formatted(self, texts):
        self.embed_calls += 1
        return tuple((float(index), 1.0) for index, _ in enumerate(texts))


class FakeParser:
    parser_id = "fixture-parser"
    parser_version = "1"
    parser_config_hash = "fixture-parser-config"

    def __init__(self) -> None:
        self.parse_calls: list[str] = []
        self.fail_for: set[str] = set()
        self.mutate_for: set[str] = set()

    def parse(self, resource, staging_dir):
        self.parse_calls.append(resource.relative_path)
        if resource.relative_path in self.fail_for:
            raise RuntimeError("fixture parser failure")
        if resource.relative_path in self.mutate_for:
            (staging_dir / "partial-parser-output.txt").write_text("partial", encoding="utf-8")
            resource.path.write_bytes(b"changed data")
        return ParsedDocument(
            document_id=document_id_for(resource.source_hash),
            source_hash=resource.source_hash,
            metadata=DocumentMetadata(
                title=resource.path.stem,
                source_filename=resource.path.name,
                source_relative_path=resource.relative_path,
                format=resource.format,
            ),
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            parser_config_hash=self.parser_config_hash,
            blocks=(ParsedBlock(block_id="body", text="order flow imbalance", content_type=ContentType.PROSE, reading_order=0),),
        )


class JsonVectorBackend:
    identity = "fixture-vector"

    def write(self, location: Path, rows):
        location.mkdir(parents=True, exist_ok=True)
        (location / "rows.json").write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")

    def read(self, location: Path):
        return tuple(json.loads((location / "rows.json").read_text(encoding="utf-8")))


class InterruptingManager:
    def __init__(self, inner, *, interrupt_after: str | None = None) -> None:
        self.inner = inner
        self.interrupt_after = interrupt_after
        self.fail_after_build = False
        self.catalog = inner.catalog
        self.embedding_spec = inner.embedding_spec

    def build_generation(self, *args, **kwargs):
        result = self.inner.build_generation(*args, **kwargs)
        if self.interrupt_after == "lexical":
            raise InterruptedError("fixture interruption after lexical projection")
        if self.fail_after_build:
            raise RuntimeError("fixture index publication failure")
        return result

    def activate_generation(self, generation):
        return self.inner.activate_generation(generation)

    def active_generation(self):
        return self.inner.active_generation()

    def resolve_active_generation(self):
        return self.inner.resolve_active_generation()


@dataclass
class Harness:
    source: Path
    catalog: KnowledgeCatalog
    parser: FakeParser
    embedder: FakeEmbedder
    ingestor: object
    shared_document_id: str


def _hold_ingestion_lock(source_root, artifact_root, ready, release):
    """Run in a separate process so the advisory lock has a live owner."""

    from tradingagents.knowledge.ingestion import KnowledgeIngestor

    config = KnowledgeConfig(source_root=source_root, artifact_root=artifact_root)
    ingestor = KnowledgeIngestor(config, parser=FakeParser(), embedder=FakeEmbedder())
    with ingestor._artifact_lock():
        ready.set()
        release.wait(10)


def ingestion_harness(tmp_path, *, duplicate=False, interrupt_after=None):
    from tradingagents.knowledge.ingestion import KnowledgeIngestor

    source = tmp_path / "source"
    source.mkdir()
    (source / "book.pdf").write_bytes(b"book bytes")
    if duplicate:
        (source / "copy.pdf").write_bytes(b"book bytes")
    else:
        (source / "second.pdf").write_bytes(b"second bytes")
        (source / "copy.pdf").write_bytes(b"book bytes")
    config = KnowledgeConfig(source_root=source, artifact_root=tmp_path / "artifacts")
    catalog = KnowledgeCatalog(config.artifact_root / "catalog.sqlite3")
    parser = FakeParser()
    embedder = FakeEmbedder()
    manager = IndexGenerationManager(
        config=config,
        catalog=catalog,
        embedding_spec=embedder.spec,
        vector_writer=VectorIndexWriter(backend=JsonVectorBackend()),
        lexical_writer=LexicalIndexWriter(),
    )
    manager = InterruptingManager(manager, interrupt_after=interrupt_after)
    ingestor = KnowledgeIngestor(
        config,
        scanner=SourceScanner(config),
        catalog=catalog,
        parser=parser,
        embedder=embedder,
        generation_manager=manager,
    )
    from hashlib import sha256

    return Harness(source, catalog, parser, embedder, ingestor, document_id_for(sha256(b"book bytes").hexdigest()))


def test_incremental_run_skips_unchanged_and_deduplicates_bytes(tmp_path):
    from tradingagents.knowledge.ingestion import IngestionMode

    harness = ingestion_harness(tmp_path)
    first = harness.ingestor.run(IngestionMode.INCREMENTAL)
    harness.parser.parse_calls.clear()
    harness.embedder.embed_calls = 0

    second = harness.ingestor.run(IngestionMode.INCREMENTAL)

    assert first.counts[IngestionState.INDEXED] == 2
    assert second.counts[IngestionState.UNCHANGED] == 2
    assert second.counts[IngestionState.DUPLICATE] == 1
    assert harness.parser.parse_calls == []
    assert harness.embedder.embed_calls == 0


def test_remove_one_alias_then_last_alias_updates_default_activity(tmp_path):
    from tradingagents.knowledge.ingestion import IngestionMode

    harness = ingestion_harness(tmp_path, duplicate=True)
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    (harness.source / "book.pdf").unlink()

    one_removed = harness.ingestor.run(IngestionMode.INCREMENTAL)
    assert one_removed.counts[IngestionState.REMOVED] == 1
    assert harness.catalog.document_is_active(harness.shared_document_id) is True

    (harness.source / "copy.pdf").unlink()
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    assert harness.catalog.document_is_active(harness.shared_document_id) is False


def test_failed_changed_alias_does_not_deactivate_other_duplicate(tmp_path):
    from tradingagents.knowledge.ingestion import IngestionMode

    harness = ingestion_harness(tmp_path, duplicate=True)
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    (harness.source / "book.pdf").write_bytes(b"changed bytes")
    harness.parser.fail_for.add("book.pdf")

    result = harness.ingestor.run(IngestionMode.INCREMENTAL)

    assert result.counts[IngestionState.PARSE_FAILED] == 1
    assert harness.catalog.get_alias(resource_id_for("copy.pdf")).relation is AliasRelation.CURRENT
    assert harness.catalog.document_is_active(harness.shared_document_id) is True
    assert harness.catalog.get_alias(resource_id_for("book.pdf")).relation is AliasRelation.RETAINED_PREVIOUS


def test_interrupted_run_recovery_removes_partial_projection_and_keeps_active_pair(tmp_path):
    from tradingagents.knowledge.ingestion import IngestionMode

    harness = ingestion_harness(tmp_path)
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    old_generation = harness.catalog.active_generation().generation_id
    (harness.source / "third.pdf").write_bytes(b"third bytes")
    harness.ingestor.generation_manager.interrupt_after = "lexical"
    harness.ingestor.run(IngestionMode.INCREMENTAL)

    harness.ingestor.recover_interrupted_runs()

    assert harness.catalog.active_generation().generation_id == old_generation
    assert harness.catalog.list_runs()[-1].state is IngestionState.INTERRUPTED


def test_ingestion_derives_chunk_policy_from_provider_spec_and_tokenizer(tmp_path, monkeypatch):
    import tradingagents.knowledge.ingestion as ingestion
    from tradingagents.knowledge.chunking import ChunkPolicy

    captured = {}

    class SpyChunker:
        def __init__(self, policy, tokenizer):
            captured["policy"] = policy
            captured["tokenizer"] = tokenizer

    monkeypatch.setattr(ingestion, "StructureAwareChunker", SpyChunker)
    harness = ingestion_harness(tmp_path)

    assert captured["policy"] == ChunkPolicy.for_embedding(harness.embedder.spec)
    assert captured["tokenizer"] is harness.embedder.tokenizer


def test_recovery_preserves_pair_published_before_interruption(tmp_path, monkeypatch):
    """A crash after catalog publication cannot make recovery delete the active pair."""

    from tradingagents.knowledge.ingestion import IngestionMode

    harness = ingestion_harness(tmp_path)
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    (harness.source / "third.pdf").write_bytes(b"third bytes")
    original_publish = harness.catalog.publish_generation

    def interrupt_after_publish(*args, **kwargs):
        original_publish(*args, **kwargs)
        raise InterruptedError("fixture crash immediately after publication")

    monkeypatch.setattr(harness.catalog, "publish_generation", interrupt_after_publish)
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    active = harness.catalog.active_generation()

    assert active is not None
    assert active.vector_location.is_dir()
    assert active.lexical_location.is_file()
    harness.ingestor.recover_interrupted_runs()
    assert harness.catalog.active_generation().generation_id == active.generation_id
    assert active.vector_location.is_dir()
    assert active.lexical_location.is_file()
    assert harness.catalog.list_runs()[-1].state == "SUCCEEDED"


def test_publication_refreshes_readiness_for_every_current_document(tmp_path):
    from tradingagents.knowledge.ingestion import IngestionMode

    harness = ingestion_harness(tmp_path, duplicate=True)
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    (harness.source / "second.pdf").write_bytes(b"second bytes")
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    assert all(harness.catalog.document_is_retrieval_ready(document_id) for document_id in harness.catalog.current_document_ids())

    (harness.source / "third-copy.pdf").write_bytes(b"book bytes")
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    assert all(harness.catalog.document_is_retrieval_ready(document_id) for document_id in harness.catalog.current_document_ids())

    (harness.source / "book.pdf").unlink()
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    assert all(harness.catalog.document_is_retrieval_ready(document_id) for document_id in harness.catalog.current_document_ids())


def test_source_mutation_discards_resource_staging_artifacts(tmp_path):
    from tradingagents.knowledge.ingestion import IngestionMode

    harness = ingestion_harness(tmp_path)
    harness.parser.mutate_for.add("second.pdf")

    result = harness.ingestor.run(IngestionMode.INCREMENTAL)

    assert result.counts[IngestionState.SOURCE_CHANGED] == 1
    assert not (harness.catalog.path.parent / ".ingestion-staging" / result.run_id).exists()


def test_index_failure_discards_unpublished_catalog_rows_and_artifacts(tmp_path):
    from tradingagents.knowledge.ingestion import IngestionMode

    harness = ingestion_harness(tmp_path)
    harness.ingestor.generation_manager.fail_after_build = True

    result = harness.ingestor.run(IngestionMode.INCREMENTAL)

    assert result.counts[IngestionState.INDEX_FAILED] >= 1
    assert not (harness.catalog.path.parent / ".ingestion-staging" / result.run_id).exists()
    assert not (harness.catalog.path.parent / "vector" / "lancedb" / f"gen_{result.run_id}").exists()
    assert not (harness.catalog.path.parent / "keyword" / f"gen_{result.run_id}").exists()
    with sqlite3.connect(harness.catalog.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_documents WHERE ingestion_run_id = ?", (result.run_id,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_chunks WHERE ingestion_run_id = ?", (result.run_id,)
        ).fetchone()[0] == 0


def test_failed_rebuild_preserves_prior_active_generation_and_current_aliases(tmp_path):
    from tradingagents.knowledge.ingestion import IngestionMode

    harness = ingestion_harness(tmp_path)
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    old_generation = harness.catalog.active_generation()
    assert old_generation is not None
    old_document_ids = harness.catalog.current_document_ids()
    old_aliases = {
        name: harness.catalog.get_alias(resource_id_for(name))
        for name in ("book.pdf", "copy.pdf", "second.pdf")
    }
    assert all(alias is not None and alias.relation is AliasRelation.CURRENT for alias in old_aliases.values())

    harness.ingestor.generation_manager.fail_after_build = True
    result = harness.ingestor.run(IngestionMode.REBUILD)

    assert result.counts[IngestionState.INDEX_FAILED] >= 1
    assert harness.catalog.active_generation().generation_id == old_generation.generation_id
    assert harness.catalog.current_document_ids() == old_document_ids
    assert all(harness.catalog.document_is_retrieval_ready(document_id) for document_id in old_document_ids)
    for name, old_alias in old_aliases.items():
        alias = harness.catalog.get_alias(resource_id_for(name))
        assert alias.relation is AliasRelation.CURRENT
        assert alias.document_id == old_alias.document_id
        assert alias.source_hash == old_alias.source_hash
    with sqlite3.connect(harness.catalog.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_documents WHERE ingestion_run_id = ?", (result.run_id,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_chunks WHERE ingestion_run_id = ?", (result.run_id,)
        ).fetchone()[0] == 0


def test_successful_ingestion_writes_and_repairs_active_pointer(tmp_path):
    from tradingagents.knowledge.ingestion import IngestionMode, KnowledgeIngestor

    harness = ingestion_harness(tmp_path)
    harness.ingestor.run(IngestionMode.INCREMENTAL)
    active = harness.catalog.active_generation()
    assert active is not None
    pointer = harness.catalog.path.parent / "state" / "active-index.json"

    assert json.loads(pointer.read_text(encoding="utf-8")) == active.to_dict()

    pointer.write_text('{"generation_id":"stale"}', encoding="utf-8")
    manager = IndexGenerationManager(
        config=harness.ingestor.config,
        catalog=harness.catalog,
        embedding_spec=harness.embedder.spec,
        vector_writer=VectorIndexWriter(backend=JsonVectorBackend()),
        lexical_writer=LexicalIndexWriter(),
    )
    KnowledgeIngestor(
        harness.ingestor.config,
        scanner=SourceScanner(harness.ingestor.config),
        catalog=harness.catalog,
        parser=FakeParser(),
        embedder=FakeEmbedder(),
        generation_manager=manager,
    )

    assert json.loads(pointer.read_text(encoding="utf-8")) == active.to_dict()


def test_recovery_uses_crash_safe_lock_for_stale_runs_and_excludes_live_writer(tmp_path):
    from tradingagents.knowledge.ingestion import IngestionLockedError, KnowledgeIngestor

    harness = ingestion_harness(tmp_path)
    stale_run_id = "stale-running-run"
    harness.catalog.start_run(stale_run_id)
    stale_lock = harness.catalog.path.parent / "locks" / "index.lock"
    stale_lock.parent.mkdir(parents=True, exist_ok=True)
    stale_lock.write_text("dead process", encoding="ascii")

    recovered = KnowledgeIngestor(
        harness.ingestor.config,
        scanner=SourceScanner(harness.ingestor.config),
        catalog=harness.catalog,
        parser=FakeParser(),
        embedder=FakeEmbedder(),
        generation_manager=harness.ingestor.generation_manager,
    )
    assert stale_run_id not in recovered.catalog.running_run_ids()
    assert recovered.catalog.list_runs()[-1].state is IngestionState.INTERRUPTED

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_ingestion_lock,
        args=(harness.source, harness.catalog.path.parent, ready, release),
    )
    holder.start()
    assert ready.wait(10)
    try:
        with pytest.raises(IngestionLockedError):
            recovered.recover_interrupted_runs()
    finally:
        release.set()
        holder.join(10)
    assert holder.exitcode == 0
