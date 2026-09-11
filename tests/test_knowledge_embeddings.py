"""Contract tests for local, no-download Phase 7 embedding adapters."""

from __future__ import annotations

import json
import math
import socket
import sys
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from tradingagents.knowledge.config import KnowledgeConfig
from tradingagents.knowledge.embeddings import EmbeddingProvider
from tradingagents.knowledge.models import ChunkRecord, EmbeddingSpec


@dataclass
class FakeTokenizer:
    model_max_input_tokens: int = 512
    special_token_budget: int = 2
    calls: list[dict[str, object]] = field(default_factory=list)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        purpose: str,
        truncation: bool,
    ) -> tuple[str, ...]:
        self.calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "purpose": purpose,
                "truncation": truncation,
            }
        )
        tokens = tuple(text.split())
        if add_special_tokens:
            tokens = ("<s>",) * self.special_token_budget + tokens
        return tokens


def make_embedding_spec(**overrides: object) -> EmbeddingSpec:
    values: dict[str, object] = {
        "model_id": "BAAI/bge-small-en-v1.5",
        "resolved_model_version": "fixture-v1",
        "runtime": "onnx-cpu",
        "artifact_hash": "sha256:fixture-model",
        "dimensions": 3,
        "normalization_policy": "l2",
        "tokenizer_fingerprint": "fixture-tokenizer-v1",
        "model_max_input_tokens": 512,
        "special_token_budget": 2,
        "effective_corpus_content_token_limit": 510,
        "corpus_instruction_policy": "none-v1",
        "corpus_instruction_version": "v1",
        "query_instruction_policy": "bge-search-prefix-v1",
        "query_instruction_version": "v1",
        "truncation": False,
    }
    values.update(overrides)
    return EmbeddingSpec(**values)


class FakeEmbeddingProvider(EmbeddingProvider):
    """Small real-contract fake; the base class performs all validation."""

    def __init__(
        self,
        *,
        dimensions: int = 3,
        spec: EmbeddingSpec | None = None,
        tokenizer: FakeTokenizer | None = None,
        vectors: tuple[tuple[float, ...], ...] | None = None,
        batch_size: int = 16,
    ) -> None:
        self.spec = spec or make_embedding_spec(dimensions=dimensions)
        self.tokenizer = tokenizer or FakeTokenizer(
            model_max_input_tokens=self.spec.model_max_input_tokens,
            special_token_budget=self.spec.special_token_budget,
        )
        self.batch_size = batch_size
        self.vectors = vectors
        self.embed_calls = 0
        self.seen_texts: list[str] = []

    def _embed_formatted(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.embed_calls += 1
        self.seen_texts.extend(texts)
        if self.vectors is not None:
            return self.vectors
        return tuple(tuple(float(index + 1) for index in range(self.spec.dimensions)) for _ in texts)


class RecordingFakeEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.seen_content_tokens: list[int] = []
        self.seen_model_input_tokens: list[int] = []
        self.truncation_requested = False

    def _embed_formatted(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        for text in texts:
            content = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                purpose="corpus",
                truncation=False,
            )
            model_input = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                purpose="corpus",
                truncation=False,
            )
            self.seen_content_tokens.append(len(content))
            self.seen_model_input_tokens.append(len(model_input))
        self.truncation_requested = any(bool(call["truncation"]) for call in self.tokenizer.calls)
        return super()._embed_formatted(texts)


def make_chunk(text: str, *, document_id: str = "doc-fixture", source_hash: str = "source-sha") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"chunk-{text}",
        document_id=document_id,
        source_hash=source_hash,
        text=text,
        content_hash=f"content-{text}",
    )


def test_embedding_provider_rejects_wrong_dimension_and_non_finite_values():
    from tradingagents.knowledge.embeddings import EmbeddingContractError

    wrong_dimension = FakeEmbeddingProvider(dimensions=3, vectors=((1.0, 2.0),))
    with pytest.raises(EmbeddingContractError, match="dimensions"):
        wrong_dimension.embed(("OFI",))

    non_finite = FakeEmbeddingProvider(dimensions=3, vectors=((1.0, math.nan, 3.0),))
    with pytest.raises(EmbeddingContractError, match="finite"):
        non_finite.embed(("OFI",))


def test_embedding_cache_skips_repeat_work_only_for_matching_complete_model_spec(tmp_path):
    from tradingagents.knowledge.embeddings import EmbeddingArtifactStore

    provider = FakeEmbeddingProvider(dimensions=3)
    store = EmbeddingArtifactStore(tmp_path / "embeddings")
    chunks = (make_chunk("VPIN"),)

    first = store.load_or_compute(chunks, provider)
    second = store.load_or_compute(chunks, provider)

    assert provider.embed_calls == 1
    assert first == second

    changed_spec = replace(provider.spec, artifact_hash="sha256:other-local-model")
    changed_provider = FakeEmbeddingProvider(spec=changed_spec)
    regenerated = store.load_or_compute(chunks, changed_provider)
    assert changed_provider.embed_calls == 1
    assert regenerated == first

    sidecars = sorted((tmp_path / "embeddings").rglob("*.json"))
    assert len(sidecars) == 2
    persisted_specs = {
        json.loads(sidecar.read_text(encoding="utf-8"))["embedding_spec"]["artifact_hash"]
        for sidecar in sidecars
    }
    assert persisted_specs == {provider.spec.artifact_hash, changed_spec.artifact_hash}
    assert store.load(chunks, provider) == first
    assert store.load(chunks, changed_provider) == regenerated


def test_cache_reuse_check_rejects_any_complete_spec_mismatch(tmp_path):
    from tradingagents.knowledge.embeddings import (
        EmbeddingArtifactStore,
        EmbeddingVersionMismatch,
    )

    store = EmbeddingArtifactStore(tmp_path / "embeddings")
    chunks = (make_chunk("VPIN"),)
    original = FakeEmbeddingProvider()
    store.load_or_compute(chunks, original)

    incompatible = FakeEmbeddingProvider(
        spec=replace(original.spec, query_instruction_policy="another-query-policy-v1")
    )
    with pytest.raises(EmbeddingVersionMismatch, match="embedding specification"):
        store.load(chunks, incompatible)


def test_cache_reuse_probe_does_not_alias_document_ids_with_dots(tmp_path):
    from tradingagents.knowledge.embeddings import EmbeddingArtifactStore

    store = EmbeddingArtifactStore(tmp_path / "embeddings")
    provider = FakeEmbeddingProvider()
    store.load_or_compute((make_chunk("VPIN", document_id="doc"),), provider)

    with pytest.raises(FileNotFoundError):
        store.load((make_chunk("VPIN", document_id="doc.other"),), provider)


def test_missing_local_model_fails_without_network(tmp_path, monkeypatch):
    from tradingagents.knowledge.embeddings import FastEmbedProvider, LocalModelUnavailable

    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs: pytest.fail("network"))
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(LocalModelUnavailable):
        FastEmbedProvider.from_config(
            KnowledgeConfig(
                source_root=source_root,
                artifact_root=tmp_path / "artifacts",
                embedding_model_path=tmp_path / "missing",
            )
        )


def _install_fake_fastembed(monkeypatch, fake_text_embedding: type[object]) -> None:
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=fake_text_embedding))


def _local_model_config(tmp_path, *, embedding_model_path=None) -> KnowledgeConfig:
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    model_path = embedding_model_path or tmp_path / "local-bge"
    model_path.mkdir(exist_ok=True)
    (model_path / "tokenizer.json").write_text('{"version": 1}', encoding="utf-8")
    (model_path / "config.json").write_text('{"max_position_embeddings": 512}', encoding="utf-8")
    return KnowledgeConfig(
        source_root=source_root,
        artifact_root=tmp_path / "artifacts",
        embedding_model_path=model_path,
        embedding_dimensions=3,
    )


def test_fastembed_binds_the_exact_configured_local_model_path_without_network(tmp_path, monkeypatch):
    from tradingagents.knowledge.embeddings import FastEmbedProvider

    class LocalTokenizer:
        model_max_length = 512

        def encode(self, text, *, add_special_tokens, truncation=False):
            return ("<s>", "</s>") + tuple(text.split()) if add_special_tokens else tuple(text.split())

    class LocalOnlyTextEmbedding:
        calls: list[dict[str, object]] = []

        def __init__(
            self,
            *,
            model_name,
            specific_model_path,
            local_files_only,
            cache_dir,
            threads,
            providers,
        ):
            type(self).calls.append(locals().copy())
            self.tokenizer = LocalTokenizer()
            self.embedding_size = 3

        def embed(self, texts, *, batch_size):
            return ((1.0, 2.0, 3.0) for _ in texts)

    config = _local_model_config(tmp_path)
    _install_fake_fastembed(monkeypatch, LocalOnlyTextEmbedding)
    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs: pytest.fail("network"))

    provider = FastEmbedProvider.from_config(config)

    assert provider.spec.artifact_hash.startswith("sha256:")
    assert LocalOnlyTextEmbedding.calls[0]["specific_model_path"] == str(config.embedding_model_path)
    assert LocalOnlyTextEmbedding.calls[0]["local_files_only"] is True
    assert LocalOnlyTextEmbedding.calls[0]["providers"] == ["CPUExecutionProvider"]


def test_fastembed_rejects_adapter_without_exact_local_path_argument(tmp_path, monkeypatch):
    from tradingagents.knowledge.embeddings import FastEmbedProvider, LocalModelUnavailable

    class CacheOnlyTextEmbedding:
        constructed = False

        def __init__(self, *, model_name, cache_dir, local_files_only, threads, providers):
            type(self).constructed = True

    _install_fake_fastembed(monkeypatch, CacheOnlyTextEmbedding)

    with pytest.raises(LocalModelUnavailable, match="specific_model_path"):
        FastEmbedProvider.from_config(_local_model_config(tmp_path))
    assert CacheOnlyTextEmbedding.constructed is False


def test_tokenizer_fingerprint_hashes_tokenizer_asset_contents(tmp_path, monkeypatch):
    from tradingagents.knowledge.embeddings import FastEmbedProvider

    class LocalTokenizer:
        model_max_length = 512

        def encode(self, text, *, add_special_tokens, truncation=False):
            return ("<s>", "</s>") + tuple(text.split()) if add_special_tokens else tuple(text.split())

    class LocalOnlyTextEmbedding:
        def __init__(
            self,
            *,
            model_name,
            specific_model_path,
            local_files_only,
            cache_dir,
            threads,
            providers,
        ):
            self.tokenizer = LocalTokenizer()
            self.embedding_size = 3

    _install_fake_fastembed(monkeypatch, LocalOnlyTextEmbedding)
    config = _local_model_config(tmp_path)
    first = FastEmbedProvider.from_config(config).spec.tokenizer_fingerprint
    (config.embedding_model_path / "tokenizer.json").write_text('{"version": 2}', encoding="utf-8")
    second = FastEmbedProvider.from_config(config).spec.tokenizer_fingerprint

    assert first != second


def test_embedding_rejects_over_limit_without_truncation():
    from tradingagents.knowledge.embeddings import EmbeddingInputTooLong

    provider = FakeEmbeddingProvider(
        spec=make_embedding_spec(
            model_max_input_tokens=512,
            effective_corpus_content_token_limit=510,
        ),
        tokenizer=FakeTokenizer(model_max_input_tokens=512, special_token_budget=2),
    )
    with pytest.raises(EmbeddingInputTooLong):
        provider.embed(("token " * 511,), purpose="corpus")
    assert all(call["truncation"] is False for call in provider.tokenizer.calls)
    assert provider.embed_calls == 0


def test_embedding_spec_persists_tokenizer_limit_and_instruction_policies():
    spec = make_embedding_spec(
        model_id="BAAI/bge-small-en-v1.5",
        dimensions=384,
        model_max_input_tokens=512,
        effective_corpus_content_token_limit=510,
        tokenizer_fingerprint="tok-v1",
        corpus_instruction_policy="none-v1",
        query_instruction_policy="bge-search-prefix-v1",
    )
    assert spec.model_max_input_tokens == 512
    assert spec.effective_corpus_content_token_limit == 510
    assert spec.tokenizer_fingerprint == "tok-v1"
    assert spec.query_instruction_policy == "bge-search-prefix-v1"
    assert spec.to_dict()["resolved_model_version"] == spec.embedding_model_version
    assert spec.to_dict()["artifact_hash"] == spec.embedding_artifact_hash


def test_allowed_embedding_input_reaches_model_without_truncation():
    provider = RecordingFakeEmbeddingProvider(
        spec=make_embedding_spec(
            model_max_input_tokens=512,
            effective_corpus_content_token_limit=510,
        ),
        tokenizer=FakeTokenizer(model_max_input_tokens=512, special_token_budget=2),
    )
    text = "token " * 510
    provider.embed((text,), purpose="corpus")
    assert provider.seen_content_tokens == [510]
    assert provider.seen_model_input_tokens == [512]
    assert provider.truncation_requested is False


def test_bge_query_prefix_is_versioned_and_tokenized_before_model_invocation():
    provider = FakeEmbeddingProvider()

    provider.embed(("order flow imbalance",), purpose="query")

    assert provider.seen_texts == [
        "Represent this sentence for searching relevant passages: order flow imbalance"
    ]
    assert any(
        call["text"] == provider.seen_texts[0]
        and call["add_special_tokens"] is True
        and call["purpose"] == "query"
        and call["truncation"] is False
        for call in provider.tokenizer.calls
    )


def test_provider_splits_model_calls_at_bounded_batch_size():
    provider = FakeEmbeddingProvider(batch_size=2)

    vectors = provider.embed(("one", "two", "three", "four", "five"))

    assert len(vectors) == 5
    assert provider.embed_calls == 3
