from __future__ import annotations

import json
from pathlib import Path

import pytest
import tomllib

from tradingagents.knowledge.models import ContentType, EmbeddingSpec, IndexGeneration, KnowledgeHit


def forbidden_constructor():
    class Forbidden:
        calls = 0

        def __init__(self, *args, **kwargs):
            type(self).calls += 1
            raise AssertionError("forbidden component was constructed")

    return Forbidden


def test_status_does_not_construct_parser_or_embedding_provider(tmp_path, monkeypatch, capsys):
    from tradingagents.knowledge.cli import main

    forbidden = forbidden_constructor()
    monkeypatch.setattr("tradingagents.knowledge.cli.KnowledgeIngestor", forbidden)
    monkeypatch.setattr("tradingagents.knowledge.cli.EmbeddingProvider", forbidden)

    assert main(["status", "--artifact-root", str(tmp_path)]) == 0
    assert "INDEXED" in capsys.readouterr().out
    assert forbidden.calls == 0


@pytest.mark.parametrize("command", ["status", "list", "document", "quarantine"])
def test_metadata_commands_construct_no_parser_embedder_ingestor_or_writers(command, tmp_path, monkeypatch):
    from tradingagents.knowledge.cli import main

    forbidden = forbidden_constructor()
    for name in (
        "DocumentParser",
        "EmbeddingProvider",
        "KnowledgeIngestor",
        "SourceScanner",
        "VectorIndexWriter",
        "LexicalIndexWriter",
    ):
        monkeypatch.setattr(f"tradingagents.knowledge.cli.{name}", forbidden)

    argv = [command, "--artifact-root", str(tmp_path)]
    if command == "document":
        argv.insert(1, "doc-test")
    assert main(argv) == 0
    assert forbidden.calls == 0


def _spec() -> EmbeddingSpec:
    return EmbeddingSpec(
        model_id="fixture/bge",
        resolved_model_version="fixture-v1",
        runtime="onnx-cpu",
        artifact_hash="sha256:fixture-model",
        dimensions=3,
        normalization_policy="l2",
        tokenizer_fingerprint="sha256:fixture-tokenizer",
        model_max_input_tokens=512,
        special_token_budget=2,
        effective_corpus_content_token_limit=510,
        corpus_instruction_policy="none-v1",
        corpus_instruction_version="v1",
        query_instruction_policy="bge-search-prefix-v1",
        query_instruction_version="v1",
        truncation=False,
    )


def test_query_config_allows_local_model_below_user_profile(monkeypatch):
    from tradingagents.knowledge.cli import _query_config

    artifact_root = Path.cwd() / "knowledge-query-artifacts"
    model_path = Path.home() / "prefetched-knowledge-model"
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MODEL_PATH", str(model_path))

    config = _query_config(artifact_root, _spec())

    assert config.embedding_model_path == model_path.resolve(strict=False)
    assert config.source_root not in (Path.home(), artifact_root)
    assert config.artifact_root == artifact_root.resolve(strict=False)
    with pytest.raises(ValueError):
        config.embedding_model_path.relative_to(config.source_root)


class _Catalog:
    def __init__(self, *_args, **_kwargs):
        self.generation = IndexGeneration(
            generation_id="gen-1",
            vector_location=Path("vector") / "lancedb" / "gen-1",
            lexical_location=Path("keyword") / "gen-1" / "bm25.sqlite3",
            embedding_spec=_spec(),
            population_hash="population",
            population_identity=(
                '{"chunk_population_hash":"chunks","document_population_hash":"documents",'
                '"population_hash":"population"}'
            ),
            vector_ready=True,
            lexical_ready=True,
            status="VALIDATED",
            component_versions={"vector": "fixture", "lexical": "fixture"},
        )

    def active_generation(self):
        return self.generation


class _Reader:
    def __init__(self, *_args, **_kwargs):
        pass


class _Embedder:
    calls = 0

    @classmethod
    def from_config(cls, _config):
        cls.calls += 1
        return cls()


class _Service:
    def __init__(self, *_args, **_kwargs):
        self.request = None

    def search(self, request):
        self.request = request
        return (
            KnowledgeHit(
                chunk_id="chunk-1",
                document_id="doc-1",
                content_type=ContentType.EQUATION,
                text="order flow imbalance equation",
                score=1.0,
                source_hash="sha256:document",
                source_filename="doc-1.pdf",
                source_relative_path="books/doc-1.pdf",
                parser_version="fixture-parser-v1",
                chunker_version="structure-v2",
                index_version="index-v1",
                extra={"projection_generation": "gen-1"},
            ),
        )


def _search_doubles(monkeypatch):
    monkeypatch.setattr("tradingagents.knowledge.cli.KnowledgeCatalog", _Catalog)
    monkeypatch.setattr("tradingagents.knowledge.cli.VectorIndexReader", _Reader)
    monkeypatch.setattr("tradingagents.knowledge.cli.LexicalIndexReader", _Reader)
    monkeypatch.setattr("tradingagents.knowledge.cli.EmbeddingProvider", _Embedder)
    monkeypatch.setattr("tradingagents.knowledge.cli.KnowledgeQueryService", _Service)


def test_search_prints_provenance_and_supports_content_type_filter(tmp_path, monkeypatch, capsys):
    from tradingagents.knowledge.cli import main

    _search_doubles(monkeypatch)
    assert main([
        "search", "order flow imbalance", "--content-type", "EQUATION", "--json",
        "--artifact-root", str(tmp_path),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["chunk_id"]
    assert payload[0]["source_hash"]
    assert payload[0]["content_type"] == "EQUATION"


def test_search_constructs_only_read_only_local_query_embedder(tmp_path, monkeypatch):
    from tradingagents.knowledge.cli import main

    _search_doubles(monkeypatch)
    forbidden = forbidden_constructor()
    for name in (
        "DocumentParser",
        "KnowledgeIngestor",
        "SourceScanner",
        "VectorIndexWriter",
        "LexicalIndexWriter",
    ):
        monkeypatch.setattr(f"tradingagents.knowledge.cli.{name}", forbidden)
    _Embedder.calls = 0

    assert main(["search", "order flow imbalance", "--artifact-root", str(tmp_path)]) == 0
    assert _Embedder.calls == 1
    assert forbidden.calls == 0


def test_search_rejects_unimplemented_include_stale(tmp_path, monkeypatch):
    """Do not expose stale retrieval before the catalog has a safe path for it."""

    from tradingagents.knowledge.cli import main

    _search_doubles(monkeypatch)
    with pytest.raises(SystemExit) as error:
        main([
            "search", "order flow imbalance", "--include-stale", "--artifact-root", str(tmp_path),
        ])
    assert error.value.code == 2


def test_index_refuses_an_artifact_root_inside_source_root(tmp_path, capsys):
    from tradingagents.knowledge.cli import main

    source_root = tmp_path / "source"
    source_root.mkdir()
    assert main([
        "index", "--source-root", str(source_root), "--artifact-root", str(source_root / "cache"),
    ]) == 2
    assert "outside source_root" in capsys.readouterr().err


def test_stock_console_entrypoint_is_unchanged():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["tradingagents"] == "cli.main:app"
    assert project["project"]["scripts"]["knowledge"] == "tradingagents.knowledge.cli:main"
