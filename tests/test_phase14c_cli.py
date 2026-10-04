from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli import phase14c
from tradingagents.self_enhancement.phase14c_models import Phase14CSourcePaths
from tradingagents.self_enhancement.phase14c_runner import (
    ArtifactRootValidationError,
    validate_discovery_paths,
)


def _block_runtime_imports(monkeypatch: pytest.MonkeyPatch, *, block_runner: bool = True) -> None:
    original = builtins.__import__
    blocked = [
        "tradingagents.knowledge.embeddings",
        "tradingagents.self_enhancement.book_drafter",
        "MetaTrader5",
    ]
    if block_runner:
        blocked.extend(
            (
                "tradingagents.self_enhancement.phase14c_runner",
                "tradingagents.self_enhancement.orchestrator",
            )
        )

    def guarded_import(name, *args, **kwargs):
        if any(name == value or name.startswith(value + ".") for value in blocked):
            raise AssertionError(f"forbidden runtime import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_status_reads_only_manifest_and_report_without_runtime_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    manifest = {"schema_version": "phase14c-run.v1", "status": "COMPLETE"}
    report = {"status": "COMPLETE", "selected_group_count": 3}
    (root / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "phase14c-report.json").write_text(json.dumps(report), encoding="utf-8")
    _block_runtime_imports(monkeypatch)

    code = phase14c.main(["status", "--artifact-root", str(root), "--json"])

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "artifact_root": str(root.resolve()),
        "manifest": {
            "schema_version": "phase14c-run.v1",
            "status": "COMPLETE",
            "identity_fingerprint": None,
        },
        "report": {"status": "COMPLETE", "selected_group_count": 3},
        "status": "COMPLETE",
    }


def test_plan_only_uses_pinned_deterministic_plan_and_creates_no_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    embedding = tmp_path / "embedding"
    embedding.mkdir()
    artifact = tmp_path / "must-not-exist"
    invoked = []

    def fake_plan(**kwargs):
        invoked.append(kwargs)
        return {
            "status": "PLANNED",
            "raw_hit_count": 12,
            "selected_group_count": 4,
            "deferred_group_count": 1,
            "llm_calls": 0,
            "conservative_estimated_runtime_seconds": 600,
        }

    monkeypatch.setattr(phase14c, "plan_phase14c", fake_plan)
    _block_runtime_imports(monkeypatch)

    code = phase14c.main(
        [
            "plan-only",
            "--knowledge-root",
            str(knowledge),
            "--expected-generation",
            "gen-fixture",
            "--expected-fingerprint",
            "a" * 64,
            "--expected-population-hash",
            "b" * 64,
            "--embedding-model-path",
            str(embedding),
        ]
    )

    assert code == 0
    assert invoked and invoked[0]["knowledge_root"] == knowledge.resolve()
    assert invoked[0]["expected_generation_id"] == "gen-fixture"
    assert invoked[0]["expected_generation_fingerprint"] == "a" * 64
    assert invoked[0]["expected_population_hash"] == "b" * 64
    assert not artifact.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["llm_calls"] == 0
    assert payload["deferred_group_count"] == 1


def test_plan_phase14c_runs_read_only_retrieval_with_local_embedder_and_no_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradingagents.knowledge import embeddings, query, vector_index
    from tradingagents.self_enhancement import book_atomic_extraction, phase14c_runner as runner
    from tradingagents.self_enhancement.phase14c_models import (
        CorpusInventory,
        Phase14CGenerationPin,
    )

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "catalog.sqlite3").write_bytes(b"readonly fixture")
    embedding_path = tmp_path / "local-embedding"
    embedding_path.mkdir()
    (embedding_path / "model.onnx").write_bytes(b"local model")
    spec_payload = {"model_id": "fixture", "resolved_model_version": "v1"}
    generation = SimpleNamespace(
        generation_id="gen-1",
        embedding_spec=SimpleNamespace(to_dict=lambda: spec_payload),
        vector_location=tmp_path / "vector",
        lexical_location=tmp_path / "lexical.sqlite3",
    )
    pin = Phase14CGenerationPin(
        generation_id="gen-1",
        generation_fingerprint="a" * 64,
        population_hash="sha256:" + "b" * 64,
        status="VALIDATED",
        vector_ready=True,
        lexical_ready=True,
        vector_location=tmp_path / "vector",
        lexical_location=tmp_path / "lexical.sqlite3",
        inventory_schema_version="phase14c-inventory.v1",
        document_count=2,
        chunk_count=3,
    )
    inventory = CorpusInventory(
        unique_source_count=2,
        indexed_document_count=2,
        queryable_document_count=2,
        alias_count=0,
        needs_ocr_count=0,
        unavailable_resource_count=0,
        active_chunk_count=3,
        source_manifest_fingerprint="c" * 64,
    )
    monkeypatch.setattr(
        runner,
        "read_phase7_inventory",
        lambda *args, **kwargs: (pin, inventory),
    )
    monkeypatch.setattr(
        runner,
        "_ReadOnlyKnowledgeCatalog",
        lambda _path: SimpleNamespace(active_generation=lambda: generation),
    )
    monkeypatch.setattr(runner, "_knowledge_config_for_generation", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        embeddings.FastEmbedProvider,
        "from_config",
        classmethod(lambda cls, _config: SimpleNamespace(spec=SimpleNamespace(to_dict=lambda: spec_payload))),
    )
    monkeypatch.setattr(query, "KnowledgeQueryService", lambda *args: object())
    monkeypatch.setattr(vector_index, "VectorIndexReader", lambda _path: object())
    monkeypatch.setattr(runner, "_ReadOnlyLexicalIndexReader", lambda _path: object())
    monkeypatch.setattr(runner, "build_discovery_query_bank", lambda: ("query",))
    monkeypatch.setattr(runner, "retrieve_discovery_evidence", lambda *args, **kwargs: SimpleNamespace(
        status="COMPLETE",
        query_bank_version="query-v1",
        query_bank_fingerprint="d" * 64,
        completed_query_count=1,
        raw_hit_count=12,
        unique_hit_count=9,
    ))
    monkeypatch.setattr(runner, "select_diverse_evidence", lambda _report: SimpleNamespace(
        selected_group_count=4,
        deferred_group_count=2,
        unclassified_chunk_count=1,
        selected_groups_by_family={"momentum": 4},
        deferred_groups_by_family={"momentum": 2},
    ))
    monkeypatch.setattr(
        book_atomic_extraction,
        "AtomicStrategyExtractor",
        lambda *args, **kwargs: pytest.fail("plan-only must not construct the local teacher"),
    )
    before = (knowledge / "catalog.sqlite3").read_bytes()

    result = runner.plan_phase14c(
        knowledge_root=knowledge,
        expected_generation_id="gen-1",
        expected_generation_fingerprint="a" * 64,
        expected_population_hash="sha256:" + "b" * 64,
        embedding_model_path=embedding_path,
        timeout_seconds=25,
    )

    assert result["llm_calls"] == 0
    assert result["selected_group_count"] == 4
    assert result["deferred_group_count"] == 2
    assert result["conservative_estimated_teacher_calls"] == 60
    assert result["conservative_estimated_runtime_seconds"] == 1500
    assert (knowledge / "catalog.sqlite3").read_bytes() == before
    assert sorted(path.name for path in knowledge.iterdir()) == ["catalog.sqlite3"]


def test_discover_rejects_non_loopback_before_constructing_model_or_touching_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    cache = tmp_path / "external-cache.sqlite3"
    sources = [tmp_path / f"{name}.sqlite3" for name in ("phase14a", "hft", "demo")]
    for source in sources:
        source.write_bytes(b"source")
    artifact = tmp_path / "new-run"

    def forbidden_model(*args, **kwargs):
        raise AssertionError("model must not be constructed before endpoint validation")

    monkeypatch.setattr(phase14c, "create_local_teacher", forbidden_model)
    code = phase14c.main(
        [
            "discover",
            "--knowledge-root", str(knowledge),
            "--expected-generation", "gen-1",
            "--expected-fingerprint", "a" * 64,
            "--expected-population-hash", "b" * 64,
            "--phase14c-artifact-root", str(artifact),
            "--phase14a-db", str(sources[0]),
            "--hft-db", str(sources[1]),
            "--demo-db", str(sources[2]),
            "--embedding-model-path", str(model),
            "--atomic-cache-path", str(cache),
            "--model", "qwen3.5:2b",
            "--model-version", "sha256:fixture",
            "--ollama-endpoint", "http://example.com/v1",
        ]
    )

    assert code == 2
    assert "loopback" in capsys.readouterr().err.lower()
    assert not artifact.exists()


def test_missing_required_discovery_paths_fail_at_argument_boundary(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        phase14c.main(["discover", "--model", "qwen3.5:2b"])

    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err.lower()


def test_existing_discovery_root_is_preserved_and_model_is_not_constructed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache = cache_root / "cache.sqlite3"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    sources = [source_root / f"{name}.sqlite3" for name in ("phase14a", "hft", "demo")]
    for source in sources:
        source.write_bytes(b"source")
    artifact = tmp_path / "existing"
    artifact.mkdir()
    sentinel = artifact / "user-file.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    def forbidden_model(*args, **kwargs):
        raise AssertionError("existing-root failure must precede model construction")

    monkeypatch.setattr(phase14c, "create_local_teacher", forbidden_model)
    code = phase14c.main(
        [
            "discover",
            "--knowledge-root", str(knowledge),
            "--expected-generation", "gen-1",
            "--expected-fingerprint", "a" * 64,
            "--expected-population-hash", "b" * 64,
            "--phase14c-artifact-root", str(artifact),
            "--phase14a-db", str(sources[0]),
            "--hft-db", str(sources[1]),
            "--demo-db", str(sources[2]),
            "--embedding-model-path", str(model),
            "--atomic-cache-path", str(cache),
            "--model", "qwen3.5:2b",
            "--model-version", "sha256:fixture",
            "--ollama-endpoint", "http://localhost:11434",
        ]
    )

    assert code == 2
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in artifact.iterdir()) == ["user-file.txt"]


def _path_set(tmp_path: Path):
    knowledge = tmp_path / "knowledge"
    embedding = tmp_path / "embedding"
    output = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    source_dir = tmp_path / "sources"
    for path in (knowledge, embedding, cache_dir, source_dir):
        path.mkdir()
    (embedding / "model.onnx").write_bytes(b"model")
    sources = [source_dir / f"{name}.sqlite3" for name in ("phase14a", "hft", "demo")]
    for source in sources:
        source.write_bytes(b"source")
    return knowledge, embedding, output, cache_dir / "atomic.sqlite3", Phase14CSourcePaths(*sources)


@pytest.mark.parametrize("cache_under", ["knowledge", "source"])
def test_discovery_rejects_atomic_cache_inside_phase7_or_source_tree(
    tmp_path: Path, cache_under: str
) -> None:
    knowledge, embedding, output, cache, sources = _path_set(tmp_path)
    cache = (knowledge if cache_under == "knowledge" else sources.phase14a_path.parent) / "atomic.sqlite3"

    with pytest.raises(ArtifactRootValidationError, match="atomic cache overlaps"):
        validate_discovery_paths(
            knowledge_root=knowledge,
            embedding_model_path=embedding,
            artifact_root=output,
            atomic_cache_path=cache,
            source_paths=sources,
        )


def test_discovery_rejects_artifact_root_overlapping_frozen_phase7(
    tmp_path: Path,
) -> None:
    knowledge, embedding, _output, cache, sources = _path_set(tmp_path)

    with pytest.raises(ArtifactRootValidationError, match="overlaps protected data"):
        validate_discovery_paths(
            knowledge_root=knowledge,
            embedding_model_path=embedding,
            artifact_root=knowledge / "phase14c-run",
            atomic_cache_path=cache,
            source_paths=sources,
        )


def test_evaluate_uses_integrity_checked_artifacts_without_model_or_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    artifact = tmp_path / "run"
    artifact.mkdir()
    (artifact / "run-manifest.json").write_text(
        json.dumps({"schema_version": "phase14c-run.v1", "status": "COMPLETE"}),
        encoding="utf-8",
    )
    specs_bytes = json.dumps(
        {"schema_version": "phase14c-validated-specs.v1", "specs": []},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    (artifact / "validated-specs.json").write_bytes(specs_bytes)
    import hashlib

    identity = {"generation_id": "gen-1"}
    identity_fingerprint = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (artifact / "run-manifest.json").write_text(
        json.dumps({"schema_version": "phase14c-run.v1", "status": "COMPLETE", "identity": identity, "identity_fingerprint": identity_fingerprint}),
        encoding="utf-8",
    )
    (artifact / "phase14c-report.json").write_text(
        json.dumps({
            "status": "COMPLETE",
            "artifact_integrity": "VALID",
            "validated_specs_file": "validated-specs.json",
            "validated_specs_sha256": hashlib.sha256(specs_bytes).hexdigest(),
        }),
        encoding="utf-8",
    )
    sources = [tmp_path / f"{name}.sqlite3" for name in ("phase14a", "hft", "demo")]
    for source in sources:
        source.write_bytes(b"source")
    invoked = []

    def fake_evaluate(specs, **kwargs):
        invoked.append((specs, kwargs))
        return ()

    monkeypatch.setattr(phase14c, "evaluate_phase14c_candidates", fake_evaluate)
    _block_runtime_imports(monkeypatch, block_runner=False)
    code = phase14c.main(
        [
            "evaluate",
            "--artifact-root", str(artifact),
            "--phase14a-db", str(sources[0]),
            "--hft-db", str(sources[1]),
            "--demo-db", str(sources[2]),
            "--source-commit", "fixture-commit",
        ]
    )

    assert code == 0
    assert invoked[0][0] == ()
    assert invoked[0][1]["source_paths"].phase14a_path == sources[0].resolve()


def test_status_missing_root_returns_missing_without_creating_it(tmp_path: Path, capsys) -> None:
    root = tmp_path / "absent"
    code = phase14c.main(["status", "--artifact-root", str(root), "--json"])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "MISSING"
    assert not root.exists()
