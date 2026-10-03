from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli import phase14b
from tradingagents.knowledge.embeddings import LocalModelUnavailable


@dataclass(frozen=True)
class _Generation:
    generation_id: str = "gen-test"
    population_hash: str = "sha256:" + "a" * 64
    status: str = "VALIDATED"
    vector_ready: bool = True
    lexical_ready: bool = True

    def to_dict(self):
        return {
            "generation_id": self.generation_id,
            "population_hash": self.population_hash,
            "status": self.status,
            "vector_ready": self.vector_ready,
            "lexical_ready": self.lexical_ready,
        }


class _Catalog:
    def __init__(self, generation: _Generation):
        self.generation = generation

    def active_generation(self):
        return self.generation


class _Pipeline:
    def __init__(self, _query_service, drafter, **_kwargs):
        self.drafter = drafter

    def extract(self, _queries, *, max_specs):
        draft_result = self.drafter.draft((), max_specs=max_specs)
        return SimpleNamespace(
            specs=(),
            rejected_rules=(),
            query_count=1,
            retrieved_count=1,
            draft_count=0,
            error_code=draft_result.error_code,
        )


class _Drafter:
    provider = "ollama-local"
    model = "qwen3.5:2b"
    timeout_seconds = 60.0
    max_output_tokens = 512
    context_tokens = 4096

    def __init__(self):
        self.calls = 0

    def draft(self, *_args, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            specs=(),
            provider="ollama-local",
            model="qwen3.5:2b",
            prompt_version="phase14b-source-grounded-draft-v1",
            schema_version="phase14b-strategy-draft-v1",
            timeout_seconds=60.0,
            max_output_tokens=512,
            context_tokens=4096,
            elapsed_seconds=0.2,
            input_tokens=20,
            output_tokens=0,
            finish_reason="stop",
            draft_digest=hashlib.sha256(b"{}").hexdigest(),
            error_code=None,
            error_type=None,
            ok=True,
        )


def _source_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = tuple(tmp_path / name for name in ("phase14a.sqlite3", "hft.sqlite3", "demo.sqlite3"))
    for path in paths:
        path.write_bytes(b"read-only-source-fixture")
    return paths  # type: ignore[return-value]


def _run_args(tmp_path: Path, *, artifact_root: Path | None = None) -> list[str]:
    phase14a, hft, demo = _source_files(tmp_path)
    return [
        "draft-and-evaluate",
        "--knowledge-root",
        str(tmp_path / "knowledge"),
        "--expected-generation",
        "gen-test",
        "--expected-fingerprint",
        phase14b._fingerprint_generation(_Generation()),
        "--phase14a-db",
        str(phase14a),
        "--hft-db",
        str(hft),
        "--demo-db",
        str(demo),
        "--phase14-artifact-root",
        str(artifact_root or (tmp_path / "phase14b")),
        "--model",
        "qwen3.5:2b",
    ]


def _patch_run_preflight(monkeypatch):
    monkeypatch.setattr(
        phase14b,
        "_preflight_replay_sources",
        lambda _paths: {
            "candidate_replay_ready": True,
            "reason_code": None,
            "verified_experience_count": 1,
            "quarantined_experience_excluded": 0,
            "tick_count": 10,
            "segment_count": 1,
            "dataset_fingerprint": "sha256:" + "c" * 64,
            "demo_reconciliation_status": "CLEAN",
        },
    )
    monkeypatch.setattr(phase14b, "_source_commit", lambda: "a" * 40)


def test_status_does_not_construct_a_model_or_drafter(tmp_path, monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("status must not construct a model or drafter")

    monkeypatch.setattr(phase14b, "_create_drafter", forbidden, raising=False)
    assert phase14b.main(["status", "--artifact-root", str(tmp_path / "absent"), "--json"]) == 0
    output = capsys.readouterr().out
    assert "artifact_root_initialized" in output


def test_draft_command_requires_an_explicit_model(tmp_path):
    args = _run_args(tmp_path)
    model_index = args.index("--model")
    del args[model_index : model_index + 2]
    with pytest.raises(SystemExit):
        phase14b.build_parser().parse_args(args)


def test_embedding_model_default_handles_string_knowledge_path(tmp_path):
    args = SimpleNamespace(embedding_model_path=None, knowledge_root=str(tmp_path))
    assert phase14b._embedding_model_path(args) is None


def test_generation_or_fingerprint_mismatch_fails_before_model_or_output(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "phase14b"
    monkeypatch.setattr(phase14b, "_open_knowledge", lambda _path: (_Catalog(_Generation()), _Generation()))
    monkeypatch.setattr(
        phase14b,
        "_create_drafter",
        lambda *_args, **_kwargs: pytest.fail("drafter must not be constructed on pin mismatch"),
    )
    args = _run_args(tmp_path, artifact_root=output)
    args[args.index("--expected-fingerprint") + 1] = "0" * 64
    assert phase14b.main(args) != 0
    assert not output.exists()
    assert "PINNED_GENERATION_MISMATCH" in capsys.readouterr().err


def test_existing_artifact_root_is_refused_before_model_construction(tmp_path, monkeypatch):
    output = tmp_path / "phase14b"
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(phase14b, "_open_knowledge", lambda _path: (_Catalog(_Generation()), _Generation()))
    monkeypatch.setattr(
        phase14b,
        "_create_drafter",
        lambda *_args, **_kwargs: pytest.fail("must refuse existing output before model construction"),
    )
    assert phase14b.main(_run_args(tmp_path, artifact_root=output)) != 0
    assert (output / "keep.txt").read_text(encoding="utf-8") == "preserve"


def test_explicit_local_model_runs_once_and_source_databases_remain_unchanged(
    tmp_path, monkeypatch
):
    output = tmp_path / "phase14b"
    catalog = _Catalog(_Generation())
    monkeypatch.setattr(phase14b, "_open_knowledge", lambda _path: (catalog, catalog.active_generation()))
    query_service = SimpleNamespace(catalog=catalog, search=lambda _request: ())
    monkeypatch.setattr(phase14b, "_build_query_service", lambda *_args, **_kwargs: query_service)
    _patch_run_preflight(monkeypatch)
    drafter = _Drafter()
    seen = {}

    def make_drafter(args):
        seen["endpoint"] = args.ollama_endpoint
        seen["model"] = args.model
        return drafter

    monkeypatch.setattr(phase14b, "_create_drafter", make_drafter)
    monkeypatch.setattr(phase14b, "_create_pipeline", lambda query, drafter, **_kwargs: _Pipeline(query, drafter))
    source_paths = _source_files(tmp_path)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    args = _run_args(tmp_path, artifact_root=output)

    assert phase14b.main(args) == 0
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    assert before == after
    assert seen == {"endpoint": "http://127.0.0.1:11434", "model": "qwen3.5:2b"}
    assert drafter.calls == 1
    report = json.loads((output / "phase14b-run.json").read_text(encoding="utf-8"))
    assert report["provider"] == "ollama-local"
    assert report["model"] == "qwen3.5:2b"
    assert report["source_unchanged"] is True
    serialized = json.dumps(report).lower()
    assert '"prompt"' not in serialized
    assert '"completion"' not in serialized
    assert '"reasoning"' not in serialized


def test_non_loopback_endpoint_is_rejected_before_pipeline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(phase14b, "_open_knowledge", lambda _path: (_Catalog(_Generation()), _Generation()))
    catalog = _Catalog(_Generation())
    monkeypatch.setattr(phase14b, "_build_query_service", lambda *_args, **_kwargs: SimpleNamespace(catalog=catalog, search=lambda _request: ()))
    _patch_run_preflight(monkeypatch)
    monkeypatch.setattr(phase14b, "_create_pipeline", lambda *_args, **_kwargs: pytest.fail("must fail closed"))
    args = _run_args(tmp_path)
    args.extend(["--ollama-endpoint", "https://example.com"])
    assert phase14b.main(args) != 0
    assert "loopback" in capsys.readouterr().err.lower()


def test_output_report_serializes_only_safe_model_telemetry(tmp_path, monkeypatch):
    output = tmp_path / "phase14b"
    monkeypatch.setattr(phase14b, "_open_knowledge", lambda _path: (_Catalog(_Generation()), _Generation()))
    catalog = _Catalog(_Generation())
    monkeypatch.setattr(phase14b, "_build_query_service", lambda *_args, **_kwargs: SimpleNamespace(catalog=catalog, search=lambda _request: ()))
    _patch_run_preflight(monkeypatch)
    monkeypatch.setattr(phase14b, "_create_drafter", lambda _args: _Drafter())
    monkeypatch.setattr(phase14b, "_create_pipeline", lambda query, drafter, **_kwargs: _Pipeline(query, drafter))
    assert phase14b.main(_run_args(tmp_path, artifact_root=output)) == 0
    report = json.loads((output / "phase14b-run.json").read_text(encoding="utf-8"))
    assert set(report["model_telemetry"]) >= {
        "provider",
        "model",
        "elapsed_seconds",
        "input_tokens",
        "output_tokens",
        "finish_reason",
    }
    assert "prompt" not in report
    assert "completion" not in report
    assert "reasoning" not in report
    assert report["llm_calls"] == 1
    assert report["mt5_calls"] == 0


def test_demo_reconciliation_blocker_suppresses_replay_without_exposing_details(
    tmp_path, monkeypatch
):
    from tradingagents.self_enhancement import causal

    monkeypatch.setattr(phase14b, "_read_verified_phase14a", lambda _path: ((object(),), 0))

    def unresolved(_path):
        raise ValueError("DEMO source has unresolved reconciliation evidence")

    monkeypatch.setattr(phase14b, "_validate_demo_source_readonly", unresolved)
    monkeypatch.setattr(
        causal,
        "load_causal_tick_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid_rows=10,
            invalid_rows=0,
            segments=(object(),),
            source_fingerprint="sha256:" + "d" * 64,
        ),
    )
    result = phase14b._preflight_replay_sources(
        {"phase14a": tmp_path / "a", "hft": tmp_path / "h", "demo": tmp_path / "d"}
    )
    assert result["candidate_replay_ready"] is False
    assert result["reason_code"] == "DEMO_RECONCILIATION_REQUIRED"
    assert result["demo_reconciliation_status"] == "RECONCILIATION_REQUIRED"


def test_missing_local_embedding_artifact_fails_before_drafter_or_output(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "phase14b"
    catalog = _Catalog(_Generation())
    monkeypatch.setattr(phase14b, "_open_knowledge", lambda _path: (catalog, catalog.active_generation()))
    _patch_run_preflight(monkeypatch)
    monkeypatch.setattr(
        phase14b,
        "_build_query_service",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LocalModelUnavailable("embedding_model_path must reference a non-empty local model")
        ),
    )
    monkeypatch.setattr(
        phase14b,
        "_create_drafter",
        lambda *_args, **_kwargs: pytest.fail("Ollama must not be invoked without Phase 7 retrieval"),
    )
    assert phase14b.main(_run_args(tmp_path, artifact_root=output)) != 0
    assert not output.exists()
    assert "LocalModelUnavailable" in capsys.readouterr().err
