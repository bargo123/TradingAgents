from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli import phase14b
from tradingagents.forex.hft.demo_store import DemoExecutionStore
from tradingagents.knowledge.embeddings import LocalModelUnavailable
from tradingagents.self_enhancement import book_drafter


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


def test_cli_passes_explicit_model_version_to_local_drafter(monkeypatch, tmp_path):
    captured = {}

    class _DrafterFactory:
        def __init__(self, endpoint, model, **kwargs):
            captured.update(endpoint=endpoint, model=model, **kwargs)

    monkeypatch.setattr(book_drafter, "OllamaStrategyDrafter", _DrafterFactory)
    args = phase14b.build_parser().parse_args(
        _run_args(tmp_path)
        + ["--model-version", "sha256:qwen-local-revision"]
    )
    args._resolved_atomic_cache_path = str(tmp_path / "phase14b" / "atomic.sqlite3")

    phase14b._create_drafter(args)

    assert captured["model_version"] == "sha256:qwen-local-revision"
    assert captured["cache_path"] == args._resolved_atomic_cache_path


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


def test_draft_performance_reports_stage_latency_tokens_and_failures_only():
    result = phase14b._draft_performance(
        {
            "cache_hits": 1,
            "cache_write_failures": 0,
            "call_telemetry": [
                {
                    "stage": "PRESENCE",
                    "cache_hit": False,
                    "elapsed_seconds": 1.5,
                    "input_tokens": 40,
                    "output_tokens": 8,
                    "retry_depth": 0,
                    "outcome": "SUCCESS",
                },
                {
                    "stage": "ATOMIC_RULE_EXTRACTION",
                    "cache_hit": False,
                    "elapsed_seconds": 2.5,
                    "input_tokens": 60,
                    "output_tokens": 20,
                    "retry_depth": 1,
                    "outcome": "TRUNCATED",
                },
                {
                    "stage": "CONCEPT_GROUPING",
                    "cache_hit": True,
                    "elapsed_seconds": 0,
                    "input_tokens": None,
                    "output_tokens": None,
                    "retry_depth": 0,
                    "outcome": "CACHE_HIT",
                },
            ],
        }
    )

    assert result["model_calls"] == 2
    assert result["cache_hits"] == 1
    assert result["input_tokens_total"] == 100
    assert result["output_tokens_total"] == 28
    assert result["runtime_seconds_total"] == 4.0
    assert result["truncations"] == 1
    assert result["retry_calls"] == 1
    assert result["stage_call_counts"] == {"ATOMIC_RULE_EXTRACTION": 1, "PRESENCE": 1}
    assert "prompt" not in result and "completion" not in result and "reasoning" not in result


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


def test_persisted_reconciliation_resolution_supersedes_older_required_events(tmp_path):
    database = tmp_path / "demo.sqlite3"
    store = DemoExecutionStore(database)
    store.initialize()
    event_ids = []
    for _ in range(3):
        event_ids.append(store.record_reconciliation(
            status="RECONCILIATION_REQUIRED",
            reason="prior broker position observation was missing",
            details={"ticket": 152717255467, "symbol": "EURUSD"},
        ))
    event_ids.append(store.record_reconciliation(
        status="RECONCILED",
        reason="authoritative broker history confirmed closed position",
        details={"ticket": 152717255467, "symbol": "EURUSD", "terminal_state": "CLOSED"},
    ))
    with sqlite3.connect(database) as connection:
        for event_id, timestamp in zip(
            event_ids,
            (
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:01+00:00",
                "2026-01-01T00:00:02+00:00",
                "2026-01-01T00:00:03+00:00",
            ),
            strict=True,
        ):
            connection.execute(
                "UPDATE demo_reconciliation SET observed_at=? WHERE event_id=?",
                (timestamp, event_id),
            )

    phase14b._validate_demo_source_readonly(database)


def test_latest_unresolved_reconciliation_event_still_blocks_replay(tmp_path):
    database = tmp_path / "demo.sqlite3"
    store = DemoExecutionStore(database)
    store.initialize()
    store.record_reconciliation(
        status="RECONCILIATION_REQUIRED",
        reason="broker history was unavailable",
        details={"ticket": 152717255467, "symbol": "EURUSD"},
    )

    with pytest.raises(ValueError, match="unresolved reconciliation"):
        phase14b._validate_demo_source_readonly(database)


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
