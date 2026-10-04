from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tradingagents.self_enhancement.phase14c_models import Phase14CRunIdentity
from tradingagents.self_enhancement.phase14c_runner import (
    Phase14CResumeError,
    append_resume_result,
    load_resume_results,
    validate_resume_identity,
    write_run_manifest,
)


def _identity(**changes) -> Phase14CRunIdentity:
    identity = Phase14CRunIdentity(
        generation_id="gen-1",
        generation_fingerprint="a" * 64,
        population_hash="b" * 64,
        embedding_spec_fingerprint="c" * 64,
        query_bank_version="query-v1",
        query_bank_fingerprint="d" * 64,
        top_k_per_formulation=10,
        selection_version="selection-v1",
        selection_config_fingerprint="e" * 64,
        assembly_version="assembly-v1",
        assembly_schema_version="spec-v1",
        artifact_schema_fingerprint="f" * 64,
        provider="ollama-local",
        ollama_endpoint="http://localhost:11434",
        model_id="qwen3.5:2b",
        model_version="sha256:fixture-model",
        temperature=0.0,
        timeout_seconds=30.0,
        max_output_tokens=1024,
        context_tokens=8192,
        atomic_cache_path=Path("C:/phase14b/atomic.sqlite3"),
        atomic_cache_schema_version="atomic-v1",
        presence_prompt_version="presence-prompt-v1",
        presence_schema_version="presence-schema-v1",
        extraction_prompt_version="extract-prompt-v1",
        segmentation_version="segmentation-v1",
        stage_budgets_fingerprint="1" * 64,
        resume_schema_version="phase14c-resume.v2",
        source_paths={
            "phase14a": "C:/sources/phase14a.sqlite3",
            "hft": "C:/sources/hft.sqlite3",
            "demo": "C:/sources/demo.sqlite3",
        },
        source_fingerprints={
            "phase14a": "1" * 64,
            "hft": "2" * 64,
            "demo": "3" * 64,
        },
    )
    return replace(identity, **changes)


def test_resume_accepts_exact_run_identity(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    identity = _identity()
    write_run_manifest(root, identity)

    manifest = validate_resume_identity(root, identity)

    assert manifest["status"] == "IN_PROGRESS"
    assert manifest["identity"] == identity.to_dict()
    assert manifest["identity_fingerprint"] == identity.fingerprint


def test_new_run_manifest_never_overwrites_an_existing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    write_run_manifest(root, _identity())
    before = (root / "run-manifest.json").read_bytes()

    with pytest.raises(Phase14CResumeError, match="refusing overwrite"):
        write_run_manifest(root, _identity(model_id="qwen3.5:2b-new"))

    assert (root / "run-manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "field, value",
    [
        ("generation_id", "gen-2"),
        ("model_id", "qwen3.5:2b-instruct"),
        ("query_bank_fingerprint", "e" * 64),
        ("top_k_per_formulation", 11),
        ("selection_version", "selection-v2"),
        ("selection_config_fingerprint", "f" * 64),
        ("assembly_schema_version", "spec-v2"),
        ("artifact_schema_fingerprint", "0" * 64),
        ("ollama_endpoint", "http://127.0.0.1:11435"),
        ("max_output_tokens", 2048),
        ("context_tokens", 16384),
        ("presence_prompt_version", "presence-prompt-v2"),
        ("stage_budgets_fingerprint", "2" * 64),
        ("atomic_cache_path", Path("C:/other/atomic.sqlite3")),
        (
            "source_fingerprints",
            {"phase14a": "4" * 64, "hft": "2" * 64, "demo": "3" * 64},
        ),
    ],
)
def test_resume_rejects_every_changed_identity_component(
    tmp_path: Path, field: str, value: object
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    write_run_manifest(root, _identity())

    with pytest.raises(Phase14CResumeError, match="identity"):
        validate_resume_identity(root, _identity(**{field: value}))


def test_completed_resume_keys_are_not_appended_twice(tmp_path: Path) -> None:
    path = tmp_path / "resume-results.jsonl"
    result = {"status": "VALIDATED", "result_id": "spec-1", "reason_code": "SUPPORTED"}

    assert append_resume_result(path, "a" * 64, result) is True
    first = path.read_bytes()
    assert append_resume_result(path, "a" * 64, result) is False
    assert path.read_bytes() == first
    assert load_resume_results(path) == ({"cache_key": "a" * 64, **result},)


def test_resume_result_round_trips_only_bounded_scalar_telemetry(tmp_path: Path) -> None:
    path = tmp_path / "resume-results.jsonl"
    telemetry = {
        "stage": "PRESENCE",
        "evidence_count": 2,
        "input_tokens": 32,
        "output_tokens": 3,
        "elapsed_seconds": 0.25,
        "finish_reason": "stop",
        "outcome": "VALIDATED",
        "retry_depth": 0,
        "cache_hit": False,
        "error_type": None,
    }
    result = {"status": "ACTIONABLE", "result_id": "group-1", "telemetry": [telemetry]}

    assert append_resume_result(path, "a" * 64, result) is True
    assert load_resume_results(path) == ({"cache_key": "a" * 64, **result},)


def test_resume_result_rejects_unbounded_telemetry_text(tmp_path: Path) -> None:
    path = tmp_path / "resume-results.jsonl"
    telemetry = {
        "stage": "PRESENCE",
        "evidence_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "elapsed_seconds": 0.1,
        "finish_reason": "stop",
        "outcome": "VALIDATED",
        "retry_depth": 0,
        "cache_hit": False,
        "error_type": "completion body: private text",
    }

    with pytest.raises(Phase14CResumeError, match="telemetry"):
        append_resume_result(
            path,
            "a" * 64,
            {"status": "ACTIONABLE", "telemetry": [telemetry]},
        )


def test_interrupted_trailing_record_is_recovered_without_losing_completed_rows(tmp_path: Path) -> None:
    path = tmp_path / "resume-results.jsonl"
    completed = json.dumps(
        {"cache_key": "a" * 64, "status": "VALIDATED", "result_id": "spec-1"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path.write_bytes(completed + b'{"cache_key":"b')

    records = load_resume_results(path)

    assert records == ({"cache_key": "a" * 64, "status": "VALIDATED", "result_id": "spec-1"},)
    assert path.read_bytes() == completed


def test_complete_unterminated_trailing_record_is_kept_and_terminated(tmp_path: Path) -> None:
    path = tmp_path / "resume-results.jsonl"
    payload = {"cache_key": "a" * 64, "status": "VALIDATED", "result_id": "spec-1"}
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    assert load_resume_results(path) == (payload,)
    assert path.read_bytes().endswith(b"\n")


def test_corrupt_completed_resume_record_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "resume-results.jsonl"
    path.write_text('{"cache_key":"broken"}\n', encoding="utf-8")

    with pytest.raises(Phase14CResumeError, match="corrupt"):
        load_resume_results(path)


@pytest.mark.parametrize("forbidden", ["prompt", "completion", "reasoning", "api_key", "credential"])
def test_resume_result_refuses_prompt_completion_reasoning_and_secret_fields(
    tmp_path: Path, forbidden: str
) -> None:
    with pytest.raises(Phase14CResumeError, match="sensitive"):
        append_resume_result(
            tmp_path / "resume-results.jsonl",
            "a" * 64,
            {"status": "VALIDATED", forbidden: "must never persist"},
        )


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_resume_reader_rejects_external_file_alias_before_repair(
    tmp_path: Path, link_kind: str
) -> None:
    external = tmp_path / "outside.jsonl"
    original = b'{"cache_key":"interrupted'
    external.write_bytes(original)
    root = tmp_path / "run"
    root.mkdir()
    resume_path = root / "resume-results.jsonl"
    try:
        if link_kind == "symlink":
            resume_path.symlink_to(external)
        else:
            os.link(external, resume_path)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"filesystem does not support {link_kind}: {exc}")

    with pytest.raises(Phase14CResumeError, match="resume log path"):
        load_resume_results(resume_path, artifact_root=root)

    assert external.read_bytes() == original


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_resume_writer_rejects_external_file_alias_before_append(
    tmp_path: Path, link_kind: str
) -> None:
    external = tmp_path / "outside.jsonl"
    external.write_bytes(b"")
    root = tmp_path / "run"
    root.mkdir()
    resume_path = root / "resume-results.jsonl"
    try:
        if link_kind == "symlink":
            resume_path.symlink_to(external)
        else:
            os.link(external, resume_path)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"filesystem does not support {link_kind}: {exc}")

    with pytest.raises(Phase14CResumeError, match="resume log path"):
        append_resume_result(
            resume_path,
            "a" * 64,
            {"status": "VALIDATED", "result_id": "spec-1"},
            artifact_root=root,
        )

    assert external.read_bytes() == b""
