from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.fixtures.phase11a_knowledge import fixture_source, grounded_candidate
from tradingagents.distillation.factory import _coerce_candidate
from tradingagents.distillation.writer import validate_generation, write_generation


def _examples():
    source = fixture_source()
    from tradingagents.distillation.models import SourcePacket

    return [
        _coerce_candidate(
            grounded_candidate(SourcePacket(f"p-{i}", (block,))), SourcePacket(f"p-{i}", (block,))
        )
        for i, block in enumerate(source.blocks())
    ][:3]


def test_writer_publishes_canonical_hashed_files_and_validates(tmp_path: Path):
    rows = _examples()
    destination = write_generation(
        tmp_path,
        rows,
        [],
        {
            row.example_id: split
            for row, split in zip(rows, ("train", "validation", "test"), strict=True)
        },
        metadata={"source_fingerprints": {"phase7": "fixture"}},
    )
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {
        "examples.jsonl",
        "excluded.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "source_index.json",
    }
    assert validate_generation(destination).valid
    with pytest.raises(FileExistsError):
        write_generation(tmp_path, rows, [], {}, metadata={"generation_id": destination.name})


def test_writer_rejects_tampered_example(tmp_path: Path):
    destination = write_generation(tmp_path, _examples(), [], {}, metadata={})
    path = destination / "examples.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert not validate_generation(destination).valid


def test_writer_rejects_output_inside_declared_phase7_source_root(tmp_path: Path):
    source_root = tmp_path / "phase7"
    output_root = source_root / "distilled"
    source_root.mkdir()
    with pytest.raises(ValueError, match="source root"):
        write_generation(output_root, _examples(), [], {}, metadata={"source_root": str(source_root)})


def test_validation_rejects_unlisted_rows_and_bad_bytes_metadata(tmp_path: Path):
    destination = write_generation(tmp_path, _examples(), [], {}, metadata={})
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["examples.jsonl"]["bytes"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_generation(destination)
    assert not report.valid
    assert any("byte" in error for error in report.errors)


def test_validation_binds_source_index_to_example_provenance(tmp_path: Path):
    destination = write_generation(tmp_path, _examples(), [], {}, metadata={})
    index_path = destination / "source_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index.pop()
    index_path.write_text(json.dumps(index), encoding="utf-8")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["source_index.json"]["sha256"] = __import__("hashlib").sha256(
        index_path.read_bytes()
    ).hexdigest()
    manifest["files"]["source_index.json"]["bytes"] = index_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not validate_generation(destination).valid


def test_writer_source_index_includes_claim_only_references(tmp_path: Path):
    rows = _examples()
    from dataclasses import replace

    from tradingagents.distillation.models import GroundingClaim

    claim_ref = rows[1].source_refs[0]
    row = replace(rows[0], claims=(GroundingClaim("The concept is defined by the source.", (claim_ref,)),))
    destination = write_generation(tmp_path, [row], [], {}, metadata={})
    index = json.loads((destination / "source_index.json").read_text(encoding="utf-8"))
    assert {item["chunk_id"] for item in index} == {claim_ref.chunk_id, row.source_refs[0].chunk_id}


def test_validation_returns_bounded_invalid_report_for_missing_file(tmp_path: Path):
    destination = write_generation(tmp_path, _examples(), [], {}, metadata={})
    (destination / "examples.jsonl").unlink()
    report = validate_generation(destination)
    assert not report.valid
    assert any("missing file: examples.jsonl" in error for error in report.errors)


def test_validation_rejects_malformed_exclusion_row(tmp_path: Path):
    destination = write_generation(tmp_path, _examples(), [], {}, metadata={})
    path = destination / "excluded.jsonl"
    path.write_text(json.dumps({"reason": "NOT_A_REASON"}) + "\n", encoding="utf-8")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["excluded.jsonl"]["sha256"] = __import__("hashlib").sha256(
        path.read_bytes()
    ).hexdigest()
    manifest["files"]["excluded.jsonl"]["bytes"] = path.stat().st_size
    manifest["counts"]["exclusions"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not validate_generation(destination).valid


def test_validation_rejects_private_exclusion_diagnostic(tmp_path: Path):
    destination = write_generation(tmp_path, _examples(), [], {}, metadata={})
    path = destination / "excluded.jsonl"
    path.write_text(
        json.dumps({"reason": "TEACHER_FAILED", "diagnostic": "private reasoning trace"})
        + "\n",
        encoding="utf-8",
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["excluded.jsonl"]["sha256"] = __import__("hashlib").sha256(
        path.read_bytes()
    ).hexdigest()
    manifest["files"]["excluded.jsonl"]["bytes"] = path.stat().st_size
    manifest["counts"]["exclusions"] = 1
    manifest["counts"]["excluded"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not validate_generation(destination).valid


def test_writer_rejects_sensitive_manifest_metadata(tmp_path: Path):
    with pytest.raises(ValueError, match="unsafe|sensitive"):
        write_generation(
            tmp_path,
            _examples(),
            [],
            {},
            metadata={"diagnostics": {"private_reasoning": "must not persist"}},
        )


def test_writer_rejects_raw_text_in_source_index(tmp_path: Path):
    with pytest.raises(ValueError, match="source index"):
        write_generation(
            tmp_path,
            _examples(),
            [],
            {},
            metadata={"source_index": [{"text": "raw book content"}]},
        )


def test_writer_rejects_caller_source_index_not_bound_to_examples(tmp_path: Path):
    rows = _examples()
    from tradingagents.distillation.writer import _source_index

    supplied = json.loads(json.dumps(_source_index(rows)))
    supplied.append(dict(supplied[0], chunk_id="fabricated-chunk"))
    with pytest.raises(ValueError, match="source index.*provenance|provenance.*source index"):
        write_generation(tmp_path, rows, [], {}, metadata={"source_index": supplied})
    assert not list(tmp_path.glob("generation-*"))


def test_writer_rejects_fabricated_action_lesson(tmp_path: Path):
    row = replace(_examples()[0], assistant="BUY EURUSD now")
    with pytest.raises(ValueError, match="action|unsafe"):
        write_generation(tmp_path, [row], [], {row.example_id: "train"}, metadata={})


def test_writer_rejects_schema_invalid_mapping_before_publication(tmp_path: Path):
    row = _examples()[0].to_dict()
    row.pop("lesson_type")
    with pytest.raises(ValueError, match="lesson_type"):
        write_generation(tmp_path, [row], [], {row["example_id"]: "train"}, metadata={})


def test_writer_rejects_action_text_in_lesson_topic(tmp_path: Path):
    row = replace(_examples()[0], topic="SELL EURUSD")
    with pytest.raises(ValueError, match="action|unsafe"):
        write_generation(tmp_path, [row], [], {row.example_id: "train"}, metadata={})


def test_validation_rejects_tampered_fabricated_action_even_with_new_hash(tmp_path: Path):
    destination = write_generation(tmp_path, _examples(), [], {}, metadata={})
    path = destination / "examples.jsonl"
    value = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    value["assistant"] = "SELL EURUSD now"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["examples.jsonl"]["sha256"] = __import__("hashlib").sha256(
        path.read_bytes()
    ).hexdigest()
    manifest["files"]["examples.jsonl"]["bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not validate_generation(destination).valid


def test_validation_rejects_split_content_divergence_even_with_new_hash(tmp_path: Path):
    rows = _examples()
    destination = write_generation(
        tmp_path,
        rows,
        [],
        {row.example_id: "train" for row in rows},
        metadata={},
    )
    path = destination / "train.jsonl"
    value = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    value["assistant"] = "A different canonical lesson"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["train.jsonl"]["sha256"] = __import__("hashlib").sha256(
        path.read_bytes()
    ).hexdigest()
    manifest["files"]["train.jsonl"]["bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_generation(destination)
    assert not report.valid
    assert any("differs from examples" in error for error in report.errors)


def test_validation_rejects_unknown_manifest_states(tmp_path: Path):
    destination = write_generation(tmp_path, _examples(), [], {}, metadata={})
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "MAYBE"
    manifest["split_status"] = "MAYBE"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not validate_generation(destination).valid



def test_writer_allows_safe_completion_word_in_topic_distribution(tmp_path: Path):
    row = replace(
        _examples()[0],
        topic="Pattern Completion Rule Limitations",
    )

    destination = write_generation(
        tmp_path,
        [row],
        [],
        {},
        metadata={},
    )

    report = validate_generation(destination)

    assert report.valid, report.errors



def test_writer_allows_legitimate_topic_distribution_labels(tmp_path: Path):
    rows = _examples()

    destination = write_generation(
        tmp_path,
        rows,
        [],
        {},
        metadata={
            "topic_distribution": {
                "Pattern Completion Rule Limitations": 1
            }
        },
    )

    assert validate_generation(destination).valid


def test_writer_allows_legitimate_completion_topic_label(tmp_path: Path):
    """A public lesson topic may contain 'Completion' without being teacher output."""
    import json
    from dataclasses import replace

    rows = _examples()
    row = replace(
        rows[0],
        topic="Pattern Completion Rule Limitations",
    )

    destination = write_generation(
        tmp_path,
        [row],
        [],
        {row.example_id: "train"},
        metadata={},
    )

    manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )

    # Writer no longer needs to persist topic strings as metadata keys.
    assert manifest["metadata"]["topic_count"] == 1

    report = validate_generation(destination)
    assert report.valid, report.errors


def test_writer_allows_safe_explicit_topic_distribution_label(tmp_path: Path):
    """If a caller supplies topic_distribution, its keys are public labels."""
    row = _examples()[0]

    destination = write_generation(
        tmp_path,
        [row],
        [],
        {row.example_id: "train"},
        metadata={
            "topic_distribution": {
                "Pattern Completion Rule Limitations": 1,
            }
        },
    )

    report = validate_generation(destination)
    assert report.valid, report.errors


@pytest.mark.parametrize(
    "metadata",
    [
        {"completion": "raw teacher output"},
        {"private_reasoning": "hidden reasoning trace"},
        {"diagnostics": {"scratchpad": "hidden work"}},
        {"diagnostics": {"chain_of_thought": "hidden work"}},
        {"api_key": "do-not-persist"},
        {"credential": "do-not-persist"},
    ],
)
def test_writer_still_rejects_sensitive_metadata_after_topic_fix(
    tmp_path: Path,
    metadata,
):
    """The topic-label exception must not weaken normal metadata protection."""
    row = _examples()[0]

    with pytest.raises(ValueError, match="unsafe metadata"):
        write_generation(
            tmp_path,
            [row],
            [],
            {row.example_id: "train"},
            metadata=metadata,
        )


@pytest.mark.parametrize(
    "unsafe_topic_label",
    [
        "private reasoning trace",
        "chain of thought notes",
        "scratchpad dump",
        "API key details",
        "credential secret",
    ],
)
def test_writer_rejects_sensitive_topic_distribution_labels(
    tmp_path: Path,
    unsafe_topic_label: str,
):
    """Even public-label handling remains fail-closed for private/sensitive labels."""
    row = _examples()[0]

    with pytest.raises(ValueError, match="unsafe metadata"):
        write_generation(
            tmp_path,
            [row],
            [],
            {row.example_id: "train"},
            metadata={
                "topic_distribution": {
                    unsafe_topic_label: 1,
                }
            },
        )

