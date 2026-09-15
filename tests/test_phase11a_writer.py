from __future__ import annotations

import json
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
