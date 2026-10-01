"""Create-only publication and validation for Phase 11A generations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

FILES = (
    "manifest.json",
    "examples.jsonl",
    "excluded.jsonl",
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "source_index.json",
)

_FORBIDDEN_METADATA = re.compile(
    r"(?:prompt|completion|reasoning|chain[._ -]?of[._ -]?thought|\bcot\b|scratchpad|secret|password|credential|api[._ -]?key|private[._ -]?key)",
    re.IGNORECASE,
)
_FORBIDDEN_LESSON_FIELD = re.compile(
    r"(?:buy|sell|hold|pnl|profit|reward|entry|exit|future|outcome|label)",
    re.IGNORECASE,
)
_ACTION_TEXT = re.compile(r"(?<![-\w])(?:BUY|SELL|HOLD)(?![-\w])", re.IGNORECASE)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    return value


def _json(value: Any) -> bytes:
    return (
        json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _validate_metadata(value: Any, path: str = "metadata") -> None:
    """Reject private teacher material before it can reach a manifest."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if _FORBIDDEN_METADATA.search(key_text):
                raise ValueError(f"unsafe metadata field: {child_path}")
            _validate_metadata(child, child_path)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, child in enumerate(value):
            _validate_metadata(child, f"{path}[{index}]")
    elif isinstance(value, str) and _FORBIDDEN_METADATA.search(value):
        raise ValueError(f"unsafe metadata value: {path}")


def _validate_source_index(value: Any) -> None:
    """Keep the provenance index reference-only; never publish source text."""

    if not isinstance(value, list):
        raise ValueError("source index must be a list")
    from .models import SourceRef

    forbidden = {"text", "content", "raw_text", "source_text"}
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"source index row {index} must be an object")
        if forbidden.intersection(str(key).lower() for key in row):
            raise ValueError(f"source index row {index} must not contain text")
        try:
            SourceRef.from_dict(row)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid source index row {index}") from exc


def _validate_lesson_safety(row: Mapping[str, Any]) -> None:
    if row.get("source_type", "BOOK_KNOWLEDGE") != "BOOK_KNOWLEDGE":
        raise ValueError("source_type must be BOOK_KNOWLEDGE")
    for key in row:
        if _FORBIDDEN_METADATA.search(str(key)) or _FORBIDDEN_LESSON_FIELD.search(str(key)):
            raise ValueError(f"unsafe lesson field: {key}")
    for key in (
        "topic",
        "system",
        "user",
        "assistant",
        "system_instruction",
        "user_instruction",
        "assistant_target",
    ):
        value = row.get(key)
        if isinstance(value, str):
            if _ACTION_TEXT.search(value):
                raise ValueError(f"unsafe action text in lesson field: {key}")


def _row_id(row: Any) -> str:
    value = getattr(row, "example_id", None)
    if value is None and isinstance(row, Mapping):
        value = row.get("example_id") or row.get("id")
    return str(value or hashlib.sha256(_json(row)).hexdigest())


def _rows(items: Iterable[Any]) -> list[Any]:
    return sorted(
        (_plain(x) for x in items),
        key=lambda x: (
            str(x.get("example_id", x.get("id", ""))) if isinstance(x, Mapping) else _row_id(x)
        ),
    )


def _canonical_knowledge_examples(items: Iterable[Any]) -> list[dict[str, Any]]:
    """Coerce every accepted row through the closed KnowledgeExample contract."""
    from .models import KnowledgeExample

    canonical: list[dict[str, Any]] = []
    for index, item in enumerate(_rows(items)):
        if not isinstance(item, Mapping):
            raise ValueError(f"example {index} must be an object")
        try:
            example = (
                item if isinstance(item, KnowledgeExample) else KnowledgeExample.from_dict(item)
            )
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise ValueError(f"invalid KnowledgeExample at index {index}: {exc}") from exc
        canonical.append(_plain(example))
    return canonical


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_generation(
    output_root: str | Path,
    examples: Iterable[Any],
    exclusions: Iterable[Any],
    assignments: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Publish one immutable generation using a staging directory and atomic rename."""
    examples = _canonical_knowledge_examples(examples)
    exclusions = _rows(exclusions)
    metadata = dict(metadata or {})
    _validate_metadata(metadata)
    for example in examples:
        if isinstance(example, Mapping):
            _validate_lesson_safety(example)
    _validate_metadata(exclusions, "exclusions")
    # A Phase 7 root is an immutable input.  Refuse destinations that would
    # place generated artifacts in (or beneath) that root, even when the
    # directory does not exist yet.
    source_root = metadata.get("source_root") or metadata.get("phase7_root")
    if source_root:
        source_path = Path(source_root).resolve()
        output_path = Path(output_root).resolve()
        try:
            output_path.relative_to(source_path)
        except ValueError:
            pass
        else:
            raise ValueError("output root must not be inside the Phase 7 source root")
    generation_id = str(
        metadata.pop(
            "generation_id",
            "generation-"
            + _hash(
                _json(
                    {
                        "examples": examples,
                        "excluded": exclusions,
                        "assignments": _plain(assignments),
                        "metadata": metadata,
                    }
                )
            )[:24],
        )
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / generation_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"generation already exists: {generation_id}")
    by_id = {_row_id(x): x for x in examples}
    if len(by_id) != len(examples):
        raise ValueError("duplicate example id")
    split_map: dict[str, str] = {}
    if isinstance(assignments, Mapping):
        split_map = {str(k): str(v) for k, v in assignments.items()}
    elif assignments is not None:
        values = getattr(assignments, "assignments", assignments)
        for item in values:
            item = _plain(item)
            if isinstance(item, Mapping):
                split_map[str(item.get("example_id"))] = str(item.get("split"))
    unknown_assignments = set(split_map) - set(by_id)
    invalid_splits = set(split_map.values()) - {"train", "validation", "test"}
    if unknown_assignments:
        raise ValueError("split assignment references unknown example")
    if invalid_splits:
        raise ValueError("split assignment contains unsupported split")
    if (
        metadata.get("split_status", "COMPLETE" if split_map else "INSUFFICIENT_DATA") == "COMPLETE"
        and set(split_map) != set(by_id)
    ):
        raise ValueError("complete split must assign every example exactly once")
    files_data: dict[str, bytes] = {
        "examples.jsonl": b"".join(_json(x) for x in examples),
        "excluded.jsonl": b"".join(_json(x) for x in exclusions),
    }
    for name, split in (
        ("train.jsonl", "train"),
        ("validation.jsonl", "validation"),
        ("test.jsonl", "test"),
    ):
        files_data[name] = b"".join(
            _json(by_id[key])
            for key, value in sorted(split_map.items())
            if value == split and key in by_id
        )
    source_index = metadata.get("source_index")
    if source_index is None:
        source_index = _source_index(examples)
    elif _json(source_index) != _json(_source_index(examples)):
        raise ValueError("source index is not bound to accepted example provenance")
    _validate_source_index(source_index)
    files_data["source_index.json"] = _json(source_index)
    # Keep distributions in the manifest metadata so inspect/reporting can be
    # done without loading lesson rows.  Caller-provided values remain
    # authoritative for forward-compatible extensions.
    lesson_distribution = Counter(
        str(x.get("lesson_type", "")) for x in examples if isinstance(x, Mapping)
    )
    topic_distribution = Counter(
        str(x.get("topic", "")) for x in examples if isinstance(x, Mapping)
    )
    difficulty_distribution = Counter(
        str(x.get("difficulty", "")) for x in examples if isinstance(x, Mapping)
    )
    source_documents = {
        str(ref.get("document_id", ""))
        for ref in source_index
        if isinstance(ref, Mapping) and ref.get("document_id")
    }
    source_sections = {
        (str(ref.get("document_id", "")), str(ref.get("section", "")))
        for ref in source_index
        if isinstance(ref, Mapping) and ref.get("document_id")
    }
    metadata.setdefault("lesson_type_distribution", dict(sorted(lesson_distribution.items())))
    metadata.setdefault("topic_distribution", dict(sorted(topic_distribution.items())))
    metadata.setdefault("difficulty_distribution", dict(sorted(difficulty_distribution.items())))
    metadata.setdefault("candidate_count", len(examples) + len(exclusions))
    metadata.setdefault(
        "source_coverage",
        {
            "documents": len(source_documents),
            "sections": len(source_sections),
            "chunks": len(source_index) if isinstance(source_index, list) else 0,
        },
    )
    reasons = Counter(str(x.get("reason", "")) for x in exclusions if isinstance(x, Mapping))
    manifest = {
        "schema_version": "phase11a-manifest.v1",
        "generation_id": generation_id,
        "status": "PUBLISHED" if examples else "EMPTY_ELIGIBLE_SET",
        "counts": {
            "examples": len(examples),
            "accepted": len(examples),
            "exclusions": len(exclusions),
            "excluded": len(exclusions),
        },
        "split_status": metadata.get(
            "split_status", "COMPLETE" if split_map else "INSUFFICIENT_DATA"
        ),
        "split_counts": dict(Counter(split_map.values())),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "source_fingerprints": metadata.get("source_fingerprints", {}),
        "policy_versions": metadata.get("policy_versions", {}),
        "metadata": metadata,
        "files": {
            name: {"sha256": _hash(data), "bytes": len(data)} for name, data in files_data.items()
        },
        "safety": metadata.get(
            "safety", {"network_attempts": 0, "llm_calls": 0, "tool_calls": 0, "mt5_calls": 0}
        ),
    }
    for name in (
        "phase7_generation_id",
        "teacher",
        "request_fingerprint",
        "safety",
        "llm_calls",
        "network_attempts",
        "mt5_calls",
        "tool_calls",
    ):
        if name in metadata:
            manifest[name] = metadata[name]
    files_data["manifest.json"] = _json(manifest)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{generation_id}-", dir=str(root)))
    try:
        for name, data in files_data.items():
            (staging / name).write_bytes(data)
        os.replace(staging, destination)
    except Exception:
        # Preserve staging for diagnosis; never publish partial generations.
        raise
    return destination


def validate_generation(path: str | Path):
    """Validate hashes and required files without modifying the generation."""
    root = Path(path).resolve()
    errors: list[str] = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"manifest unreadable: {exc}")
        manifest = {}
    if not isinstance(manifest, Mapping):
        errors.append("manifest must be an object")
        manifest = {}
    for name in FILES:
        if not (root / name).is_file():
            errors.append(f"missing file: {name}")
    expected_hashed = set(FILES) - {"manifest.json"}
    manifest_files = manifest.get("files", {})
    if not isinstance(manifest_files, Mapping):
        errors.append("manifest file metadata must be an object")
        manifest_files = {}
    if set(manifest_files) != expected_hashed:
        errors.append("manifest file hash set is incomplete")
    if manifest.get("schema_version") != "phase11a-manifest.v1":
        errors.append("unknown or incompatible manifest version")
    if manifest.get("generation_id") != root.name:
        errors.append("generation identity does not match directory")
    raw_counts = manifest.get("counts", {})
    counts_for_status = raw_counts if isinstance(raw_counts, Mapping) else {}
    example_count = counts_for_status.get("examples", 0)
    status = manifest.get("status")
    if status not in {"PUBLISHED", "EMPTY_ELIGIBLE_SET"}:
        errors.append("unknown manifest status")
    elif isinstance(example_count, bool) or not isinstance(example_count, int):
        errors.append("manifest example count must be an integer")
    elif status == "PUBLISHED" and example_count < 1:
        errors.append("published generation must contain examples")
    elif status == "EMPTY_ELIGIBLE_SET" and example_count != 0:
        errors.append("empty generation must not contain examples")
    split_status = manifest.get("split_status")
    if split_status not in {"COMPLETE", "INSUFFICIENT_DATA"}:
        errors.append("unknown split status")
    try:
        _validate_metadata(manifest.get("metadata", {}))
    except ValueError as exc:
        errors.append(str(exc))
    for name, info in manifest_files.items():
        if name not in expected_hashed:
            errors.append(f"unexpected file metadata: {name}")
            continue
        target = root / name
        if not isinstance(info, Mapping):
            errors.append(f"invalid file metadata: {name}")
            continue
        if not isinstance(info.get("sha256"), str) or len(info["sha256"]) != 64:
            errors.append(f"invalid file hash metadata: {name}")
        if (
            target.is_file()
            and info.get("sha256") != _hash(target.read_bytes())
        ):
            errors.append(f"hash mismatch: {name}")
        if target.is_file() and info.get("bytes") != target.stat().st_size:
            errors.append(f"byte size mismatch: {name}")
    try:
        # Validate the canonical rows when the contracts are available.  This
        # catches missing provenance rather than merely checking file hashes.
        from .models import GroundingClaim, KnowledgeExample, SourceRef

        parsed: dict[str, list[dict[str, Any]]] = {}
        source_index_rows: list[Mapping[str, Any]] = []
        source_index_path = root / "source_index.json"
        if source_index_path.is_file():
            source_index_value = json.loads(source_index_path.read_text(encoding="utf-8"))
            if not isinstance(source_index_value, list) or any(
                not isinstance(row, Mapping) for row in source_index_value
            ):
                errors.append("source index must contain only objects")
            else:
                source_index_rows = list(source_index_value)
        for filename in (
            "examples.jsonl",
            "excluded.jsonl",
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
        ):
            parsed[filename] = []
            source_path = root / filename
            if not source_path.is_file():
                continue
            for _line_no, line in enumerate(
                source_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError(f"{filename} row must be an object")
                    if filename == "excluded.jsonl":
                        _validate_metadata(row, f"{filename}:{_line_no}")
                        raw_refs = row.get("source_refs", ())
                        if not isinstance(raw_refs, (list, tuple)) or any(
                            not isinstance(ref, Mapping) for ref in raw_refs
                        ):
                            raise ValueError("exclusion source_refs must contain objects")
                        refs = tuple(
                            SourceRef.from_dict(ref) for ref in raw_refs
                        )
                        from .models import DatasetExclusion

                        DatasetExclusion(
                            reason=row.get("reason"),
                            diagnostic=row.get("diagnostic", ""),
                            packet_id=row.get("packet_id"),
                            candidate_id=row.get("candidate_id"),
                            source_refs=refs,
                        )
                        parsed[filename].append(row)
                        continue
                    row["source_refs"] = tuple(
                        SourceRef(**ref) if isinstance(ref, Mapping) else ref
                        for ref in row.get("source_refs", ())
                    )
                    row["claims"] = tuple(
                        GroundingClaim(
                            claim=item.get("claim", ""),
                            source_refs=tuple(
                                SourceRef(**ref) if isinstance(ref, Mapping) else ref
                                for ref in item.get("source_refs", item.get("refs", ()))
                            ),
                        )
                        if isinstance(item, Mapping)
                        else item
                        for item in row.get("claims", ())
                    )
                    _validate_lesson_safety(row)
                    KnowledgeExample.from_dict(row)
                    parsed[filename].append(row)
        expected_refs = {
            _json(ref).decode("utf-8").rstrip("\n")
            for row in parsed["examples.jsonl"]
            for ref in (
                list(row.get("source_refs", ()))
                + [
                    ref
                    for claim in row.get("claims", ())
                    for ref in claim.source_refs
                ]
            )
        }
        actual_refs = {_json(ref).decode("utf-8").rstrip("\n") for ref in source_index_rows}
        if expected_refs != actual_refs:
            errors.append("source index provenance mismatch")
        phase7_generation_id = manifest.get("phase7_generation_id")
        if phase7_generation_id:
            for row in parsed["examples.jsonl"]:
                for ref in row.get("source_refs", ()):
                    if getattr(ref, "generation_id", None) != phase7_generation_id:
                        errors.append("source reference generation mismatch")
                        break
            for row in parsed["excluded.jsonl"]:
                for ref in row.get("source_refs", ()):
                    if isinstance(ref, Mapping) and ref.get("generation_id") != phase7_generation_id:
                        errors.append("excluded source reference generation mismatch")
                        break
        counts = manifest.get("counts", {})
        if not isinstance(counts, Mapping):
            raise ValueError("manifest counts must be an object")
        if counts.get("examples") != len(parsed["examples.jsonl"]):
            errors.append("example count mismatch")
        if counts.get("exclusions") != len(parsed["excluded.jsonl"]):
            errors.append("exclusion count mismatch")
        if counts.get("accepted") != counts.get("examples"):
            errors.append("accepted count mismatch")
        if counts.get("excluded") != counts.get("exclusions"):
            errors.append("excluded count mismatch")
        split_counts = manifest.get("split_counts", {})
        for name, split in (
            ("train.jsonl", "train"),
            ("validation.jsonl", "validation"),
            ("test.jsonl", "test"),
        ):
            if split_counts.get(split, 0) != len(parsed[name]):
                errors.append(f"split count mismatch: {split}")
        # Split files are projections of examples.jsonl, not independent
        # datasets.  Every projected row must be unique and known.
        all_ids = [_row_id(row) for row in parsed["examples.jsonl"]]
        canonical_by_id = {
            _row_id(row): _json(row).decode("utf-8").rstrip("\n")
            for row in parsed["examples.jsonl"]
        }
        if len(set(all_ids)) != len(all_ids):
            errors.append("duplicate example id")
        for filename in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            ids = [_row_id(row) for row in parsed[filename]]
            if len(set(ids)) != len(ids):
                errors.append(f"duplicate split row: {filename}")
            if not set(ids).issubset(set(all_ids)):
                errors.append(f"unknown split row: {filename}")
            for row in parsed[filename]:
                row_id = _row_id(row)
                if row_id in canonical_by_id and (
                    _json(row).decode("utf-8").rstrip("\n") != canonical_by_id[row_id]
                ):
                    errors.append(f"{filename} row differs from examples: {row_id}")
        split_status = manifest.get("split_status")
        if split_status == "COMPLETE" and set().union(
            *(
                {_row_id(row) for row in parsed[name]}
                for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
            )
        ) != set(all_ids):
            errors.append("complete split does not cover all examples")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid generation rows: {exc}")
    try:
        from .models import ValidationReport

        return ValidationReport(valid=not errors, errors=tuple(errors))
    except (ImportError, TypeError):
        return {"valid": not errors, "errors": errors}


__all__ = ["write_generation", "validate_generation"]


def _source_index(examples: list[Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for example in examples:
        value = _plain(example)
        refs = list(value.get("source_refs", [])) if isinstance(value, Mapping) else []
        if isinstance(value, Mapping):
            refs.extend(
                ref
                for claim in value.get("claims", [])
                if isinstance(claim, Mapping)
                for ref in claim.get("source_refs", claim.get("refs", []))
            )
        for ref in refs:
            if not isinstance(ref, Mapping):
                continue
            key = "|".join(
                str(ref.get(field, ""))
                for field in ("generation_id", "document_id", "chunk_id", "source_hash")
            )
            if key:
                rows[key] = dict(ref)
    return [rows[key] for key in sorted(rows)]
