"""Explicit, read-only knowledge handoff for the Phase 11 trainer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DistillationError
from .writer import validate_generation


class KnowledgeGenerationInvalidError(DistillationError):
    """Published lesson generation is missing, mutable, or incompatible."""


def _read_rows(path: Path, expected_source_type: str = "BOOK_KNOWLEDGE") -> tuple[dict, ...]:
    if not path.is_file():
        raise KnowledgeGenerationInvalidError(f"missing dataset split: {path.name}")
    rows = []
    try:
        for _line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row must be an object")
            if row.get("source_type", expected_source_type) != expected_source_type:
                raise ValueError("source type is not BOOK_KNOWLEDGE")
            rows.append(row)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise KnowledgeGenerationInvalidError(f"invalid {path.name}: {exc}") from exc
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class CurriculumPlan:
    stages: tuple[str, ...] = ("KNOWLEDGE", "EXPERIENCE")


@dataclass(frozen=True, slots=True)
class KnowledgeTrainingSource:
    path: Path
    source_type: str = "BOOK_KNOWLEDGE"
    _train: tuple[dict, ...] = field(default_factory=tuple, repr=False, compare=False)
    _validation: tuple[dict, ...] = field(default_factory=tuple, repr=False, compare=False)

    @property
    def train_rows(self) -> tuple[dict, ...]:
        return self._train

    @property
    def validation_rows(self) -> tuple[dict, ...]:
        return self._validation

    @property
    def test_rows(self) -> tuple[dict, ...]:
        # The test partition belongs to Phase 12 evaluation and is deliberately
        # not exposed through a trainer binding.
        return ()

    def rows(self) -> Iterator[dict]:
        yield from self._train
        yield from self._validation


@dataclass(frozen=True, slots=True)
class KnowledgeDatasetBinding:
    path: Path
    generation_id: str
    manifest: Mapping[str, object]
    manifest_hash: str
    curriculum: CurriculumPlan = field(default_factory=CurriculumPlan)
    training_source: KnowledgeTrainingSource = field(init=False)

    def __post_init__(self) -> None:
        train = _read_rows(self.path / "train.jsonl")
        validation = _read_rows(self.path / "validation.jsonl")
        object.__setattr__(
            self,
            "training_source",
            KnowledgeTrainingSource(self.path, _train=train, _validation=validation),
        )

    @classmethod
    def open(cls, path: str | Path) -> KnowledgeDatasetBinding:
        root = Path(path).resolve()
        report = validate_generation(root)
        if not getattr(report, "valid", False):
            errors = "; ".join(str(x) for x in getattr(report, "errors", ()))
            raise KnowledgeGenerationInvalidError(errors or "generation validation failed")
        try:
            manifest_bytes = (root / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise KnowledgeGenerationInvalidError(f"invalid manifest: {exc}") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != "phase11a-manifest.v1"
        ):
            raise KnowledgeGenerationInvalidError("incompatible manifest schema")
        if manifest.get("split_status") != "COMPLETE":
            raise KnowledgeGenerationInvalidError("knowledge split is not complete")
        generation_id = manifest.get("generation_id")
        if not isinstance(generation_id, str) or generation_id != root.name:
            raise KnowledgeGenerationInvalidError("generation identity does not match directory")
        binding = cls(root, generation_id, manifest, hashlib.sha256(manifest_bytes).hexdigest())
        counts = manifest.get("split_counts", {})
        if counts.get("train", 0) != len(binding.train_rows) or counts.get("validation", 0) != len(
            binding.validation_rows
        ):
            raise KnowledgeGenerationInvalidError("knowledge split counts do not match manifest")
        return binding

    @property
    def train_rows(self) -> tuple[dict, ...]:
        return self.training_source.train_rows

    @property
    def validation_rows(self) -> tuple[dict, ...]:
        return self.training_source.validation_rows

    @property
    def test_rows(self) -> tuple[dict, ...]:
        return ()


__all__ = [
    "KnowledgeDatasetBinding",
    "KnowledgeTrainingSource",
    "CurriculumPlan",
    "KnowledgeGenerationInvalidError",
]
