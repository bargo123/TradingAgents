"""Dependency-free evidence helpers for local knowledge retrieval quality.

These helpers deliberately operate on the public read-only query contracts so
benchmark CI does not provision a model, index backend, or external service.
"""

from __future__ import annotations

import ast
import json
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import ContentType, KnowledgeHit, KnowledgeQuery
from .provenance import ProvenanceError, validate_hit_provenance


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One locally labeled retrieval expectation."""

    query: str
    documents: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    content_type: ContentType | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BenchmarkCase:
        content_type = value.get("content_type")
        return cls(
            query=str(value["query"]),
            documents=tuple(str(item) for item in value.get("documents", ())),
            terms=tuple(str(item) for item in value.get("terms", ())),
            content_type=ContentType(content_type) if content_type is not None else None,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Retrieval-only aggregate metrics; none imply trading performance."""

    recall_at: dict[int, float]
    mrr: float
    exact_keyword_hit_rate: float
    content_type_filter_precision: float
    provenance_correctness: float
    duplicate_result_rate: float
    deterministic_ordering: bool
    forbidden_result_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceSnapshotEntry:
    """A source file's relative path, raw bytes, and stable filesystem metadata."""

    relative_path: str
    bytes: bytes
    size_bytes: int
    modified_ns: int
    mode: int


class QueryService(Protocol):
    def search(self, request: KnowledgeQuery) -> Sequence[KnowledgeHit]: ...


def load_cases(path: Path | str) -> tuple[BenchmarkCase, ...]:
    """Load local benchmark labels without depending on an index or model runtime."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("benchmark fixture must contain a JSON list")
    return tuple(BenchmarkCase.from_mapping(item) for item in payload)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _forbidden_fields(hit: KnowledgeHit) -> tuple[str, ...]:
    return tuple(
        key
        for key in hit.to_dict()
        if key.casefold() == "action" or "trade" in key.casefold()
    )


def run_benchmark(
    cases: Iterable[BenchmarkCase | Mapping[str, Any]],
    service: QueryService,
    *,
    ranks: tuple[int, ...] = (1, 5, 10),
) -> BenchmarkReport:
    """Evaluate rank, exact-term, filter, provenance, uniqueness, and order evidence."""
    labels = tuple(
        item if isinstance(item, BenchmarkCase) else BenchmarkCase.from_mapping(item)
        for item in cases
    )
    if not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError("ranks must contain positive integers")
    ranks = tuple(sorted(set(ranks)))
    max_rank = max(ranks)
    relevant_cases = 0
    recalls = dict.fromkeys(ranks, 0)
    reciprocal_rank_sum = 0.0
    exact_cases = exact_hits = 0
    filtered_total = filtered_matches = 0
    provenance_total = provenance_valid = 0
    result_total = duplicate_total = 0
    deterministic_ordering = True
    forbidden_fields: set[str] = set()

    for case in labels:
        content_types = (case.content_type,) if case.content_type is not None else ()
        request = KnowledgeQuery(text=case.query, top_k=max_rank, content_types=content_types)
        hits = tuple(service.search(request))
        repeated = tuple(service.search(request))
        if tuple(hit.chunk_id for hit in hits) != tuple(hit.chunk_id for hit in repeated):
            deterministic_ordering = False

        ids = tuple(hit.chunk_id for hit in hits)
        result_total += len(ids)
        duplicate_total += len(ids) - len(set(ids))
        for hit in hits:
            forbidden_fields.update(_forbidden_fields(hit))
            provenance_total += 1
            try:
                validate_hit_provenance(hit)
            except ProvenanceError:
                continue
            provenance_valid += 1

        if case.documents:
            relevant_cases += 1
            expected = set(case.documents)
            first_rank = next(
                (index for index, hit in enumerate(hits, 1) if hit.document_id in expected),
                None,
            )
            if first_rank is not None:
                reciprocal_rank_sum += 1.0 / first_rank
                for rank in ranks:
                    if first_rank <= rank:
                        recalls[rank] += 1

        if case.terms:
            exact_cases += 1
            text = "\n".join(hit.text for hit in hits).casefold()
            if all(term.casefold() in text for term in case.terms):
                exact_hits += 1

        if case.content_type is not None:
            filtered_total += len(hits)
            filtered_matches += sum(hit.content_type is case.content_type for hit in hits)

    return BenchmarkReport(
        recall_at={rank: _rate(recalls[rank], relevant_cases) for rank in ranks},
        mrr=_rate(reciprocal_rank_sum, relevant_cases),
        exact_keyword_hit_rate=_rate(exact_hits, exact_cases),
        content_type_filter_precision=_rate(filtered_matches, filtered_total),
        provenance_correctness=_rate(provenance_valid, provenance_total),
        duplicate_result_rate=_rate(duplicate_total, result_total),
        deterministic_ordering=deterministic_ordering,
        forbidden_result_fields=tuple(sorted(forbidden_fields)),
    )


def snapshot_tree(root: Path | str) -> tuple[SourceSnapshotEntry, ...]:
    """Capture every regular source file without normalizing its bytes or metadata."""
    root = Path(root)
    entries: list[SourceSnapshotEntry] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        metadata = path.stat()
        entries.append(
            SourceSnapshotEntry(
                relative_path=path.relative_to(root).as_posix(),
                bytes=path.read_bytes(),
                size_bytes=metadata.st_size,
                modified_ns=metadata.st_mtime_ns,
                mode=stat.S_IMODE(metadata.st_mode),
            )
        )
    return tuple(entries)


def imported_modules_under(package_root: Path | str, pyproject_path: Path | str | None = None) -> frozenset[str]:
    """Return static imports plus console entry modules without importing target code."""
    modules: set[str] = set()
    for path in Path(package_root).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module)
    if pyproject_path is not None:
        in_scripts = False
        for line in Path(pyproject_path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_scripts = stripped == "[project.scripts]"
                continue
            if not in_scripts:
                continue
            match = re.match(r'^[-\w]+\s*=\s*"([^"\n]+)"\s*$', stripped)
            if match and ":" in match.group(1):
                modules.add(match.group(1).split(":", 1)[0])
    return frozenset(modules)


__all__ = [
    "BenchmarkCase",
    "BenchmarkReport",
    "SourceSnapshotEntry",
    "imported_modules_under",
    "load_cases",
    "run_benchmark",
    "snapshot_tree",
]
