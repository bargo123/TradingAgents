"""Deterministic dense/lexical candidate fusion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def _finite_score(value: float | int, name: str) -> float:
    """Normalize a retrieval score without allowing non-finite ordering keys."""

    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} must be finite")
    return score


def _chunk_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("chunk_id must be a non-empty string")
    return value


def _rank(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} rank must be a positive integer")
    return value


@dataclass(frozen=True, slots=True, init=False)
class DenseCandidate:
    chunk_id: str
    rank: int
    score: float
    chunk: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(self, chunk_id: str, rank: int, score: float | None = None, chunk: Any = None,
                 metadata: Mapping[str, Any] | None = None, *, semantic_score: float | None = None) -> None:
        if score is None:
            score = semantic_score
        if score is None:
            raise TypeError("dense candidate requires semantic score")
        object.__setattr__(self, "chunk_id", _chunk_id(chunk_id))
        rank = _rank(rank, "dense candidate")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "score", _finite_score(score, "dense candidate score"))
        object.__setattr__(self, "chunk", chunk)
        object.__setattr__(self, "metadata", metadata or {})

    @property
    def semantic_score(self) -> float:
        return self.score


@dataclass(frozen=True, slots=True, init=False)
class LexicalCandidate:
    chunk_id: str
    rank: int
    score: float
    chunk: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(self, chunk_id: str, rank: int, score: float | None = None, chunk: Any = None,
                 metadata: Mapping[str, Any] | None = None, *, lexical_score: float | None = None) -> None:
        if score is None:
            score = lexical_score
        if score is None:
            raise TypeError("lexical candidate requires lexical score")
        object.__setattr__(self, "chunk_id", _chunk_id(chunk_id))
        rank = _rank(rank, "lexical candidate")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "score", _finite_score(score, "lexical candidate score"))
        object.__setattr__(self, "chunk", chunk)
        object.__setattr__(self, "metadata", metadata or {})

    @property
    def lexical_score(self) -> float:
        return self.score


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    chunk_id: str
    fused_score: float
    semantic_score: float | None = None
    lexical_score: float | None = None
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    chunk: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.fused_score


@dataclass(frozen=True, slots=True)
class RRFConfig:
    k: int = 60

    def __post_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("RRF k must be positive")


def _candidate(item: Any, cls: type[DenseCandidate] | type[LexicalCandidate], rank: int) -> Any:
    if isinstance(item, cls):
        return item
    if isinstance(item, Mapping):
        chunk_id = str(item.get("chunk_id", ""))
        score_key = "semantic_score" if cls is DenseCandidate else "lexical_score"
        score = item.get(score_key, item.get("score", 0.0))
        return cls(
            chunk_id=chunk_id,
            rank=int(item.get("rank", rank)),
            score=float(score),
            chunk=item.get("chunk"),
            metadata=item.get("metadata", item),
        )
    raise TypeError(f"unsupported {cls.__name__} value: {type(item).__name__}")


def reciprocal_rank_fuse(
    dense: Sequence[DenseCandidate | Mapping[str, Any]],
    lexical: Sequence[LexicalCandidate | Mapping[str, Any]],
    config: RRFConfig | None = None,
) -> tuple[FusedCandidate, ...]:
    """Fuse candidates by reciprocal rank while retaining both score streams."""

    config = config or RRFConfig()
    merged: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(dense, 1):
        if isinstance(item, Mapping) and not str(item.get("chunk_id", "")).strip():
            continue
        candidate = _candidate(item, DenseCandidate, index)
        if not candidate.chunk_id:
            continue
        row = merged.setdefault(candidate.chunk_id, {"chunk_id": candidate.chunk_id})
        row.update(
            semantic_score=candidate.semantic_score,
            semantic_rank=candidate.rank,
            chunk=candidate.chunk if candidate.chunk is not None else row.get("chunk"),
            metadata=candidate.metadata or row.get("metadata", {}),
        )
        row["fused_score"] = row.get("fused_score", 0.0) + 1.0 / (config.k + candidate.rank)
    for index, item in enumerate(lexical, 1):
        if isinstance(item, Mapping) and not str(item.get("chunk_id", "")).strip():
            continue
        candidate = _candidate(item, LexicalCandidate, index)
        if not candidate.chunk_id:
            continue
        row = merged.setdefault(candidate.chunk_id, {"chunk_id": candidate.chunk_id})
        row.update(
            lexical_score=candidate.lexical_score,
            lexical_rank=candidate.rank,
            chunk=candidate.chunk if candidate.chunk is not None else row.get("chunk"),
            metadata=candidate.metadata or row.get("metadata", {}),
        )
        row["fused_score"] = row.get("fused_score", 0.0) + 1.0 / (config.k + candidate.rank)
    result = tuple(FusedCandidate(**row) for row in merged.values())
    return tuple(
        sorted(
            result,
            key=lambda item: (
                -item.fused_score,
                -(item.semantic_score if item.semantic_score is not None else float("-inf")),
                item.chunk_id,
            ),
        )
    )


__all__ = [
    "DenseCandidate",
    "FusedCandidate",
    "LexicalCandidate",
    "RRFConfig",
    "RankedCandidate",
    "reciprocal_rank_fuse",
]

RankedCandidate = FusedCandidate
