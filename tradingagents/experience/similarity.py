"""Exact, deterministic NumPy similarity over persisted feature vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SimilarityHit:
    experience_id: str
    distance: float
    similarity_score: float
    comparable_feature_count: int
    analysis_snapshot_timestamp: datetime | None = None


def _get(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


class ExactSimilarityIndex:
    """In-memory read-only index; callers provide already-loaded projections."""

    def __init__(
        self, vectors: Mapping[str, Any] | Sequence[Any], metadata: Mapping[str, Any] | None = None
    ):
        if isinstance(vectors, Mapping):
            self._vectors = dict(vectors)
        else:
            self._vectors = {_get(row, "experience_id", _get(row, "id")): row for row in vectors}
        self._metadata = metadata or {}

    def search(self, query_vector, query_mask, candidate_ids, top_k, profile):
        names = tuple(profile.feature_order)
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        qm = np.asarray(query_mask, dtype=bool).reshape(-1)
        if len(q) != len(names) or len(qm) != len(names):
            return ()
        rows: list[SimilarityHit] = []
        for eid in candidate_ids:
            row = self._vectors.get(eid)
            if row is None:
                continue
            vector = _get(row, "values", _get(row, "vector"))
            mask = _get(row, "mask")
            if vector is not None and hasattr(vector, "values"):
                mask = mask if mask is not None else vector.mask
                vector = vector.values
            if isinstance(vector, Mapping):
                vector = [vector.get(n, np.nan) for n in names]
            c = np.asarray(vector, dtype=np.float32).reshape(-1)
            cm = np.asarray(mask if mask is not None else np.isfinite(c), dtype=bool).reshape(-1)
            if len(c) != len(names) or len(cm) != len(names):
                continue
            comparable = qm & cm & np.isfinite(q) & np.isfinite(c)
            count = int(comparable.sum())
            if count < 8 or count < (len(names) + 1) // 2:
                continue
            qn = np.empty(len(names), dtype=np.float32)
            cn = np.empty(len(names), dtype=np.float32)
            for i, name in enumerate(names):
                scale = float(profile.scales.get(name, profile.fallback_scales.get(name, 1.0)))
                scale = max(scale, float(profile.epsilon))
                qn[i] = np.clip(
                    (q[i] - float(profile.medians.get(name, 0.0))) / scale, *profile.clipping
                )
                cn[i] = np.clip(
                    (c[i] - float(profile.medians.get(name, 0.0))) / scale, *profile.clipping
                )
            weights = np.asarray(
                [float(profile.weights.get(n, 1.0)) for n in names], dtype=np.float32
            )
            delta = qn[comparable] - cn[comparable]
            distance = float(
                np.sqrt(
                    np.sum(weights[comparable] * delta * delta, dtype=np.float32)
                    / np.sum(weights[comparable], dtype=np.float32)
                )
            )
            meta = self._metadata.get(eid, {})
            rows.append(
                SimilarityHit(
                    str(eid),
                    distance,
                    1.0 / (1.0 + distance),
                    count,
                    _get(meta, "analysis_snapshot_timestamp"),
                )
            )
        rows.sort(
            key=lambda h: (
                h.distance,
                h.analysis_snapshot_timestamp or datetime.min.replace(tzinfo=timezone.utc),
                h.experience_id,
            )
        )
        return tuple(rows[:top_k])


__all__ = ["ExactSimilarityIndex", "SimilarityHit"]
