from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.phase7_fast_reconstruct import CacheAwareIngestor


def test_cache_vectors_reorder_by_chunk_id_without_reembedding(tmp_path: Path) -> None:
    spec = {"model_id": "fixture", "dimensions": 2}
    payload = {
        "embedding_spec": spec,
        "chunk_ids": ["chunk-b", "chunk-a"],
        "vectors": [[2.0, 0.0], [1.0, 0.0]],
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    ingestor = CacheAwareIngestor.__new__(CacheAwareIngestor)
    ingestor.embedder = SimpleNamespace(spec=SimpleNamespace(to_dict=lambda: spec, dimensions=2))
    chunks = (SimpleNamespace(chunk_id="chunk-a"), SimpleNamespace(chunk_id="chunk-b"))

    assert ingestor._load_cache(path, "doc-fixture", chunks) == ((1.0, 0.0), (2.0, 0.0))
