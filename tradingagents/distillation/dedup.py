"""Deterministic exact and near-duplicate protection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.lower())).strip()


def canonical_example_id(example: Any) -> str:
    try:
        from .models import canonical_json

        text = canonical_json(example)
    except Exception:
        import json

        text = json.dumps(example, sort_keys=True, default=lambda x: vars(x), separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class DedupIndex:
    near_threshold: float = 0.90
    _keys: set[str] = field(default_factory=set)
    _tokens: list[set[str]] = field(default_factory=list)

    def check(self, example: Any) -> str | None:
        try:
            from .models import canonical_json

            text = _norm(canonical_json(example))
        except Exception:
            text = _norm(str(example))
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self._keys:
            return "DUPLICATE"
        tokens = set(text.split())
        for prior in self._tokens:
            score = len(tokens & prior) / max(1, len(tokens | prior))
            if score >= self.near_threshold:
                return "NEAR_DUPLICATE"
        self._keys.add(key)
        self._tokens.append(tokens)
        return None
