"""Offline adapter from frozen Phase 7 evidence to extracted candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from .models import CandidateSpec, CandidateState, ExitPolicyConfig, StrategyVersion


class BookStrategyFactory:
    """Turn auditable Phase 7 hits into proposals, never active strategies."""

    _required = ("document_id", "chunk_id", "source_filename", "source_hash", "content_type", "text")

    def from_hits(
        self,
        hits: Sequence[Mapping[str, object]],
        *,
        parent: StrategyVersion,
    ) -> tuple[CandidateSpec, ...]:
        if isinstance(hits, (str, bytes, bytearray, Mapping)):
            raise ValueError("hits must be a sequence")
        evidence = tuple(hits)
        for hit in evidence:
            if not isinstance(hit, Mapping) or any(not str(hit.get(key, "")).strip() for key in self._required):
                raise ValueError("book evidence requires exact provenance")
        canonical = json.dumps([dict(item) for item in evidence], sort_keys=True, separators=(",", ":"), default=str)
        candidate_id = "book-" + hashlib.sha256((parent.config_hash + canonical).encode()).hexdigest()[:24]
        return (
            CandidateSpec(
                candidate_id=candidate_id,
                parent=parent,
                strategy_id=parent.strategy_id,
                exit_policy=ExitPolicyConfig(),
                hypothesis="Book-derived proposal; requires causal replay before any deployment.",
                source_evidence=evidence,
                state=CandidateState.EXTRACTED,
            ),
        )


__all__ = ["BookStrategyFactory"]
