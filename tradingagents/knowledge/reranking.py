"""Small deterministic CPU reranker for hybrid knowledge candidates."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from .fusion import FusedCandidate
from .models import KnowledgeQuery

_TOKEN = re.compile(r"[\w]+(?:[-_][\w]+)*", re.UNICODE)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(value.casefold()))


class Reranker:
    """Rank candidates using explainable lexical/structure features only."""

    def rerank(self, query: KnowledgeQuery, candidates: Sequence[FusedCandidate]) -> tuple[FusedCandidate, ...]:
        query_tokens = _tokens(query.text)
        phrase = query.text.casefold()
        scored: list[FusedCandidate] = []
        seen_sources: set[str] = set()
        for candidate in candidates:
            chunk = candidate.chunk
            text = str(getattr(chunk, "text", "") if chunk is not None else candidate.metadata.get("text", ""))
            title = str(getattr(chunk, "title", "") if chunk is not None else candidate.metadata.get("title", ""))
            section = " ".join(getattr(chunk, "section_path", ()) if chunk is not None else candidate.metadata.get("section_path", ()))
            text_tokens = set(_tokens(text))
            coverage = len(set(query_tokens) & text_tokens) / max(1, len(set(query_tokens)))
            phrase_score = 1.0 if phrase in text.casefold() else 0.0
            title_score = len(set(query_tokens) & set(_tokens(title + " " + section))) / max(1, len(set(query_tokens)))
            requested_type = query.content_types and chunk is not None and getattr(chunk, "content_type", None) in query.content_types
            source = str(getattr(chunk, "source_hash", "") if chunk is not None else candidate.metadata.get("source_hash", ""))
            diversity = 1.0 if source and source not in seen_sources else 0.0
            if source:
                seen_sources.add(source)
            score = (
                candidate.fused_score
                + 0.35 * coverage
                + 0.35 * phrase_score
                + 0.15 * title_score
                + 0.10 * bool(requested_type)
                + 0.05 * diversity
            )
            scored.append(replace(candidate, rerank_score=score))
        return tuple(
            sorted(
                scored,
                key=lambda item: (
                    -(item.rerank_score if item.rerank_score is not None else float("-inf")),
                    -item.fused_score,
                    -(item.semantic_score if item.semantic_score is not None else float("-inf")),
                    -(item.lexical_score if item.lexical_score is not None else float("-inf")),
                    item.chunk_id,
                ),
            )
        )


__all__ = ["Reranker"]
