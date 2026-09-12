"""Tokenizer-aware, structure-preserving knowledge chunking.

This module intentionally stops at deterministic chunk artifacts.  It neither
embeds text nor writes an index; a later ingestion coordinator supplies the
same embedding provider's specification and tokenizer to this boundary.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .models import (
    ChunkRecord,
    ContentType,
    EmbeddingSpec,
    EquationMetadata,
    ParsedBlock,
    ParsedDocument,
    TableMetadata,
)


class EmbeddingTokenizer(Protocol):
    """The minimum tokenizer boundary used by the chunker.

    Concrete providers may return token IDs, strings, or a HuggingFace-style
    mapping.  The chunker only needs their length, but every candidate is
    encoded with the provider's corpus policy and special tokens enabled.
    """

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        purpose: str,
        truncation: bool,
    ) -> Sequence[Any] | dict[str, Sequence[Any]]:
        ...


class ChunkTooLargeForEmbedding(RuntimeError):
    """A semantic unit cannot fit the active model without truncation."""

    def __init__(
        self,
        message: str,
        *,
        document_id: str | None = None,
        block_id: str | None = None,
        content_type: ContentType | None = None,
    ) -> None:
        super().__init__(message)
        self.document_id = document_id
        self.block_id = block_id
        self.content_type = content_type


class ChunkPolicyError(ValueError):
    """The configured embedding specification cannot produce safe chunks."""


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    """Size policy derived from one complete, active embedding specification."""

    soft_token_target: int
    hard_content_token_limit: int
    overlap_tokens: int
    version: str
    model_max_input_tokens: int = 512
    special_token_budget: int = 2
    corpus_instruction_tokens: int = 0
    embedding_spec: EmbeddingSpec | None = None

    def __post_init__(self) -> None:
        soft = int(self.soft_token_target)
        hard = int(self.hard_content_token_limit)
        overlap = int(self.overlap_tokens)
        model_limit = int(self.model_max_input_tokens)
        special = int(self.special_token_budget)
        instruction = int(self.corpus_instruction_tokens)
        if soft <= 0 or hard <= 0 or model_limit <= 0:
            raise ChunkPolicyError("chunk token limits must be positive")
        if overlap < 0 or overlap > 64:
            raise ChunkPolicyError("overlap_tokens must be between 0 and 64")
        if special < 0 or instruction < 0:
            raise ChunkPolicyError("special and instruction token budgets must be non-negative")
        if hard != model_limit - special - instruction:
            raise ChunkPolicyError("hard content limit must match the embedding context budget")
        if soft > hard:
            raise ChunkPolicyError("soft token target cannot exceed the hard content limit")
        if not str(self.version).strip():
            raise ChunkPolicyError("chunker version must be non-empty")
        object.__setattr__(self, "soft_token_target", soft)
        object.__setattr__(self, "hard_content_token_limit", hard)
        object.__setattr__(self, "overlap_tokens", overlap)
        object.__setattr__(self, "model_max_input_tokens", model_limit)
        object.__setattr__(self, "special_token_budget", special)
        object.__setattr__(self, "corpus_instruction_tokens", instruction)

    @classmethod
    def for_embedding(cls, spec: EmbeddingSpec) -> ChunkPolicy:
        """Derive the only safe policy from a loaded provider specification."""

        if not isinstance(spec, EmbeddingSpec):
            raise ChunkPolicyError("ChunkPolicy.for_embedding requires an EmbeddingSpec")
        if spec.truncation:
            raise ChunkPolicyError("knowledge chunking requires truncation=False")
        available_after_special_tokens = spec.model_max_input_tokens - spec.special_token_budget
        computed_limit = spec.effective_corpus_content_token_limit
        corpus_instruction_tokens = available_after_special_tokens - computed_limit
        if computed_limit <= 0 or corpus_instruction_tokens < 0:
            raise ChunkPolicyError("embedding specification has no usable corpus token budget")
        return cls(
            soft_token_target=min(448, computed_limit),
            hard_content_token_limit=computed_limit,
            overlap_tokens=min(64, computed_limit),
            version="structure-v2",
            model_max_input_tokens=spec.model_max_input_tokens,
            special_token_budget=spec.special_token_budget,
            corpus_instruction_tokens=corpus_instruction_tokens,
            embedding_spec=spec,
        )


def normalized_chunk_text(value: str) -> str:
    """Canonical text for chunk/content hashing without altering display text."""

    return unicodedata.normalize("NFC", str(value).replace("\r\n", "\n").replace("\r", "\n"))


def deterministic_chunk_id(
    document_id: str,
    source_hash: str,
    policy_version: str,
    content_type: ContentType | str,
    section_path: Sequence[str],
    block_range: tuple[int, int],
    ordinal: int,
    normalized_text: str,
) -> str:
    """Return the full, schema-versioned SHA-256 content/provenance identifier."""

    kind = content_type.value if isinstance(content_type, ContentType) else ContentType(content_type).value
    payload = {
        "schema_version": str(policy_version),
        "document_id": str(document_id),
        "source_hash": str(source_hash),
        "chunker_version": str(policy_version),
        "content_type": kind,
        "section_path": list(section_path),
        "block_range": [int(block_range[0]), int(block_range[1])],
        "ordinal": int(ordinal),
        "normalized_text": normalized_chunk_text(normalized_text),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "chk_" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _ChunkDraft:
    text: str
    content_type: ContentType
    block_start: int
    block_end: int
    block: ParsedBlock
    table_metadata: dict[str, Any] | None = None
    equation_metadata: dict[str, Any] | None = None


class StructureAwareChunker:
    """Emit only structure-safe, model-fitting chunks for one document."""

    def __init__(self, policy: ChunkPolicy, tokenizer: EmbeddingTokenizer) -> None:
        if not isinstance(policy, ChunkPolicy):
            raise TypeError("policy must be a ChunkPolicy derived from EmbeddingSpec")
        if tokenizer is None:
            raise TypeError("tokenizer is required")
        self.policy = policy
        self.tokenizer = tokenizer
        declared_limit = getattr(tokenizer, "model_max_input_tokens", None)
        if declared_limit is not None and int(declared_limit) != policy.model_max_input_tokens:
            raise ChunkPolicyError("tokenizer model limit does not match the active EmbeddingSpec")

    def chunk(self, document: ParsedDocument) -> tuple[ChunkRecord, ...]:
        drafts: list[_ChunkDraft] = []
        ordered_blocks = sorted(enumerate(document.blocks), key=lambda item: (item[1].reading_order, item[0]))
        position = 0
        while position < len(ordered_blocks):
            index, block = ordered_blocks[position]
            if self._is_parser_heading(block):
                position += 1
                continue
            if block.content_type is ContentType.DEFINITION:
                leading_definitions, after_leading = self._definition_sequence(
                    ordered_blocks, position, block.section_path
                )
                if (
                    after_leading < len(ordered_blocks)
                    and ordered_blocks[after_leading][1].content_type is ContentType.EQUATION
                    and not self._is_parser_heading(ordered_blocks[after_leading][1])
                    and ordered_blocks[after_leading][1].section_path == block.section_path
                ):
                    equation_index, equation_block = ordered_blocks[after_leading]
                    trailing_definitions, after_equation = self._definition_sequence(
                        ordered_blocks, after_leading + 1, equation_block.section_path
                    )
                    drafts.extend(
                        self._equation_bundle_drafts(
                            document,
                            equation_block,
                            equation_index,
                            leading_definitions,
                            trailing_definitions,
                        )
                    )
                    position = after_equation
                    continue
                drafts.extend(self._drafts_for_block(document, block, index))
                position += 1
                continue
            if block.content_type is ContentType.PROSE:
                paragraphs = [(index, block)]
                position += 1
                while position < len(ordered_blocks):
                    next_index, next_block = ordered_blocks[position]
                    if (
                        next_block.content_type is not ContentType.PROSE
                        or next_block.section_path != block.section_path
                        or self._is_parser_heading(next_block)
                    ):
                        break
                    paragraphs.append((next_index, next_block))
                    position += 1
                drafts.extend(self._paragraph_group_drafts(document, paragraphs))
                continue
            if block.content_type is ContentType.EQUATION:
                trailing_definitions, after_equation = self._definition_sequence(
                    ordered_blocks, position + 1, block.section_path
                )
                drafts.extend(
                    self._equation_bundle_drafts(document, block, index, (), trailing_definitions)
                )
                position = after_equation
                continue
            drafts.extend(self._drafts_for_block(document, block, index))
            position += 1

        return tuple(self._record(document, draft, ordinal) for ordinal, draft in enumerate(drafts))

    @staticmethod
    def _is_parser_heading(block: ParsedBlock) -> bool:
        return bool(block.metadata.get("is_heading"))

    def _definition_sequence(
        self,
        ordered_blocks: Sequence[tuple[int, ParsedBlock]],
        position: int,
        section_path: tuple[str, ...],
    ) -> tuple[list[tuple[int, ParsedBlock]], int]:
        definitions: list[tuple[int, ParsedBlock]] = []
        while position < len(ordered_blocks):
            index, block = ordered_blocks[position]
            if (
                block.content_type is not ContentType.DEFINITION
                or self._is_parser_heading(block)
                or block.section_path != section_path
            ):
                break
            definitions.append((index, block))
            position += 1
        return definitions, position

    def _equation_bundle_drafts(
        self,
        document: ParsedDocument,
        equation_block: ParsedBlock,
        equation_index: int,
        leading_definitions: Sequence[tuple[int, ParsedBlock]],
        trailing_definitions: Sequence[tuple[int, ParsedBlock]],
    ) -> list[_ChunkDraft]:
        equation = equation_block.equation or EquationMetadata()
        definition_texts = tuple(
            block.text
            for _, block in (*leading_definitions, *trailing_definitions)
            if block.text.strip()
        )
        bundled_block = replace(
            equation_block,
            equation=replace(
                equation,
                variable_definitions=(
                    *equation.variable_definitions,
                    *definition_texts,
                ),
            ),
        )
        first_index = leading_definitions[0][0] if leading_definitions else equation_index
        last_index = trailing_definitions[-1][0] if trailing_definitions else equation_index
        return self._equation_drafts(document, bundled_block, first_index, last_index)

    def _drafts_for_block(
        self, document: ParsedDocument, block: ParsedBlock, block_index: int
    ) -> list[_ChunkDraft]:
        if block.content_type is ContentType.PROSE:
            return self._prose_drafts(document, block, block_index)
        if block.content_type is ContentType.TABLE:
            return self._table_drafts(document, block, block_index)
        if block.content_type is ContentType.EQUATION:
            return self._equation_drafts(document, block, block_index)
        if block.content_type is ContentType.FIGURE_CAPTION:
            figure = block.figure_metadata
            text = self._join_text(block.text, getattr(figure, "caption", None), getattr(figure, "nearby_explanation", None))
            return [self._bounded_draft(document, text, block, block_index)]
        if block.content_type is ContentType.REFERENCE:
            reference = block.reference_metadata
            text = self._join_text(block.text, getattr(reference, "citation", None), getattr(reference, "identifier", None))
            return [self._bounded_draft(document, text, block, block_index)]
        return [self._bounded_draft(document, block.text, block, block_index)]

    @staticmethod
    def _join_text(*values: str | None) -> str:
        return "\n".join(str(value).strip() for value in values if value and str(value).strip())

    def _bounded_draft(
        self, document: ParsedDocument, text: str, block: ParsedBlock, block_index: int
    ) -> _ChunkDraft:
        self._require_fit(document, text, block)
        return _ChunkDraft(text, block.content_type, block_index, block_index, block)

    def _paragraph_group_drafts(
        self, document: ParsedDocument, paragraphs: Sequence[tuple[int, ParsedBlock]]
    ) -> list[_ChunkDraft]:
        """Accumulate complete same-section paragraphs up to the soft target.

        Overlap belongs only to a single paragraph that is intrinsically too
        large.  It is never carried over from a preceding paragraph into the
        next one.
        """

        drafts: list[_ChunkDraft] = []
        current: list[tuple[int, ParsedBlock]] = []
        for paragraph in paragraphs:
            _, block = paragraph
            if not block.text.strip():
                continue
            if not self._fits(block.text, self.policy.soft_token_target):
                if current:
                    drafts.append(self._complete_paragraph_draft(document, current))
                    current = []
                drafts.extend(self._prose_drafts(document, block, paragraph[0]))
                continue
            candidate = (*current, paragraph)
            candidate_text = self._paragraph_text(candidate)
            if current and not self._fits(candidate_text, self.policy.soft_token_target):
                drafts.append(self._complete_paragraph_draft(document, current))
                current = [paragraph]
            else:
                current.append(paragraph)
        if current:
            drafts.append(self._complete_paragraph_draft(document, current))
        return drafts

    @staticmethod
    def _paragraph_text(paragraphs: Sequence[tuple[int, ParsedBlock]]) -> str:
        return "\n\n".join(block.text for _, block in paragraphs)

    def _complete_paragraph_draft(
        self, document: ParsedDocument, paragraphs: Sequence[tuple[int, ParsedBlock]]
    ) -> _ChunkDraft:
        first_index, first_block = paragraphs[0]
        last_index, last_block = paragraphs[-1]
        text = self._paragraph_text(paragraphs)
        grouped_block = replace(
            first_block,
            text=text,
            page_end=last_block.page_end or last_block.page_start or first_block.page_end,
        )
        self._require_fit(document, text, grouped_block)
        return _ChunkDraft(text, ContentType.PROSE, first_index, last_index, grouped_block)

    def _prose_drafts(
        self, document: ParsedDocument, block: ParsedBlock, block_index: int, block_end: int | None = None
    ) -> list[_ChunkDraft]:
        block_end = block_index if block_end is None else block_end
        if self._fits(block.text, self.policy.soft_token_target):
            return [_ChunkDraft(block.text, block.content_type, block_index, block_end, block)]
        words = block.text.split()
        if not words:
            return []
        chunks: list[_ChunkDraft] = []
        start = 0
        while start < len(words):
            end = start
            while end < len(words) and self._fits(" ".join(words[start : end + 1]), self.policy.soft_token_target):
                end += 1
            if end == start:
                if not self._fits(words[start], self.policy.hard_content_token_limit):
                    self._raise_too_large(document, block)
                end = start + 1
            text = " ".join(words[start:end])
            self._require_fit(document, text, block)
            chunks.append(_ChunkDraft(text, block.content_type, block_index, block_end, block))
            if end == len(words):
                break
            overlap_start = self._overlap_start(words, start, end)
            start = overlap_start if overlap_start < end else end
        return chunks

    def _overlap_start(self, words: list[str], start: int, end: int) -> int:
        overlap_start = end
        for candidate in range(end - 1, start - 1, -1):
            if self._content_token_count(" ".join(words[candidate:end])) <= self.policy.overlap_tokens:
                overlap_start = candidate
            else:
                break
        return overlap_start

    def _table_drafts(
        self, document: ParsedDocument, block: ParsedBlock, block_index: int
    ) -> list[_ChunkDraft]:
        table = block.table or TableMetadata()
        context = self._table_context(table)
        rows = table.cells or ((block.text,),)
        drafts: list[_ChunkDraft] = []
        cursor = 0
        while cursor < len(rows):
            end = cursor
            while end < len(rows) and self._fits(
                self._table_text(context, rows[cursor : end + 1]), self.policy.hard_content_token_limit
            ):
                end += 1
            if end == cursor:
                self._raise_too_large(document, block)
            selected_rows = rows[cursor:end]
            metadata = table.to_dict()
            metadata.update(
                {
                    "cells": [list(row) for row in selected_rows],
                    "row_count": len(selected_rows),
                    "row_group_is_complete": True,
                }
            )
            drafts.append(
                _ChunkDraft(
                    self._table_text(context, selected_rows),
                    ContentType.TABLE,
                    block_index,
                    block_index,
                    block,
                    table_metadata=metadata,
                )
            )
            cursor = end
        return drafts

    @staticmethod
    def _table_context(table: TableMetadata) -> tuple[str, ...]:
        return tuple(
            value
            for value in (
                table.caption,
                " | ".join(table.headers) if table.headers else None,
                " | ".join(table.units) if table.units else None,
                "\n".join(table.notes) if table.notes else None,
                table.nearby_explanation,
            )
            if value
        )

    @staticmethod
    def _table_text(context: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
        return "\n".join((*context, *(" | ".join(row) for row in rows)))

    def _equation_drafts(
        self, document: ParsedDocument, block: ParsedBlock, block_index: int, block_end: int | None = None
    ) -> list[_ChunkDraft]:
        block_end = block_index if block_end is None else block_end
        equation = block.equation or EquationMetadata()
        representation = self._join_text(block.text, equation.latex, equation.mathml, equation.parser_native, equation.plain_text)
        definitions = equation.variable_definitions
        if not definitions:
            self._require_fit(document, representation, block)
            return [
                _ChunkDraft(
                    representation,
                    ContentType.EQUATION,
                    block_index,
                    block_end,
                    block,
                    equation_metadata=equation.to_dict(),
                )
            ]
        drafts: list[_ChunkDraft] = []
        cursor = 0
        while cursor < len(definitions):
            end = cursor
            while end < len(definitions) and self._fits(
                self._join_text(representation, *definitions[cursor : end + 1]),
                self.policy.hard_content_token_limit,
            ):
                end += 1
            if end == cursor:
                self._raise_too_large(document, block)
            selected = definitions[cursor:end]
            metadata = equation.to_dict()
            metadata["variable_definitions"] = list(selected)
            drafts.append(
                _ChunkDraft(
                    self._join_text(representation, *selected),
                    ContentType.EQUATION,
                    block_index,
                    block_end,
                    block,
                    equation_metadata=metadata,
                )
            )
            cursor = end
        return drafts

    def _record(self, document: ParsedDocument, draft: _ChunkDraft, ordinal: int) -> ChunkRecord:
        content_tokens, model_tokens = self._token_counts(draft.text)
        if content_tokens > self.policy.hard_content_token_limit or model_tokens > self.policy.model_max_input_tokens:
            self._raise_too_large(document, draft.block)
        metadata = document.metadata
        content_hash = hashlib.sha256(normalized_chunk_text(draft.text).encode("utf-8")).hexdigest()
        return ChunkRecord(
            chunk_id=deterministic_chunk_id(
                document.document_id,
                document.source_hash,
                self.policy.version,
                draft.content_type,
                draft.block.section_path,
                (draft.block_start, draft.block_end),
                ordinal,
                draft.text,
            ),
            document_id=document.document_id,
            source_hash=document.source_hash,
            text=draft.text,
            content_type=draft.content_type,
            source_filename=metadata.source_filename,
            source_relative_path=metadata.source_relative_path,
            title=metadata.title,
            authors=metadata.authors,
            publication_year=metadata.publication_year,
            document_type=metadata.document_type,
            format=metadata.format,
            page_start=draft.block.page_start,
            page_end=draft.block.page_end,
            chapter=draft.block.chapter,
            section_path=draft.block.section_path,
            epub_spine_item=draft.block.epub_spine_item,
            anchor=draft.block.anchor,
            reading_order=draft.block.reading_order,
            block_range=(draft.block_start, draft.block_end),
            chunk_ordinal=ordinal,
            content_hash=content_hash,
            table_metadata=draft.table_metadata,
            equation_metadata=draft.equation_metadata,
            embedding_spec=self.policy.embedding_spec,
            parser_id=document.parser_id,
            parser_version=document.parser_version,
            parser_config_hash=document.parser_config_hash,
            chunker_version=self.policy.version,
            content_tokens=content_tokens,
            embedding_input_tokens=model_tokens,
            embedding_truncation=False,
        )

    def _fits(self, text: str, limit: int) -> bool:
        content_tokens, model_tokens = self._token_counts(text)
        return content_tokens <= limit and model_tokens <= self.policy.model_max_input_tokens

    def _require_fit(self, document: ParsedDocument, text: str, block: ParsedBlock) -> None:
        if not self._fits(text, self.policy.hard_content_token_limit):
            self._raise_too_large(document, block)

    def _raise_too_large(self, document: ParsedDocument, block: ParsedBlock) -> None:
        raise ChunkTooLargeForEmbedding(
            "indivisible semantic unit exceeds the active embedding context limit",
            document_id=document.document_id,
            block_id=block.block_id,
            content_type=block.content_type,
        )

    def _content_token_count(self, text: str) -> int:
        content_tokens, _ = self._token_counts(text)
        return content_tokens

    def _token_counts(self, text: str) -> tuple[int, int]:
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            purpose="corpus",
            truncation=False,
        )
        if isinstance(encoded, dict):
            tokens = encoded.get("input_ids")
            if tokens is None:
                raise ChunkPolicyError("tokenizer output must include input_ids")
        else:
            tokens = encoded
        model_tokens = len(tokens)
        content_tokens = model_tokens - self.policy.special_token_budget - self.policy.corpus_instruction_tokens
        if content_tokens < 0:
            raise ChunkPolicyError("tokenizer returned fewer tokens than the configured special-token budget")
        return content_tokens, model_tokens


__all__ = [
    "ChunkPolicy",
    "ChunkPolicyError",
    "ChunkTooLargeForEmbedding",
    "EmbeddingTokenizer",
    "StructureAwareChunker",
    "deterministic_chunk_id",
    "normalized_chunk_text",
]
